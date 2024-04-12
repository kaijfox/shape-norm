from typing import NamedTuple, Tuple, Protocol, Callable
from jaxtyping import Scalar, Float, Array, Integer
from functools import partial
from fnmatch import fnmatch
from pathlib import Path
import jax.tree_util as pt
import jax.random as jr
import jax.numpy as jnp
import joblib as jl

import numpy as np
import functools
import logging
import optax
import tqdm
import time
import jax

from ..models import pose
from ..io.dataset_refactor import Dataset
from ..models.joint import JointModel, JointModelParams
from ..logging import ArrayTrace, _keystr
from ..io.utils import split_sessions


def fit_model(model, dataset, config):
    """Fit a model to a dataset.

    Parameters
    ----------
    model : JointModel
    dataset : FeatureDataset
    config : dict
        Config dictionary for fitting.
    """
    pass


def calibrate_base_model(dataset, config):
    """
    Calibrate a morph model from a dataset.

    Parameters
    ----------
    config : dict
        Full model and project config.
    """
    return config


def _pytree_sum(tree):
    return pt.tree_reduce(lambda x, y: x.sum() + y.sum(), tree)


def _save_checkpoint(save_dir, contents):
    if save_dir is None:
        return
    checkpoint_file = save_dir / f"checkpoint.p"
    save_dir.mkdir(exist_ok=True)
    jl.dump(contents, checkpoint_file)
    return contents


def _check_should_stop_early(loss_hist, step_i, tol, stop_window):
    """
    Check for median decrease in loss that is worse than `tol`.

    Args:
        loss_hist: np.ndarray, shape: (N,)
            Last `N` observations of loss.
        tol: Scalar
            Required average improvement to continue.
    """
    if tol is not None and step_i > stop_window:
        loss_hist = loss_hist[step_i - stop_window : step_i + 1]
        diff = np.diff(loss_hist)
        median = np.median(diff)
        return median > -tol
    return False


def _path_to_name(model, param_path):
    """
    Parameters are exposed to jitted functions as a tuple
        ((pose_param_0, pose_param_0, ...), (morph_param_0, morph_param_1, ...))
    and so pytree paths for the params look like [IntKey(0), IntKey(1)]. This
    function converts paths of this form to strings like "pose/pose_param_0".
    """
    param_group = ["pose", "morph"][param_path[0].idx]
    param_name = "..."
    param_name = getattr(model, param_group).ParamClass._trained[
        param_path[1].idx
    ]
    return param_group + "/" + param_name


def _mask_gradients_by_path(model, grads, blacklist, verbose=True):
    if verbose:
        pt.tree_map_with_path(
            lambda pth, grad: (
                logging.info(
                    f"Locking param: {_path_to_name(model, pth), grad.shape}"
                )
                if any(fnmatch(_path_to_name(model, pth), p) for p in blacklist)
                # else None
                else logging.info(
                    f"Fitting param: {_path_to_name(model, pth), grad.shape}"
                )
            ),
            grads,
        )
    return pt.tree_map_with_path(
        lambda pth, grad: (
            jnp.zeros_like(grad)
            if any(fnmatch(_path_to_name(model, pth), p) for p in blacklist)
            else grad
        ),
        grads,
    )


def _estep(
    model: JointModel,
    observations: Dataset,
    estimated_params: JointModelParams,
) -> JointModelParams:
    est_poses = model.morph.inverse_transform(
        estimated_params.morph, observations
    )
    aux_pdf = model.pose.aux_distribution(estimated_params.pose, est_poses)

    return aux_pdf


def _mstep_objective(
    model: JointModel,
    observations: Dataset,
    params: JointModelParams,
    aux_pdf: Float[Array, "n_samp n_discrete"],
) -> Scalar:
    """
    Calculate objective for M-step to maximize.
    """

    poses, jacobian_logdets = model.morph.inverse_transform(
        params.morph, observations, return_determinants=True
    )
    pose_probs = model.pose.pose_logprob(params.pose, poses)
    jacobian_logdets = jacobian_logdets[..., None]  # (n_samp, n_discrete)
    keypt_probs = pose_probs + jacobian_logdets

    dataset_prob = (keypt_probs * aux_pdf).sum()

    morph_prior = model.morph.log_prior(params.morph, observations, poses)
    posespace_prior = model.pose.log_prior(params.pose, observations, poses)

    aux_entropy = -jnp.where(
        aux_pdf > 1e-12, aux_pdf * jnp.log(aux_pdf), 0
    ).sum()
    elbo = dataset_prob + aux_entropy

    # return morph_prior
    return {
        "objectives": {
            "dataset": dataset_prob,
            "morph": morph_prior,
            "pose": posespace_prior,
        },
        "aux": {"logjac": jacobian_logdets.mean(), "elbo": elbo},
    }


def _mstep_loss(model: JointModel, use_priors: bool = True) -> Scalar:
    def step_loss_(
        observations: Dataset,
        static_params: dict,
        hyper_params: dict,
        trained_params: optax.Params,
        aux_pdf: Float[Array, "n_samp n_discrete"],
    ):
        params = JointModelParams.from_types(
            model, static_params, hyper_params, trained_params
        )
        scalars = _mstep_objective(model, observations, params, aux_pdf)
        if use_priors:
            loss = -_pytree_sum(scalars["objectives"])
        else:
            logging.log("Ignored priors.")
            loss = -scalars["objectives"]["dataset"]
        objectives = {**scalars["objectives"], **scalars["aux"]}
        return loss, objectives

    return step_loss_


def lr_const(step_i, lr, n_samp, **kw):
    return lr / n_samp


def lr_exp(step_i, n_samp, lr, hl, min=0, **kw):
    return (lr / n_samp) * 2 ** (-step_i / hl) + min / n_samp


rates = dict(const=lr_const, exp=lr_exp)


def construct_jitted_mstep(
    model: JointModel,
    optimizer: optax.GradientTransformation,
    update_max: float,
    update_blacklist: list = None,
    use_priors: bool = True,
):
    loss_func = _mstep_loss(model, use_priors)

    @partial(jax.jit, static_argnums=(2, 3))
    def step(
        opt_state,
        observations_arr,
        observations_meta,
        static_params,
        hyper_params,
        trained_params,
        aux_pdf,
    ):
        observations = Dataset({"data": observations_arr, **observations_meta})
        (loss_value, objectives), grads = jax.value_and_grad(
            loss_func, argnums=3, has_aux=True
        )(observations, static_params, hyper_params, trained_params, aux_pdf)

        if update_max is not None:
            grads = pt.tree_map(
                (lambda a: a.clip(-update_max, update_max)), grads
            )

        if update_blacklist is not None:
            grads = _mask_gradients_by_path(model, grads, update_blacklist)

        updates, opt_state = optimizer.update(grads, opt_state, trained_params)
        trained_params = optax.apply_updates(trained_params, updates)
        return trained_params, opt_state, (loss_value, objectives)

    return step


def construct_jitted_estep(model: JointModel):
    @partial(jax.jit, static_argnums=(1, 2))
    def step(
        observations_arr,
        observations_meta,
        static_params,
        hyper_params,
        trained_params,
    ):
        estimated_params = JointModelParams.from_types(
            model, static_params, hyper_params, trained_params
        )
        observations = Dataset({"data": observations_arr, **observations_meta})
        return _estep(model, observations, estimated_params)

    return step


def _mstep(
    init_params: tuple,
    static_params: tuple,
    hyper_params: tuple,
    aux_pdf: Float[Array, "Nt L"],
    observations: Dataset,
    jitted_step: Callable,
    opt_state,
    session_names: tuple,
    config: dict,
    batch_seed: int = 29,
    trace_params=False,
    log_every=-1,
) -> Tuple[Float[Array, "n_steps"], JointModelParams, dict, ArrayTrace, dict]:
    """
    Parameters
    ----------
    tol: float > 0
        Stop iterations if average improvement over `stop_window` is
        not better than `tol`.
    stop_window: int
        Number of iterations over which to assess early stopping.
    """

    # ----- Set up variables for iteration

    step = jitted_step
    n_steps = config["n_steps"]
    batch_size = config["batch_size"]
    curr_trained = init_params
    loss_hist = np.full([n_steps], np.nan, dtype=np.float32)

    if trace_params:
        param_trace = ArrayTrace(n_steps)
    else:
        param_trace = None

    if batch_size is not None:
        batch_rkey_seed = jr.PRNGKey(batch_seed)
        generate_batch = observations.batch_generator(
            batch_size, replace=False, session_names=session_names
        )

    # ---- Run M-step iterations

    for step_i in range(n_steps):
        if batch_size is not None:
            batch_rkey_seed, step_obs, ixs = generate_batch(batch_rkey_seed)
            step_aux = aux_pdf[ixs]
        else:
            step_obs = observations
            step_aux = aux_pdf

        curr_trained, opt_state, (loss_value, objectives) = step(
            opt_state,
            *step_obs.serialize(),
            static_params,
            hyper_params,
            curr_trained,
            step_aux,
        )
        loss_hist[step_i] = float(loss_value)
        if trace_params:
            param_trace.record(curr_trained, step_i)

        if (log_every > 0) and (not step_i % log_every):
            logging.info(f"Step {step_i} : loss = {loss_value}")

        # evaluate early stopping
        if _check_should_stop_early(
            loss_hist, step_i, config["tol"], config["stop_window"]
        ):
            loss_hist = loss_hist[: step_i + 1]
            break

    loss_hist = jnp.array(loss_hist)
    return (
        loss_hist,
        curr_trained,
        objectives,
        param_trace,
        opt_state,
    )


def _initialize_or_continue_metadata(
    meta,
    config,
    n_steps,
    init_params,
    init_opt_state,
    return_param_hist,
    return_mstep_losses,
    return_reports,
):
    # meta is None if we are starting from scratch
    if meta is None:
        meta = {
            "loss": jnp.full([n_steps], jnp.nan),
            "gd_step": jnp.full([n_steps], jnp.nan),
            "mstep_length": jnp.full([n_steps], jnp.nan),
            "walltime": jnp.full([n_steps], jnp.nan),
            "opt_state": init_opt_state,
        }
        if return_mstep_losses:
            meta["mstep_losses"] = jnp.full(
                [n_steps, config["mstep"]["n_steps"]], jnp.nan
            )
        if return_param_hist is True:
            meta["param_hist"] = [init_params]
        elif return_param_hist == "trace":
            meta["param_hist"] = ArrayTrace(n_steps + 1)
            meta["param_hist"].record(init_params, 0)
        elif return_param_hist == "mstep":
            meta["param_hist"] = ArrayTrace(n_steps)
        if return_reports:
            meta["reports"] = ArrayTrace(n_steps)

    # extend metadata to potentially longer n_steps if n_steps has been adjusted
    # since the metadata objects were created
    if meta is not None and len(meta["loss"]) < n_steps:
        old_n = len(meta["loss"])
        meta_new = {
            "loss": jnp.concatenate(
                [meta["loss"], jnp.full([n_steps - old_n], jnp.nan)]
            ),
            "gd_step": jnp.concatenate(
                [meta["gd_step"], jnp.full([n_steps - old_n], jnp.nan)]
            ),
            "mstep_length": jnp.concatenate(
                [meta["mstep_length"], jnp.full([n_steps - old_n], jnp.nan)]
            ),
            "walltime": jnp.concatenate(
                [meta["walltime"], jnp.full([n_steps - old_n], jnp.nan)]
            ),
            "opt_state": meta["opt_state"],
        }
        if return_mstep_losses:
            meta_new["mstep_losses"] = jnp.concatenate(
                [
                    meta["mstep_losses"],
                    jnp.full(
                        [n_steps - old_n, config["mstep"]["n_steps"]], jnp.nan
                    ),
                ]
            )
        if return_param_hist is True:
            meta_new["param_hist"] = meta["param_hist"]
        elif return_param_hist == "trace":
            meta_new["param_hist"] = ArrayTrace(n_steps + 1)
            meta_new["param_hist"].initialize(meta["param_hist"][0])
            meta_new["param_hist"].record(
                meta["param_hist"].as_dict(), slice(0, old_n + 1)
            )
        elif return_param_hist == "mstep":
            meta_new["param_hist"] = ArrayTrace(n_steps)
            meta_new["param_hist"].initialize(meta["param_hist"][0])
            meta_new["param_hist"].record(
                meta["param_hist"].as_dict(), slice(0, old_n + 1)
            )
        if return_reports:
            meta_new["reports"] = ArrayTrace(n_steps)
            meta_new["reports"].initialize(meta["reports"][0])
            meta_new["reports"].record(
                meta["reports"].as_dict(), slice(0, old_n)
            )
        meta = meta_new

    return meta


def iterate_em(
    model: JointModel,
    init_params: JointModelParams,
    observations: Dataset,
    config: dict,
    meta: dict = None,
    first_step: int = 0,
    checkpoint_dir: Path = None,
    checkpoint_every: int = 10,
    checkpoint_extra=dict(),
    log_every: int = -1,
    progress=False,
    return_mstep_losses=True,
    return_param_hist="trace",
    return_reports=True,
) -> Tuple[
    Float[Array, "n_steps"],
    JointModelParams,
    Float[Array, "n_steps mstep_n_steps"],
    ArrayTrace,
]:
    """
    Perform EM on a JointModel.
    """

    n_steps = config["n_steps"]

    static, hyper, curr_trained = init_params.by_type()

    # set up learning rate schedule
    learning_rate = config["learning_rate"]
    if isinstance(learning_rate, int) or isinstance(learning_rate, float):
        learning_rate = dict(kind="const", lr=learning_rate)

    if config["scale_lr"]:
        if config["mstep"]["batch_size"] is not None:
            step_data_size = config["mstep"]["batch_size"]
        elif config["batch_size"] is not None:
            step_data_size = config["batch_size"]
        else:
            step_data_size = len(observations)
        logging.info(
            "Adjusting learning rate:"
            f"{learning_rate['lr']} -> {learning_rate['lr'] / step_data_size}"
        )
    logging.info(f"Loading LR schedule: {learning_rate['kind']}")

    lr_sched = functools.partial(
        rates[learning_rate["kind"]],
        n_steps=n_steps,
        n_samp=step_data_size,
        **learning_rate,
    )

    optimizer = optax.inject_hyperparams(optax.adam)(
        learning_rate=learning_rate["lr"]
    )
    opt_state = optimizer.init(curr_trained)

    # set up metadata dictionary if not provided
    meta = _initialize_or_continue_metadata(
        meta,
        config,
        n_steps,
        curr_trained,
        opt_state,
        return_param_hist,
        return_mstep_losses,
        return_reports,
    )

    jitted_mstep = construct_jitted_mstep(
        model,
        optimizer,
        config["mstep"]["update_max"],
        config["update_blacklist"],
        config["use_priors"],
    )
    jitted_estep = construct_jitted_estep(model)

    # create functions used each step: batch generation / checkpointing
    if config["batch_size"] is not None:
        batch_rkey_seed = jr.PRNGKey(config["batch_seed"])
        generate_batch = observations.batch_generator(
            config["batch_size"],
            replace=False,
            session_names=observations.ordered_session_names(),
        )

    save_with_status = lambda status: _save_checkpoint(
        checkpoint_dir,
        dict(
            params=curr_params,
            meta=meta,
            step=step_i,
            status=status,
            **checkpoint_extra,
        ),
    )
    curr_params = JointModelParams.from_types(
        model, static, hyper, curr_trained
    )

    step_iter = (
        range(first_step, n_steps)
        if not progress
        else tqdm.trange(first_step, n_steps)
    )
    gd_step = 0 if first_step == 0 else meta["gd_step"][first_step]
    walltime = 0 if first_step == 0 else meta["walltime"][first_step]

    for step_i in step_iter:
        step_start_time = time.time()
        aux_pdf = jitted_estep(
            *observations.serialize(),
            static,
            hyper,
            curr_trained,
        )

        if config["batch_size"] is not None:
            batch_rkey_seed, step_obs, ixs = generate_batch(batch_rkey_seed)
            step_aux = aux_pdf[ixs]
        else:
            step_obs = observations
            step_aux = aux_pdf

        if config["mstep"]["reinit_opt"]:
            meta["opt_state"] = optimizer.init(curr_trained)
        meta["opt_state"].hyperparams["learning_rate"] = lr_sched(step_i)
        (
            loss_hist_mstep,
            trained_params_mstep,
            mstep_end_objective,
            mstep_param_trace,
            meta["opt_state"],
        ) = _mstep(
            init_params=curr_trained,
            static_params=static,
            hyper_params=hyper,
            aux_pdf=step_aux,
            observations=step_obs,
            jitted_step=jitted_mstep,
            opt_state=meta["opt_state"],
            session_names=observations.ordered_session_names(),
            batch_seed=(config["mstep"]["batch_seed"] + step_i),
            trace_params=return_param_hist == "mstep",
            config=config["mstep"],
        )

        mstep_len = len(loss_hist_mstep)
        meta["loss"] = meta["loss"].at[step_i].set(loss_hist_mstep[-1])
        meta["mstep_length"] = meta["mstep_length"].at[step_i].set(mstep_len)
        meta["gd_step"] = (
            meta["gd_step"].at[step_i].set(gd_step := gd_step + mstep_len)
        )
        curr_trained = trained_params_mstep
        if return_mstep_losses:
            meta["mstep_losses"] = (
                meta["mstep_losses"].at[step_i, :mstep_len].set(loss_hist_mstep)
            )
        if return_param_hist is True:
            meta["param_hist"].append(curr_trained)
        elif return_param_hist == "trace":
            meta["param_hist"].record(curr_trained, step_i + 1)
        elif return_param_hist == "mstep":
            meta["param_hist"].record(mstep_param_trace.as_dict(), step_i)

        curr_params = JointModelParams.from_types(
            model, static, hyper, curr_trained
        )

        if return_reports:
            aux_reports = {}
            if config["full_dataset_objectives"]:
                aux_reports["dataset_logprob"] = _mstep_objective(
                    model,
                    observations,
                    curr_params,
                    aux_pdf=jitted_estep(
                        *observations.serialize(),
                        static,
                        hyper,
                        curr_trained,
                    ),
                )["objectives"]["dataset"]
            meta["reports"].record(
                dict(
                    logprobs=mstep_end_objective,
                    lr=jnp.array(lr_sched(step_i)),
                    **aux_reports,
                ),
                step_i,
            )

        if (log_every > 0) and (not step_i % log_every):
            logging.info(f"Step {step_i} : loss = {meta['loss'][step_i]}")

        if (checkpoint_every > 0) and (step_i % checkpoint_every == 0):
            save_with_status("in_progress")

        # evaluate early stopping and divergence
        converged = _check_should_stop_early(
            meta["loss"], step_i, config["tol"], config["stop_window"]
        )
        if converged:
            meta["loss"] = meta["loss"][: step_i + 1]
            logging.info("Stopping due to early convergence or divergence.")
            save_with_status("finished.early_stop")
            break
        if not jnp.isfinite(meta["loss"][step_i]):
            meta["loss"] = meta["loss"][: step_i + 1]
            logging.warning("Stopping, diverged.")
            save_with_status("finished.diverged")
            break

        meta["walltime"] = (
            meta["walltime"]
            .at[step_i]
            .set(walltime := (walltime + (time.time() - step_start_time)))
        )

    final_ckpt = save_with_status("finished")
    return final_ckpt
