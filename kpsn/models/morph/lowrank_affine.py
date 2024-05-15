from ..joint import MorphModelParams, MorphModel, _getter
from ...io.dataset_refactor import Dataset
from ...config import get_entries, load_calibration_data, save_calibration_data
from ...util import (
    broadcast_batch,
)
from ...pca import fit_with_center
from ...project.paths import Project

from typing import Tuple, Union
from jaxtyping import Array, Float, Scalar, Integer
import tensorflow_probability.substrates.jax as tfp
import jax.numpy as jnp
import jax.numpy.linalg as jla
import matplotlib.pyplot as plt
import logging


type_name = "lowrank_affine"
defaults = dict(
    n_dims=None,
    upd_var_modes=None,
    upd_var_ofs=None,
    dist_var=None,
    prior_mode="params",
    init=dict(seed=0, offsets=True),
    calibration=dict(tgt_variance=0.9),
)


def calibrate_base_model(dataset: Dataset, config, n_dims=None):
    """
    Calibrate a morph model from a dataset.

    Parameters
    ----------
    dataset : FeatureDataset
    config : dict
        Full model and project config.
    """

    # Fit PCA model to morph dataset and select number of dimensions to explain
    # target amount of variance
    full_config = config
    config = config["morph"]
    init_pts = dataset.get_session(dataset.ref_session)
    arr = fit_with_center(init_pts)._pcadata.variances()
    scree = jnp.cumsum(arr) / arr.sum()
    if n_dims is None:
        selected_ix = jnp.argmax(scree > config["calibration"]["tgt_variance"])
    else:
        selected_ix = n_dims - 1

    if config["prior_mode"] == "params":
        # variance of each component in feature space
        mean_vars = jnp.var(
            jnp.array(
                [dataset.get_session(s).mean(axis=0) for s in dataset.sessions]
            ),
            axis=0,
        )
        config["upd_var_ofs"] = float(mean_vars.mean())
        config["upd_var_modes"] = 0.1
    else:
        config["dist_var"] = 1.0

    # set config and save calculated calibration
    config["n_dims"] = int(selected_ix) + 1
    config["calibration_data"] = dict(
        variance_explained=scree,
        n_dims=selected_ix,
        mean_variances_by_dim=mean_vars,
    )

    return full_config


def plot_calibration(config: dict, colors):
    """Plot calibration data.

    Parameters
    ----------
    project : Project
    config : dict
        Full project config.
    """
    from ...viz.general import scree

    config = config["morph"]
    calibration_data = config["calibration_data"]
    selected_ix = calibration_data["n_dims"]
    cumsum = calibration_data["variance_explained"]

    fig, ax = scree(cumsum, selected_ix, config["calibration"]["tgt_variance"])
    ax.set_title("Morph model dimension selection")

    return fig


# conversion from string parameter modes to int
PriorModes = dict(params=0, distance=1)


class LRAParams(MorphModelParams):
    # --- static/shape params
    n_dims: int = _getter("n_dims")
    prior_mode: int = _getter("prior_mode")
    # --- hyperparams
    upd_var_modes: Scalar = _getter("upd_var_modes")
    upd_var_ofs: Scalar = _getter("upd_var_ofs")
    dist_var: Scalar = _getter("dist_var")
    modes: Float[Array, "n_feats n_dims"] = _getter("modes")
    offset: Float[Array, "n_feats"] = _getter("offset")
    # --- trainable params
    _mode_updates: Float[Array, "n_bodies n_feats n_dims"] = _getter(
        "_mode_updates"
    )
    _offset_updates: Float[Array, "n_bodies n_feats"] = _getter(
        "_offset_updates"
    )

    # --- specify how to split params during jitting & differentiation
    _static = MorphModelParams._static + ["n_dims", "prior_mode"]
    _hyper = MorphModelParams._hyper + [
        "upd_var_modes",
        "upd_var_ofs",
        "dist_var",
        "modes",
        "offset",
    ]
    _trained = MorphModelParams._trained + ["_mode_updates", "_offset_updates"]

    # --- derived parameters
    @property
    def mode_updates(self):
        return self._mode_updates.at[..., self.ref_body, :, :].set(0)

    @property
    def offset_updates(self):
        return self._offset_updates.at[..., self.ref_body, :].set(0)


def get_transform(
    params: LRAParams,
) -> Tuple[
    Float[Array, "n_bodies n_feats n_feats"],
    Float[Array, "n_feats"],
    Float[Array, "n_bodies n_feats"],
]:
    mode_pinv = params.modes.T[None]  # (1, n_dims, n_feats)
    I = jnp.eye(mode_pinv.shape[-1])
    linear_parts = I + params.mode_updates @ mode_pinv

    pop_offset = params.offset
    sess_offsets = params.offset[None] + params.offset_updates

    return linear_parts, pop_offset, sess_offsets


def transform(params: LRAParams, poses: Dataset) -> Float[Array, "*#K n_feats"]:
    """
    Calculate the linear transform defining the morph model
    using a given set of parameters.

    Parameters
    ----------
    sess_ids: jax array, integer
        Session indices ($n_bodies$, in parameter matrices) to apply to each
        batch element.

    Returns
    -------
    morphed_poses:
        Array of poses under the morph transform.
    """

    linear_parts, pop_offset, sess_offsets = get_transform(params)

    # broadcast transform arrays
    batch_shape = poses.data.shape[:-1]
    linear_parts = linear_parts[
        poses.stack_body_ids
    ]  # (batch, n_feats, n_feats)
    pop_offset = broadcast_batch(pop_offset, batch_shape)  # (batch, n_feats)
    sess_offsets = sess_offsets[poses.stack_body_ids]  # (batch, n_feats)

    # apply transform
    centered = (poses.data - pop_offset)[..., None]  # (batch, n_feats, 1)
    updated = (linear_parts @ centered)[..., 0]  # (batch, n_feats)
    uncentered = updated + sess_offsets  # (batch, n_feats)

    return poses.update(data=uncentered)


def inverse_transform(
    params: LRAParams,
    observations: Dataset,
    return_determinants: bool = False,
) -> Tuple[Dataset, Float[Array, "*#K"]]:
    linear_parts, pop_offset, sess_offsets = get_transform(params)
    linear_invs = jla.inv(linear_parts)

    # broadcast transform arrays
    batch_shape = observations.data.shape[:-1]
    linear_inv = linear_invs[
        observations.stack_body_ids
    ]  # (batch, n_feats, n_feats)
    pop_offset = broadcast_batch(pop_offset, batch_shape)  # (batch, n_feats)
    sess_offsets = sess_offsets[observations.stack_body_ids]  # (batch, n_feats)

    # apply transform
    centered = (observations.data - sess_offsets)[
        ..., None
    ]  # (batch, n_feats, 1)
    updated = (linear_inv @ centered)[..., 0]  # (batch, n_feats)
    uncentered = updated + pop_offset  # (batch, n_feats)

    poses = observations.update(data=uncentered)
    if return_determinants:
        linear_invs_logdet = jnp.log(jla.det(linear_invs))
        linear_inv_logdet = linear_invs_logdet[observations.stack_body_ids]
        return poses, linear_inv_logdet
    else:
        return poses


def log_prior(params: LRAParams, observations: Dataset, poses: Dataset) -> dict:
    if params.prior_mode == PriorModes["params"]:
        offset_logp = tfp.distributions.MultivariateNormalDiag(
            scale_diag=params.upd_var_ofs * jnp.ones(params.n_feats)
        ).log_prob(params.offset_updates)

        flat_updates = params.mode_updates.reshape(
            [params.n_bodies, params.n_feats * params.n_dims]
        )
        mode_logp = tfp.distributions.MultivariateNormalDiag(
            scale_diag=params.upd_var_modes
            * jnp.ones(params.n_feats * params.n_dims)
        ).log_prob(flat_updates)
        dist_logp = jnp.array(0.0)
    else:
        offset_logp = jnp.array(0.0)
        mode_logp = jnp.array(0.0)
        induced_dist = ((observations.data - poses.data) ** 2).mean()
        dist_logp = tfp.distributions.Normal(
            loc=0.0, scale=params.dist_var
        ).log_prob(induced_dist)

    return dict(offset=offset_logp, mode=mode_logp, induced_dist=dist_logp)


def reports(params: LRAParams) -> dict:
    return dict()


def init_hyperparams(
    observations: Dataset,
    config: dict,
) -> LRAParams:

    ref_keypts = observations.get_session(observations.ref_session)
    pcs = fit_with_center(ref_keypts)

    # if prior mode given as a string, convert to int
    prior_mode = config.get("prior_mode", "params")
    if prior_mode not in PriorModes:
        raise ValueError(
            f"Invalid prior mode for lowank_affine morph: {prior_mode}. "
            f"Options are {repr(list(PriorModes.keys()))}."
        )
    prior_mode = PriorModes[prior_mode]

    return LRAParams(
        dict(
            n_bodies=observations.n_bodies,
            n_feats=observations.data.shape[-1],
            ref_body=observations.session_body_id(observations.ref_session),
            modes=pcs._pcadata.pcs()[: config["n_dims"], :].T,
            offset=pcs._center,
            prior_mode=prior_mode,
            dist_var=config.get("dist_var", 1.0),
            **get_entries(
                config,
                [
                    "n_dims",
                    "upd_var_modes",
                    "upd_var_ofs",
                ],
            ),
        )
    )


def init(
    hyperparams: LRAParams, observations: Dataset, config: dict
) -> LRAParams:
    params = hyperparams

    # Calculate offsets
    if config["init"].get("offsets", True):
        offset_updates = jnp.stack(
            [
                (observations.get_all_with_body(i) - params.offset).mean(axis=0)
                for i in range(params.n_bodies)
            ]
        )
    else:
        logging.warning("Not initializing offsets.")
        offset_updates = jnp.zeros([params.n_bodies, params.n_feats])

    mode_updates = jnp.zeros([params.n_bodies, params.n_feats, params.n_dims])

    params._tree.update(
        dict(
            _offset_updates=offset_updates,
            _mode_updates=mode_updates,
        )
    )
    return params


def mode_components(
    params: LRAParams, poses: Float[Array, "*#K n_feats"]
) -> Tuple[
    Float[Array, "*#K n_dims"],
    Float[Array, "*#K n_feats n_dims"],
    Float[Array, "*#K n_feats-n_dims"],
]:
    """
    Returns:
        coords:
            Coordinates of poses along the morph modes.
        components:
            Components of poses along the subspace of each morph mode.
        complement:
            Component of poses in the complement of the span of the morph
            modes.
    """
    dim_expand = (None,) * (len(poses.shape) - 1)
    modes = params.modes[dim_expand]
    coords = (poses[..., None] * modes).sum(axis=1)  # (*#K, n_dims)
    components = coords[..., None, :] * modes  # (*#K, n_feats, n_dims)
    complement = poses - components.sum(axis=-1)
    return coords, components, complement


def apply_bodies(
    params: LRAParams,
    observations: Dataset,
    target_bodies: dict[Union[str, int], Union[str, int]],
):
    """
    Morph dataset so that each session has a given body.

    Parameters
    ----------
    target_bodies: array, integer
        Array of body indices to assign to each session (note that in a
        Dataset, sessions are indexed by integers not strings, so this is
        an array instead of a dictionary.)
    """
    poses: Dataset = inverse_transform(params, observations)
    # form new dataset with new bodies assigned to each session
    tgt_dataset = poses.update(
        session_meta=observations.session_meta.update(
            session_bodies=target_bodies
        )
    )
    # map to observation space using the new assignment of bodies
    return transform(params, tgt_dataset)
    # tgt_dataset = Dataset(
    #     poses.data,
    #     observations.slices,
    #     target_bodies,
    #     ref_session=observations.ref_session,
    #     body_ids=new_body_ids,
    #     session_ids=observations.stack_session_ids,
    #     bodies_inv=tuple(
    #         tuple(jnp.where(target_bodies == b)[0])
    #         for b in range(observations.n_bodies)
    #     ),
    # )


model = MorphModel(
    type_name=type_name,
    defaults=defaults,
    ParamClass=LRAParams,
    transform=transform,
    inverse_transform=inverse_transform,
    log_prior=log_prior,
    init_hyperparams=init_hyperparams,
    init=init,
    reports=reports,
    apply_bodies=apply_bodies,
    plot_calibration=plot_calibration,
)
