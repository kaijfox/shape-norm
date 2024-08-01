from ..joint import MorphModelParams, MorphModel, _getter
from ...io.dataset import PytreeDataset, FeatureDataset
from ...io import armature
from ...config import get_entries, load_calibration_data, save_calibration_data
from ...util import (
    broadcast_batch,
)
from ...pca import fit_with_center
from ...project.paths import Project

from typing import Tuple
from jaxtyping import Array, Float, Scalar, Integer
import tensorflow_probability.substrates.jax as tfp
import jax.numpy as jnp
import jax.numpy.linalg as jla
import matplotlib.pyplot as plt
import logging

from .lowrank_affine import (
    reports,
    apply_bodies,
)


def get_avg_length_ratios(
    dataset: FeatureDataset, n_spatial: int, n_bones: int
):
    bodies = (
        dataset.bodies
        if hasattr(dataset, "bodies")
        else range(dataset.n_bodies)
    )
    avg_lengths = {
        b: dataset.get_all_with_body(b)[
            ..., n_spatial : n_spatial + n_bones
        ].mean(axis=0)
        for b in bodies
    }
    ref_body = dataset.sess_bodies[dataset.ref_session]
    ratios = {
        b: jnp.log(avg_lengths[b] / avg_lengths[ref_body])
        for b in avg_lengths
        if b != ref_body
    }
    return ratios


def calibrate_base_model(dataset: FeatureDataset, config, n_dims=None):
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
    arms = armature.Armature.from_config(full_config["dataset"])
    n_bones = len(arms.bones)
    # n_feats = n_spatial + n_bones + (n_spatial * n_bones)
    # => n_spatial = (n_feats - n_bones) / (1 + n_bones)
    n_spatial = (dataset.data.shape[-1] - n_bones) // (1 + n_bones)

    # set config and save calculated values
    if config["logscale_std"] is None:
        logratios = get_avg_length_ratios(dataset, n_spatial, n_bones)
        ratio_variance = jnp.std(jnp.concatenate(list(logratios.values())))
        config["logscale_std"] = 3 * float(ratio_variance)

    # save info about calibration for plotting later
    config["calibration_data"] = dict(
        n_bones=n_bones,
        n_spatial=n_spatial,
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
    return None


class BLSParams(MorphModelParams):
    # --- static/shape params
    n_bones: int = _getter("n_bones")
    n_spatial: int = _getter("n_spatial")
    # --- hyperparams
    logscale_std: Scalar = _getter("logscale_std")
    # --- trainable params
    _logscales: Float[Array, "n_bodies-1 n_bones"] = _getter("_logscales")
    # --- specify how to split params during jitting & differentiation
    _static = MorphModelParams._static + ["n_bones", "n_spatial", "ref_body"]
    _hyper = [p for p in MorphModelParams._hyper if p != "ref_body"] + [
        "logscale_std",
    ]
    _trained = MorphModelParams._trained + ["_logscales"]

    # --- derived parameters
    @property
    def scales(self):
        s = jnp.exp(self._logscales)
        return jnp.concatenate(
            [
                s[..., : self.ref_body, :],
                jnp.ones_like(s[..., :1, :]),
                s[..., self.ref_body :, :],
            ],
            axis=-2,
        )


def get_transform(
    params: BLSParams,
) -> Tuple[Float[Array, "n_bodies n_spatial+n_bones*(n_spatial+1)"]]:

    return jnp.concatenate(
        [
            jnp.broadcast_to(
                jnp.ones(params.n_spatial)[None],
                (params.n_bodies, params.n_spatial),
            ),
            params.scales,
            jnp.broadcast_to(
                jnp.ones(params.n_spatial * params.n_bones)[None],
                (params.n_bodies, params.n_spatial * params.n_bones),
            ),
        ],
        axis=-1,
    )


def transform(
    params: BLSParams, poses: PytreeDataset
) -> Float[Array, "*#K n_feats"]:
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

    transform = get_transform(params)[poses.body_ids]  # (batch, n_feats)
    return poses.with_data(poses.data * transform)


def inverse_transform(
    params: BLSParams,
    observations: PytreeDataset,
    return_determinants: bool = False,
) -> Tuple[PytreeDataset, Float[Array, "*#K"]]:
    transform = get_transform(params)[observations.body_ids]  # (batch, n_feats)
    poses = observations.with_data(observations.data / transform)

    if return_determinants:
        logdets = jnp.log(jnp.prod(transform, axis=-1))
        return poses, logdets[observations.body_ids]
    else:
        return poses


def log_prior(params: BLSParams) -> dict:
    scales_logp = tfp.distributions.MultivariateNormalDiag(
        scale_diag=params.logscale_std * jnp.ones(params.n_bones)
    ).log_prob(params._logscales)

    return dict(
        scales=scales_logp,
    )


def init_hyperparams(
    observations: PytreeDataset,
    config: dict,
) -> BLSParams:
    # ref_keypts = observations.get_session(observations.ref_session)
    calib = config["calibration_data"]

    N, M = calib["n_spatial"], calib["n_bones"]
    return BLSParams(
        dict(
            n_bodies=observations.n_bodies,
            n_spatial=N,
            n_bones=M,
            n_feats=N + M * (N + 1),
            ref_body=observations.sess_bodies[observations.ref_session],
            logscale_std=config["logscale_std"],
        )
    )


def init(
    hyperparams: BLSParams, observations: PytreeDataset, config: dict
) -> BLSParams:
    params = hyperparams

    # Calculate offsets
    if config["init"].get("scales", True):
        ratios = get_avg_length_ratios(
            observations, params.n_spatial, params.n_bones
        )
        logscales = jnp.stack(
            [
                ratios.get(i, jnp.zeros(params.n_bodies))
                for i in range(params.n_bodies)
                if i != params.ref_body
            ]
        )
    else:
        logging.warning("Not initializing offsets.")
        logscales = jnp.zeros([params.n_bodies - 1, params.n_bones])

    params._tree.update(
        dict(
            _logscales=logscales,
        )
    )
    return params


type_name = "bone_length"
defaults = dict(logscale_std=None, init=dict(scales=True))
model = MorphModel(
    type_name=type_name,
    defaults=defaults,
    ParamClass=BLSParams,
    transform=transform,
    inverse_transform=inverse_transform,
    log_prior=log_prior,
    init_hyperparams=init_hyperparams,
    init=init,
    reports=reports,
    apply_bodies=apply_bodies,
    plot_calibration=plot_calibration,
)
