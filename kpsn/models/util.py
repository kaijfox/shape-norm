from .joint import MorphModelParams, MorphModel
from ..io.dataset import FeatureDataset, PytreeDataset
from ..io.alignment import align
from ..io.features import reduce_to_features
from .instantiation import get_model

import jax.numpy as jnp
import jax.numpy.linalg as jla


def apply_bodies(
    morph_model: MorphModel,
    params: MorphModelParams,
    observations: FeatureDataset,
    target_bodies: dict,
):
    """Transform each session in a dataset to a targer body."""
    # transform to integer-indexed sessions required by model functions
    pt_obs = PytreeDataset.from_pythonic(observations)
    target_bod_arr = jnp.array(
        [
            (
                # session in `observations` and in `target_bodies`
                observations._body_names.inverse[
                    target_bodies[observations._session_names[i]]
                ]
                if observations._session_names[i] in target_bodies
                # no target set, retain the same body
                else observations._body_names.inverse[
                    observations.sess_bodies[observations._session_names[i]]
                ]
            )
            for i in range(observations.n_sessions)
        ]
    )
    mapped = morph_model.apply_bodies(params, pt_obs, target_bod_arr)
    return FeatureDataset.from_pytree(
        mapped, observations._body_names, observations._session_names
    )


def reconst_errs(kpts_a, kpts_b):
    """
    Parameters
    ----------
    kpts_a, kpts_b : array shape (n_frame, n_keypoints, n_spatial)
        Array of keypoints, such as a session from a KeypointDataset.
    """
    return jla.norm(kpts_a - kpts_b, axis=-1).mean(axis=0)
