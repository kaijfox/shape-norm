from .joint import MorphModelParams, MorphModel
from ..io.dataset import FeatureDataset, PytreeDataset, KeypointDataset
from ..io.alignment import align
from ..io.features import reduce_to_features, inflate
from ..io.utils import stack_dict
from .instantiation import get_model
from ..fitting.methods import prepare_dataset

from typing import Union
import jax.numpy as jnp
import jax.numpy.linalg as jla
import numpy as np
import tqdm


def apply_bodies(
    morph_model: MorphModel,
    params: MorphModelParams,
    observations: FeatureDataset,
    target_bodies: dict,
):
    """Transform each session in a dataset to a target body."""
    # transform to integer-indexed sessions required by model functions
    pt_obs = PytreeDataset.from_pythonic(observations)
    target_bod_arr = np.zeros(max(observations._session_names.keys()) + 1, int)
    for sess_id, sess_name in observations._session_names.items():
        # session in `observations` and in `target_bodies`
        if sess_name in target_bodies:
            tgt_bod = target_bodies[sess_name]
        # no target set, retain the same body
        else:
            tgt_bod = observations.sess_bodies[sess_name]
        target_bod_arr[sess_id] = observations._body_names.inverse[tgt_bod]
    target_bod_arr = jnp.array(target_bod_arr)

    mapped = morph_model.apply_bodies(params, pt_obs, target_bod_arr)
    return FeatureDataset.from_pytree(
        mapped, observations._body_names, observations._session_names
    )


def induced_reference_keypoints(
    dataset: Union[KeypointDataset, FeatureDataset],
    config: dict,
    morph_model: MorphModel,
    params: Union[MorphModelParams, list, tuple],
    to_body=None,
    include_reference: bool = False,
):
    """Map reference session keypoints each or a specific body under different
    morph parameters.

    Returns
    -------
    ref_kpts : dict
        Dictionary of keypoints for each body, with the same structure as
        `dataset`."""

    # set up dataset with n_bodies copy of reference session
    session = dataset.ref_session
    if to_body is None:
        ref_body = dataset.sess_bodies[session]
        ref_frames = dataset.get_session(session)
        all_bodies = {
            b
            for b in dataset._body_names.values()
            if (b != ref_body or include_reference)
        }
        new_data, slices = stack_dict(
            {
                f"s-{ref_body}": ref_frames[
                    :0
                ],  # overriden if `include_reference`
                **{f"s-{b}": ref_frames for b in all_bodies},
            }
        )
        new_bodies = {
            f"s-{ref_body}": ref_body,  # overriden if `include_reference`
            **{f"s-{b}": ref_body for b in all_bodies},
        }
        dataset = dataset.with_sessions(
            new_data,
            slices,
            new_bodies,
            f"s-{ref_body}",
            _body_names=dataset._body_names,
        )
        # drop empty reference session if not included
        if not include_reference:
            dataset = dataset.session_subset(
                [f"s-{b}" for b in all_bodies], bad_ref_ok=True
            )
        # in this mode we should map each session to the corresponding body for
        # which it is named
        bodymap = {f"s-{b}": b for b in all_bodies}
    # set up dataset with only (one copy of) the reference session
    else:
        session = dataset.ref_session
        dataset = dataset.session_subset([session])
        # in this mode we should map `session` onto to_body
        bodymap = {session: to_body}

    was_list = True
    if not isinstance(params, (list, tuple)):
        was_list = False
        params = [params]

    # morph by each set of parameters
    if dataset.ndim > 1:
        dataset, _ = prepare_dataset(dataset, config)
    morphed = [
        inflate(
            apply_bodies(morph_model, p, dataset, bodymap),
            config["features"],
        )
        for p in params
    ]

    # extract data by body if we mapped onto multiple bodies and match inpput
    # format of params (list / not list)
    by_params = lambda s: (
        [m.get_session(s) for m in morphed]
        if was_list
        else morphed[0].get_session(s)
    )
    if to_body is None:
        return {b: by_params(f"s-{b}") for b in all_bodies}
    else:
        return by_params(session)


def induced_keypoint_distances(
    dataset: KeypointDataset,
    config: dict,
    morph_model: MorphModel,
    params_a: MorphModelParams,
    body_a: str,
    params_b: MorphModelParams = None,
    body_b: str = None,
):

    if params_b is None:
        params_b = params_a
    if body_b is None:
        body_b = body_a

    if body_a is None:
        kpts = induced_reference_keypoints(
            dataset, config, morph_model, [params_a, params_b], to_body=None
        )
        return {
            body: reconst_errs(kpts[0][body], kpts[1][body])
            for body in kpts_a.keys()
        }
    else:
        kpts_a = induced_reference_keypoints(
            dataset, config, morph_model, params_a, to_body=body_a
        )
        kpts_b = induced_reference_keypoints(
            dataset, config, morph_model, params_b, to_body=body_b
        )
        return reconst_errs(kpts_a, kpts_b)


def _deprecated_induced_keypoint_distances(
    dataset: KeypointDataset,
    config: dict,
    morph_model: MorphModel,
    params_a: MorphModelParams,
    body_a: str,
    params_b: MorphModelParams = None,
    body_b: str = None,
):
    """Compute the distance between reference session trasnformed onto separate
    bodies under different morph parameters.

    Returns
    -------
    reconst_errs : array shape (n_keypoints,)
        Mean (in frames) distance between keypoints of the two bodies."""

    if params_b is None:
        params_b = params_a
    if body_b is None:
        body_b = body_a

    # set up dataset with n_bodies copy of reference session
    session = dataset.ref_session
    if body_a is None:
        ref_body = dataset.sess_bodies[session]
        ref_frames = dataset.get_session(session)
        all_bodies = {b for b in dataset._body_names.values() if b != ref_body}
        new_data, slices = stack_dict(
            {
                f"s-{ref_body}": ref_frames[:0],
                **{f"s-{b}": ref_frames for b in all_bodies},
            }
        )
        new_bodies = {
            f"s-{ref_body}": ref_body,
            **{f"s-{b}": ref_body for b in all_bodies},
        }
        dataset = dataset.with_sessions(
            new_data,
            slices,
            new_bodies,
            f"s-{ref_body}",
            _body_names=dataset._body_names,
        )
        dataset = dataset.session_subset(
            [f"s-{b}" for b in all_bodies], bad_ref_ok=True
        )
        # in this mode we should map each session to the corresponding body for
        # which it is named (so we get identical data mapped to each known body)
        bodymap = lambda to: {f"s-{b}": b for b in all_bodies}
    # set up dataset with only (one copy of) the reference session
    else:
        session = dataset.ref_session
        dataset = dataset.session_subset([session])
        # in this mode we should map `session` onto body_a or body_b, as given
        # by `to` when this function is called later
        bodymap = lambda to: {session: to}

    prepped, _ = prepare_dataset(dataset, config)
    morphed_a = apply_bodies(morph_model, params_a, prepped, bodymap(body_a))
    morphed_b = apply_bodies(morph_model, params_b, prepped, bodymap(body_b))
    kpts_a = inflate(morphed_a, config["features"])
    kpts_b = inflate(morphed_b, config["features"])

    if body_a is None:
        return {
            b: reconst_errs(
                kpts_a.get_session(f"s-{b}"), kpts_b.get_session(f"s-{b}")
            )
            for b in all_bodies
        }
    else:
        return reconst_errs(
            kpts_a.get_session(session), kpts_b.get_session(session)
        )


def reconst_errs(kpts_a, kpts_b, average=True):
    """
    Parameters
    ----------
    kpts_a, kpts_b : array shape (n_frame, n_keypoints, n_spatial)
        Array of keypoints, such as a session from a KeypointDataset.
    """
    if average:
        return jla.norm(kpts_a - kpts_b, axis=-1).mean(axis=0)
    else:
        return jla.norm(kpts_a - kpts_b, axis=-1)


def _optional_pbar(iterable, flag):
    if flag is False:
        return iterable
    if isinstance(flag, (str, int)) and not isinstance(flag, bool):
        return tqdm.tqdm(iterable, desc=str(flag))
    return tqdm.tqdm(iterable)
