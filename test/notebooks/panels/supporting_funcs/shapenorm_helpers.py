from shape_norm.project.paths import Project, create_model
from shape_norm.io import loaders
from shape_norm.models import joint
from shape_norm import config
from shape_norm.models import (
    instantiation,
    setup,
    pose,
    morph,
    util as model_util,
)
from shape_norm import fitting
from shape_norm.fitting import em
from shape_norm.io import alignment, features
from shape_norm.io.armature import Armature
from shape_norm.fitting import methods
from shape_norm.fitting import scans
from shape_norm.pca import fit_with_center, CenteredPCA, PCAData
from shape_norm.io.dataset import PytreeDataset
from shape_norm.io.dataset_refactor import (
    Dataset,
    SessionMetadata,
    StackedArrayMetadata,
)
from shape_norm.models.morph.lowrank_affine import LRAParams, model as lra_model
from shape_norm import pca
from blscale_loader import loader, linear_skeletal as blscale_ls
import os, sys, shutil
import numpy as np
from pprint import pprint
from matplotlib import colors as mpl_col
import jax.numpy as jnp
import itertools as iit
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
import matplotlib as mpl
import seaborn as sns
from shape_norm import viz
from shape_norm.viz import styles
from scipy.spatial.distance import jensenshannon
from tensorflow_probability.substrates import jax as tfp
from ruamel.yaml import YAML
from bidict import bidict
import matplotlib.pyplot as plt
from collections import defaultdict
import os, re
import tqdm
from cmap import Colormap
import joblib as jl

from pathlib import Path
import logging

logging.getLogger().setLevel(logging.INFO)


def single_session_dataset(data, body, session, meta: SessionMetadata):
    session_id = meta.session_id(session)
    body_id = meta.body_id(body)
    return Dataset(
        data=data,
        session_meta=meta.update(
            session_bodies={**meta._session_bodies, session_id: body_id}
        ),
        stack_meta=StackedArrayMetadata(
            slices={session_id: (0, len(data))}, length=len(data)
        ),
    )


# model manipulation
def lra_anchor_poses(params: LRAParams, magnitudes=90):
    """
    Parameters:
    params (LRAParams):
        Parameters used for calculating anchor poses.
    magnitudes (float, numpy.ndarray):
        The magnitude value used in calculating anchor poses. If a float, the
        same magnitude is used for all anchors. If an array, the magnitude
        for each dimension is specified separately. Not applied to the centroid
        anchor.

    Returns, (array, shape (n_bodies, n_dims, n_modes + 1)):
    numpy.ndarray:
        The anchor poses in the canonical pose space.
    """

    L = params.n_dims

    if isinstance(magnitudes, (int, float)):
        magnitudes = magnitudes * np.ones(L)

    # (n_bodies, n_dims, 1)
    anchors = (params.offset + params.offset_updates).reshape(
        params.n_bodies, -1, 1
    )
    # (n_bodies, n_dims, n_modes + 1)
    anchors = np.concatenate(
        [
            anchors,
            anchors
            + magnitudes[None, None] * (params.modes + params.mode_updates),
        ],
        axis=-1,
    )

    return anchors


def anchor_keypoints(
    params: LRAParams,
    align_meta: dict,
    magnitudes=90,
    _inflate=None,
    config=None,
    session_meta: SessionMetadata = None,
):
    """
    Calculate and return the anchor poses from a model in the original keypoint
    space.

    This function computes anchor poses based on the provided parameters and
    alignment metadata, inflates the poses based on a given configuration (if
    any), and then applies inverse alignment scaling to return the poses in
    their original keypoint space.

    Parameters
    ----------
    params, dict:
        Parameters used for calculating anchor poses.
    session_meta, SessionMetadata, optional:
        Session metadata object (accessible via dataset.session_meta) for
        mapping body indices to session indices for rescaling. Required if
        align_meta is provided. If provided, first index of output is over
        session ids, not body ids.
    align_meta, dict:
        Metadata for alignment, optionally including scaling factors under the
        key `scale`.
    magnitudes, int, optional:
        The magnitude value used in calculating anchor poses. Defaults to 90.
    _inflate, function, optional:
        A custom function to inflate anchor poses. If None, a default inflation
        based on the 'features' configuration is used.
    config, dict, optional:
        Configuration dictionary that, if provided, is used to inflate the
        anchor poses using the '_inflate' function. It should contain a
        'features' key if'_inflate' is not provided.

    Returns
    -------
    numpy.ndarray, shape (n_bodies or n_sessions, n_anchor, n_keypt, n_spatial):
        The anchor poses in the original keypoint space after applying inverse
        alignment scaling. If session_meta is provided, the first index is over
        sessions instead of bodies.



    """
    if _inflate is None:
        _inflate = lambda x: features.inflate(x, config["features"])
    anchor_poses = lra_anchor_poses(
        params, magnitudes
    )  # (n_bodies, n_feat, n_anch)
    inflated = _inflate(anchor_poses.transpose(0, 2, 1))
    if session_meta is not None:
        inflated = jnp.array(
            [
                inflated[session_meta.session_body_id(i)]
                for i in range(len(session_meta._session_ids))
            ]
        )
        if "scale" in align_meta:
            inflated = alignment._inverse_align_scales(
                inflated, align_meta["scale"][:, None], stacked=True
            )
    return inflated


def unalign_scales(align_meta, dataset, split_meta=None, base_dataset=None):
    """
    Invert scaling alignment, potentially on a split dataset

    Parameters
    ----------
    align_meta : dict
        The alignment metadata, containing an array under key 'scale' whose
        $i$th element is the scale factor to be applied to sesssion with id $i$.
    dataset : Dataset
        The dataset to invert the scaling on.
    split_meta : tuple, optional
        Tuple whos second element (index 1) maps original session names (in
        `dataset`) to the names of sessions in `dataset`.
    base_dataset : Dataset, optional
        Dataset with session ids that match the indices of
        `align_meta['scale']`. Can be provided if `dataset` is a split dataset
        where alignment was already performed before splitting.
    """
    if base_dataset is not None:
        scan_align_meta = {"scale": np.array([-1.0 for _ in dataset.sessions])}
        for src_sess, splits in split_meta[1].items():
            for s in splits:
                scan_align_meta["scale"][dataset.session_id(s)] = align_meta[
                    "scale"
                ][base_dataset.session_id(src_sess)]
        align_meta = scan_align_meta
    return (
        alignment._inverse_align_scales(dataset, align_meta["scale"]),
        align_meta,
    )


def anchor_magnitudes(
    dataset: Dataset, params: LRAParams, q=0.9, canonicalized=False
):
    """
    Sample the `q`th percentile in absolute value from distributions of anchor
    pose coordinates.

    Parameters
    ----------
    dataset : Dataset or jnp.array
        Feature data, in canonical pose space if `canonicalized` is True. If an
        array, it is assumed to be in canonical pose space.
    model : MorphModel
    params : LRAParams
        Parameters of the model.
    q : float, default 0.9
        The quantile to sample from.
    canonicalized : bool, default False
        Whether the dataset is already in canonical pose space (magnitudes are
        not session- or body-dependent)

    Returns
    -------
    magnitudes : array, shape (n_modes,)
        The selected magnitude in each morph mode.
    """
    if canonicalized:
        canonical = dataset.data
    elif isinstance(dataset, jnp.ndarray):
        canonical = dataset
    else:
        canonical = model_util.apply_bodies(
            lra_model,
            params,
            dataset,
            {
                s: dataset.session_body_name(dataset.ref_session)
                for s in dataset.sessions
            },
        ).data

    coords = canonical @ params.modes  # (n_pts, n_modes)
    qix = jnp.argsort(coords, axis=0)[int(coords.shape[0] * q)]  # argquantile
    selected_coord = coords[qix, jnp.arange(coords.shape[1])]
    return selected_coord


def merge_morphs(
    params: LRAParams,
    merged_session_meta: SessionMetadata,
    split_session_meta: SessionMetadata,
    split_meta: dict,
):
    """
    Merge morph parameters from a split dataset into a single dataset.

    Parameters
    ----------
    params : LRAParams
        The morph parameters.
    merged_session_meta, split_session_metadata : SessionMetadata
        Session metadata object (accessible via dataset.session_meta) for
        identifying body ids.
    split_meta : dict
        Metadata for the split dataset, containing a mapping original body names
        to sessions in the split dataset.

    Returns
    -------
    numpy.ndarray, shape (n_sessions, n_modes, n_dims)
        The merged morph parameters.
    """
    body_inv = {
        b: [split_session_meta.session_body_name(s) for s in split_meta[b]]
        for b in split_meta
    }
    new_nbod = len(merged_session_meta._body_ids)
    new_ref = None
    mode_upds = np.zeros((new_nbod, params.n_feats, params.n_dims))
    ofs_upds = np.zeros((new_nbod, params.n_feats))
    for i in range(new_nbod):
        merged_bod = merged_session_meta.body_name(i)
        i_mode_upds = []
        i_ofs_upds = []
        for split_bod in body_inv[merged_bod]:
            j = split_session_meta.body_id(split_bod)
            i_mode_upds.append(params.mode_updates[j])
            i_ofs_upds.append(params.offset_updates[j])
            if j == params.ref_body:
                new_ref = i
        mode_upds[i] = np.mean(i_mode_upds, axis=0)
        ofs_upds[i] = np.mean(i_ofs_upds, axis=0)

    return LRAParams(
        {
            **params._tree,
            **dict(
                n_bodies=new_nbod,
                ref_body=new_ref,
                _mode_updates=jnp.array(mode_upds),
                _offset_updates=jnp.array(ofs_upds),
            ),
        }
    )


def naive_model_params(ref_pc, upds_best, ref_center, alt_center, ref_body_id):
    return LRAParams(
        dict(
            n_bodies=len(alt_center) + 1,
            n_feats=ref_pc.shape[0],
            ref_body=ref_body_id,
            n_dims=ref_pc.shape[1],
            prior_mode="distance",
            # --- hyperparams
            upd_var_modes=1.0,
            upd_var_ofs=1.0,
            dist_var=1.0,
            modes=jnp.array(ref_pc),
            offset=jnp.array(ref_center),
            # --- trainable params
            _mode_updates=jnp.array(
                np.insert(upds_best, ref_body_id, 0, axis=0)
            ),
            _offset_updates=jnp.array(
                np.insert(alt_center - ref_center[None], ref_body_id, 0, axis=0)
            ),
        )
    )


def naive_model_anchors(
    canonicalized_dict, session_params, config, magnitudes=None
):
    """Compute anchor poses for a collection of naive models fit to session
    half pairs.

    Parameters
    ----------
    canonicalized_dict : dict
        Dictionary of canonicalized keypoints, with keys <session>.<part>
    session_params : dict
        Dictionary of session morph parameters, with keys <session>
    config : dict
        Configuration used to inflate from features to keypoints

    Returns
    -------
    anchors : dict[str, array (2,n_anchor, n_kpt, 3)]
        Dictionary of anchor poses, with keys <session>, containing anchor poses
        for each parameter set.
    """
    if magnitudes is None:
        magnitudes = {
            s: anchor_magnitudes(
                jnp.concatenate(
                    [
                        canonicalized_dict[f"{s}.0"],
                        canonicalized_dict[f"{s}.1"],
                    ]
                ),
                session_params[s],
                q=0.1,
            )
            for s in session_params
        }
    if not isinstance(magnitudes, dict):
        magnitudes = {s: magnitudes for s in session_params}
    return {
        s: anchor_keypoints(
            session_params[s],
            {},
            magnitudes=magnitudes[s],
            config=config,
        )
        for s in session_params
    }


def create_canonicalized_and_raw(align_meta, model, cfg, split_meta):
    def canonicalized_and_raw(
        dataset,
        params,
        base_dataset=None,
        rescale=True,
        scale_to=None,
        _align_meta=None,
    ):

        if _align_meta is None:
            _align_meta = align_meta
        if hasattr(params, "morph"):
            params = params.morph

        canonicalized = model_util.apply_bodies(
            model.morph,
            params,
            dataset,
            {
                s: dataset.session_body_name(dataset.ref_session)
                for s in dataset.sessions
            },
        )
        canonicalized = features.inflate(canonicalized, cfg["features"])
        raw = features.inflate(dataset, cfg["features"])

        if rescale:
            raw, split_align_meta = unalign_scales(
                _align_meta, raw, split_meta, base_dataset
            )
            align_to = split_align_meta["scale"][dataset.session_id(scale_to)]
            canonical_meta = {"scale": np.full(dataset.n_sessions, align_to)}
            canonicalized, _ = unalign_scales(canonical_meta, canonicalized)
        return canonicalized, raw, (split_align_meta if rescale else {})

    return canonicalized_and_raw


def create_canonicalized_raw_bls(arms):
    _bone_lengths = lambda kpts, arms: jnp.linalg.norm(
        kpts[..., arms.bones[:, 0], :] - kpts[..., arms.bones[:, 1], :], axis=-1
    )

    def canonicalized_raw_bls(
        canonicalized, raw, sessions, parts="01", group_bodies=True, sep="."
    ):
        groups = (
            {s: re.search(r"(\d+)bod", s).group(1) for s in sessions}
            if group_bodies
            else {s: s for s in sessions}
        )
        bls_canon = {g: [] for g in set(groups.values())}
        bls_raw = {g: [] for g in set(groups.values())}
        for s in sessions:
            k = groups[s]
            bls_canon[k].extend(
                [
                    _bone_lengths(
                        canonicalized.get_session(f"{s}{sep}{part}"), arms
                    )
                    for part in parts
                ]
            )
            bls_raw[k].extend(
                [
                    _bone_lengths(raw.get_session(f"{s}{sep}{part}"), arms)
                    for part in parts
                ]
            )
        bls_canon = {k: jnp.concatenate(v) for k, v in bls_canon.items()}
        bls_raw = {k: jnp.concatenate(v) for k, v in bls_raw.items()}
        return bls_canon, bls_raw, groups

    return canonicalized_raw_bls
