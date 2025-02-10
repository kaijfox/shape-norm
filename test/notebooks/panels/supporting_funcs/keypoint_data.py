import jax.random as jr

jr.KeyArray = jr.PRNGKey
import shape_norm

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
from shape_norm.io.dataset_refactor import Dataset, SessionMetadata
from shape_norm.models.morph.lowrank_affine import LRAParams, model as lra_model
from shape_norm.viz import styles

from blscale_loader import loader, linear_skeletal as blscale_ls

from collections import defaultdict
from shape_norm.viz import util as vu
from scipy import spatial
import itertools as iit
import matplotlib.colors as mpl_col
from sklearn.linear_model import LogisticRegression
import matplotlib as mpl
import numpy as np
import re
import jax
import os
import jax.numpy as jnp
from pprint import pprint
import numpy.linalg as la
from ruamel.yaml import YAML
import scipy.stats
from pathlib import Path
from cmap import Colormap
from bidict import bidict
import matplotlib as mpl
import joblib as jl
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import itertools as iit
import pandas as pd
import seaborn as sns


_bone_lengths = lambda kpts, arms: jnp.linalg.norm(
    kpts[..., arms.bones[:, 0], :] - kpts[..., arms.bones[:, 1], :], axis=-1
)
cos = lambda a, b, **k: (a * b).sum(**k) / jnp.sqrt(
    (a * a).sum(**k) * (b * b).sum(**k)
)
_cos_angle = lambda a, b, **k: np.arccos(np.clip(cos(a, b, **k), -1, 1))
_joint_angles = lambda kpts, joint_kp_ixs: np.array(
    [
        (180 / np.pi)
        * _cos_angle(
            kpts[:, c, :] - kpts[:, p, :],
            kpts[:, g, :] - kpts[:, p, :],
            axis=-1,
        )
        for c, p, g in joint_kp_ixs
    ]
).T


def mouse_data_armature_meta():
    """Organize metadata on keypoints and skeleton of mouse data."""
    names = [
        "shldr",
        "back",
        "hips",
        "t_base",
        "t_tip",
        "head",
        "l_ear",
        "r_ear",
        "nose",
        "lr_knee",
        "lr_foot",
        "rr_knee",
        "rr_foot",
        "lf_foot",
        "rf_foot",
    ]
    parents = dict(
        zip(
            names,
            [
                "back",
                "hips",
                None,
                "hips",
                "t_base",
                "shldr",
                "head",
                "head",
                "head",
                "hips",
                "lr_knee",
                "hips",
                "rr_knee",
                "shldr",
                "shldr",
            ],
        )
    )
    keypt_ix = lambda name: names.index(name)
    bones = np.array(
        [(keypt_ix(c), keypt_ix(p)) for c, p in parents.items() if p]
    )
    root = keypt_ix("shldr")
    bones = blscale_ls.reroot(bones, root)
    bones = bones[np.argsort(bones[:, 0])]
    return names, parents, bones, root


def raw_data_armature_meta_no_tail():
    """Organize metadata on keypoints and skeleton of mouse data."""
    names = [
        "shldr",
        "back",
        "hips",
        "t_base",
        "head",
        "l_ear",
        "r_ear",
        "nose",
        "lr_knee",
        "lr_foot",
        "rr_knee",
        "rr_foot",
        "lf_foot",
        "rf_foot",
    ]
    parents = dict(
        zip(
            names,
            [
                "back",
                "hips",
                None,
                "hips",
                "shldr",
                "head",
                "head",
                "head",
                "hips",
                "lr_knee",
                "hips",
                "rr_knee",
                "shldr",
                "shldr",
            ],
        )
    )
    keypt_ix = lambda name: names.index(name)
    bones = np.array(
        [(keypt_ix(c), keypt_ix(p)) for c, p in parents.items() if p]
    )
    root = keypt_ix("shldr")
    bones = blscale_ls.reroot(bones, root)
    bones = bones[np.argsort(bones[:, 0])]
    return names, parents, bones, root


def mouse_data_full_metadata():

    keypt_names = [
        "shldr",
        "back",
        "hips",
        "t_base",
        "head",
        "l_ear",
        "r_ear",
        "nose",
        "lr_knee",
        "lr_foot",
        "rr_knee",
        "rr_foot",
        "lf_foot",
        "rf_foot",
    ]
    keypt_ix = lambda k: keypt_names.index(k)
    _names, _parents, _bones, _root = raw_data_armature_meta_no_tail()
    arms = Armature(_names, _bones, _root, anterior=[], posterior=[])

    # bone_names = [f'{keypt_names[i]}-{keypt_names[j]}' for i, j in armature.bones]
    bone_names = [
        "hi back",  #  'back-shldr',
        "lo back",  #  'hips-back',
        "rump",  #  't_base-hips',
        "neck",  #  'head-shldr',
        "left ear",  #  'l_ear-head',
        "right ear",  #  'r_ear-head',
        "head",  #  'nose-head',
        "left hindlimb",  #  'lr_knee-hips',
        "left hind foot",  #  'lr_foot-lr_knee',
        "right hindlimb",  #  'rr_knee-hips',
        "right hind foot",  #  'rr_foot-rr_knee',
        "left forelimb",  #  'lf_foot-shldr',
        "right forelimb",  #  'rf_foot-shldr'
    ]
    keypt_parent = lambda ix, armature=arms: armature.bones[
        armature.bones[:, 0] == ix, 1
    ][0]
    has_parent = lambda ix, armature=arms: np.any(armature.bones[:, 0] == ix)
    bone_ix = lambda keypt_ix, armature=arms: np.where(
        armature.bones[:, 0] == keypt_ix
    )[0][0]

    joints = np.array(
        [
            (child, parent, keypt_parent(parent))
            for child, parent in arms.bones
            if has_parent(parent)
        ]
    )
    joint_names = [
        f"{keypt_names[c]}-{keypt_names[p]}-{keypt_names[g]}"
        for c, p, g in joints
    ]
    joint_names = [
        "back",  #  'hips-back-shldr',
        "hips",  #  't_base-hips-back',
        "left ear",  #  'l_ear-head-shldr',
        "right ear",  #  'r_ear-head-shldr',
        "head",  #  'nose-head-shldr',
        "left hindlimb",  #  'lr_knee-hips-back',
        "left hind foot",  #  'lr_foot-lr_knee-hips',
        "right hindlimb",  #  'rr_knee-hips-back',
        "right hind foot",  #  'rr_foot-rr_knee-hips'
    ]

    short_names = [keypt_names[i] for i, j in arms.bones]
    full_joint_angles_func = (
        lambda kpts, joint_ix, joints: 180
        / np.pi
        * np.arccos(
            np.clip(
                cos(
                    kpts[:, joints[joint_ix, 0], :]
                    - kpts[:, joints[joint_ix, 1], :],
                    kpts[:, joints[joint_ix, 2], :]
                    - kpts[:, joints[joint_ix, 1], :],
                ),
                -1,
                1,
            )
        )
    )
    joint_angles = lambda kpts, joint_ix: full_joint_angles_func(
        kpts, joint_ix, joints
    )

    return (
        keypt_names,
        keypt_ix,
        arms,
        bone_names,
        keypt_parent,
        has_parent,
        bone_ix,
        joints,
        joint_names,
        short_names,
        joint_angles,
    )


_with_match = lambda test, pattern, f: (
    f(m) if (m := re.search(pattern, test)) else None
)
_name_func = lambda path, *a: _with_match(
    path,
    r"(?:/.*)+/\d{2}_\d{2}_\d{2}_(\d+wk_m\d+)\.npy",
    lambda m: f"{m.group(1)}",
)


def full_ontogeny_data(data_dir, session_filter=None, ref_session=None):
    _sources = dict(
        map(lambda x: (_name_func(str(x)),) * 2, data_dir.glob(f"*.npy"))
    )
    if session_filter is not None:
        _sources = {k: v for k, v in _sources.items() if session_filter(k)}
    ont_keypoints, _ = loader.from_sources_dict(
        data_dir,
        _sources,
        extension=".npy",
        name_func=_name_func,
    )
    names, parents, _, _ = mouse_data_armature_meta()

    if ref_session is None:
        ref_session = list(ont_keypoints.keys())[0]

    project_config_kws = {
        "session_names": ont_keypoints.keys(),
        "bodies": {s: f"b-{s}" for s in ont_keypoints},
        "ref_session": ref_session,
        "keypoint_names": names,
        "keypoint_parents": parents,
    }

    return project_config_kws, ont_keypoints


def full_ontogeny_data_aligned(
    data_dir, session_filter=None, use_keypoints=None, scale_ntt=False
):
    project_config_kws, ont_keypoints = full_ontogeny_data(
        data_dir, session_filter
    )
    if use_keypoints is None:
        use_keypoints = project_config_kws["keypoint_names"]
    dataset = loaders.arrays.from_arrays(
        ont_keypoints,
        dict(
            type="arrays",
            subsample=None,
            subsample_to=None,
            ref_session=project_config_kws["ref_session"],
            sessions={
                s: dict(body=b) for s, b in project_config_kws["bodies"].items()
            },
            keypoint_names=project_config_kws["keypoint_names"],
            use_keypoints=use_keypoints,
            anterior=None,
            posterior=None,
            invert_axes=None,
        ),
    )
    aligned, align_inv = alignment.align(
        dataset,
        dict(
            type="sagittal",
            origin="hips",
            anterior="head",
            rescale=scale_ntt,
            rescale_mode="session",
        ),
    )
    return aligned, dataset, align_inv


def load_all_mouse_sessions(data_root, age_whitelist=None):
    # data_root = '../../../../data_explore/data'

    # Create mapping of session names (ie 3wk_m0) to keypoint data paths

    modata_npy_name_func = lambda path, *a: (
        match.group(1)
        if (
            (match := re.search(r"\d{2}_\d{2}_\d{2}_(\d+wk_m\d+)\.npy", path))
            is not None
        )
        else None
    )
    session_paths = dict(
        filter(
            lambda x: x[0] is not None,
            [
                (modata_npy_name_func(m), str(Path(data_root) / m))
                for m in os.listdir(data_root)
            ],
        )
    )
    # throw out 9-12 week
    session_ages = {s: s.split("_")[0].strip("wk") for s in session_paths}
    ages = set(session_ages.values())
    if age_whitelist is not None:
        ages = list(filter(lambda x: x in age_whitelist, ages))
    session_paths = {
        s: p for s, p in session_paths.items() if session_ages[s] in ages
    }
    session_bodies = {s: f"body-{session_ages[s]}wk" for s in session_paths}
    names, parents, _, _ = mouse_data_armature_meta()

    age_dates = {
        "3": "10_11_22",
        "5": "10_24_22",
        "7": "10_24_22",
        "24": "11_03_22",
        "52": "11_03_22",
        "72": "10_14_22",
    }

    print("sessions:", session_paths.keys())

    sesssion_kps = {s: np.load(p) for s, p in session_paths.items()}

    project_config_kws = {
        "session_names": list(sesssion_kps.keys()),
        "bodies": session_bodies,
        "keypoint_names": names,
        "keypoint_parents": parents,
    }

    return project_config_kws, sesssion_kps


def blscale_dataset(source_dict, scale_dict, data_dir, ref_session=None):
    """
    Create a dataset using `blscale_loader`

    Parameters
    ----------
    source_dict : str or Path
        Path to the source dictionary
    scale_dict : str or Path
        Path to the scale dictionary

    Returns
    -------
    project_config_kws : dict
        Keyword args for `loaders.arrays.setup_project_config`
    coords : dict
        Dictionary mapping session names to (unaligned) keypoint data
    scales : dict
        Dictionary mapping session names to scale bone-wise scale factors and
        uniform scale factors.
    """
    session_names = loader.find_sessions(
        sources=source_dict,
        scales=scale_dict,
        ext=".npy",
        name_func=_name_func,
    )
    session_bodies = {s: f"body-{s}" for s in session_names}

    # skeleton without dropped keypoints
    names, parents, bones, root = mouse_data_armature_meta()

    coords, _, _ = loader.external_scale(
        data_dir,
        sources=source_dict,
        scales=scale_dict,
        bones=bones,
        root_keypoint_ix=root,
        ext=".npy",
        name_func=_name_func,
    )
    _, scales = loader._align_scales_sources(
        source_dict, scale_dict, ext=".npy", name_func=_name_func
    )

    ref_session = list(coords.keys())[0]

    project_config_kws = {
        "session_names": list(coords.keys()),
        "bodies": session_bodies,
        "ref_session": ref_session,
        "keypoint_names": names,
        "keypoint_parents": parents,
    }

    return project_config_kws, coords, scales


def blscale_dataset_aligned(
    source_dict,
    scale_dict,
    data_dir,
    ref_session,
    use_keypoints=None,
    scale_ntt=False,
):

    project_config_kws, obs_keypoints, _ = blscale_dataset(
        source_dict, scale_dict, data_dir, ref_session=ref_session
    )
    if use_keypoints is None:
        use_keypoints = project_config_kws["keypoint_names"]
    dataset = loaders.arrays.from_arrays(
        obs_keypoints,
        dict(
            type="arrays",
            subsample=None,
            subsample_to=None,
            ref_session=project_config_kws["ref_session"],
            sessions={
                s: dict(body=b) for s, b in project_config_kws["bodies"].items()
            },
            keypoint_names=project_config_kws["keypoint_names"],
            use_keypoints=use_keypoints,
            anterior=None,
            posterior=None,
            invert_axes=None,
        ),
    )
    aligned, align_inv = alignment.align(
        dataset,
        dict(
            type="sagittal",
            origin="hips",
            anterior="head",
            rescale=scale_ntt,
            rescale_mode="session",
        ),
    )
    return aligned, dataset, align_inv
