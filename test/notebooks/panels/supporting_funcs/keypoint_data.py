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
