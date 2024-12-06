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
from sklearn import metrics
import numpy as np
import re
from scipy.spatial import distance
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

from .plotting import getc
from .keypoint_data import mouse_data_armature_meta


def plotting_metadata(
    params_file,
):

    plot_meta = YAML(typ="safe").load(open(params_file, "r"))
    age_pal = {k: getc(c) for k, c in plot_meta["colors"]["age"].items()}
    sim_pal = {k: getc(c) for k, c in plot_meta["colors"]["sim_age"].items()}

    kp_order = []
    kp_colors = {}
    kp_colormap = Colormap(plot_meta["colors"]["keypoint"][0])
    for kp_color in plot_meta["colors"]["keypoint"][1:]:
        for kp in kp_color[1:]:
            kp_order.append(kp)
            kp_colors[kp] = kp_colormap(kp_color[0])

    use_kpts = [
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
    keypt_ix = lambda name: use_kpts.index(name)

    _bones = jnp.array(
        [
            [keypt_ix(c), keypt_ix(mouse_data_armature_meta()[1][c])]
            for c in use_kpts
            if mouse_data_armature_meta()[1][c] is not None
        ]
    )
    arms = Armature(
        keypoint_names=bidict(zip(range(14), use_kpts)),
        bones=blscale_ls.reroot(_bones, keypt_ix("hips")),
        root=keypt_ix("hips"),
        anterior=["shldr"],
        posterior=["hips"],
    )

    # --- Organize bone names and order
    bone_names = [
        f"{arms.keypoint_names[int(c)]}-{arms.keypoint_names[int(p)]}"
        for c, p in arms.bones
    ]
    short_names = bidict(plot_meta["armature"]["short_names"])
    bones_by_short = {v: bone_names.index(k) for k, v in short_names.items()}
    bone_ix_order = [
        bone_names.index(short_names.inverse[b])
        for b in plot_meta["armature"]["order"]
    ]
    bone_groups = plot_meta["armature"]["groups"]
    bone_group_names = plot_meta["armature"]["group_names"]

    # organize joint names and order
    joint_names = plot_meta["armature"]["joint_order"]
    joint_short_names = bidict(plot_meta["armature"]["joint_short_names"])
    joint_by_short = {v: i for i, v in enumerate(joint_names)}
    joint_kp_ixs = np.array(
        [
            [k for k in joint_short_names.inverse[j].split("-")]
            for j in joint_names
        ]
    )
    joint_kp_ixs = np.array(
        [
            [
                arms.keypoint_names.inverse[k]
                for k in joint_short_names.inverse[j].split("-")
            ]
            for j in joint_names
        ]
    )

    return (
        sim_pal,
        age_pal,
        kp_order,
        kp_colors,
        kp_colormap,
        use_kpts,
        keypt_ix,
        _bones,
        arms,
        bone_names,
        short_names,
        bones_by_short,
        bone_ix_order,
        bone_groups,
        bone_group_names,
        joint_names,
        joint_short_names,
        joint_by_short,
        joint_kp_ixs,
    )


def load_ontogeny_keypoint_data(data_dir, src_sess):
    data_dir = "/Users/kaifox/projects/mph/data_explore/data"
    ont_dataset, unaligned_dataset, align_inv = full_ontogeny_data_aligned(
        Path(data_dir),
        session_filter=lambda s: re.search(r"(\d+)wk", s).group(1)
        not in ["9", "12"],
        use_keypoints=[
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
        ],
    )
    obs_keypts = {s: ont_dataset.get_session(s) for s in ont_dataset.sessions}
    unaligned = {
        s: unaligned_dataset.get_session(s) for s in unaligned_dataset.sessions
    }
    sessions = ont_dataset.sessions
    src_keypts = obs_keypts[src_sess]

    # sessions = metadata['session_slice'].keys()
    # slices = metadata['session_slice']
    # src_sess = '24wk_m0'
    # obs_keypts = {s: gt_obs.keypts[slices[s]].reshape([-1, 14, 3]) for s in sessions}
    # unaligned = {s: inverse_saggital_align(obs_keypts[s], metadata['centroid'][s], metadata['rotation'][s]) for s in sessions}
    # src_kpts = obs_keypts[src_sess]

    age_groups = defaultdict(list)
    for s in sessions:
        age_groups[s.split("_")[0].rstrip("wk")].append(s)
    age_groups = dict(age_groups)
    ages = sorted(list(age_groups.keys()), key=lambda x: int(x))

    print("Age groups:")
    pprint(age_groups)

    return (
        ont_dataset,
        unaligned_dataset,
        align_inv,
        obs_keypts,
        unaligned,
        sessions,
        src_keypts,
        age_groups,
        ages,
    )


def ontogeny_data_ethograms(ethogram_path, params_file):
    plot_meta = YAML(typ="safe").load(open(params_file, "r"))
    bhv_labels = jl.load(ethogram_path)
    bhv_pal = {k: getc(c) for k, c in plot_meta["colors"]["bhv"].items()}
    bhv_keys = plot_meta["behavior"]["keys"]
    bhv_names = plot_meta["behavior"]["names"]
    bhv_order = list(bhv_keys.keys())
    bhv_masks = {k: bhv_labels["masks"][n] for k, n in bhv_keys.items()}
    rear_mask = bhv_labels["masks"]["absolute_reer"]
    loco_mask = bhv_labels["masks"]["locomotion"]

    return (
        bhv_pal,
        bhv_keys,
        bhv_names,
        bhv_order,
        bhv_masks,
        rear_mask,
        loco_mask,
        bhv_labels,
    )
