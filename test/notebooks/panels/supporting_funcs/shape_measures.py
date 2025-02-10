import jax.random as jr

jr.KeyArray = jr.PRNGKey
import numpy as np
import numpy.linalg as la
from scipy.signal import savgol_filter
import matplotlib.colors as mpl_col

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
import numpy as np
import re
from functools import reduce
import jax
from cmap import Colormap
import jax.numpy as jnp
from pprint import pprint
from ruamel.yaml import YAML
from bidict import bidict
import scipy.stats
from sklearn.linear_model import LinearRegression
from pathlib import Path
import matplotlib as mpl
import joblib as jl
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import itertools as iit
import pandas as pd
import seaborn as sns


# calculus and statistics
val_and_deriv = lambda x, w=5: (
    x,
    savgol_filter(x, w, polyorder=1, deriv=1, axis=0),
)
vel_acc = lambda x, w=5: val_and_deriv(  # x, shape (t, ..., spatial)
    la.norm(savgol_filter(x, w, polyorder=1, deriv=1, axis=0), axis=-1), w=w
)
cos = lambda a, b: np.clip(
    (a * b).sum(axis=-1) / (la.norm(a, axis=-1) * la.norm(b, axis=-1)), -1, 1
)
cov = lambda x, y: (x * y).mean(axis=-1) - x.mean(axis=-1) * y.mean(axis=-1)
corr = lambda a, b: cov(a, b) / np.sqrt(cov(a, a) * cov(b, b))
coef = lambda a, b: cov(a, b) / cov(a, a)
windowed_cov = lambda x, y, w: cov(
    *np.lib.stride_tricks.sliding_window_view(
        np.pad(
            np.stack([x, y]),
            ((0, 0), (int(np.ceil((w - 1) / 2)), int(np.floor((w - 1) / 2)))),
        ),
        window_shape=w,
        axis=1,
    )
)

# skeleton functions
bone_locations = lambda keypts, armature: np.stack(
    [
        keypts[..., armature.bones[i, 1], :]
        - keypts[..., armature.bones[i, 0], :]
        for i in range(len(armature.bones))
    ],
    axis=-2,
)
bone_lengths = lambda keypts, armature: la.norm(
    bone_locations(keypts, armature), axis=-1
)
elevation = (
    lambda arr: (
        np.arccos(cos(arr, arr * np.array([1, 1, 0])[None]))
        * np.sign(arr[:, 2])
    )
    * 180
    / np.pi
)
rotation = (
    lambda arr: (
        np.arccos(cos(arr, arr * np.array([1, 0, 1])[None]))
        * np.sign(arr[:, 1])
    )
    * 180
    / np.pi
)
short_bone_names = lambda armature: [
    armature.keypoint_names[int(i)] for i, j in armature.bones
]
long_bone_names = lambda armature: [
    f"{armature.keypoint_names[int(i)]}-{armature.keypoint_names[int(j)]}"
    for i, j in armature.bones
]


# --- Rolling-mean time series, groupwise stripplot utiliies ---


def rolling_line(ax, x, ys, window, color, lighten=0, **kws):
    ydf = pd.DataFrame(np.array(ys).T, index=x).rolling(window).mean()
    y = ydf.mean(axis=1)  # .iloc[::window // 2]
    yerr = ydf.std(axis=1)  # .iloc[::window // 2]
    print(y)
    ax.plot(y.index, y, color=color, **kws)
    ax.fill_between(
        y.index,
        y - yerr,
        y + yerr,
        color=vu.lighten(color, lighten),
        alpha=0.3,
        linewidth=0,
    )


def agewise_stripstat(func, groups, order, expand=True):
    ret = [np.array([func(s) for s in groups[k]]) for k in order]
    if expand:
        ret = vu.expand_groups(ret, order)
    return ret


def stripstat(func, keys):
    return [np.array([func(s) for s in keys])]


def group_and_summary_strips(
    ax,
    func,
    groups,
    order,
    all_keys,
    summary_color,
    group_colors,
    x_ofs=3,
    **kws,
):
    vu.grouped_stripplot(
        agewise_stripstat(func, groups, order),
        ax=ax,
        colors=group_colors,
        **{"offset": False, "lighten_points": 0.5, **kws},
    )
    vu.grouped_stripplot(
        stripstat(func, all_keys),
        x=[-x_ofs],
        ax=ax,
        colors=summary_color,
        jitter=0.2,
        **{"offset": False, "lighten_points": 0.5, **kws},
    )

    ax.set_xticks([-x_ofs] + list(range(len(order))))
    ax.set_xlim([-x_ofs - 2, len(order) + 1])
    ax.set_xticklabels([""] * (len(order) + 1))
