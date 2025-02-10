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


def get_commonly_used(u, thresh):
    """
    at least than `pop_thresh` of subjects use a syllable more than
    `usage_thresh` of the time
    AND
    no more than `pop_thresh` of subjects use a syllable less than
    `min_thresh` of the time
    """
    usage_thresh, pop_thresh, min_thresh, min_pop_thresh = thresh or (
        0,
        0,
        None,
        None,
    )
    n_subj_total = u["session"].nunique()
    pop_usage = (
        u.query(f"usage > {usage_thresh}").groupby("index").usage.count()
        / n_subj_total
    )
    mask = pop_usage > pop_thresh
    if min_thresh is not None:
        min_usage = (
            u.assign(keep=u.usage < min_thresh).groupby("index").keep.sum()
            / n_subj_total
        )
        mask = mask & (min_usage <= min_pop_thresh)
    return mask


def mask_by_bool_series(df, mask, on="index"):
    return df.join(mask.rename("keep"), on=on).dropna().query("keep == True")


def usage_pivot(u, keys=("session",), thresh=None, reorder=False):
    if thresh is not None:
        used = get_commonly_used(u, thresh)
        u = mask_by_bool_series(u, used)
    pvt = u.pivot(columns="index", values="usage", index=keys)
    if reorder:
        pvt = pvt[pvt.columns[np.argsort(pvt.mean().values)[::-1]]]
    return pvt


def get_target_comparison_pivots(
    usages, dset_root, src, ref, tgt, first_seed, second_seed
):

    datasets = [f"{dset_root}-s{first_seed}", f"{dset_root}-s{second_seed}"]
    u = usages.query("dataset == @datasets[0]")
    u = mask_by_bool_series(
        u, get_commonly_used(u, thresh=(0.02, 0.2, None, 0))
    )
    u = usage_pivot(u, ("session", "body"))
    data_seed0 = u[u.mean().sort_values(ascending=False).index]
    used_sylls = data_seed0.columns

    u = usages.query("dataset == @datasets[1]")
    data_seed1 = usage_pivot(u, ("session", "body"))[used_sylls]

    data_seed0 = data_seed0.groupby("body")
    data_seed1 = data_seed1.groupby("body")
    ref_seed0 = data_seed0.get_group(ref)
    ref_seed1 = data_seed1.get_group(ref)
    tgt_seed1 = data_seed1.get_group(tgt)

    return ref_seed0, ref_seed1, tgt_seed1
