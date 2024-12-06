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


def indiv_holdout(feats, groups, seed, shuffle):
    rng = np.random.default_rng(seed)
    feat_val = []
    feat_trn = []
    tgt_val = []
    tgt_trn = []
    tgt_keys = dict(zip(groups.keys(), range(len(groups))))
    for grp_key, grp in groups.items():
        holdout = rng.integers(len(grp))
        feat_val.append(feats[grp[holdout]])
        feat_trn.extend([feats[s] for i, s in enumerate(grp) if i != holdout])
        lens = [len(feats[s]) for s in grp]
        tgt_val.append(np.full(lens[holdout], tgt_keys[grp_key]))
        tgt_trn.extend(
            [
                np.full(l, tgt_keys[grp_key])
                for i, l in enumerate(lens)
                if i != holdout
            ]
        )
    feat_val = np.concatenate(feat_val, axis=0)
    feat_trn = np.concatenate(feat_trn, axis=0)
    tgt_val = np.concatenate(tgt_val, axis=0)
    tgt_trn = np.concatenate(tgt_trn, axis=0)
    if shuffle:
        tgt_trn = rng.permutation(tgt_trn)
    return feat_trn, feat_val, tgt_trn, tgt_val


def frame_holdout(feats, groups, seed, shuffle, train_pct=0.5):
    if isinstance(seed, int):
        rng = np.random.default_rng(seed)
    else:
        rng = seed
    feat_val = []
    feat_trn = []
    tgt_val = []
    tgt_trn = []
    tgt_keys = dict(zip(groups.keys(), range(len(groups))))
    for grp_key, grp in groups.items():
        for s in grp:
            x = feats[s]
            trn, val = np.split(
                rng.permutation(len(x)), [int(train_pct * len(x))]
            )
            feat_trn.append(x[trn])
            feat_val.append(x[val])
            tgt_trn.append(np.full(len(trn), tgt_keys[grp_key]))
            tgt_val.append(np.full(len(val), tgt_keys[grp_key]))
    feat_val = np.concatenate(feat_val, axis=0)
    feat_trn = np.concatenate(feat_trn, axis=0)
    tgt_val = np.concatenate(tgt_val, axis=0)
    tgt_trn = np.concatenate(tgt_trn, axis=0)
    if shuffle:
        tgt_trn = rng.permutation(tgt_trn)
    return feat_trn, feat_val, tgt_trn, tgt_val


def pairwise_group_acc(feats, groups, ref, nonref, n_iter=40):
    scores = {a: [] for a in nonref}
    scores_null = {a: [] for a in nonref}
    for is_null, scores_dict in enumerate([scores, scores_null]):
        for nonref_age in tqdm(nonref):
            for seed in range(n_iter):
                xt, xv, yt, yv = indiv_holdout(
                    feats,
                    {ref: groups[ref], nonref_age: groups[nonref_age]},
                    seed,
                    shuffle=is_null,
                )
                model = LogisticRegression(
                    max_iter=1000, class_weight="balanced"
                )
                scores_dict[nonref_age].append(model.fit(xt, yt).score(xv, yv))
    return scores, scores_null


def allway_group_acc(
    feats, groups, n_iter=40, subsample=1, clf_kws={}, prescale=False
):
    scores_list = [[], []]
    for is_null in range(2):
        for seed in tqdm.trange(n_iter):
            xt, xv, yt, yv = indiv_holdout(
                {k: v[::subsample] for k, v in feats.items()},
                groups,
                seed,
                shuffle=is_null,
            )
            if prescale:
                m, sd = xt.mean(axis=0), xt.std(axis=0)
                xt = (xt - m) / sd
                xv = (xv - m) / sd
            model = LogisticRegression(
                **{**dict(max_iter=1000, class_weight="balanced"), **clf_kws}
            )
            scores_list[is_null].append(model.fit(xt, yt).score(xv, yv))
    return scores_list


def pairwise_framewise_acc(
    feats, groups, n_iter=1, across_mode="none", rng=None
):
    """
    Returns
    -------
    across_group : list
        Frames taken from sessions in different groups forming positive and
        negative example sets.
    across_group_null : list
        As in `across_group` but with shuffled labels.
    within_group : list
        Frames from different sessions within the same group forming positive
        and negative example sets.
    within_session : list
        Sets of frames taken from within the same session forming positive and
        negative example sets.
    """
    scores_list = [[], [], [], []]
    combos = list(iit.combinations(groups.keys(), 2)) + [
        (k, k) for k in groups.keys()
    ]
    if rng is None:
        rng = np.random.default_rng()
    for g1, g2 in tqdm(combos):
        within_group = g1 == g2
        if across_mode == "across-only" and within_group:
            continue
        if across_mode == "within-only" and not within_group:
            continue
        for is_null in range(2 if not within_group else 1):
            for i, (s1, s2) in enumerate(iit.product(groups[g1], groups[g2])):
                within_animal = s1 == s2
                for j in range(n_iter):
                    xt, xv, yt, yv = frame_holdout(
                        feats, {"s1": [s1], "s2": [s2]}, rng, shuffle=is_null
                    )
                    # print(np.unique(yt, return_counts=True), np.unique(yv, return_counts=True))
                    model = LogisticRegression(
                        max_iter=1000, class_weight="balanced"
                    )
                    score_ix = (
                        3 if within_animal else (2 if within_group else is_null)
                    )
                    scores_list[score_ix].append(
                        model.fit(xt, yt).score(xv, yv)
                    )
    return scores_list


def named_group_pairwise_framewise_acc(
    feats, groups, n_iter=1, across_mode="none", rng=None
):
    """
    Returns
    -------
    across_group : dict
        Frames taken from sessions in different groups forming positive and
        negative example sets.
    across_group_null : dict
        As in `across_group` but with shuffled labels.
    within_group : dict
        Frames from different sessions within the same group forming positive
        and negative example sets.
    within_session : dict
        Sets of frames taken from within the same session forming positive and
        negative example sets.
    """
    scores_list = [defaultdict(list) for _ in range(4)]
    combos = list(iit.combinations(groups.keys(), 2)) + [
        (k, k) for k in groups.keys()
    ]
    if rng is None:
        rng = np.random.default_rng()
    for g1, g2 in tqdm(combos):
        within_group = g1 == g2
        if across_mode == "across-only" and within_group:
            continue
        if across_mode == "within-only" and not within_group:
            continue
        for is_null in range(2 if not within_group else 1):
            for i, (s1, s2) in enumerate(iit.product(groups[g1], groups[g2])):
                within_animal = s1 == s2
                for seed in range(n_iter):
                    xt, xv, yt, yv = frame_holdout(
                        feats, {"s1": [s1], "s2": [s2]}, rng, shuffle=is_null
                    )
                    model = LogisticRegression(
                        max_iter=1000, class_weight="balanced"
                    )
                    score_ix = (
                        3 if within_animal else (2 if within_group else is_null)
                    )
                    scores_list[score_ix][frozenset({g1, g2})].append(
                        model.fit(xt, yt).score(xv, yv)
                    )
    return scores_list


def calculate_feat_scaled_kp_normed(
    kpts_dict, arms, subsample=1, mode="keypoints"
):
    """Convert dictionary of keypoints to normalized feature vectors"""
    sessions = list(kpts_dict.keys())
    if mode == "keypoints":
        feat_shape = lambda a: a.reshape(-1, arms.n_kpts * 3)
    elif mode == "angles":
        feat_shape = lambda a: a
    else:
        raise ValueError(f"Unrecognized mode: {mode}")
    feat_scaled_kp = {
        s: feat_shape(kpts_dict[s][::subsample]) for s in sessions
    }
    norm_stats = [
        np.mean(np.concatenate(list(feat_scaled_kp.values())), axis=0),
        np.std(np.concatenate(list(feat_scaled_kp.values())), axis=0),
    ]
    keep_ixs = np.where(norm_stats[1] > 1e-3)[0]
    norm_stats[1] = np.where(norm_stats[1] == 0, 1, norm_stats[1])
    return {
        s: ((feat_scaled_kp[s] - norm_stats[0]) / norm_stats[1])[:, keep_ixs]
        for s in sessions
    }


def calc_all_scores(feats_dict, groups, ref_age, tgt_ages, n_iter=1, seed=9731):
    """Compute accuracy for all for cohort and individual comparisons"""
    rng = np.random.default_rng(seed)
    c_scores, c_scores_null = pairwise_group_acc(
        feats_dict, groups, ref_age, tgt_ages, n_iter=n_iter
    )
    c_indiv_across, c_indiv_null, c_indiv_within, c_indiv_self = (
        pairwise_framewise_acc(feats_dict, groups, n_iter=n_iter, rng=rng)
    )
    c_indiv = c_indiv_across, c_indiv_within, c_indiv_self, c_indiv_null
    return c_scores, c_scores_null, c_indiv


def calc_indiv_filtered_scores(
    feats_dict, groups, ref_age, tgt_ages, indiv_groups, n_iter=1, seed=9729
):
    """
    `calc_all_scores`, with comparisons across group only performed within
    sessions in the same `indiv_group`

    This avoids comparing across both distortion and behavior variance
    """
    scores, scores_null = [defaultdict(list) for _ in range(2)]
    rng = np.random.default_rng(seed)
    for indiv, indiv_sessions in indiv_groups.items():
        indiv_groups = {
            g: [s for s in v if s in indiv_sessions] for g, v in groups.items()
        }
        indiv_groups = {g: v for g, v in indiv_groups.items() if len(v)}
        indiv_scores, indiv_null, _, _ = named_group_pairwise_framewise_acc(
            feats_dict,
            indiv_groups,
            n_iter=n_iter,
            across_mode="across-only",
            rng=rng,
        )
        for k in indiv_scores:
            if ref_age in k:
                k_tgt = [g for g in k if g != ref_age][0]
                scores[k_tgt].extend(indiv_scores[k])
                scores_null[k_tgt].extend(indiv_null[k])
    allway_scores = pairwise_framewise_acc(
        feats_dict, groups, n_iter=n_iter, across_mode="none", rng=rng
    )
    return scores, scores_null, allway_scores


classif_strip = lambda ax, keys, data, colors, offset, point_ms=0.5, **kws: vu.grouped_stripplot(
    vu.expand_groups([np.array(data[a]) for a in keys], keys),
    x=np.arange(len(keys)) + offset,
    ax=ax,
    colors=dict(zip(keys, colors)),
    points_kw=dict(ms=point_ms, mew=0),
    errorbar_kw=dict(ms=2, elinewidth=0),
    **{
        **dict(
            xticks="",
            lighten_points=0.5,
            offset=False,
        ),
        **kws,
    },
)

classif_strip_points = lambda ax, keys, data, colors, offset, point_ms=0.7, **kws: vu.grouped_stripplot(
    vu.expand_groups([np.array(data[a]) for a in keys], keys),
    x=np.arange(len(keys)) + offset,
    ax=ax,
    colors=dict(zip(keys, colors)),
    points_kw=dict(ms=point_ms),
    errorbar_kw=dict(ms=0, elinewidth=0),
    jitter=0.1,
    **{
        **dict(
            xticks="",
            lighten_points=False,
            offset=False,
        ),
        **kws,
    },
)
