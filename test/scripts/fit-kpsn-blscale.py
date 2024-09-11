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
from shape_norm.io.dataset_refactor import Dataset, SessionMetadata
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


### -------------------------  Arg parsing  -----------------------------------

import sys

blscale_arrays_dir = Path(
    "/Users/kaifox/projects/mph/data_explore/testsets/blscale-arrays"
)
project_dir = Path(
    "/Users/kaifox/projects/mph/generative_api/test/projects/blscale-script/3wk-to-07-52-npyloader-runnable"
)

# currently at subsample_to=20 and n_steps=10 for base testing


### -------------------------  Supporting functions  --------------------------


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


def raw_data_armature_meta():
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
    names, parents, bones, root = raw_data_armature_meta()

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
        "session_names": coords.keys(),
        "bodies": session_bodies,
        "ref_session": ref_session,
        "keypoint_names": names,
        "keypoint_parents": parents,
    }

    return project_config_kws, coords, scales


def getc(spec):
    try:
        from cmap import Colormap, Color

        try:
            return Color(spec)
        except ValueError as e:
            stop = spec.split(":")[-1]
            if stop.startswith("."):
                stop = float(stop)
            else:
                stop = int(stop)
            cm = Colormap(":".join(spec.split(":")[:-1]))
            return cm(stop)
    except ImportError:
        return spec


def plot_mouse_3d(
    frame,
    ax,
    armature,
    elev,
    rot,
    colors=None,
    bone_n=40,
    point_size=10,
    line_size=2,
    line_colors=None,
    point_kws={},
    line_kws={},
    boundary=True,
    set_aspect=True,
    label=None,
):
    """all keys in point kws should be present in line kws"""

    if colors is None:
        import seaborn as sns

        colors = mpl_col.to_rgba(
            sns.color_palette("Blues", 10 + frame.shape[0])[10:]
        )
    if line_colors is None:
        line_colors = colors
    colors = np.array([mpl_col.to_rgba(x) for x in colors])
    line_colors = np.array([mpl_col.to_rgba(x) for x in line_colors])
    point_kws = {
        **dict(
            linewidths=np.array([0] * armature.n_kpts),
        ),
        **point_kws,
    }
    line_kws = {
        **dict(
            linewidths=np.array([0] * armature.n_kpts),
        ),
        **line_kws,
    }

    # --- points

    x, y, z, c, s = [], [], [], [], []
    x.append(frame[:, 0])
    y.append(frame[:, 1])
    z.append(frame[:, 2])
    c.append(colors)
    s.append([point_size] * armature.n_kpts)
    kws = {k: [] for k in point_kws}
    concat_keys = []
    for k, v in point_kws.items():
        if np.array(v).shape[:1] == (armature.n_kpts,):
            concat_keys.append(k)
            kws[k].append(v)
        else:
            kws[k] = v

    x = np.concatenate(x)
    y = np.concatenate(y)
    z = np.concatenate(z)
    c = np.concatenate(c)
    s = np.concatenate(s)
    for k, v in kws.items():
        if k in concat_keys:
            kws[k] = np.concatenate(kws[k])
    point_artist = ax.scatter(x, y, z, c=c, s=s, depthshade=False, **kws)

    # --- "lines"

    x, y, z, c, s = [], [], [], [], []
    kws = {k: [] for k in line_kws}
    concat_keys = []
    for i, (ch, pa) in enumerate(armature.bones):
        x.append(np.linspace(frame[ch, 0], frame[pa, 0], bone_n)[1:-1])
        y.append(np.linspace(frame[ch, 1], frame[pa, 1], bone_n)[1:-1])
        z.append(np.linspace(frame[ch, 2], frame[pa, 2], bone_n)[1:-1])
        c.append(np.full([bone_n - 2, 4], line_colors[ch]))
        s.append(np.full([bone_n - 2], line_size))
        for k, v in line_kws.items():
            if np.array(v).shape[:1] == (armature.n_kpts,):
                concat_keys.append(k)
                kws[k].append(
                    np.full((bone_n - 2,) + np.array(v).shape[1:], v[ch])
                )
            else:
                kws[k] = v
    x = np.concatenate(x)
    y = np.concatenate(y)
    z = np.concatenate(z)
    c = np.concatenate(c)
    s = np.concatenate(s)
    for k, v in kws.items():
        if k in concat_keys:
            kws[k] = np.concatenate(kws[k])

    line_artist = ax.scatter(x, y, z, c=c, s=s, depthshade=False, **kws)

    ax.view_init(elev, rot)

    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.zaxis.set_ticks([])
    if not boundary:
        ax.xaxis.pane.set_linewidth(0)
        ax.yaxis.pane.set_linewidth(0)
        ax.zaxis.pane.set_linewidth(0)
        ax.xaxis.line.set_linewidth(0)
        ax.yaxis.line.set_linewidth(0)
        ax.zaxis.line.set_linewidth(0)

    # use a.set_box_aspect to make the aspect equal to the ratio of the limits
    if set_aspect:
        xrng = np.ptp(ax.get_xlim())
        yrng = np.ptp(ax.get_ylim())
        zrng = np.ptp(ax.get_zlim())
        ax.set_box_aspect([xrng, yrng, zrng])

    return point_artist, line_artist


### -------------------------  Project / dataset setup  -----------------------

ref_age = "3"
tgt_ages = ["7", "52"]
animals = ["0", "1"]
sessions = [
    f"{ref_age}wk_{t}bod_m{a}" for a in animals for t in [ref_age] + tgt_ages
]
names, parents, _, _ = raw_data_armature_meta()
project_config_kws = dict(
    filepaths={s: blscale_arrays_dir / f"{s}.npy" for s in sessions},
    keypoint_names=names,
    keypoint_parents=parents,
    bodies={s: f"body-{s}" for s in sessions},
    ref_session=f"{ref_age}wk_52bod_m1",
)

project_dir.mkdir(exist_ok=True)
project = Project(project_dir)

plot_dir = project_dir / "plots"
plot_dir.mkdir(exist_ok=True)
plotter, colors = styles.init_nb(
    str(plot_dir),
    style="default",
    fmt="pdf",
    display=False,
)


### -------------------------  Calibration / training  ------------------------

loaders.raw_npy.setup_project_config(
    project.main_config(),
    feature_type="pcs",
    alignment_type="sagittal",
    exclude_keypoints=[
        "t_base",
        "lr_knee",
        "lr_foot",
        "rr_knee",
        "rr_foot",
        "lf_foot",
        "rf_foot",
        "t_tip",
    ],
    subsample_to=20,
    **project_config_kws,
)

setup.setup_base_model_config(
    project.main_config(), project.base_model_config()
)

# --- calibration: alignment and feature reduction
cfg = config.load_project_config(project.main_config())
dataset = loaders.load_dataset(cfg["dataset"])
cfg["features"]["calibration"]["tgt_variance"] = 0.98

dataset, cfg = alignment.sagittal.calibrate(
    dataset, cfg, origin="hips", anterior="head"
)

cfg = features.pcs.calibrate(dataset, cfg)
dataset = features.reduce_to_features(dataset, cfg["features"])
config.save_project_config(project.main_config(), cfg, write_calib=True)

# --- calibration: pose and morph models
cfg = config.load_model_config(project.base_model_config())
cfg["morph"]["prior_mode"] = "distance"
cfg["morph"]["calibration"]["tgt_variance"] = 0.99
cfg["pose"]["calibration"]["n_iter"] = 9
cfg["pose"]["calibration"]["max_components"] = 10
cfg["pose"]["subj_weight_uniformity"] = 1
cfg["fit"]["n_steps"] = 1000
cfg["fit"]["learning_rate"] = dict(kind="stair", hl=250, lr=5e-1, min=1e-5)
cfg["fit"]["mstep"]["tol"] = None
cfg["fit"]["update_scales"] = {"pose/*": 1e1}

cfg = pose.gmm.calibrate_base_model(dataset, cfg)
cfg = morph.lowrank_affine.calibrate_base_model(dataset, cfg)
cfg = em.calibrate_base_model(dataset, cfg)
config.save_model_config(project.base_model_config(), cfg, write_calib=True)


# plotting calibration
cfg = config.load_model_config(project.base_model_config())

fig = features.pcs.plot_calibration(cfg)
plotter.finalize(fig, "feature_calibration")

figs = viz.model.plot_calibration(cfg)
plotter.finalize(figs["morph"], "morph_calibration")
plotter.finalize(figs["pose"], "pose_calibration")


# set up parameter scan
scan_cfg, model_cfg = scans.setup_scan_config(
    project,
    "morph-prior-scan",
    {
        "morph.dist_var": [1e-1, 8e-1, 3e2],
        "fit.em.learning_rate.lr": [1e0, 8e-1, 8e-1],
        "fit.em.n_steps": [100, 100, 100],
    },
)

# Load data and prep for training
cfg = config.load_project_config(project.main_config())
scan_cfg = config.load_config(project.scan("morph-prior-scan") / "scan.yml")
dataset = loaders.load_dataset(cfg["dataset"])
scan_dataset, split_meta, _ = scans.prepare_scan_dataset(
    dataset, project, "morph-prior-scan", return_session_inv=True
)


# Run parameter scan!
scans.run_scan(
    project, "morph-prior-scan", scan_dataset, log_every=20, force_restart=False
)

# -- training diagnostics

progress = True
models = list(scan_cfg["models"].keys())

for model_name in model_util._optional_pbar(models, progress):

    print("Plotting", model_name)
    ckpt = methods.load_fit(project.model(model_name))
    ckpt["config"] = config.load_model_config(project.model_config(model_name))
    (project.model(model_name) / "plots").mkdir(exist_ok=True)
    plotter.plot_dir = str(project.model(model_name) / "plots")

    print("- reports")
    fig = viz.model.report_plots(ckpt, first_step=0, ax_size=(2, 1.2))
    fig.suptitle(model_name)
    plotter.finalize(fig, "reports")

    fig = viz.model.em_loss(ckpt, final_mstep=True)
    fig.suptitle(model_name)
    plotter.finalize(fig, "em_loss")

    print("- morph")
    fig = viz.model.lra_param_convergence(
        ckpt,
        progress=True,
        dataset=scan_dataset,
        magnitude_only=True,
        ax_size=(1.5, 1.5),
        legend=False,
    )
    fig.suptitle(model_name)
    plotter.finalize(fig, "morph_convergence")

    print("- gmm")
    fig = viz.model.gmm_param_convergence(
        ckpt,
        progress=True,
        dataset=scan_dataset,
        magnitude_only=True,
        normalize=False,
        ax_size=(1.5, 1.5),
    )
    fig.suptitle(model_name)
    plotter.finalize(fig, "gmm_convergence")

    mean_fig, wt_fig = viz.model.gmm_components(ckpt, dataset=scan_dataset)
    mean_fig.suptitle(model_name)
    wt_fig.suptitle(model_name)
    plotter.finalize(mean_fig, "gmm_means")

# -- Scan diagnostics

scan_name = "morph-prior-scan"
fig = viz.scans.withinbody_induced_errs(
    project, models, progress=True, dataset=scan_dataset, split_meta=split_meta
)
plot_dir = project.scan(scan_name) / "plots"
plotter.finalize(fig, "reconst-errs", path=plot_dir)

fig = viz.scans.jsds_to_reference(
    project, models, progress=True, dataset=scan_dataset, split_meta=split_meta
)
plotter.finalize(fig, "jsd-to-ref", path=plot_dir)


# -- BLscale-specific plots

# once canonicalizeed, frames/means of sessions A.0 and B.0 should (approximately) match
# since they only vary by the bone scaling morph

n_kpt = Armature.from_config(cfg["dataset"]).n_kpts
kp_colors = Colormap("greens")(np.linspace(0.3, 0.8, n_kpt))

models = list(scan_cfg["models"].keys())
parts = "01"

animal_sessions = {
    animal: [
        s
        for s in dataset.sessions
        if s.endswith(f"m{animal}") and f"{ref_age}bod" not in s
    ]
    for animal in animals
}
ncol = max([len(s_list) for s_list in animal_sessions.values()]) * len(parts)
nanim = len(animals)
n_animcol = min(nanim, 3)
n_animrow = int(np.ceil(nanim / n_animcol))
figure = plt.figure(
    figsize=((1.2 * ncol) * n_animcol + 1, (1.2 * len(models)) * n_animrow)
)
_, figs, fig_grid = viz.util.flat_subfig_grid(nanim, n_animcol, fig=figure)
for animal, fig in zip(animals, figs):

    sessions = animal_sessions[animal]
    pairs = {s: re.sub("(\d+)bod", f"{ref_age}bod", s) for s in sessions}

    ax = fig.subplots(len(models), ncol, sharex=True, sharey=True)

    base_errs = {s: [[] for _ in range(len(parts))] for s in sessions}
    inflated = features.inflate(scan_dataset, cfg["features"])
    for i_part, part in enumerate(parts):
        for s in pairs:
            base_errs[s][i_part] = model_util.reconst_errs(
                inflated.get_session(f"{pairs[s]}.{part}"),
                inflated.get_session(f"{s}.{part}"),
                average=False,
            ).mean(axis=0)
    base_errs = {s: np.array(base_errs[s]) for s in sessions}

    for i_mod, model_name in enumerate(models):

        session_pal = dict(zip(sessions, [colors.C[0], colors.C[1]]))

        ckpt = methods.load_fit(project.model(model_name))
        cfg = ckpt["config"]
        max_iter = cfg["fit"]["em"]["n_steps"]
        plot_iters = list(range(0, max_iter - 10, max_iter // 40)) + ["final"]
        model = instantiation.get_model(cfg)
        arms = Armature.from_config(cfg["dataset"])
        _x = [
            (x_ if isinstance(x_, int) else len(ckpt["meta"]["loss"]))
            for x_ in plot_iters
        ]

        errs = {s: [[] for _ in range(len(parts))] for s in sessions}
        for row, plot_iter in enumerate(plot_iters):
            if isinstance(plot_iter, int):
                step_params = scans.select_param_step(
                    model, ckpt["params"], ckpt["meta"]["param_hist"], plot_iter
                )
            else:
                step_params = ckpt["params"]
            canonicalized = model_util.apply_bodies(
                model.morph,
                step_params.morph,
                scan_dataset,
                {
                    s: scan_dataset.session_body_name(scan_dataset.ref_session)
                    for s in scan_dataset.sessions
                },
            )
            # canonicalized = scan_dataset
            inflated = features.inflate(canonicalized, cfg["features"])

            for s in sessions:
                for i_part, part in enumerate(parts):
                    errs[s][i_part].append(
                        model_util.reconst_errs(
                            inflated.get_session(f"{pairs[s]}.{part}"),
                            inflated.get_session(f"{s}.{part}"),
                            average=False,
                        ).mean(axis=0)
                    )
        errs = {s: np.array(errs[s]) for s in sessions}

        for i_s, s in enumerate(sessions):
            for i_part, part in enumerate(parts):
                i_col = i_s * len(parts) + i_part
                for i_kp in range(errs[s].shape[-1]):
                    if cfg["dataset"]["use_keypoints"][i_kp] == "hips":
                        continue
                    ax[i_mod, i_col].plot(
                        _x,
                        errs[s][i_part, :, i_kp],
                        color=kp_colors[i_kp],
                        lw=1,
                        ms=0,
                        label="Canonicalized",
                    )
                    ax[i_mod, i_col].plot(
                        [-_x[1]],
                        [base_errs[s][i_part, i_kp]],
                        "o",
                        color=kp_colors[i_kp],
                        lw=0,
                        ms=3,
                        label="Raw",
                    )
                ax[i_mod, 0].set_ylabel(model_name, fontsize=6)
                ax[0, i_col].set_title(
                    f"{s}.{part} v.\n{pairs[s]}.{part}", fontsize=7
                )

    for a in ax.ravel():
        sns.despine(ax=a)

    if fig is fig_grid[0, -1]:
        kp_artists = [
            ax[1, -1].plot(
                [np.nan],
                [np.nan],
                "o",
                color=kp_colors[i_kp],
                ms=3,
                label=cfg["dataset"]["use_keypoints"][i_kp],
            )[0]
            for i_kp in range(errs[sessions[0]].shape[-1])
        ]
        kp_handles, kp_labels = zip(
            *[
                [h, l]
                for h, l in zip(*ax[1, -1].get_legend_handles_labels())
                if h in kp_artists
            ]
        )

        ax[0, -1].legend(
            *viz.util.unique_handles(ax[0, -1]),
            loc="lower left",
            bbox_to_anchor=(
                1,
                0.5,
            ),
            fontsize=6,
            frameon=False,
        )
        ax[1, -1].legend(
            kp_handles,
            kp_labels,
            loc="upper left",
            bbox_to_anchor=(1, 1),
            fontsize=6,
            frameon=False,
        )

plotter.finalize(
    figure,
    "reconst-error-over-training",
    tight=False,
    despine=False,
    path=plot_dir,
)


# bone length over training

# once canonicalizeed, frames/means of sessions A.0 and B.0 should (approximately) match
# since they only vary by the bone scaling morph

bo_colors = Colormap("whorbr")(np.linspace(0.3, 0.8, n_kpt))

cfg = config.load_project_config(project.main_config())
arms = Armature.from_config(cfg["dataset"])
_bone_lengths = lambda kpts, arms: jnp.linalg.norm(
    kpts[:, arms.bones[:, 0]] - kpts[:, arms.bones[:, 1]], axis=-1
)
_bone_name = (
    lambda i: f"{arms.keypoint_names[int(arms.bones[i, 0])]}-{arms.keypoint_names[int(arms.bones[i, 1])]}"
)
bone_names = [_bone_name(i) for i in range(arms.bones.shape[0])]
inflated = features.inflate(scan_dataset, cfg["features"])
base_bls = _bone_lengths(
    inflated.get_session(scan_dataset.ref_session), arms
).mean(axis=0)
base_rtos = {
    s: _bone_lengths(inflated.get_session(s), arms).mean(axis=0) / base_bls
    for s in scan_dataset.sessions
}
sessions = [
    s
    for s in scan_dataset.sessions
    if scan_dataset.session_id(s) != scan_dataset.ref_session
]


fig, ax = plt.subplots(
    len(models),
    len(sessions),
    figsize=(1.3 * len(sessions) + 1, 1.3 * len(models)),
    sharex=True,
    sharey=True,
)

for i_mod, model_name in enumerate(models):

    session_pal = dict(zip(sessions, [colors.C[0], colors.C[1]]))

    ckpt = methods.load_fit(project.model(model_name))
    cfg = ckpt["config"]
    max_iter = cfg["fit"]["em"]["n_steps"]
    plot_iters = list(range(0, max_iter - 10, max_iter // 40)) + ["final"]
    model = instantiation.get_model(cfg)
    arms = Armature.from_config(cfg["dataset"])
    _x = [
        (x_ if isinstance(x_, int) else len(ckpt["meta"]["loss"]))
        for x_ in plot_iters
    ]

    bls = {s: [] for s in sessions}
    for row, plot_iter in enumerate(plot_iters):
        if isinstance(plot_iter, int):
            step_params = scans.select_param_step(
                model, ckpt["params"], ckpt["meta"]["param_hist"], plot_iter
            )
        else:
            step_params = ckpt["params"]
        canonicalized = model_util.apply_bodies(
            model.morph,
            step_params.morph,
            scan_dataset,
            {
                s: scan_dataset.session_body_name(scan_dataset.ref_session)
                for s in scan_dataset.sessions
            },
        )
        # canonicalized = scan_dataset
        inflated = features.inflate(canonicalized, cfg["features"])

        for s in bls:
            bls[s].append(
                _bone_lengths(inflated.get_session(s), arms).mean(axis=0)
            )
    bls = {k: np.array(a) for k, a in bls.items()}

    for i_s, s in enumerate(bls):
        for i_b, b_name in enumerate(bone_names):
            ax[i_mod, i_s].plot(
                _x,
                bls[s][:, i_b] / base_bls[i_b],
                color=bo_colors[i_b],
                lw=1,
                ms=0,
                label="Canonicalized",
            )
            ax[i_mod, i_s].plot(
                [-_x[1]],
                [base_rtos[s][i_b]],
                "o",
                color=bo_colors[i_b],
                lw=0,
                ms=3,
                label="Raw",
            )
            ax[i_mod, i_s].axhline(1.0, color="k", lw=0.5, ls="--", zorder=-1)
        ax[i_mod, 0].set_ylabel(model_name, fontsize=8)
        ax[0, i_s].set_title(s, fontsize=7)

kp_artists = [
    ax[0, 1].plot(
        [np.nan], [np.nan], "o", color=bo_colors[i_b], ms=3, label=b_name
    )[0]
    for i_b, b_name in enumerate(bone_names)
]
kp_handles, kp_labels = zip(
    *[
        [h, l]
        for h, l in zip(*ax[0, 1].get_legend_handles_labels())
        if h in kp_artists
    ]
)

ax[0, -1].legend(
    *viz.util.unique_handles(ax[0, -1]),
    loc="lower left",
    bbox_to_anchor=(
        1,
        0,
    ),
    fontsize=6,
    frameon=False,
)
ax[1, -1].legend(
    kp_handles,
    kp_labels,
    loc="upper left",
    bbox_to_anchor=(1, 1),
    fontsize=6,
    frameon=False,
)
plt.tight_layout()

plotter.finalize(
    fig,
    "bone-lengths-over-training",
    path=plot_dir,
)
