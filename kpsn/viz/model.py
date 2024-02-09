from ..logging import ArrayTrace
from .util import (
    plot_mouse,
    find_nearest_frames,
    select_frame_gallery,
    plot_mouse_views,
    legend,
    axes_off,
    flat_grid,
)
from ..models.morph.lowrank_affine import LRAParams, model as lra_model
from .styles import colorset
from ..io.loaders import load_dataset
from ..fitting.methods import load_and_prepare_dataset, load_fit
from ..io.armature import Armature
from ..models.util import apply_bodies
from ..io.features import inflate
from ..project.paths import Project
from ..config import load_model_config
from ..models.instantiation import get_model
from ..fitting.scans import merge_param_hist_with_hyperparams

import jax.numpy as jnp
from pathlib import Path
import numpy as np
import jax.numpy.linalg as jla
import matplotlib.pyplot as plt
import seaborn as sns


def report_plots(checkpoint, n_col=5, colors: colorset = None):
    if colors is None:
        colors = colorset.active
    reports: ArrayTrace = checkpoint["meta"]["reports"]
    n_row = int(jnp.ceil(reports.n_leaves() / n_col))
    fig, ax = plt.subplots(n_row, n_col, figsize=(3 * n_col, 2 * n_row))
    reports.plot(ax.ravel()[: reports.n_leaves()], color=colors.neutral, lw=1)
    for a in ax.ravel()[reports.n_leaves() :]:
        a.set_axis_off()
    sns.despine()
    fig.tight_layout()
    return fig


def em_loss(checkpoint, mstep_relative=True, colors: colorset = None):
    if colors is None:
        colors = colorset.active
    loss_hist = checkpoint["meta"]["loss"]

    # if loss during each mstep was not recorded, just plot the loss history
    if "mstep_losses" not in checkpoint["meta"]:
        fig, ax = plt.subplots(figsize=(3, 1.7))
        ax.plot(loss_hist, "k-")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Step")
        fig.tight_layout()
        return fig

    mstep_losses = checkpoint["meta"]["mstep_losses"]
    fig, ax = plt.subplots(figsize=(9, 1.7), ncols=3)

    pal = colors.cts(len(mstep_losses))

    mstep_lengths = []
    for i in range(0, len(mstep_losses)):
        if jnp.any(~jnp.isfinite(mstep_losses[i])):
            curr_loss = mstep_losses[i][
                : jnp.argmax(~jnp.isfinite(mstep_losses[i]))
            ]
            if len(curr_loss) == 0:
                continue
        else:
            curr_loss = mstep_losses[i]
        mstep_lengths.append(len(curr_loss))

        if mstep_relative:
            plot_y = (curr_loss - curr_loss.min()) / (
                curr_loss.max() - curr_loss.min()
            )
        else:
            plot_y = curr_loss

        ax[1].plot(
            jnp.linspace(0, 1, len(curr_loss)), plot_y, color=pal[i], lw=1
        )

    if not mstep_relative:
        ax[1].set_yscale("log")

    ax[0].plot(loss_hist, color=colors.neutral)
    if loss_hist.max() > 2 * loss_hist[0]:
        ax[0].set_ylim(None, 2 * loss_hist[0])
    ax[2].plot(jnp.arange(len(mstep_lengths)), mstep_lengths, colors.neutral)

    ax[1].set_xlabel("Loss profile")
    ax[2].set_ylabel("M step length")
    ax[2].set_xlabel("Step")
    ax[0].set_ylabel("Loss")
    ax[0].set_xlabel("Step")

    fig.tight_layout()
    sns.despine()
    return fig


def lra_centroid_and_modes(checkpoint: dict, colors: colorset = None):
    if colors is None:
        colors = colorset.active
    params: LRAParams = checkpoint["params"].morph
    config = _insert_calibration_data_to_checkpoint_config(checkpoint)
    dataset, _ = load_and_prepare_dataset(config)
    armature = Armature.from_config(config["dataset"])

    fig, ax = plt.subplots(
        (params.n_dims * 2 + 2),
        params.n_bodies,
        figsize=(1.7 * params.n_bodies, 1.2 * (params.n_dims * 2 + 2)),
        sharex=True,
        sharey=True,
    )

    _inflate = lambda arr: inflate(arr, config["features"])
    _plot = lambda row, col, arr, yaxis, color, **kw: plot_mouse(
        ax[row, col], _inflate(arr), armature, 0, yaxis, color=color, **kw
    )

    pal = colors.cts1(params.n_bodies)
    neut = colors.neutral
    sbtl = colors.subtle
    mode_scale = 10
    for i_body in range(params.n_bodies):
        ax[0, i_body].set_title(dataset._body_names[i_body])
        for i_row, row_y in enumerate([2, 1]):
            # -- plot centroid
            ctr = params.offset + params.offset_updates[i_body]
            ref = f"{dataset.ref_session}\n(ref)"
            _plot(i_row, i_body, ctr, row_y, pal[i_body])
            ax[0, 0].set_ylabel("centroid")
            if i_body != params.ref_body:
                ref_ctr = params.offset + params.offset_updates[params.ref_body]
                _plot(i_row, i_body, ref_ctr, row_y, sbtl, zorder=-3, label=ref)

            # -- plot modes
            for i_mode in range(params.n_dims):
                mode = ctr + mode_scale * (
                    params.modes[:, i_mode]
                    + params.mode_updates[i_body, :, i_mode]
                )
                _ax = (2 + 2 * i_mode + i_row, i_body)
                _plot(
                    *_ax,
                    ctr,
                    row_y,
                    neut,
                    zorder=-3,
                    label="centroid",
                    line_kw={"lw": 0.5},
                )
                _plot(*_ax, mode, row_y, pal[i_body], label=f"mode {i_mode}")
                if i_row == 0:
                    ax[_ax[0], 0].set_ylabel(f"mode {i_mode}")

    axes_off(ax)
    for a in ax[::2, -1]:
        legend(a)

    return fig


def lra_param_convergence(checkpoint: dict, colors: colorset = None):
    if colors is None:
        colors = colorset.active

    cfg = _insert_calibration_data_to_checkpoint_config(checkpoint)
    _dataset, _ = load_and_prepare_dataset(cfg)
    model = get_model(cfg)
    param_hist = merge_param_hist_with_hyperparams(
        model, checkpoint["params"], checkpoint["meta"]["param_hist"]
    )
    n_bodies = int(param_hist.morph.n_bodies[0])
    n_dims = int(param_hist.morph.n_dims[0])
    pal = [colors.C[0], colors.C[2], colors.C[1], colors.C[3]]
    fig, ax = plt.subplots(
        n_dims + 1,
        n_bodies - 1,
        figsize=(n_bodies * 1.5, n_dims * 1.5 + 1.5),
        sharex=True,
        sharey="row",
    )
    ax = np.atleast_2d(ax.T).T
    i_ax = 0
    for i_body in range(n_bodies):
        if i_body == param_hist.morph.ref_body[0]:
            continue
        for i_mode in range(n_dims):
            upds = param_hist.morph.mode_updates[:, i_body, :, i_mode]
            ax[i_mode, i_ax].plot(
                upds,
                color=pal[i_body],
                lw=0.5,
                label=[i_body] + [None] * (upds.shape[-1] - 1),
            )
            ax[0, i_ax].set_title(_dataset._body_names[i_body])
            ax[i_mode, 0].set_ylabel(f"pc {i_mode}")
        upds = param_hist.morph.offset_updates[:, i_body, :]
        ax[-1, i_ax].plot(upds, color=pal[i_body], lw=0.5)
        i_ax += 1

    ax[-1, 0].set_xlabel("iteration [m-steps]")
    ax[-1, 0].set_ylabel("centroid")
    return fig


def gmm_param_convergence(checkpoint: dict, colors: colorset = None):
    if colors is None:
        colors = colorset.active
    cfg = _insert_calibration_data_to_checkpoint_config(checkpoint)
    _dataset, _ = load_and_prepare_dataset(cfg)
    model = get_model(cfg)
    param_hist = merge_param_hist_with_hyperparams(
        model, checkpoint["params"], checkpoint["meta"]["param_hist"]
    )
    n_sessions = int(param_hist.pose.n_sessions[0])
    n_components = int(param_hist.pose.n_components[0])
    pal = colors.make("Spectral")(n_sessions)
    fig, ax = plt.subplots(
        n_components,
        n_sessions + 1,
        figsize=(1.5 * n_sessions + 1.5, 1.5 * n_components),
    )
    for i_comp in range(n_components):
        weights = param_hist.pose.subj_weights[..., i_comp]
        means = param_hist.pose.means[..., i_comp, :]
        for i_sess in range(n_sessions):
            ax[i_comp, i_sess].plot(weights[:, i_sess], color=pal[i_sess])
            ax[i_comp, i_sess].set_ylim(0, param_hist.pose.subj_weights.max())
            ax[0, i_sess].set_title(_dataset._session_names[i_sess])
        ax[i_comp, -1].plot(means, color=colors.neutral, lw=0.5)
        ax[i_comp, 0].set_ylabel(f"comp {i_comp}")
    ax[0, -1].set_title("Mean")

    return fig


def gmm_components(checkpoint: dict, colors: colorset = None):
    """Component means and weights of a GMM model."""
    if colors is None:
        colors = colorset.active

    cfg = checkpoint["config"]
    arm = Armature.from_config(cfg["dataset"])
    dataset, _ = load_and_prepare_dataset(cfg)
    _inflate = lambda arr: inflate(arr, cfg["features"])
    params = checkpoint["params"].pose

    # means figure
    means_fig, ax, ax_grid = flat_grid(params.n_components * 2, 6, (2, 2))
    for i in range(params.n_components):
        pose = _inflate(params.means[i : i + 1])[0]
        plot_mouse_views(
            ax[i * 2 : (i + 1) * 2], pose, arm, color=colors.neutral
        )
        ax[i * 2].set_ylabel(f"Component {i}")
    axes_off(ax)

    # weights figure
    weights_fig, ax = plt.subplots(figsize=(3.5, 2))
    mappable = ax.imshow(params.subj_weights)
    ax.set_yticks(range(len(dataset.sessions)))
    ax.set_yticklabels(
        [
            (
                "(ref) "
                if dataset._session_names[i] == dataset.ref_session
                else ""
            )
            + dataset._session_names[i]
            for i in range(len(dataset.sessions))
        ]
    )
    weights_fig.colorbar(mappable)

    return means_fig, weights_fig


def _insert_calibration_data_to_checkpoint_config(checkpoint):
    from .. import config

    model_cfg = checkpoint["config"]
    if "calibration_data" in checkpoint["config"]["morph"]:
        return model_cfg
    project = Project(Path(model_cfg["project"]).parent)

    model_cfg["calibration_file"] = config.load_project_config(
        project.main_config()
    )["calibration_file"]
    calib = config.load_calibration_data(model_cfg["calibration_file"])
    model_cfg = config._add_calibration_data(
        model_cfg, config.load_calibration_data(model_cfg["calibration_file"])
    )
    return model_cfg


def compare_nearest_frames(
    checkpoint: dict, colors: colorset = None, group_lines=False
):
    """Nearest frames of each session to the reference session before and after
    morphing.
    """
    if colors is None:
        colors = colorset.active

    # set up model/dataset
    params: LRAParams = checkpoint["params"].morph
    config = _insert_calibration_data_to_checkpoint_config(checkpoint)
    model = get_model(config)
    dataset, _ = load_and_prepare_dataset(config)
    _inflate = lambda arr: inflate(arr, config["features"])
    aligned = _inflate(dataset)
    armature = Armature.from_config(config["dataset"])

    # map all sessions to the reference session's body
    ref_body = dataset.sess_bodies[dataset.ref_session]
    mapped_reduced = apply_bodies(
        model.morph, params, dataset, {s: ref_body for s in dataset.sessions}
    )
    mapped = _inflate(mapped_reduced)
    ref_poses = select_frame_gallery(
        aligned.get_session(aligned.ref_session), armature, as_list=True
    )
    unmapped_nbrs = find_nearest_frames(ref_poses, aligned)
    mapped_nbrs = find_nearest_frames(ref_poses, mapped)

    fig, ax = plt.subplots(
        dataset.n_sessions,
        len(ref_poses) * 2,
        figsize=(len(ref_poses) * 3, dataset.n_sessions * 1.5),
        sharex=True,
        sharey=True,
    )
    _plot_mouse = lambda pose, color, label, **kw: plot_mouse_views(
        axs, pose, armature, color=color, label=label, **kw
    )
    pal = colors.cts1(dataset.n_sessions - 1)

    for i, ref_pose in enumerate(ref_poses):
        # iterate over all sessions except the reference session
        j = 0
        ref_s = dataset.ref_session
        for s in dataset.sessions:
            if s == ref_s:
                continue
            # plot reference and nearest poses
            unmapped_nbr = unmapped_nbrs[s][i]
            mapped_nbr = mapped_nbrs[s][i]
            axs = ax[j + 1, 2 * i : 2 * (i + 1)]
            _plot_mouse(ref_pose, colors.neutral, ref_s, line_kw={"lw": 0.5})
            _plot_mouse(unmapped_nbr, colors.subtle, s, line_kw={"lw": 0.5})
            _plot_mouse(mapped_nbr, pal[j], "mapped")
            j += 1
        # plot reference poses in top two rows
        axs = ax[0, 2 * i : 2 * (i + 1)]
        _plot_mouse(ref_pose, colors.neutral, ref_s)

    axes_off(ax)
    for a in ax[:, -1]:
        legend(a)

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    # plot lines above axes before their bounds change due to aspect = 1
    if group_lines:
        for i, ref_pose in enumerate(ref_poses):
            ax0_lim = (ax[0, 2 * i].get_xlim(), ax[0, 2 * i].get_ylim())
            ax1_lim = (ax[0, 2 * i + 1].get_xlim(), ax[0, 2 * i + 1].get_ylim())
            ax0_topleft = fig.transFigure.inverted().transform(
                ax[0, 2 * i].transData.transform([ax0_lim[0][0], ax1_lim[1][1]])
            )
            ax1_topright = fig.transFigure.inverted().transform(
                ax[0, 2 * i + 1].transData.transform(
                    [ax0_lim[0][1], ax1_lim[1][1]]
                )
            )
            fig.add_artist(
                plt.Line2D(
                    [ax0_topleft[0] + 0.01, ax1_topright[0] - 0.01],
                    [ax0_topleft[1] + 0.03, ax1_topright[1] + 0.03],
                    lw=1,
                    color=colors.neutral,
                    # transform=fig.dpi_scale_trans,
                )
            )

    return fig


def plot_calibration(project: Project, config: dict, colors=None):
    """Plot explanations of model calibration values.

    Parameters
    ----------
    project : Project
    config : dict
        Full project config.
    """
    if colors is None:
        colors = colorset.active
    model = get_model(config)
    return {
        "morph": model.morph.plot_calibration(project, config, colors),
        "pose": model.pose.plot_calibration(project, config, colors),
    }
