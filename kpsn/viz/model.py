from ..logging import ArrayTrace
from .util import (
    plot_mouse,
    find_nearest_frames,
    select_frame_gallery,
    plot_mouse_views,
    legend,
    axes_off,
    flat_grid,
    select_frame_gallery,
    stack_lines,
)
from .videos import (
    load_videos,
    _egocentric_crop,
    _overlay_keypoints,
    _scalar_summaries,
    _egocentric_window_align,
    write_video,
)
from ..io.utils import stack_dict
from ..io.dataset_refactor import Dataset
from ..models.morph.lowrank_affine import LRAParams, model as lra_model
from ..models.morph.bone_length import BLSParams, model as bls_model
from .styles import colorset
from ..io.armature import Armature
from ..fitting.methods import load_and_prepare_dataset
from ..models.util import apply_bodies, _optional_pbar
from ..io.features import inflate
from ..project.paths import Project
from ..models.instantiation import get_model
from ..fitting.scans import merge_param_hist_with_hyperparams

import jax.numpy as jnp
from pathlib import Path
import numpy as np
import jax.numpy.linalg as jla
import matplotlib.pyplot as plt
from matplotlib import colors as plt_colors
import seaborn as sns
import tqdm
import cv2


def report_plots(checkpoint, n_col=5, first_step=0, colors: colorset = None):
    if colors is None:
        colors = colorset.active
    reports: ArrayTrace = checkpoint["meta"]["reports"]
    n_row = int(jnp.ceil(reports.n_leaves() / n_col))
    fig, ax = plt.subplots(n_row, n_col, figsize=(3 * n_col, 2 * n_row))
    reports.plot(
        ax.ravel()[: reports.n_leaves()],
        color=colors.neutral,
        lw=1,
        first_step=first_step,
    )
    for a in ax.ravel()[reports.n_leaves() :]:
        a.set_axis_off()
    sns.despine()
    fig.tight_layout()
    return fig


def em_loss(
    checkpoint, mstep_relative=True, colors: colorset = None, progress=True
):
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
    xs, ys, cs = [], [], []
    for i in _optional_pbar(np.arange(len(mstep_losses)), progress):
        if jnp.any(~jnp.isfinite(mstep_losses[i])):
            curr_loss = mstep_losses[i][
                : jnp.argmax(~jnp.isfinite(mstep_losses[i]))
            ]
            if len(curr_loss) == 0:
                continue
        else:
            curr_loss = mstep_losses[i]
        curr_loss = np.array(curr_loss)
        mstep_lengths.append(len(curr_loss))

        if mstep_relative:
            plot_y = (curr_loss - curr_loss.min()) / (
                curr_loss.max() - curr_loss.min()
            )
        else:
            plot_y = curr_loss

        xs.append(np.linspace(0, 1, len(curr_loss)))
        ys.append(plot_y)
        cs.append(pal[i])

    lines = stack_lines(xs, ys, cs, lw=1)
    ax[1].add_artist(lines)

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


def lra_centroid_and_modes(
    checkpoint: dict,
    dataset: Dataset = None,
    colors: colorset = None,
    progress=False,
    body_whitelist=None,
    pal=None,
    params=None,
):
    if colors is None:
        colors = colorset.active
    config = checkpoint["config"]
    if dataset is None:
        dataset, _ = load_and_prepare_dataset(config)
    armature = Armature.from_config(config["dataset"])
    bodies = dataset.bodies if body_whitelist is None else body_whitelist

    if params is None:
        params: LRAParams = checkpoint["params"].morph

    fig, ax = plt.subplots(
        (params.n_dims * 2 + 2),
        len(bodies),
        figsize=(1.7 * len(bodies), 1.2 * (params.n_dims * 2 + 2)),
        sharex=True,
        sharey=True,
    )

    _inflate = lambda arr: inflate(arr, config["features"])
    _plot = lambda row, col, arr, yaxis, color, **kw: plot_mouse(
        ax[row, col], _inflate(arr), armature, 0, yaxis, color=color, **kw
    )

    if pal is None:
        pal = dict(zip(bodies, colors.cts1(len(bodies))))
    neut = colors.neutral
    sbtl = colors.subtle
    mode_scale = 10
    i_body = 0
    for body_id, body in _optional_pbar(dataset._body_names.items(), progress):
        if body not in body_whitelist:
            continue
        ax[0, i_body].set_title(body)
        for i_row, row_y in enumerate([2, 1]):
            # -- plot centroid
            ctr = params.offset + params.offset_updates[body_id]
            ref = f"{dataset.ref_session}\n(ref)"
            _plot(i_row, i_body, ctr, row_y, pal[body])
            ax[0, 0].set_ylabel("centroid")
            if i_body != params.ref_body:
                ref_ctr = params.offset + params.offset_updates[params.ref_body]
                _plot(i_row, i_body, ref_ctr, row_y, sbtl, zorder=-3, label=ref)

            # -- plot modes
            for i_mode in range(params.n_dims):
                mode = ctr + mode_scale * (
                    params.modes[:, i_mode]
                    + params.mode_updates[body_id, :, i_mode]
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
                _plot(*_ax, mode, row_y, pal[body], label=f"mode {i_mode}")
                if i_row == 0:
                    ax[_ax[0], 0].set_ylabel(f"mode {i_mode}")
        i_body += 1

    axes_off(ax)
    for a in ax[::2, -1]:
        legend(a)

    return fig


def lra_param_convergence(
    checkpoint: dict,
    dataset: Dataset = None,
    colors: colorset = None,
    stepsize=1,
    progress=False,
    first_step=0,
):
    if colors is None:
        colors = colorset.active

    cfg = _insert_calibration_data_to_checkpoint_config(checkpoint)
    if dataset is None:
        dataset, _ = load_and_prepare_dataset(cfg)
    model = get_model(cfg)
    param_hist = merge_param_hist_with_hyperparams(
        model, checkpoint["params"], checkpoint["meta"]["param_hist"]
    )

    # hacky: param_hist.ref_body becomes an array, but will be used in .at[].set
    # so we would prefer it be a scalar
    param_hist.morph._tree["ref_body"] = param_hist.morph.ref_body[0]

    step = jnp.concatenate(
        [
            jnp.array([0]),
            checkpoint["meta"].get(
                "gd_step", np.arange(1, len(param_hist.morph.offset_updates))
            ),
        ]
    )[first_step::stepsize]
    n_bodies = int(param_hist.morph.n_bodies[0])
    n_dims = int(param_hist.morph.n_dims[0])
    pal = colors.make("Spectral")(n_bodies)
    fig, ax = plt.subplots(
        n_dims + 1,
        n_bodies - 1,
        figsize=(n_bodies * 1.5, n_dims * 1.5 + 1.5),
        sharex=True,
        sharey="row",
    )
    ax = np.atleast_2d(ax.T).T
    i_ax = 0
    for i_body in _optional_pbar(range(n_bodies), progress):
        if i_body == param_hist.morph.ref_body:
            continue
        for i_mode in range(n_dims):
            upds = param_hist.morph.mode_updates[:, i_body, :, i_mode]
            ax[i_mode, i_ax].plot(
                step,
                upds[first_step::stepsize],
                color=pal[i_body],
                lw=0.5,
                label=[i_body] + [None] * (upds.shape[-1] - 1),
            )
            ax[0, i_ax].set_title(dataset.body_name(i_body))
            ax[i_mode, 0].set_ylabel(f"pc {i_mode}")
        upds = param_hist.morph.offset_updates[:, i_body, :]
        ax[-1, i_ax].plot(
            step, upds[first_step::stepsize], color=pal[i_body], lw=0.5
        )
        i_ax += 1

    ax[-1, 0].set_xlabel("iteration [m-steps]")
    ax[-1, 0].set_ylabel("centroid")
    return fig


def bls_param_convergence(
    checkpoint: dict,
    dataset: Dataset = None,
    colors: colorset = None,
    stepsize=1,
    progress=False,
    first_step=0,
):
    if colors is None:
        colors = colorset.active

    cfg = _insert_calibration_data_to_checkpoint_config(checkpoint)
    if dataset is None:
        dataset, _ = load_and_prepare_dataset(cfg)
    model = get_model(cfg)
    param_hist: BLSParams = merge_param_hist_with_hyperparams(
        model, checkpoint["params"], checkpoint["meta"]["param_hist"]
    )
    arms = Armature.from_config(cfg["dataset"])

    # hacky: param_hist.ref_body becomes an array, but will be used in .at[].set
    # so we would prefer it be a scalar
    param_hist.morph._tree["ref_body"] = param_hist.morph.ref_body[0]

    step = jnp.concatenate(
        [
            jnp.array([0]),
            checkpoint["meta"].get(
                "gd_step", np.arange(1, len(param_hist.morph._logscales))
            ),
        ]
    )[first_step::stepsize]
    n_bodies = int(param_hist.morph.n_bodies[0])
    n_bones = int(param_hist.morph.n_bones[0])
    pal = colors.make("Spectral")(n_bodies)
    fig, ax = plt.subplots(
        n_bones,
        n_bodies - 1,
        figsize=(n_bodies * 1.5, n_bones * 0.75),
        sharex=True,
        sharey="row",
    )
    ax = np.atleast_2d(ax.T).T
    i_ax = 0
    for i_body in _optional_pbar(range(n_bodies), progress):
        if i_body == param_hist.morph.ref_body:
            continue
        for i_bone in range(n_bones):
            scales = param_hist.morph.scales[:, i_body, i_bone]
            ax[i_bone, i_ax].plot(
                step,
                scales[first_step::stepsize],
                color=pal[i_body],
                lw=0.5,
                label=i_body,
            )
            ax[0, i_ax].set_title(dataset.body_name(i_body))
            ax[i_bone, 0].set_ylabel(
                arms.keypoint_names[int(arms.bones[i_bone, 0])]
            )
        i_ax += 1

    ax[-1, 0].set_xlabel("iteration [m-steps]")
    return fig


def gmm_param_convergence(
    checkpoint: dict,
    dataset: Dataset = None,
    colors: colorset = None,
    stepsize=1,
    progress=False,
    first_step=0,
):
    if colors is None:
        colors = colorset.active
    cfg = _insert_calibration_data_to_checkpoint_config(checkpoint)
    if dataset is None:
        dataset, _ = load_and_prepare_dataset(cfg)
    model = get_model(cfg)
    param_hist = merge_param_hist_with_hyperparams(
        model, checkpoint["params"], checkpoint["meta"]["param_hist"]
    )
    n_sessions = int(param_hist.pose.n_sessions[0])
    n_components = int(param_hist.pose.n_components[0])
    step = jnp.concatenate(
        [
            jnp.array([0]),
            checkpoint["meta"].get(
                "gd_step", np.arange(1, len(param_hist.pose.subj_weights))
            ),
        ]
    )[first_step::stepsize]
    pal = colors.make("Spectral")(n_sessions)
    fig, ax = plt.subplots(
        n_components,
        n_sessions + 1,
        figsize=(1.5 * n_sessions + 1.5, 1.5 * n_components),
    )
    for i_comp in _optional_pbar(range(n_components), progress):
        weights = param_hist.pose.subj_weights[..., i_comp]
        means = param_hist.pose.means[..., i_comp, :]
        for i_sess in range(n_sessions):
            ax[i_comp, i_sess].plot(
                step, weights[first_step::stepsize, i_sess], color=pal[i_sess]
            )
            ax[i_comp, i_sess].set_ylim(0, param_hist.pose.subj_weights.max())
            ax[0, i_sess].set_title(dataset.session_name(i_sess))
        ax[i_comp, -1].plot(
            step, means[first_step::stepsize], color=colors.neutral, lw=0.5
        )
        ax[i_comp, 0].set_ylabel(f"comp {i_comp}")
    ax[0, -1].set_title("Mean")

    return fig


def gmm_components(
    checkpoint: dict, dataset: Dataset = None, colors: colorset = None
):
    """Component means and weights of a GMM model."""
    if colors is None:
        colors = colorset.active

    cfg = checkpoint["config"]
    arm = Armature.from_config(cfg["dataset"])
    if dataset is None:
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
    weights_fig, ax = plt.subplots(
        figsize=(params.n_sessions * 0.2 + 1, params.n_components * 0.2 + 0.5)
    )
    mappable = ax.imshow(params.subj_weights.T)
    ax.set_xticks(range(len(dataset.sessions)))
    ax.set_xticklabels(
        [
            (
                "(ref) "
                if dataset.session_name(i) == dataset.ref_session
                else ""
            )
            + dataset._session_names[i]
            for i in range(len(dataset.sessions))
        ],
        rotation=80,
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
    output_path: Path,
    checkpoint: dict,
    dataset: Dataset = None,
    colors: colorset = None,
    group_lines=False,
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
    if dataset is None:
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


def plot_calibration(config: dict, colors=None):
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
        "morph": model.morph.plot_calibration(config, colors),
        "pose": model.pose.plot_calibration(config, colors),
    }


def display_clip_across_bodies(
    output_path,
    checkpoint,
    start,
    end,
    window_size,
    dataset: Dataset = None,
    source_session=None,
    n_cols=3,
    fixed_crop=False,
    subject_size=0.8,
    colors: colorset = None,
    font_scale=0.5,
    fps=30.0,
    progress=False,
    use_raw_video=True,
):
    """Display a video clip across all bodies in a session."""

    if colors is None:
        colors = colorset.active

    params: LRAParams = checkpoint["params"].morph
    config = _insert_calibration_data_to_checkpoint_config(checkpoint)
    model = get_model(config)
    if dataset is None:
        dataset, _ = load_and_prepare_dataset(config, allow_subsample=False)
    _inflate = lambda arr: inflate(arr, config["features"])
    armature = Armature.from_config(config["dataset"])

    if source_session is None:
        source_session = dataset.ref_session
    if use_raw_video:
        video, video_kpts = load_videos(
            config["dataset"], start, end, [source_session], progress=progress
        )
        frames = video[source_session]
        anterior_ix = config["dataset"]["viz"]["anterior_ix"]
        posterior_ix = config["dataset"]["viz"]["posterior_ix"]
        c, h, s = _scalar_summaries(
            video_kpts[source_session], anterior_ix, posterior_ix
        )

        raw_window_size = int(np.ceil(s.max() * 2 / subject_size))
        video_tile = _egocentric_crop(
            frames, c, h, raw_window_size, window_size, fixed_crop
        )

    # create a dataset with a session for each body, all containing identical
    # data: the clip from the source session
    ref_body = dataset.sess_bodies[source_session]
    ref_kpt_frames = dataset.get_session(source_session)[start:end]
    new_data, slices = stack_dict(
        {f"s-{b}": ref_kpt_frames for b in dataset.bodies}
    )
    new_bodies = {f"s-{b}": ref_body for b in dataset.bodies}
    short_data = dataset.with_sessions(
        new_data,
        slices,
        new_bodies,
        f"s-{ref_body}",
        _body_names=dataset._body_names,
    )
    # map each session onto the body for which the session is named
    mapped = apply_bodies(
        model.morph,
        params,
        short_data,
        {f"s-{b}": b for b in dataset.bodies},
    )

    inflated_ref = _inflate(short_data.get_session(short_data.ref_session))
    xaxis = 0
    mapped_tiles = {}
    for view_name, yaxis in zip(["top", "side"], [2, 1]):

        # align the reference an egocentric view and record params of the view
        # to align the mapped sessions as well
        rc, rh, rs = _scalar_summaries(
            inflated_ref[..., [xaxis, yaxis]],
            armature=armature,
        )
        if view_name == "side":
            rh *= 0  # do not rotate side view
        keypt_scale = rs.max() * 2 / subject_size

        ref_kpt_frames = _egocentric_window_align(
            np.array(inflated_ref[..., [xaxis, yaxis]]),
            rc,
            rh,
            keypt_scale,
            window_size,
            fixed_crop,
        )
        # plot the reference session keypoints
        backdrop = np.zeros([end - start, window_size, window_size, 3])
        ref_tile = _overlay_keypoints(
            backdrop,
            ref_kpt_frames,
            armature,
            keypoint_colors=plt_colors.to_rgb(colors.subtle),
        )
        pal = colors.cts1(dataset.n_bodies)
        mapped_tiles[view_name] = {}
        # align mapped sessions to the egocentric view from above and overlay
        # atop the reference session keypoints
        for i, b in enumerate(
            _optional_pbar(
                dataset.bodies, f"Rendering {view_name}" if progress else False
            )
        ):
            inflated = _inflate(mapped.get_session(f"s-{b}"))[
                ..., [xaxis, yaxis]
            ]
            scaled = _egocentric_window_align(
                np.array(inflated),
                rc,
                rh,
                keypt_scale,
                window_size,
                fixed_crop,
            )
            mapped_tiles[view_name][b] = _overlay_keypoints(
                ref_tile, scaled, armature, keypoint_colors=pal[i], copy=True
            )
            # mapped_tiles[view_name][b] = ref_tile

    # stack the various tiles together
    n_cols = int(min(n_cols, dataset.n_bodies))
    n_rows = int(np.ceil(dataset.n_bodies / n_cols))
    tiles = np.zeros(
        [n_rows, 2 * n_cols + 1, end - start, window_size, window_size, 3]
    )
    header_height = window_size // 4
    header_buff = np.zeros([n_rows, 1, header_height, window_size, 3])
    headers = np.zeros([n_rows, n_cols, header_height, window_size * 2, 3])
    for i, b in enumerate(dataset.bodies):
        # force reference body to be in the top left corner
        i = i + 1
        if b == ref_body:
            i = 0
        r, c = (i // n_cols, i % n_cols)
        c = 2 * c + 1
        tiles[r, c] = mapped_tiles["top"][b]
        tiles[r, c + 1] = mapped_tiles["side"][b]

        headers[r, i % n_cols] = cv2.putText(
            headers[r, i % n_cols],
            b,
            (2, header_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if use_raw_video:
        tiles[0, 0] = video_tile
    else:
        tiles = tiles[:, 1:]
        header_buff = header_buff[..., :0, :]

    # join tiles and headers into one large video
    # concatenate tiles within each row
    headers = np.concatenate(
        [header_buff[:, 0], *np.swapaxes(headers, 0, 1)], axis=-2
    )
    tiles = np.concatenate(tiles.swapaxes(0, 1), axis=-2)
    # alterate tiles and headers in columns and concatenate columns
    headers = np.broadcast_to(
        headers[:, None], tiles.shape[:2] + headers.shape[1:]
    )
    interleaved = np.concatenate([headers, tiles], axis=-3)
    tiled_vid = np.concatenate(interleaved, axis=-3)

    if output_path is not None:
        write_video(str(output_path), tiled_vid, fps=fps)

    return tiled_vid
