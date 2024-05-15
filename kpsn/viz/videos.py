from ..io.loaders import _get_root_path
from ..io.armature import Armature
from .util import plot_mouse
from ..io.utils import stack_dict
from ..models.util import _optional_pbar, apply_bodies
from ..models.instantiation import get_model
from ..fitting.methods import load_and_prepare_dataset

from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
import imageio.v3 as iio
import logging
from matplotlib import colors as plt_colors


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
)
from ..io.utils import stack_dict
from ..models.morph.lowrank_affine import LRAParams, model as lra_model
from .styles import colorset
from ..fitting.methods import load_and_prepare_dataset, load_fit
from ..io.armature import Armature
from ..models.util import apply_bodies, _optional_pbar
from ..io.features import inflate
from ..project.paths import Project
from ..config import load_model_config
from ..models.instantiation import get_model
from ..fitting.scans import merge_param_hist_with_hyperparams


def add_videos_to_config(
    config,
    paths,
    video_root=None,
    keypoint_root=None,
    keypoint_type: str = "raw_npy",
    display_range=None,
    anterior_ix=None,
    posterior_ix=None,
    choose_with_frame=0,
):
    """Add video paths
    Parameters
    ----------
    config : dict
        Full config dictionary.

    paths : dict
        Dictionary mapping session names to a path for "video" and a path for
        "keypoints".

    video_root, keypoint_root : str or Path, optional
        Root directory for videos and keypoints respectively. If None, paths are
        assumed to be absolute.
    """

    video_root = Path("/") if video_root is None else Path(video_root)
    keypt_root = Path("/") if keypoint_root is None else Path(keypoint_root)
    video_root, video_paths = _get_root_path(
        {s: video_root / paths[s]["video"] for s in paths}
    )
    keypt_root, keypt_paths = _get_root_path(
        {s: keypt_root / paths[s]["keypoints"] for s in paths}
    )

    # add config necessary to load videos
    config["dataset"]["viz"].update(
        dict(
            video_root=video_root,
            keypoint_root=keypt_root,
            keypoint_type=keypoint_type,
            video_display_range=display_range,
            videos={
                s: dict(video=video_paths[s], keypoints=keypt_paths[s])
                for s in paths
            },
        )
    )

    # allow user to specify anterior/posterior indices if not given
    if anterior_ix is None or posterior_ix is None:
        logging.error(
            "[add_videos_to_config] No anterior/posterior indices provided. "
            "Plotting keypoint indices. "
            "Please choose an anterior and posterior keypoint and rerun. "
            "Use argument `choose_with_frame` to specify the frame to display."
        )
        _choose_anterior_posterior_keypoints(config, frame=choose_with_frame)
        return config

    config["dataset"]["viz"].update(
        dict(anterior_ix=anterior_ix, posterior_ix=posterior_ix)
    )

    return config


def _choose_anterior_posterior_keypoints(config, frame=0):
    config
    example = list(config["dataset"]["viz"]["videos"].keys())[0]
    v, k = load_videos(config["dataset"], frame, frame + 1, whitelist=[example])
    c, h, s = _scalar_summaries(k[example])
    k = _egocentric_window_align(k[example], c, h, s.max() * 3, 256)
    v = _egocentric_crop(v[example], c, h, int(s.max() * 3), 256)
    fig, ax = _overlay_keypoint_numbers(v[0], k[0], point_color="r")
    ax.set_title(f"keypoint indices | {example}, frame {frame}")
    plt.show()


def load_videos(config, start, end, whitelist=None, progress=False):
    """
    Parameters
    ----------
    config : dict
        `dataset` section of config dictionary.

    start, end : int
        Start and end frame of the clip to load for each session.

    """
    viz_config = config["viz"]
    video_dict = viz_config["videos"]
    keypt_type = viz_config["keypoint_type"]
    video_root = viz_config["video_root"]
    keypt_root = viz_config["keypoint_root"]
    display_range = viz_config["video_display_range"]

    videos = {}
    keypts = {}
    for session in video_dict:
        if whitelist is not None and session not in whitelist:
            continue
        video_path = Path(video_root) / video_dict[session]["video"]
        keypt_path = Path(keypt_root) / video_dict[session]["keypoints"]

        # read segment of the video
        print("video path", video_path)
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for i in _optional_pbar(
            np.arange(end),
            f"Loading {session}" if progress is not False else False,
        ):
            res, frame = cap.read()
            if res:
                if display_range is not None:
                    l, h = display_range
                    frame = (np.clip(frame, l, h) - l) / (h - l)
                if frame.max() < 1.5:
                    frame = (frame * 255).astype(np.uint8)
                frames.append(frame)
            else:
                raise ValueError(f"Could not read frame {i} from {video_path}")
        videos[session] = np.stack(frames)

        # read keypoints

        if keypt_type == "raw_npy":
            full_keypts = np.load(keypt_path)
        elif keypt_type.startswith("h5"):
            import h5py

            with h5py.File(keypt_path, "r") as f:
                full_keypts = f[keypt_type.split(":")[1]][:]
        else:
            raise ValueError(f"Unknown keypoint type: {keypt_type}")
        keypts[session] = full_keypts[start:end]

    return videos, keypts


def _scalar_summaries(
    keypoints,
    anterior_ix=None,
    posterior_ix=None,
    armature=None,
    smooth_headings=1,
):
    """Return the centroid and heading (direciton of maximal variance) of
    keypoints.

    Parameters
    ----------
    keypoints : array, shape (n_frames, n_keypoints, 2)
        Array of keypoints.
    smooth_headings : bool or int, default=False
        If True, smooth the heading with a rolling mean of width 5. If an int,
        use the specified width.
    """
    if armature is not None:
        anterior_ix = armature.keypt_by_name[armature.anterior]
        posterior_ix = armature.keypt_by_name[armature.posterior]
    centroid = np.mean(keypoints, axis=1)
    centered = keypoints - centroid[:, None]
    if anterior_ix is None or posterior_ix is None:
        theta = np.zeros(len(keypoints))
    else:
        vec = centered[:, anterior_ix] - centered[:, posterior_ix]
        theta = np.arctan2(vec[:, 1], vec[:, 0])
    if smooth_headings:
        theta = np.unwrap(theta)
        if smooth_headings < 2:
            smooth_headings = 5
        theta = np.convolve(
            theta, np.ones(smooth_headings) / smooth_headings, mode="same"
        )

    sizes = np.linalg.norm(centered, axis=-1).max(axis=-1)
    return centroid, theta, sizes


def _egocentric_crop(
    frames,
    centroids,
    headings,
    window_size,
    scaled_window_size,
    fixed_crop=False,
):
    """Crop the frames to an egocentric view.

    Parameters
    ----------
    frames : array, shape (n_frames, height, width, 3)
        Array of frames.

    centroids : array, shape (n_frames, 2)
        Array of centroid positions.

    headings : array, shape (n_frames,)
        Array of heading angles in radians.

    window_size : int
        Size of the window in coordinates of the original video.

    scaled_window_size : int
        Size of the video to return in pixels.
    """

    if fixed_crop:
        use_ix = [len(frames) // 2] * len(frames)
        hs, cs = headings[use_ix], centroids[use_ix]
    else:
        hs, cs = headings, centroids

    rs = np.float32([[np.cos(hs), np.sin(hs)], [-np.sin(hs), np.cos(hs)]])
    rs = rs.transpose((2, 0, 1))

    tile = []

    for i, (frame, h, c, r) in enumerate(zip(frames, hs, cs, rs)):

        d = r @ c - window_size // 2
        M = [[np.cos(h), np.sin(h), -d[0]], [-np.sin(h), np.cos(h), -d[1]]]

        frame = cv2.warpAffine(frame, np.float32(M), (window_size, window_size))
        frame = cv2.resize(frame, (scaled_window_size, scaled_window_size))
        tile.append(frame)

    return np.stack(tile)


def _overlay_keypoint_numbers(
    frame, keypoints, text_color="k", point_color="w"
):
    """Overlay keypoints and their indices on a frame."""
    fig, ax = plt.subplots()
    ax.imshow(frame)
    for i, (x, y) in enumerate(keypoints):
        ax.plot(x, y, "o", ms=8, color=point_color)
        ax.text(x - 4, y + 4, str(i), color=text_color, fontsize=8)
    ax.axis("off")
    return fig, ax


def _overlay_keypoints(
    image,
    keypoints,
    armature: Armature,
    keypoint_cmap=None,
    keypoint_colors=None,
    node_size=2,
    line_width=1,
    copy=False,
    opacity=1.0,
    progress=False,
):
    """Overlay keypoints on an image.

    Parameters
    ----------
    image: ndarray of shape (height, width, 3)
        Image to overlay keypoints on.

    keypoints: ndarray of shape (n_keypoints, 2)
        Array of keypoint keypoints.

    keypoint_cmap: str, default='autumn'
        Name of a matplotlib colormap to use for coloring the keypoints.

    keypoint_colors : array-like, shape=(num_keypoints,3), default=None
        Color for each keypoint. If None, the keypoint colormap is used.
        If the dtype is int, the values are assumed to be in the range 0-255,
        otherwise they are assumed to be in the range 0-1. If `ndim` is 1, then
        all keypoints are plotted with the same color.

    node_size: int, default=5
        Size of the keypoints.

    line_width: int, default=2
        Width of the skeleton lines.

    copy: bool, default=False
        Whether to copy the image before overlaying keypoints.

    opacity: float, default=1.0
        Opacity of the overlay graphics (0.0-1.0).

    Returns
    -------
    image: ndarray of shape (height, width, 3)
        Image with keypoints overlayed.
    """
    if copy or opacity < 1.0:
        canvas = image.copy()
    else:
        canvas = image

    if canvas.ndim > 3:
        for i in _optional_pbar(range(canvas.shape[0]), progress):
            canvas[i] = _overlay_keypoints(
                canvas[i],
                keypoints[i],
                armature,
                keypoint_cmap,
                keypoint_colors,
                node_size,
                line_width,
                False,  # already copied if requested
                opacity,
            )
        return canvas

    if keypoint_colors is None:
        cmap = plt.colormaps[keypoint_cmap]
        colors = np.array(cmap(np.linspace(0, 1, keypoints.shape[0])))[:, :3]
    elif np.array(keypoint_colors).ndim == 1:
        colors = np.array([keypoint_colors] * len(keypoints))
    else:
        colors = np.array(keypoint_colors)

    if isinstance(colors[0, 0], float):
        colors = [tuple([int(c) for c in cs * 255]) for cs in colors]

    # overlay skeleton
    for i, j in armature.bones:
        if np.isnan(keypoints[i, 0]) or np.isnan(keypoints[j, 0]):
            continue
        pos1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
        pos2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
        canvas = cv2.line(
            canvas, pos1, pos2, colors[i], line_width, cv2.LINE_AA
        )

    # overlay keypoints
    for i, (x, y) in enumerate(keypoints):
        if np.isnan(x) or np.isnan(y):
            continue
        pos = (int(x), int(y))
        canvas = cv2.circle(
            canvas, pos, node_size, colors[i], -1, lineType=cv2.LINE_AA
        )

    if opacity < 1.0:
        image = cv2.addWeighted(image, 1 - opacity, canvas, opacity, 0)
    return image


def _egocentric_window_align(
    keypoints,
    centroids,
    headings,
    data_size,
    scaled_window_size,
    fixed_crop=False,
):
    """Align keypoints to the egocentric window.

    Parameters
    ----------
    keypoints : array, shape (n_frames, n_keypoints, 2)
        Array of keypoints.

    centroids, headings : array, shape (n_frames, 2)
        Arrays of centroid positions and heading angles in radians.

    data_size : int
        Size of the subject original video.

    scaled_window_size : int
        Size of the video to return in pixels.

    fixed_crop : bool, default=False
        Whether to use a fixed crop for all based on the midpoint of the clip.
    """
    if fixed_crop:
        use_ix = [len(keypoints) // 2] * len(keypoints)
        hs, cs = headings[use_ix], centroids[use_ix]
    else:
        hs, cs = headings, centroids

    rs = np.float32([[np.cos(hs), np.sin(hs)], [-np.sin(hs), np.cos(hs)]])
    rs = rs.transpose((2, 0, 1))

    tile = []

    for i, (frame, h, c, r) in enumerate(zip(keypoints, hs, cs, rs)):

        d = r @ c - data_size // 2
        M = [[np.cos(h), np.sin(h), -d[0]], [-np.sin(h), np.cos(h), -d[1]]]

        frame = cv2.transform(frame[None], np.float32(M))[0]
        frame = frame * scaled_window_size / data_size

        frame[:, 1] = scaled_window_size - frame[:, 1]

        tile.append(frame)

    return np.stack(tile)


def write_video(path, frames, fps, show_frame_numbers=False):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with iio.imopen(str(path), "w", plugin="pyav") as writer:
        writer.init_video_stream("h264", fps=fps, pixel_format="yuv420p")
        for i, frame in enumerate(frames):

            if show_frame_numbers:
                frame = cv2.putText(
                    frame,
                    f"Frame {i}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            writer.write_frame(frame)


def map_clip_across_bodies(
    checkpoint, start, end, dataset=None, model=None, source_session=None
):
    params = checkpoint["params"].morph
    cfg = checkpoint["config"]
    if dataset is None:
        dataset, _ = load_and_prepare_dataset(cfg, allow_subsample=False)
    if model is None:
        model = get_model(cfg)
    if source_session is None:
        source_session = dataset.ref_session

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
    return (
        apply_bodies(
            model.morph,
            params,
            short_data,
            {f"s-{b}": b for b in dataset.bodies},
        ),
        short_data.get_session(short_data.ref_session),
    )


def overlay_tiles_for_bodies(
    ref_keypts,
    mapped_dataset,
    bodies,
    fixed_crop=False,
    subject_size=0.8,
    window_size=200,
    progress=True,
    colors: colorset = None,
    _inflate=None,
    armature=None,
    config=None,
    pal=None,
):

    if colors is None:
        colors = colorset.active
    if _inflate is None:
        _inflate = lambda arr: inflate(arr, config["features"])
    if armature is None:
        armature = Armature.from_config(config["dataset"])
    if pal is None:
        pal = dict(zip(bodies, colors.cts1(len(bodies))))

    inflated_ref = _inflate(ref_keypts)
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
        backdrop = np.zeros([len(ref_keypts), window_size, window_size, 3])
        ref_tile = _overlay_keypoints(
            backdrop,
            ref_kpt_frames,
            armature,
            keypoint_colors=plt_colors.to_rgb(colors.subtle),
        )

        mapped_tiles[view_name] = {}
        # align mapped sessions to the egocentric view from above and overlay
        # atop the reference session keypoints
        for b in _optional_pbar(
                bodies, f"Rendering {view_name}" if progress else False
            ):
            inflated = _inflate(mapped_dataset.get_session(f"s-{b}"))[
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
                ref_tile, scaled, armature, keypoint_colors=pal[b], copy=True
            )

    return mapped_tiles


def join_view_tiles(
    bodies,
    mapped_tiles,
    n_cols,
    font_scale=0.5,
):

    n_cols = int(min(n_cols, len(bodies)))
    n_rows = int(np.ceil(len(bodies) / n_cols))
    n_frames = mapped_tiles["top"][bodies[0]].shape[0]
    window_size = mapped_tiles["top"][bodies[0]].shape[-2]

    # stack the various tiles together
    tiles = np.zeros(
        [n_rows, 2 * n_cols, n_frames, window_size, window_size, 3]
    )
    header_height = window_size // 4
    headers = np.zeros([n_rows, n_cols, header_height, window_size * 2, 3])
    for i, b in enumerate(bodies):
        r, c = (i // n_cols, i % n_cols)
        c = 2 * c
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

    # join tiles and headers into one large video
    # concatenate tiles within each row
    headers = np.concatenate(np.swapaxes(headers, 0, 1), axis=-2)
    tiles = np.concatenate(tiles.swapaxes(0, 1), axis=-2)
    # alterate tiles and headers in columns and concatenate columns
    headers = np.broadcast_to(
        headers[:, None], tiles.shape[:2] + headers.shape[1:]
    )
    interleaved = np.concatenate([headers, tiles], axis=-3)
    tiled_vid = np.concatenate(interleaved, axis=-3)

    return tiled_vid
