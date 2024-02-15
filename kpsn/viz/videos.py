from ..io.loaders import _get_root_path
from ..io.armature import Armature

from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
import imageio.v3 as iio


def add_videos_to_config(
    config,
    paths,
    video_root=None,
    keypoint_root=None,
    keypoint_type: str = "raw_npy",
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
    keypoint_root = Path("/") if keypoint_root is None else Path(keypoint_root)
    video_paths, video_root = _get_root_path(
        {s: video_root / paths[s]["video"] for s in paths}
    )
    keypt_paths, keypt_root = _get_root_path(
        {s: keypt_root / paths[s]["keypoints"] for s in paths}
    )
    config["dataset"]["viz"]["videos"] = dict(
        video_root=video_root,
        keypoint_root=keypt_root,
        keypoint_type=keypoint_type,
        **{
            s: dict(video=video_paths[s], keypoints=keypt_paths[s])
            for s in paths
        },
    )

    return config


def load_videos(config, start, end, whitelist=None):
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

    videos = {}
    keypts = {}
    for session in video_dict:
        if whitelist is not None and session not in whitelist:
            continue
        video_path = Path(video_root) / video_dict[session]["video"]
        keypt_path = Path(keypt_root) / video_dict[session]["keypoints"]

        # read segment of the video
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for i in range(end):
            res, frame = cap.read()
            if res:
                frames.append(frame)
            else:
                raise ValueError(f"Could not read frame {i} from {video_path}")
        videos[session] = np.stack(frames)

        # read keypoints

        if keypt_type == "raw_npy":
            full_keypts = np.load(keypt_path)
        else:
            raise ValueError(f"Unknown keypoint type: {keypt_type}")
        keypts[session] = full_keypts[start:end]

    return videos


def _centroid_headings_and_sizes(keypoints):
    """Return the centroid and heading (direciton of maximal variance) of
    keypoints.

    Parameters
    ----------
    keypoints : array, shape (n_frames, n_keypoints, 2)
        Array of keypoints.
    """
    centroid = np.mean(keypoints, axis=0)
    centered = keypoints - centroid[None]
    cov = (np.swapaxes(centered, -2, -1) @ centered) / centered.shape[1]
    _, vecs = np.linalg.eigh(cov)
    theta = np.arctan2(vecs[:, 1, -1], vecs[:, 0, -1])
    sizes = np.linalg.norm(centered, axies=-1).max(axis=-1)
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


def _overlay_keypoints(
    image,
    keypoints,
    armature: Armature,
    keypoint_cmap=None,
    keypoint_colors=None,
    node_size=5,
    line_width=2,
    copy=False,
    opacity=1.0,
):
    """Overlay keypoints on an image.

    Parameters
    ----------
    image: ndarray of shape (height, width, 3)
        Image to overlay keypoints on.

    keypoints: ndarray of shape (n_keypoints, 2)
        Array of keypoint keypoints.

    edges: list of tuples, default=[]
        List of edges that define the skeleton, where each edge is a
        pair of indexes.

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

    if keypoint_colors is None:
        cmap = plt.colormaps[keypoint_cmap]
        colors = np.array(cmap(np.linspace(0, 1, keypoints.shape[0])))[:, :3]
    elif keypoint_colors.ndim == 1:
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

        frame = cv2.transform(frame, np.float32(M))
        frame = frame * scaled_window_size / data_size
        tile.append(frame)

    return np.stack(tile)


def write_video(path, frames, fps, show_frame_numbers=False):
    with iio.get_writer(
        path, pixelformat="yuv420p", fps=fps, quality=5
    ) as writer:
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

            writer.append_data(frame)
