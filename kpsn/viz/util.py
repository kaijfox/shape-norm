from ..io.armature import Armature

import jax.numpy as jnp
import jax.numpy.linalg as jla
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging


def plot_mouse(
    ax,
    keypt_frame,
    armature: Armature,
    xaxis,
    yaxis,
    color="k",
    scatter_kw={},
    line_kw={},
    label=None,
    line2_kw=None,
    labelon="line",
    zorder=0,
    specialkp=None,
    debug=False,
):
    if specialkp is not None:
        ax.plot(
            [keypt_frame[specialkp, xaxis]],
            [keypt_frame[specialkp, yaxis] + 1],
            "ko",
            ms=2,
        )
    for i, parent in armature.bones:
        if debug:
            logging.info(
                f"i: {i},  {armature.keypoint_names[int(i)]} - parent: {parent} {armature.keypoint_names[int(parent)]}"
            )

        curr_child = keypt_frame[i]
        curr_parent = keypt_frame[parent]

        if line2_kw is not None:
            kws = [line2_kw, line_kw]
        else:
            kws = [line_kw]
        for kw in kws:
            ax.plot(
                (curr_child[xaxis], curr_parent[xaxis]),
                (curr_child[yaxis], curr_parent[yaxis]),
                **{
                    "color": color,
                    **kw,
                    **(
                        {}
                        if (labelon == "scatter" or i != 1)
                        else {"label": label}
                    ),
                },
                zorder=zorder,
            )

    ax.scatter(
        keypt_frame[..., xaxis],
        keypt_frame[..., yaxis],
        **{
            "s": 3,
            "color": color,
            **scatter_kw,
            **({} if labelon == "line" else {"label": label}),
        },
        zorder=zorder + 1,
    )

    ax.set_aspect(1.0)


def plot_mouse_views(
    axes,
    keypt_frame,
    armature: Armature,
    color="k",
    scatter_kw={},
    line_kw={},
    label=None,
    line2_kw=None,
    labelon="line",
    zorder=0,
    specialkp=None,
    debug=False,
):
    for ax, yaxis in zip(axes, [2, 1]):
        plot_mouse(
            ax,
            keypt_frame,
            armature,
            0,
            yaxis,
            color=color,
            scatter_kw=scatter_kw,
            line_kw=line_kw,
            label=label,
            line2_kw=line2_kw,
            labelon=labelon,
            zorder=zorder,
            specialkp=specialkp,
            debug=debug,
        )


def select_frame_gallery(
    keypoints,
    armature: Armature,
    return_ixs=False,
    as_list=False,
):
    """
    Parameters
    ----------
    keypoints : array
        Array of shape (nframe, nkeypoints, 3), such as a session from a
        KeypointDataset.
    armature : Armature
    """

    nframe = len(keypoints)
    anterior_kp = armature.keypt_by_name[armature.anterior]
    ht = jnp.argsort(keypoints[:, anterior_kp, 2])
    ln = jnp.argsort(keypoints[:, anterior_kp, 0])
    wd = jnp.argsort(keypoints[:, anterior_kp, 1])
    if return_ixs:
        quantile = lambda ix_arr, pct: ix_arr[int(pct * nframe)]
    else:
        quantile = lambda ix_arr, pct: keypoints[ix_arr[int(pct * nframe)]]

    poses = {
        "high": quantile(ht, 0.1),
        "low": quantile(ht, 0.9),
        "extend": quantile(ln, 0.2),
        "scrunch": quantile(ln, 0.9),
        "left": quantile(wd, 0.2),
        "right": quantile(wd, 0.8),
    }
    if as_list:
        poses = jnp.stack(list(poses.values()))
    return poses


def find_nearest_frames(query, library, return_ixs=False):
    """Find closest frames in a libraru to a set of query frames.

    Parameters
    ----------
    query : array, shape (n_query, n_features)
        Query frames.
    library : Dataset, or array shaped (n_library, n_features)
        Library of frames."""

    if hasattr(library, "_is_dataset"):
        return {
            s: find_nearest_frames(query, library.get_session(s), return_ixs)
            for s in library.sessions
        }
    query_flat = query.reshape((query.shape[0], -1))
    library_flat = library.reshape((library.shape[0], -1))

    dists = jla.norm(query_flat[:, None] - library_flat[None], axis=-1)
    selected_ixs = jnp.argmin(dists, axis=1)

    if return_ixs:
        return selected_ixs
    return library[selected_ixs]


def axes_off(ax):
    # run on each axis if an array of axes
    if hasattr(ax, "__len__"):
        for a in np.array(ax).ravel():
            axes_off(a)
        return
    # remove ticks and axes
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)


def legend(ax, handles_labels=None):
    if handles_labels is not None:
        ax.legend(
            *handles_labels,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            frameon=False,
        )
    else:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)


def unique_handles(ax):
    handles, labels = ax.get_legend_handles_labels()
    newLabels, newHandles = [], []
    labelSet = set()
    for handle, label in zip(handles, labels):
        if label not in labelSet:
            newLabels.append(label)
            newHandles.append(handle)
            labelSet.add(label)
    return newHandles, newLabels


def jitter_points(
    arr,
    shape=None,
    scale=0.1,
):
    if shape is None:
        shape = arr.shape
    return jnp.array(np.random.uniform(-scale, scale, shape)) + arr


def stripplot(*arrs, x=None, stacked=False, jitter=0.1):
    """
    Parameters
    ----------
    arrs : arrays
        One-dimensional arrays to plot in strips.
    x : array, optional
        Array of x-coordinates for each strip. If None, will use
        `0...len(arrs)`.
    stacked : bool, optional
        If True, will stack the arrays in `arrs`, otherwise will concatenate.
        Can allow plotting of lines between points.

    Returns
    -------
    xs, ys : array
        Coordinates to plot, concatenated or stacked.
    """
    if x is None:
        x = np.arange(len(arrs))
    if stacked:
        f = np.stack
    else:
        f = np.concatenate
    return (
        f(
            [
                jitter_points(_x, shape=arr.shape, scale=jitter)
                for _x, arr in zip(x, arrs)
            ]
        ),
        f(arrs),
    )


def flat_grid(total, n_col, ax_size, **subplot_kw):
    n_row = int(np.ceil(total / n_col))
    fig, ax = plt.subplots(
        n_row,
        n_col,
        figsize=(ax_size[0] * n_col, ax_size[1] * n_row),
        **subplot_kw,
    )
    ax = np.array(ax)
    if ax.ndim == 1:
        ax = ax[None, :]
    elif ax.ndim == 0:
        ax = ax[None, None]
    ax_ravel = ax.ravel()
    for a in ax_ravel[total:]:
        a.set_axis_off()

    return fig, ax_ravel[:total], ax
