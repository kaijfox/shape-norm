from ..io.armature import Armature

import jax.numpy as jnp
import jax.numpy.linalg as jla
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticks
import matplotlib.colors as mpl_color
from contextlib import contextmanager
from matplotlib.collections import LineCollection
import logging
import math


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
    for i_bone, (i, parent) in enumerate(armature.bones):

        curr_child = keypt_frame[i]
        curr_parent = keypt_frame[parent]

        if not isinstance(line_kw, dict):
            if line2_kw is not None:
                kws = [line2_kw[i_bone], line_kw[i_bone]]
            else:
                kws = [line_kw[i_bone]]
        else:
            if line2_kw is not None:
                kws = [line2_kw, line_kw]
            else:
                kws = [line_kw]

        if debug:
            print(
                f"bone: {i_bone}; "
                f"child: {i},  {armature.keypt_names[int(i)]} - "
                f"parent: {parent} {armature.keypt_names[int(parent)]}"
            )
            print(kws)
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


def axes_off(ax, y = True, x = True):
    # run on each axis if an array of axes
    if hasattr(ax, "__len__"):
        for a in np.array(ax).ravel():
            axes_off(a, y, x)
        return
    # remove ticks and axes
    if x:
        ax.set_xticks([])
    if y:
        ax.set_yticks([])
    sns.despine(ax=ax, left=y, bottom=x)


def legend(ax, handles_labels=None, **kw):
    kw = {
        **dict(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            frameon=False,
        ),
        **kw,
    }
    if handles_labels is not None:
        return ax.legend(
            *handles_labels,
            **kw,
        )
    else:
        return ax.legend(**kw)


def max_bounds(ax):
    vmin = np.min([ax.get_xlim()[0], ax.get_ylim()[0]])
    vmax = np.max([ax.get_xlim()[1], ax.get_ylim()[1]])
    return [vmin, vmax], [vmin, vmax]


def nticks(ax, n=2, x=True, y=True):
    if hasattr(ax, "__len__"):
        for a in np.array(ax).ravel():
            nticks(a, n=n, x=x, y=y)
        return
    if x:
        ax.xaxis.set_major_locator(
            mpl_ticks.LinearLocator(
                numticks=n,
            )
        )
    if y:
        ax.yaxis.set_major_locator(mpl_ticks.LinearLocator(numticks=n))


def _round_limits(vmin, vmax, precision=0):
    d = (vmax - vmin) / 10**precision
    order = np.floor(np.log10(d))
    scale = np.floor(d * 10 ** (-order)) * 10 ** (order)
    rvmin = np.floor(vmin / scale) * scale
    rvmax = np.ceil(vmax / scale) * scale
    return rvmin, rvmax


def round_limits(
    ax,
    precision=0,
    x=True,
    y=True,
    share=False,
    vmin=None,
    vmax=None,
    pad=0.05,
    fixed_vmin=None,
    fixed_vmax=None,
):

    # convert scalar vmin/vmax's to tuples
    if isinstance(vmin, (int, float)) or vmin is None:
        vmin = (vmin, vmin)
    if isinstance(vmax, (int, float)) or vmax is None:
        vmax = (vmax, vmax)
    if isinstance(fixed_vmin, (int, float)) or fixed_vmin is None:
        fixed_vmin = (fixed_vmin, fixed_vmin)
    if isinstance(fixed_vmax, (int, float)) or fixed_vmax is None:
        fixed_vmax = (fixed_vmax, fixed_vmax)

    # if share, get the min and max of all axes, and save as vmin, vmax
    ax_arr = np.atleast_1d(ax)
    if share:
        if vmin[0] is None and x:
            vmin = (min([a.get_xlim()[0] for a in ax_arr]), vmin[1])
        if vmin[1] is None and y:
            vmin = (vmin[0], min([a.get_ylim()[0] for a in ax_arr]))
        if vmax[0] is None and x:
            vmax = (max([a.get_xlim()[1] for a in ax_arr]), vmax[1])
        if vmax[1] is None and y:
            vmax = (vmax[0], max([a.get_ylim()[1] for a in ax_arr]))

    # apply separately to each axis in an array
    if hasattr(ax, "__len__"):
        for a in np.array(ax).ravel():
            round_limits(
                a,
                precision=precision,
                x=x,
                y=y,
                vmin=vmin,
                vmax=vmax,
                pad=pad,
                fixed_vmin=fixed_vmin,
                fixed_vmax=fixed_vmax,
            )
        return

    vmin = (
        vmin[0] if vmin[0] is not None else ax.get_xlim()[0],
        vmin[1] if vmin[1] is not None else ax.get_ylim()[0],
    )
    vmax = (
        vmax[0] if vmax[0] is not None else ax.get_xlim()[1],
        vmax[1] if vmax[1] is not None else ax.get_ylim()[1],
    )

    vrng = vmax[0] - vmin[0], vmax[1] - vmin[1]
    vmin = vmin[0] - pad * vrng[0], vmin[1] - pad * vrng[1]
    vmax = vmax[0] + pad * vrng[0], vmax[1] + pad * vrng[1]

    if x:
        xmin, xmax = _round_limits(vmin[0], vmax[0], precision=precision)
        xmin, xmax = (
            xmin if fixed_vmin[0] is None else fixed_vmin[0],
            xmax if fixed_vmax[0] is None else fixed_vmax[0],
        )
        ax.set_xlim(xmin, xmax)
    if y:
        ymin, ymax = _round_limits(vmin[1], vmax[1], precision=precision)
        ymin, ymax = (
            ymin if fixed_vmin[1] is None else fixed_vmin[1],
            ymax if fixed_vmax[1] is None else fixed_vmax[1],
        )
        ax.set_ylim(ymin, ymax)


def get_offsets(n, i=None):
    if i is None:
        i = np.arange(n)
    return (i - (n - 1) / 2) * min(0.2, 0.6 / n)


def get_aspect(ax):
    # Total figure size
    figW, figH = ax.get_figure().get_size_inches()
    # Axis size on figure
    _, _, w, h = ax.get_position().bounds
    # Ratio of display units
    disp_ratio = (figH * h) / (figW * w)
    # Ratio of data units
    # Negative over negative because of the order of subtraction
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    data_ratio = (ymax - ymin) / (xmax - xmin)

    return disp_ratio / data_ratio


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


def kde(sample, bw, resolution=200, buffer=5, density=False, eval_x = None):
    if eval_x is None:
        x = np.linspace(
            sample.min() - buffer * bw, sample.max() + buffer * bw, resolution
        )
    else:
        x = eval_x
    gauss = lambda x, sample, bw: np.exp(-0.5 * ((sample - x) / bw) ** 2)
    kde = gauss(x[:, None], sample[None, :], bw).sum(axis=-1)
    if density:
        kde /= kde.sum() * 2 * np.pi * bw**2
    return x, kde


def jitter_points(
    arr,
    shape=None,
    scale=0.1,
):
    if shape is None:
        shape = arr.shape
    return jnp.array(np.random.uniform(-scale, scale, shape)) + arr


def stripplot(*arrs, x=None, stacked=False, aslist=False, jitter=0.1, apply=()):
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
        f = lambda x: np.stack(x).squeeze()
    elif aslist:
        f = list
    else:
        f = lambda x: np.concatenate(x).squeeze()
    ret = (
        f(
            [
                jitter_points(_x, shape=arr.shape, scale=jitter)
                for _x, arr in zip(x, arrs)
            ]
        ),
        f(arrs),
    )
    if len(apply):
        applied = tuple(
            [np.atleast_1d(g(arr))[None] for arr in arrs] for g in apply
        )
        add_ret = tuple(
            (
                f([np.full(arr.shape[:-1], _x) for _x, arr in zip(x, arrs_)]),
                f(arrs_),
            )
            for arrs_ in applied
        )
        ret = ret + add_ret
    return ret


def multi_stripplot(
    arrs,
    x=None,
    stacked=False,
    aslist=False,
    jitter=0.1,
):
    """Apply stripplot to multiple arrays with shared x values
    arrs: iterable[iterable[array]]
        Lists of arrays to be passed to stripplot. Each iterable of arrays
        should contain arrays of the same shape in the same order.
    """
    kw = dict(x=x, stacked=stacked, aslist=aslist, jitter=jitter)
    strips = [stripplot(*a, **kw) for a in arrs]
    xs = strips[0][0]
    strips = [s[1] for s in strips]
    return xs, strips


def grouped_stripplot(
    data: dict,
    x=None,
    group_order=None,
    labels=None,
    colors=None,
    xticks=None,
    vlines=False,
    points=True,
    jitter=0.05,
    offset=True,
    lim_buffer=None,
    lighten_points=False,
    error=np.nanstd,
    errorbar=True,
    connect=False,
    legend=True,
    ax=None,
    errorbar_kw={},
    points_kw={},
    connection_kw={},
    vline_kw={},
    legend_kw={},
    xtick_kw={},
):
    """
    Parameters
    ----------
    data : dict,
        Dictionary of lists of arrays to plot in strips, with keys as hue
        labels, and list indices as indices in `x`. Therefore all values should
        have the same length."""
    if not isinstance(data, dict):
        data = {0: data}
    if ax is None:
        ax = plt.gca()
    if group_order is None:
        group_order = list(data.keys())
    if colors is None:
        colors = sns.color_palette(n_colors=len(data))
        colors = dict(zip(group_order, colors))
    if not isinstance(colors, dict) and colors is not None:
        colors = {k: colors for k in group_order}
    if labels is True:
        labels = {k: k for k in group_order}
    if not isinstance(labels, dict) and labels is not None:
        labels = {k: labels for k in group_order}
    if x is None:
        x = np.arange(len(data[group_order[0]]))
    if xticks is None:
        xticks = x
    if not isinstance(xticks, (list, tuple, np.ndarray, jnp.ndarray)):
        xticks = [xticks] * len(x)

    # merge kwargs with defaults
    vline_kw = {
        **{"color": ".8", "ls": "--", "lw": 0.5},
        **vline_kw,
    }
    errorbar_kw = {**{"fmt": "o", "ms": 3, "elinewidth": 1}, **errorbar_kw}
    points_kw = {**{"ms": 2}, **points_kw}
    legend_kw = {
        **{"frameon": False, "bbox_to_anchor": (1.05, 1), "loc": "upper left"},
        **legend_kw,
    }

    # plot! (vertical lines at x values)
    if vlines:
        for x_ in x:
            ax.axvline(x_, **vline_kw)

    # plot! actual data
    for i_grp, grp in enumerate(group_order):
        label = labels[grp] if labels is not None else None
        grp_color = colors[grp]

        ofs = (
            (i_grp - (len(group_order) - 1) / 2)
            * min(0.2, 0.6 / len(group_order))
            if offset
            else 0
        )

        strip_x, strip_y, (mean_x, mean_strip), (_, err_strip) = stripplot(
            *data[grp],
            x=x,
            apply=(np.mean, error),
            jitter=jitter,
            stacked=connect,
        )

        if points:
            if lighten_points is not False:
                c = lighten(grp_color, lighten_points)
            else:
                c = grp_color
            if connect:
                ax.plot(
                    strip_x + ofs,
                    strip_y,
                    **{**dict(ls="-", marker="", color=c), **connection_kw},
                )
            ax.plot(
                strip_x + ofs,
                strip_y,
                **{**dict(ls="", marker="o", color=c), **points_kw},
            )

        # symmetrize errorbars if only one value is provided by `error`
        if errorbar:
            if err_strip.ndim == 1:
                err_strip = np.stack([err_strip, err_strip], axis=-1)
            ax.errorbar(
                mean_x + ofs,
                mean_strip,
                yerr=err_strip.T,
                label=label,
                color=grp_color,
                **errorbar_kw,
            )

    if labels is not None and legend:
        ax.legend(**legend_kw)
    if xticks is not False:
        ax.set_xticks(x)
        ax.set_xticklabels(xticks, **xtick_kw)
    if lim_buffer is not None:
        ax.set_xlim([x[0] - lim_buffer, x[-1] + lim_buffer])

    return ax


def expand_groups(data, group_names, ndim = 1):
    """Convert list of arrays to dictionary of lists of arrays for grouped
    plots.
    
    data : list[array]
        List of arrays to be arranged into groups.
    group_names : list
        List of group name corresponding to each array.
    """
    e = [np.full([1] * ndim, np.nan)]
    return {k: e * i + [d] + e * (len(data) - i - 1) for i, (d, k) in enumerate(zip(data, group_names))}




def grouped_errorbar_strips(
    data: dict,
    x=None,
    group_order=None,
    labels=None,
    colors=None,
    xticks=None,
    vlines=False,
    connect=False,
    jitter=0.05,
    lim_buffer=None,
    error=np.nanstd,
    offset=True,
    ax=None,
    errorbar_kw={},
    vline_kw={},
    legend_kw={},
    xtick_kw={},
    connection_kw={},
):
    """
    Parameters
    ----------
    data : dict,
        Dictionary of lists of lists of arrays to plot in strips. The outermost
        key determines hue and offset around x. The outer index determines is an
        index into `x` (if provided) or an x value. The inner index runs over
        multiple each of which will be summarized as a mean and standard
        deviation.
    """
    if not isinstance(data, dict):
        data = {0: data}
    if ax is None:
        ax = plt.gca()
    if group_order is None:
        group_order = list(data.keys())
    if colors is None:
        colors = sns.color_palette(n_colors=len(data))
        colors = dict(zip(group_order, colors))
    if not isinstance(colors, dict) and colors is not None:
        colors = {k: colors for k in group_order}
    if not isinstance(labels, dict) and labels is not None:
        labels = {k: labels for k in group_order}
    if x is None:
        x = np.arange(len(data[group_order[0]]))
    if xticks is None:
        xticks = x

    # merge kwargs with defaults
    vline_kw = {
        **{"color": ".8", "ls": "--", "lw": 0.5},
        **vline_kw,
    }
    errorbar_kw = {**{"fmt": "o", "ms": 3, "elinewidth": 1}, **errorbar_kw}
    legend_kw = {
        **{"frameon": False, "bbox_to_anchor": (1.05, 1), "loc": "upper left"},
        **legend_kw,
    }

    # plot! (vertical lines at x values)
    if vlines:
        for x_ in x:
            ax.axvline(x_, **vline_kw)

    # plot! actual data
    for i_grp, grp in enumerate(group_order):
        label = labels[grp] if labels is not None else None
        grp_color = colors[grp]

        ofs = (i_grp - (len(group_order) - 1) / 2) * min(
            0.2, 0.6 / len(group_order)
        ) * offset

        means = [np.array([np.nanmean(arr) for arr in la]) for la in data[grp]]
        errs = [
            np.array(
                [np.broadcast_to(np.atleast_1d(error(arr)), (2,)) for arr in la]
            )
            for la in data[grp]
        ]
        strip_x, strip_ys = multi_stripplot(
            [
                means,
                [la_err[:, 0] for la_err in errs],
                [la_err[:, 1] for la_err in errs],
            ],
            x=x,
            jitter=jitter,
            stacked=connect,
        )

        if connect:
            ax.plot(
                strip_x + ofs,
                strip_ys[0],
                **{**dict(color=grp_color), **connection_kw},
            )

        ax.errorbar(
            strip_x.ravel() + ofs,
            strip_ys[0].ravel(),
            yerr=np.stack([strip_ys[1], strip_ys[2]]),
            label=label,
            color=grp_color,
            **errorbar_kw,
        )

    if labels is not None:
        ax.legend(*unique_handles(ax), **legend_kw)
    if xticks is not False:
        ax.set_xticks(x)
        ax.set_xticklabels(xticks, **xtick_kw)
    if lim_buffer is not None:
        ax.set_xlim([x[0] - lim_buffer, x[-1] + lim_buffer])

    return ax


def _grouped_plot(
    data: dict,
    func: callable,
    kws: dict,
    x=None,
    group_order=None,
    labels=None,
    colors=None,
    xticks=None,
    vlines=False,
    legend=True,
    offset=True,
    lim_buffer=None,
    ax=None,
    vline_kw={},
    legend_kw={},
    xtick_kw={},
):
    if not isinstance(data, dict):
        data = {0: data}
    if ax is None:
        ax = plt.gca()
    if group_order is None:
        group_order = list(data.keys())
    if colors is None:
        colors = sns.color_palette(n_colors=len(data))
        colors = dict(zip(group_order, colors))
    if not isinstance(colors, dict) and colors is not None:
        colors = {k: colors for k in group_order}
    if labels is True:
        labels = {k: k for k in group_order}
    if not isinstance(labels, dict) and labels is not None:
        labels = {k: labels for k in group_order}
    if x is None:
        x = np.arange(len(data[group_order[0]]))
    if xticks is None:
        xticks = x
    if not isinstance(xticks, (list, tuple, np.ndarray, jnp.ndarray)):
        xticks = [xticks] * len(x)

    # merge kwargs with defaults
    vline_kw = {
        **{"color": ".8", "ls": "--", "lw": 0.5},
        **vline_kw,
    }
    legend_kw = {
        **{"frameon": False, "bbox_to_anchor": (1.05, 1), "loc": "upper left"},
        **legend_kw,
    }

    # plot! (vertical lines at x values)
    if vlines:
        for x_ in x:
            ax.axvline(x_, **vline_kw)

    # plot! actual data
    for i_grp, grp in enumerate(group_order):
        label = labels[grp] if labels is not None else None
        grp_color = colors[grp]

        ofs = (
            (i_grp - (len(group_order) - 1) / 2)
            * min(0.2, 0.6 / len(group_order))
            if offset
            else 0
        )

        func(data[grp], ofs, x, grp_color, label, ax, **kws)

    if labels is not None and legend:
        ax.legend(**legend_kw)
    if xticks is not False:
        ax.set_xticks(x)
        ax.set_xticklabels(xticks, **xtick_kw)
    if lim_buffer is not None:
        ax.set_xlim([x[0] - lim_buffer, x[-1] + lim_buffer])

    return ax


def __grouped_stripplot(arrs, ofs, x, color, label, ax, **kws):
    error = kws["error"]
    points = kws["points"]
    lighten_points = kws["lighten_points"]
    errorbar_kw = kws["errorbar_kw"]
    points_kw = kws["points_kw"]
    jitter = kws["jitter"]

    strip_x, strip_y, (mean_x, mean_strip), (_, err_strip) = stripplot(
        *arrs,
        x=x,
        apply=(np.mean, error),
        jitter=jitter,
    )

    if points:
        if lighten_points is not False:
            c = lighten(color, lighten_points)
        else:
            c = color
        ax.plot(strip_x + ofs, strip_y, "o", color=c, **points_kw)

    # symmetrize errorbars if only one value is provided by `error`
    if err_strip.ndim == 1:
        err_strip = np.stack([err_strip, err_strip], axis=-1)
    ax.errorbar(
        mean_x + ofs,
        mean_strip,
        yerr=err_strip.T,
        label=label,
        color=color,
        **errorbar_kw,
    )


def refactor_grouped_stripplot(
    data: dict,
    x=None,
    group_order=None,
    labels=None,
    colors=None,
    xticks=None,
    vlines=False,
    points=True,
    jitter=0.05,
    offset=True,
    lim_buffer=None,
    lighten_points=False,
    error=np.nanstd,
    ax=None,
    errorbar_kw={},
    points_kw={},
    vline_kw={},
    legend_kw={},
    xtick_kw={},
):
    """
    Parameters
    ----------
    data : dict,
        Dictionary of lists of arrays to plot in strips, with keys as hue
        labels, and list indices as indices in `x`. Therefore all values should
        have the same length."""

    # merge kwargs with defaults
    errorbar_kw = {**{"fmt": "o", "ms": 3, "elinewidth": 1}, **errorbar_kw}
    points_kw = {**{"ms": 2}, **points_kw}

    return _grouped_plot(
        data,
        __grouped_stripplot,
        {
            "error": error,
            "points": points,
            "lighten_points": lighten_points,
            "errorbar_kw": errorbar_kw,
            "points_kw": points_kw,
            "jitter": jitter,
        },
        x=x,
        group_order=group_order,
        labels=labels,
        colors=colors,
        xticks=xticks,
        vlines=vlines,
        offset=offset,
        lim_buffer=lim_buffer,
        ax=ax,
        vline_kw=vline_kw,
        legend_kw=legend_kw,
        xtick_kw=xtick_kw,
    )


def __grouped_violins(arrs, ofs, x, color, label, ax, **kws):
    """
    Arrs: list of list of arrays. First index is x, second is multiple arrays to
    be plotted at x."""
    error = kws["error"]
    errorbar_kw = kws["errorbar_kw"]
    errorbar = kws["errorbar"]
    fill_kw = kws["fill_kw"]
    points = kws["points"]
    lighten_points = kws["lighten_points"]
    points_kw = kws["points_kw"]
    point_jitter = kws["point_jitter"]
    jitter = kws["jitter"]
    thumb_bandwidth = kws["thumb_bandwidth"]
    bw = kws["bw"]
    resolution = kws["resolution"]
    width = kws["width"]
    bw_buffer = kws["bw_buffer"]

    means = [np.array([np.nanmean(arr) for arr in la]) for la in arrs]
    errs = [
        np.array(
            [np.broadcast_to(np.atleast_1d(error(arr)), (2,)) for arr in la]
        )
        for la in arrs
    ]

    strip_x, strip_ys = multi_stripplot(
        [
            means,
            [la_err[:, 0] for la_err in errs],
            [la_err[:, 1] for la_err in errs],
        ],
        x=x,
        jitter=jitter,
        aslist=True,
    )

    for x__, arrs__ in zip(strip_x, arrs):
        for x_, arr in zip(x__, arrs__):
            if thumb_bandwidth:
                iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
                bw_ = bw * 0.9 * min(np.std(arr), iqr / 1.34) * len(arr) ** (-0.2)
            y, k = kde(arr, bw, resolution=resolution, buffer=bw_buffer)
            k = k / k.max() * width / 2
            k = k - k.min()
            ax.fill_betweenx(
                y,
                x_ + ofs - k,
                x_ + ofs + k,
                label=label,
                **{
                    **{
                        "fc": color,
                        "ec": color,
                    },
                    **fill_kw,
                },
            )

            if points:
                if lighten_points is not False:
                    c = lighten(color, lighten_points)
                else:
                    c = color
                ax.plot(
                    jitter_points(np.full_like(arr, x_), scale=point_jitter)
                    + ofs,
                    arr,
                    **{**dict(color=c, ls="", marker="o"), **points_kw},
                )

        if errorbar:
            ax.errorbar(
                np.concatenate(strip_x) + ofs,
                np.concatenate(strip_ys[0]),
                yerr=np.stack(
                    [np.concatenate(strip_ys[1]), np.concatenate(strip_ys[2])]
                ),
                color=color,
                **errorbar_kw,
            )


def grouped_violins(
    data: dict,
    x=None,
    group_order=None,
    labels=None,
    colors=None,
    xticks=None,
    vlines=False,
    bw=None,
    bw_buffer=False,
    resolution=200,
    width=0.3,
    jitter=0.05,
    offset=True,
    lim_buffer=None,
    fill=False,
    stroke=True,
    error=np.nanstd,
    errorbar=True,
    thumb_bandwidth=False,
    legend=True,
    points=False,
    lighten_points=False,
    point_jitter=0.1,
    ax=None,
    errorbar_kw={},
    points_kw={},
    vline_kw={},
    legend_kw={},
    xtick_kw={},
    fill_kw={},
):
    """
    Parameters
    ----------
    data : dict,
        Dictionary of lists of lists of arrays to plot in strips, with keys as hue
        labels, and list indices as indices in `x`. Therefore all values should
        have the same length."""

    # merge kwargs with defaults

    errorbar_kw = {**{"fmt": "o", "ms": 3, "elinewidth": 1}, **errorbar_kw}
    legend_kw = {
        **{"frameon": False, "bbox_to_anchor": (1.05, 1), "loc": "upper left"},
        **legend_kw,
    }
    points_kw = {**{"ms": 2}, **points_kw}
    if not fill:
        fill_kw = {**{"fc": (1, 1, 1, 0)}, **fill_kw}
    if not stroke:
        fill_kw = {**{"ec": (1, 1, 1, 0)}, **fill_kw}

    return _grouped_plot(
        data,
        __grouped_violins,
        {
            "error": error,
            "errorbar_kw": errorbar_kw,
            "errorbar": errorbar,
            "fill_kw": fill_kw,
            "points": points,
            "points_kw": points_kw,
            "lighten_points": lighten_points,
            "point_jitter": point_jitter,
            "thumb_bandwidth": thumb_bandwidth,
            "jitter": jitter,
            "bw": bw,
            "resolution": resolution,
            "width": width,
            "bw_buffer": bw_buffer,
        },
        x=x,
        group_order=group_order,
        labels=labels,
        colors=colors,
        xticks=xticks,
        vlines=vlines,
        offset=offset,
        lim_buffer=lim_buffer,
        legend=legend,
        ax=ax,
        vline_kw=vline_kw,
        legend_kw=legend_kw,
        xtick_kw=xtick_kw,
    )


def __grouped_violin_points(arrs, ofs, x, color, label, ax, **kws):
    """
    Arrs: list of list of arrays. First index is x, second is multiple arrays to
    be plotted at x."""
    error = kws["error"]
    errorbar_kw = kws["errorbar_kw"]
    errorbar = kws["errorbar"]
    points = kws["points"]
    lighten_points = kws["lighten_points"]
    points_kw = kws["points_kw"]
    point_jitter = kws["point_jitter"]
    jitter = kws["jitter"]
    bw = kws["bw"]
    resolution = kws["resolution"]
    width = kws["width"]

    means = [np.array([np.nanmean(arr) for arr in la]) for la in arrs]
    errs = [
        np.array(
            [np.broadcast_to(np.atleast_1d(error(arr)), (2,)) for arr in la]
        )
        for la in arrs
    ]

    strip_x, strip_ys = multi_stripplot(
        [
            means,
            [la_err[:, 0] for la_err in errs],
            [la_err[:, 1] for la_err in errs],
        ],
        x=x,
        jitter=jitter,
        aslist=True,
    )

    for x__, arrs__ in zip(strip_x, arrs):
        for x_, arr in zip(x__, arrs__):
            y, k = kde(arr, bw, eval_x = arr)
            k = k / k.max() * width / 2
            k = k - k.min()

            if points:
                if lighten_points is not False:
                    c = lighten(color, lighten_points)
                else:
                    c = color
                jtr = jitter_points(np.full_like(arr, 0), scale=1) * k
                ax.plot(
                    jtr + ofs + x_,
                    arr,
                    **{**dict(color=c, ls="", marker="o"), **points_kw},
                )

        if errorbar:
            ax.errorbar(
                np.concatenate(strip_x) + ofs,
                np.concatenate(strip_ys[0]),
                yerr=np.stack(
                    [np.concatenate(strip_ys[1]), np.concatenate(strip_ys[2])]
                ),
                color=color,
                **errorbar_kw,
            )



def grouped_violin_points(
    data: dict,
    x=None,
    group_order=None,
    labels=None,
    colors=None,
    xticks=None,
    vlines=False,
    bw=None,
    resolution=200,
    width=0.3,
    jitter=0.05,
    offset=True,
    lim_buffer=None,
    error=np.nanstd,
    errorbar=True,
    legend=True,
    points=True,
    lighten_points=False,
    point_jitter=0.1,
    ax=None,
    errorbar_kw={},
    points_kw={},
    vline_kw={},
    legend_kw={},
    xtick_kw={},
):
    """
    Parameters
    ----------
    data : dict,
        Dictionary of lists of lists of arrays to plot in strips, with keys as hue
        labels, and list indices as indices in `x`. Therefore all values should
        have the same length."""

    # merge kwargs with defaults

    errorbar_kw = {**{"fmt": "o", "ms": 3, "elinewidth": 1}, **errorbar_kw}
    legend_kw = {
        **{"frameon": False, "bbox_to_anchor": (1.05, 1), "loc": "upper left"},
        **legend_kw,
    }
    points_kw = {**{"ms": 2}, **points_kw}

    return _grouped_plot(
        data,
        __grouped_violin_points,
        {
            "error": error,
            "errorbar_kw": errorbar_kw,
            "errorbar": errorbar,
            "points": points,
            "points_kw": points_kw,
            "lighten_points": lighten_points,
            "point_jitter": point_jitter,
            "jitter": jitter,
            "bw": bw,
            "resolution": resolution,
            "width": width,
        },
        x=x,
        group_order=group_order,
        labels=labels,
        colors=colors,
        xticks=xticks,
        vlines=vlines,
        offset=offset,
        lim_buffer=lim_buffer,
        legend=legend,
        ax=ax,
        vline_kw=vline_kw,
        legend_kw=legend_kw,
        xtick_kw=xtick_kw,
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


def ci(
    arr,
    stat=np.mean,
    level=0.95,
    seed=0,
    n=100,
    alternative="two-sided",
    absolute = False,
    return_samples=False,
):
    sample_stat = stat(arr)
    rng = np.random.default_rng(seed)
    samples = stat(
        arr[rng.choice(arr.shape[0], (n, *arr.shape), replace=True)], axis=1
    )
    if return_samples:
        return samples
    if alternative == "two-sided":
        edge = (1 - level) / 2
        ci = np.quantile(samples, [edge, 1 - edge], axis=0)
        if absolute: return ci
        return sample_stat - ci[0], ci[1] - sample_stat
    elif alternative == "less":
        ci = np.quantile(samples, level, axis=0)
        if absolute: return ci
        return sample_stat - ci
    elif alternative == "greater":
        if absolute: return ci
        ci = np.quantile(samples, 1 - level, axis=0)
        return ci - sample_stat


def lighten(c, factor):
    c = np.array(mpl_color.to_rgba(c))
    return (1 - factor) * c + factor * np.array([1, 1, 1, c[3]])


def stem_bar(y, center, as_kw=True):
    """
    Returns
    -------
    height, bottom
    or
    dict(height, bottom)"""
    bottom = np.minimum(y, center)
    height = np.maximum(y, center) - bottom
    if as_kw:
        return dict(height=height, bottom=bottom)
    return height, bottom


def stack_lines(xs, ys, cs, **kws):
    # Ensure xs, ys, and cs are numpy arrays
    # Check that the lengths of xs, ys, and cs are the same
    if len(xs) != len(ys) or len(xs) != len(cs):
        raise ValueError("xs, ys, and cs must have the same length")

    # Find the maximum length among xs and ys
    max_len = np.max([len(x) for x in xs] + [len(y) for y in ys])
    pad_kw = dict(mode="constant", constant_values=np.nan)
    # Pad xs and ys with nans at the end to make them the same length
    xs = [
        np.pad(
            x, (0, max_len - len(x)), **pad_kw
        )
        for x in xs
    ]
    ys = [
        np.pad(
            y, (0, max_len - len(y)), **pad_kw
        )
        for y in ys
    ]

    # Create a list of (x, y) points for each pair of xs and ys
    points = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    # Create a LineCollection from the points, with colors specified by cs
    line_collection = LineCollection(points, colors=cs, **kws)
    return line_collection
