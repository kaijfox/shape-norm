from .styles import colorset
from ..io.loaders import load_dataset
from ..io.alignment import align
from ..io.armature import Armature
from .util import plot_mouse_views, select_frame_gallery, axes_off, legend

import matplotlib.pyplot as plt


def session_means(config: dict, colors: colorset = None):
    """Plot mean pose for each session.

    Parameters
    ----------
    config : dict
        Full project config dictionary.
    """
    if colors is None:
        colors = colorset.active

    dataset = load_dataset(config["dataset"])
    dataset, align_inverse = align(dataset, config["alignment"])
    fig, ax = plt.subplots(
        2, dataset.n_sessions, figsize=(dataset.n_sessions * 2, 3)
    )

    for i, session in enumerate(dataset.sessions):
        plot_mouse_views(
            ax[:, i],
            dataset.get_session(session).mean(axis=0),
            Armature.from_config(config["dataset"]),
            color=colors.neutral,
            specialkp=None,
        )
        ax[0, i].set_title(session)

    fig.suptitle("Subject mean poses")
    return fig


def pose_gallery(config: dict, colors: colorset = None):
    """Plot gallery of poses for each session from the dataset described in
    config."""
    if colors is None:
        colors = colorset.active

    dataset = load_dataset(config["dataset"])
    dataset, align_inverse = align(dataset, config["alignment"])
    arm = Armature.from_config(config["dataset"])
    fig = None

    for i, session in enumerate(dataset.sessions):

        poses = select_frame_gallery(
            dataset.get_session(session), arm, as_list=True
        )

        if fig is None:
            fig, ax = plt.subplots(
                dataset.n_sessions * 2,
                len(poses),
                figsize=(len(poses) * 1.5, dataset.n_sessions * 3),
            )

        for j in range(len(poses)):
            plot_mouse_views(
                ax[i * 2 : i * 2 + 2, j],
                poses[j],
                arm,
                color=colors.neutral,
                specialkp=None,
            )
        ax[i * 2, 0].set_ylabel(session)

    axes_off(ax)
    return fig


def scree(cumulative_variance, selected_ix, tgt_variance, colors=None, ax=None):
    """Plot calibration data.

    Parameters
    ----------
    project : Project
    config : dict
        Full project config.
    """
    if colors is None:
        colors = colorset.active

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.3, 2))
    else:
        fig = ax.figure

    if tgt_variance is not None:
        ax.axhline(
            tgt_variance,
            color=colors.subtle,
            linestyle="--",
            lw=1,
            label="Target\nvariance",
        )
    ax.axvline(
        selected_ix,
        color=colors.subtle,
        lw=1,
        label="Selected\ndimension",
    )
    ax.plot(cumulative_variance, color=colors.C[0])
    if cumulative_variance.ndim == 1:
        ax.plot(
            [selected_ix],
            [cumulative_variance[selected_ix]],
            "o",
            color=colors.C[0],
            ms=3,
        )
    ax.set_xticks([selected_ix])
    ax.set_xticklabels([selected_ix])
    ax.set_ylabel("Variance\nexplained")
    ax.set_xlabel("Number of dimensions")
    legend(ax)

    return fig, ax
