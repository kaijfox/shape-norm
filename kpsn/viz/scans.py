from ..fitting.scans import (
    withinbody_induced_errs as _withinbody_induced_errs,
    withinsession_induced_errs as _withinsession_induced_errs,
    jsds_to_reference as _jsds_to_reference,
)
from ..fitting.methods import load_and_prepare_dataset
from .styles import colorset
from .util import flat_grid, stripplot, legend, unique_handles
from ..config import load_project_config, load_model_config

from collections import defaultdict
import jax.numpy as jnp
import numpy as np


def _vertical_lineplot(data, colors: colorset = None, labels=None):
    """
    Parameters
    ----------
    data : dict[str, list[array, shape (n_models, n_samples) or (n_models)]]
        Mapping from axis titles to list of lines to plot on the axis.
    """
    if colors is None:
        colors = colorset.active

    fig, ax, ax_grid = flat_grid(
        len(data) + 1, n_col=5, ax_size=(2, 2), sharey=True, sharex=True
    )
    for _a, ttl in zip(ax[1:], data):
        scatter_and_mean = data[ttl][0].ndim > 1

        for a in [_a, ax[0]]:
            for err_data in data[ttl]:

                if scatter_and_mean:
                    strip_data = stripplot(
                        *err_data,
                        stacked=True,
                    )
                    a.plot(
                        *strip_data[::-1],
                        "o",
                        color=colors.subtle,
                        ms=2,
                        zorder=-1,
                    )
                    mean_line = err_data.mean(axis=-1)
                else:
                    mean_line = err_data

                strip_data = stripplot(
                    *mean_line,
                    stacked=True,
                    jitter=0.01,
                )
                a.plot(
                    *strip_data[::-1], "o-", color=colors.neutral, ms=3, lw=0.5
                )
        _a.set_title(ttl)
        _a.set_yticks(range(len(mean_line)))
        if labels is not None:
            _a.set_yticklabels(labels)
    ax[0].set_title("All")

    return fig, ax, ax_grid


def withinbody_induced_errs(
    project, scan_name, colors: colorset = None, progress=False
):
    if colors is None:
        colors = colorset.active

    errs = _withinbody_induced_errs(project, scan_name, progress=progress)

    models = sorted(list(errs.keys()))
    bodies = errs[models[0]].keys()

    fig, ax, ax_grid = flat_grid(
        len(bodies) + 1, n_col=5, ax_size=(2, 2), sharey=True, sharex=True
    )
    for _a, b in zip(ax[1:], bodies):
        for a in [_a, ax[0]]:
            err_data = [jnp.array(list(errs[m][b].values())) for m in models]

            strip_data = stripplot(
                *[arr.ravel() for arr in err_data],
                stacked=True,
            )
            a.plot(*strip_data[::-1], "o", color=colors.subtle, ms=2, zorder=-1)

            strip_data = stripplot(
                *[arr.mean(axis=-1) for arr in err_data],
                stacked=True,
                jitter=0.01,
            )
            a.plot(*strip_data[::-1], "o-", color=colors.neutral, ms=3, lw=0.5)
        _a.set_title(b)
        _a.set_yticks(range(len(models)))
        _a.set_yticklabels(models)
    ax[0].set_title("All")
    ax_grid[-1, 0].set_xlabel("Keypoint dist")

    return fig


def withinsession_induced_errors(
    project, scan_name, colors: colorset = None, progress=False
):
    if colors is None:
        colors = colorset.active

    _body_inv, errs = _withinsession_induced_errs(
        project, scan_name, progress=progress
    )

    models = sorted(list(errs.keys()))
    sessions = errs[models[0]].keys()

    err_data = defaultdict(list)
    for orig_sess_name in sessions:
        for sess in errs[models[0]][orig_sess_name].keys():
            # should be exactly one entry in _body_inv containing `sess`
            body = list(filter(lambda kv: sess in kv[1], _body_inv.items()))
            if len(body) == 0:
                print("No data found for", sess)
            body = body[0][0]
            err_data[body].append(
                np.array([errs[m][orig_sess_name][sess] for m in models])
            )

    fig, ax, ax_grid = _vertical_lineplot(err_data, labels=models)
    return fig, ax, ax_grid


def jsds_to_reference(
    project,
    scan_name,
    colors: colorset = None,
    progress=False,
    data_only=False,
    with_data=None,
):
    if colors is None:
        colors = colorset.active

    if with_data is None:
        jsds, base_jsds, dataset = _jsds_to_reference(
            project, scan_name, progress=progress
        )
        if data_only:
            return jsds, base_jsds, dataset
    else:
        jsds, base_jsds, dataset = with_data
    models = sorted(list(jsds.keys()))
    bodies = jsds[models[0]].keys()
    ref_body = [b for b in base_jsds if dataset.ref_session in base_jsds[b]][0]

    # labels and aesthetics
    base_ref_label = lambda i: (
        f"Within {ref_body}, unmorphed" if i == 0 else None
    )
    base_ref_kw = dict(color=colors.C[0], ls="-", lw=1, zorder=2)
    base_label = lambda i: f"To {ref_body}, unmorphed" if i == 0 else None
    base_kw = dict(color=colors.subtle, ls="--", lw=1, zorder=1)
    morphed_label = f"To {ref_body}, morphed"
    morphed_kw = dict(
        color=colors.neutral, ms=3, lw=0.5, zorder=3, label=morphed_label
    )

    fig, ax, ax_grid = flat_grid(
        len(bodies) + 1, n_col=5, ax_size=(2, 2), sharey=True, sharex=True
    )
    for _a, b in zip(ax[1:], bodies):
        for a in [_a, ax[0]]:
            jsd_data = [jnp.array(list(jsds[m][b].values())) for m in models]
            a.plot(jsd_data, jnp.arange(len(jsd_data)), "o-", **morphed_kw)
            for i, (s, base_jsd) in enumerate(base_jsds[ref_body].items()):
                if s != dataset.ref_session:
                    a.axvline(base_jsd, label=base_ref_label(i), **base_ref_kw)
        for i, base_jsd in enumerate(base_jsds[b].values()):
            _a.axvline(base_jsd, label=base_label(i), **base_kw)
        _a.set_title(b)
        _a.set_yticks(range(len(models)))
        _a.set_yticklabels(models)
    fig.tight_layout()
    ax[0].set_title("All")
    ax_grid[-1, 0].set_xlabel("JSD")
    legend(ax[-1], unique_handles(ax[-1]))
    return fig
