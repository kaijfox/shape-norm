from ..fitting.scans import (
    withinbody_reconst_errs as _withinbody_reconst_errs,
    jsds_to_reference as _jsds_to_reference,
)
from ..fitting.methods import load_and_prepare_dataset
from .styles import colorset
from .util import flat_grid, stripplot
from ..config import load_project_config, load_model_config

import jax.numpy as jnp


def withinbody_reconst_errs(project, scan_name, colors: colorset = None):
    if colors is None:
        colors = colorset.active

    errs = _withinbody_reconst_errs(project, scan_name)

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


def jsds_to_reference(project, scan_name, colors: colorset = None):
    if colors is None:
        colors = colorset.active

    jsds, base_jsds, dataset = _jsds_to_reference(project, scan_name)
    models = sorted(list(jsds.keys()))
    bodies = jsds[models[0]].keys()
    ref_body = [b for b in base_jsds if dataset.ref_session in base_jsds[b]][0]

    fig, ax, ax_grid = flat_grid(
        len(bodies) + 1, n_col=5, ax_size=(2, 2), sharey=True, sharex=True
    )
    for _a, b in zip(ax[1:], bodies):
        for a in [_a, ax[0]]:
            jsd_data = [jnp.array(list(jsds[m][b].values())) for m in models]
            a.plot(
                jsd_data,
                jnp.arange(len(jsd_data)),
                "o-",
                color=colors.neutral,
                ms=3,
                lw=0.5,
            )
            for s, base_jsd in base_jsds[ref_body].items():
                if s != dataset.ref_session:
                    a.axvline(base_jsd, color=colors.subtle, ls="-", lw=1)
        for base_jsd in base_jsds[b].values():
            _a.axvline(base_jsd, color=colors.subtle, ls="--", lw=1)
        _a.set_title(b)
        _a.set_yticks(range(len(models)))
        _a.set_yticklabels(models)
    ax[0].set_title("All")
    ax[0].set_xlabel("JSD")
    return fig
