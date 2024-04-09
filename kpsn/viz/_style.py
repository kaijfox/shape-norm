import seaborn as sns
import matplotlib.pyplot as plt
from typing import Protocol, Tuple, Callable
from pathlib import Path
import numpy as np
import logging

def palgen(name, skip=None):
    def unbound_make(n=None, **kw):
        if kw.get("as_cmap", False):
            return sns.color_palette(name, n, **kw)
        if skip is not None:
            if skip[1] is None:
                skip_end = n + skip[0]
                skip_add = skip[0]
            else:
                skip_end = -skip[1]
                skip_add = sum(skip)
            return sns.color_palette(name, n + skip_add, **kw)[
                skip[0] : skip_end
            ]
        return sns.color_palette(name, n, **kw)

    return unbound_make


class colorset(Protocol):
    neutral: any
    subtle: any
    seq: Callable[[int], np.array]
    cts: Callable[[int], np.array]
    cts1: Callable[[int], np.array]
    C: np.array

    @staticmethod
    def make(name):
        return palgen(
            name,
        )


class vscode_dark_colors(colorset):
    neutral = ".8"
    subtle = ".4"
    seq = palgen("Spectral")
    cts = palgen("Blues_r")
    cts1 = palgen("viridis", skip=(2, None))
    C = ["C0", "C2", "C3", "C4"]


class light_colors(colorset):
    neutral = ".1"
    subtle = ".8"
    seq = lambda n: sns.hls_palette(l=0.4, n_colors=n + 2)[1:-1]
    cts = palgen("magma")
    cts1 = palgen("viridis")
    C = ["C3", "C4", "C2"]


color_sets = {"vscode_dark": vscode_dark_colors, "default": light_colors}


class plot_finalizer(object):
    def __init__(self, plot_dir, **kws):
        self.plot_dir = plot_dir
        kws = {"fmt": "png", "save": True, **kws}
        self.fmt = kws.pop("fmt")
        self.save = kws.pop("save")
        self.kws = {"dpi": 300, "bbox_inches": "tight", **kws}

    def finalize(
        self, fig, name, path = None, fmt = None, display=True, tight=True, despine=True, save=True
    ):
        if despine:
            for ax in fig.get_axes():
                sns.despine(ax=ax)
        if tight:
            fig.tight_layout()

        out_file = None
        if name is not None and (save and self.save or save == "force"):
            plot_dir = Path(self.plot_dir) if path is None else Path(self.plot_dir) / path
            fmt = self.fmt if fmt is None else fmt
            if not plot_dir.exists():
                logging.warn(f"Creating plot directory: {plot_dir}")
                plot_dir.mkdir(parents=True)
            out_file = Path(plot_dir) / (name + "." + fmt)
            fig.savefig(out_file, **self.kws)
        
        if display:
            plt.show(fig)
        
        return out_file


def init_plt(
    plot_dir, style="vscode_dark", **kws
) -> Tuple[colorset, plot_finalizer]:
    plt.style.use(style)
    sns.set_context("paper")
    clr = color_sets[style]
    plotter = plot_finalizer(plot_dir, **kws)
    return clr, plotter
