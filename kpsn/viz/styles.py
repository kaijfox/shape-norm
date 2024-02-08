from ._style import *


def init_nb(
    plot_dir, style="vscode_dark", context="paper", **kws
) -> Tuple[colorset, plot_finalizer]:
    _style = color_sets[style]
    colorset.active = _style
    plt.style.use(style)
    sns.set_context(context)
    clr = color_sets[style]
    plotter = plot_finalizer(plot_dir, **kws)
    return plotter, _style
