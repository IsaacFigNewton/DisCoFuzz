import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class EvalVisualizationsMixin:
    @staticmethod
    def _make_grid(n_items: int, ncols: int = 2, w: float = 5, h: float = 4, **kwargs):
        n_items = int(max(n_items, 1))
        ncols = int(min(max(ncols, 1), n_items))
        nrows = int(np.ceil(n_items / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), **kwargs)
        axes = np.atleast_2d(axes).reshape(nrows, ncols)
        return fig, axes, axes.ravel(), nrows, ncols

    @staticmethod
    def _show_axis_labels(ax, idx: int, nrows: int, ncols: int,
                         xlabel: str = "", ylabel: str = "",
                         show_left_y_only: bool = True,
                         show_bottom_x_only: bool = True,
                         x_fontsize: int = 12, y_fontsize: int = 12):
        row, col = divmod(idx, ncols)
        is_left = (col == 0)
        is_bottom = (row == nrows - 1)

        if show_left_y_only and not is_left:
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.set_ylabel("")
        else:
            ax.set_ylabel(ylabel, fontsize=y_fontsize)

        if show_bottom_x_only and not is_bottom:
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
            ax.set_xlabel("")
        else:
            ax.set_xlabel(xlabel, fontsize=x_fontsize)

    @staticmethod
    def _hide_unused_axes(axes_flat, n_used: int):
        for j in range(n_used, len(axes_flat)):
            axes_flat[j].set_visible(False)

    @staticmethod
    def _unique_legend_from_axes(axes):
        handles, labels = [], []
        seen = set()
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                if ll not in seen:
                    seen.add(ll)
                    handles.append(hh)
                    labels.append(ll)
        return handles, labels

    @staticmethod
    def _legend_below(fig, handles, labels,
                    ncol_cap: int = 5,
                    fontsize: int = 10):
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.2),  # outside the figure
            ncol=min(len(labels), ncol_cap),
            fontsize=fontsize
        )
    
    @staticmethod
    def _finish(fig, bottom_space: float = 0.25, top_space: float = 0.85):
        fig.subplots_adjust(bottom=bottom_space, top=top_space)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _colormap_list(n: int, name: str = "viridis"):
        if n <= 0:
            return []
        cmap = plt.get_cmap(name)
        return cmap(np.linspace(0, 1, n))