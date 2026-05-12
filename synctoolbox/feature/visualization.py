import matplotlib.pyplot as plt
import numpy as np


def plot_matrix(X,
                Fs=1,
                Fs_F=1,
                T_coef=None,
                F_coef=None,
                xlabel='Time (seconds)',
                ylabel='Frequency (Hz)',
                xlim=None,
                ylim=None,
                clim=None,
                title='',
                dpi=72,
                colorbar=True,
                colorbar_aspect=20.0,
                cbar_label='',
                ax=None,
                figsize=(6, 3),
                **kwargs):
    """2D raster visualization of a matrix."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax = [ax]

    if T_coef is None:
        T_coef = np.arange(X.shape[1]) / Fs
    if F_coef is None:
        F_coef = np.arange(X.shape[0]) / Fs_F

    if 'extent' not in kwargs:
        x_ext1 = (T_coef[1] - T_coef[0]) / 2
        x_ext2 = (T_coef[-1] - T_coef[-2]) / 2
        y_ext1 = (F_coef[1] - F_coef[0]) / 2
        y_ext2 = (F_coef[-1] - F_coef[-2]) / 2
        kwargs['extent'] = [T_coef[0] - x_ext1, T_coef[-1] + x_ext2,
                            F_coef[0] - y_ext1, F_coef[-1] + y_ext2]
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'gray_r'
    if 'aspect' not in kwargs:
        kwargs['aspect'] = 'auto'
    if 'origin' not in kwargs:
        kwargs['origin'] = 'lower'
    if 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'nearest'

    im = ax[0].imshow(X, **kwargs)

    if len(ax) == 2 and colorbar:
        cbar = plt.colorbar(im, cax=ax[1])
        cbar.set_label(cbar_label)
    elif len(ax) == 2 and not colorbar:
        ax[1].set_axis_off()
    elif len(ax) == 1 and colorbar:
        plt.sca(ax[0])
        cbar = plt.colorbar(im, aspect=colorbar_aspect)
        cbar.set_label(cbar_label)

    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[0].set_title(title)
    if xlim is not None:
        ax[0].set_xlim(xlim)
    if ylim is not None:
        ax[0].set_ylim(ylim)
    if clim is not None:
        im.set_clim(clim)

    if fig is not None:
        plt.tight_layout()

    return fig, ax, im


def plot_chromagram(*args, chroma_yticks=np.arange(12), **kwargs):
    """Plot a chromagram with chroma pitch-class labels."""
    if 'ylabel' not in kwargs:
        kwargs['ylabel'] = 'Chroma'
    fig, ax, im = plot_matrix(*args, **kwargs)

    chroma_names = 'C C# D D# E F F# G G# A A# B'.split()
    ax[0].set_yticks(np.array(chroma_yticks))
    ax[0].set_yticklabels([chroma_names[i] for i in chroma_yticks])

    return fig, ax, im


def plot_signal(x,
                Fs=1,
                T_coef=None,
                ax=None,
                figsize=(6, 2),
                xlabel='Time (seconds)',
                ylabel='',
                title='',
                dpi=72,
                ylim=True,
                **kwargs):
    """Line plot visualization of a signal."""
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.subplot(1, 1, 1)
    if T_coef is None:
        T_coef = np.arange(x.shape[0]) / Fs

    if 'color' not in kwargs:
        kwargs['color'] = 'gray'

    line = ax.plot(T_coef, x, **kwargs)

    ax.set_xlim([T_coef[0], T_coef[-1]])
    if ylim is True:
        ylim_x = x[np.isfinite(x)]
        x_min, x_max = ylim_x.min(), ylim_x.max()
        if x_max == x_min:
            x_max = x_max + 1
        ax.set_ylim([min(1.1 * x_min, 0.9 * x_min), max(1.1 * x_max, 0.9 * x_max)])
    elif ylim not in [True, False, None]:
        ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if fig is not None:
        plt.tight_layout()

    return fig, ax, line


def plot_matrix_with_points(C,
                            P=np.empty((0, 2)),
                            color='r',
                            marker='o',
                            linestyle='',
                            **kwargs):
    """Plot a matrix with index pairs overlaid."""
    fig, ax, im = plot_matrix(C, **kwargs)
    line = ax[0].plot(P[:, 1], P[:, 0], marker=marker, color=color, linestyle=linestyle)

    if fig is not None:
        plt.tight_layout()

    return fig, im, line
