import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from numpy.typing import NDArray
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from .ssa_collection import BaseSSA

def _get_components(
                        ssa: BaseSSA,
                        components: list[int] | NDArray,
                    ) -> NDArray[np.int_]:
    """
    Parameters
    ----------
    ssa : BaseSSA
        Fitted SSA/MSSA model.
    components : list[int] | NDArray
        Number or explicit indices of components to plot.

    Returns
    -------
    ndarray
        Selected component indices.
    """
    if isinstance(components, int):
        return np.arange(min(components, ssa.d))

    return np.asarray(components)


def _make_grid(
                    nplots: int,
                    ncols: int,
                    figsize: tuple[float, float]
                ) -> tuple[Figure, NDArray]:
    """
    Parameters
    ----------
    nplots : int
        Number of required subplots.
    ncols : int
        Number of subplot columns.
    figsize : tuple[float, float]
        Figure size.

    Returns
    -------
    Figure, ndarray
        Figure and flattened array of axes.
    """
    ncols = min(ncols, nplots)
    nrows = int(np.ceil(nplots / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        squeeze=False
    )

    return fig, axes.ravel()


###### Eigenvectors ######
def plot_eigenvectors(
                        ssa: BaseSSA,
                        components: list[int] | NDArray = np.arange(0,5,1),
                        ncols: int = 2,
                        figsize: tuple[float, float] | None = None,
                        **plot_kwargs: Any
                    ) -> tuple[Figure, NDArray]:
    """
    Parameters
    ----------
    ssa : BaseSSA
        Fitted SSA/MSSA model.
    components : list[int] | NDArray
        Components to visualize.
    ncols : int
        Number of subplot columns.
    figsize : tuple | None
        Figure dimensions.
    **plot_kwargs : Any
        Arguments passed to matplotlib plot.

    Returns
    -------
    Figure, ndarray
        Figure and component axes.
    """
    idx = _get_components(ssa, components)

    if figsize is None:
        figsize = (5 * ncols, 3 * np.ceil(len(idx) / ncols))

    fig, axes = _make_grid(len(idx), ncols, figsize)

    for ax, i in zip(axes, idx):
        ax.plot(ssa.U[:, i], **plot_kwargs)

        ax.set_title(
            rf"$U_{{{i}}}$ "
            f"({ssa.rel_contribution[i]:.2f}%)"
        )

        ax.set_xlabel("Lag")
        ax.grid(alpha=0.3)

    for ax in axes[len(idx):]:
        ax.remove()

    fig.tight_layout()

    return fig, axes[:len(idx)]

##### Elementary matrices #####
def plot_elementary_matrices(
                                ssa: BaseSSA,
                                components: list[int] | NDArray = np.arange(0,5,1),
                                ncols: int = 3,
                                figsize: tuple[float, float] | None = None,
                                cmap: str = "viridis",
                                **imshow_kwargs: Any
                            ) -> tuple[Figure, NDArray]:
    """
    Parameters
    ----------
    ssa : BaseSSA
        Fitted SSA/MSSA model.
    components : list[int] | NDArray
        Elementary matrices to visualize.
    ncols : int
        Number of subplot columns.
    figsize : tuple | None
        Figure dimensions.
    cmap : str
        Matplotlib color map.
    **imshow_kwargs : Any
        Arguments passed to imshow.

    Returns
    -------
    Figure, ndarray
        Figure and matrix axes.
    """
    idx = _get_components(ssa, components)

    if figsize is None:
        figsize = (4 * ncols, 3 * np.ceil(len(idx) / ncols))

    fig, axes = _make_grid(len(idx), ncols, figsize)

    for ax, i in zip(axes, idx):

        im = ax.imshow(
            ssa.X_elem[i],
            cmap=cmap,
            aspect="auto",
            **imshow_kwargs
        )

        ax.set_title(
            rf"$X_{{{i}}}$ "
            f"({ssa.rel_contribution[i]:.2f}%)"
        )

        ax.set_xticks([])
        ax.set_yticks([])

        fig.colorbar(im, ax=ax)

    for ax in axes[len(idx):]:
        ax.remove()

    fig.tight_layout()

    return fig, axes[:len(idx)]



##### WCorr plot ##########

def plot_weighted_correlation(
                                        ssa: BaseSSA,
                                        figsize: tuple[float, float] = (7, 6),
                                        cmap: str = "viridis",
                                        **imshow_kwargs: Any
                            ) -> tuple[Figure, Axes]:
    """
    Parameters
    ----------
    ssa : BaseSSA
        Fitted SSA/MSSA model.
    figsize : tuple
        Figure dimensions.
    cmap : str
        Matplotlib color map.
    **imshow_kwargs : Any
        Arguments passed to imshow.

    Returns
    -------
    Figure, Axes
        Figure and correlation-matrix axis.
    """
    W = ssa.compute_weighted_correlation_matrix()

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        W,
        cmap=cmap,
        vmin=0,
        vmax=1,
        **imshow_kwargs
    )

    ax.set_xlabel("Component")
    ax.set_ylabel("Component")
    ax.set_title("Weighted correlation")

    fig.colorbar(
        im,
        ax=ax,
        label=r"$W_{ij}$"
    )

    fig.tight_layout()

    return fig, ax



### Eigenvals Contribution ###########
def plot_contributions(
                        ssa: BaseSSA,
                        n_components: int | None = None,
                        figsize: tuple[float, float] = (10, 4),
                        **plot_kwargs: Any
                    ) -> tuple[Figure, NDArray]:
    """
    Parameters
    ----------
    ssa : BaseSSA
        Fitted SSA/MSSA model.
    n_components : int | None
        Number of components to display.
    figsize : tuple
        Figure dimensions.
    **plot_kwargs : Any
        Arguments passed to matplotlib plot.

    Returns
    -------
    Figure, ndarray
        Figure and relative/cumulative contribution axes.
    """



    n = ssa.d if n_components is None else min(
        n_components,
        ssa.d
    )

    x = np.arange(n)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize
    )

    axes[0].plot(
        x,
        ssa.rel_contribution[:n],
        **plot_kwargs
    )

    axes[0].set_title("Relative contribution")
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Contribution (%)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        x,
        ssa.cumsum_contr[:n],
        **plot_kwargs
    )

    axes[1].set_title("Cumulative contribution")
    axes[1].set_xlabel("Component")
    axes[1].set_ylabel("Contribution (%)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()

    return fig, axes

def plot_esprit_roots(
                        ssa: BaseSSA,
                        idx_components: list[int] | NDArray = np.arange(0,5,1),
                        figsize: tuple[float, float] = (6, 6),
                        circle_kwargs: dict[str, Any] | None = None,
                        **scatter_kwargs: Any
                    ) -> tuple[Figure, Axes]:
    """
    Parameters
    ----------
    ssa : BaseSSA
        Fitted SSA/MSSA model.
    idx_components : list[int] | NDArray
        Components defining the ESPRIT signal subspace.
    figsize : tuple
        Figure dimensions.
    circle_kwargs : dict | None
        Styling arguments for the unit circle.
    **scatter_kwargs : Any
        Arguments passed to the root scatter plot.

    Returns
    -------
    Figure, Axes
        Figure and ESPRIT-root axis.
    """


                        
    mu, _, _ = ssa.estimate_ESPRIT(idx_components)

    fig, ax = plt.subplots(figsize=figsize)

    # Unit circle
    theta = np.linspace(0, 2*np.pi, 400)

    circle_style = {
        "linestyle": "--",
        "linewidth": 1
    }

    if circle_kwargs is not None:
        circle_style.update(circle_kwargs)

    ax.plot(
        np.cos(theta),
        np.sin(theta),
        **circle_style
    )

    # ESPRIT roots
    ax.scatter(
        mu.real,
        mu.imag,
        **scatter_kwargs
    )

    ax.axhline(0, linewidth=0.5)
    ax.axvline(0, linewidth=0.5)

    ax.set_xlabel("Real part")
    ax.set_ylabel("Imaginary part")
    ax.set_title("ESPRIT roots")

    ax.set_aspect("equal", adjustable="box")

    limit = max(1.1, 1.1 * np.max(np.abs(mu)))

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    fig.tight_layout()

    return fig, ax