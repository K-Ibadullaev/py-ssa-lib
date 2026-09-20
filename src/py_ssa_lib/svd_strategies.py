from typing import Any
from numpy.typing import NDArray

from abc import ABC, abstractmethod
import numpy as np

from sklearn.utils.extmath import randomized_svd


class SVDStrategy(ABC):

    @abstractmethod
    def decompose(
                    self,
                    X: NDArray[np.float64]
                    ) -> tuple[
                                    NDArray[np.float64],
                                    NDArray[np.float64],
                                    NDArray[np.float64]
                                ]:
        """
        Parameters
        ----------
        X : ndarray
            Matrix to decompose.

        Returns
        -------
        U : ndarray
            Left singular vectors.
        sigma : ndarray
            Singular values.
        Vt : ndarray
            Transposed right singular vectors.
        """
    
        pass

class FullSVD(SVDStrategy):

    def __init__(self, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        **kwargs : Any
            Additional arguments passed to numpy.linalg.svd.
        """
        self.kwargs = kwargs

    def decompose(
                        self,
                        X: NDArray[np.float64]
                    ) -> tuple[
                                NDArray[np.float64],
                                NDArray[np.float64],
                                NDArray[np.float64]
                             ]:
        """
        Parameters
        ----------
        X : ndarray
            Matrix to decompose.

        Returns
        -------
        U : ndarray
            Left singular vectors.
        sigma : ndarray
            Singular values.
        Vt : ndarray
            Transposed right singular vectors.
        """

        U, sigma, Vt = np.linalg.svd(
                                        X,
                                        full_matrices=False,
                                        **self.kwargs
                                    )

        return U, sigma, Vt

class RandomizedSVD(SVDStrategy):

    def __init__(
                                self,
                                n_components: int,
                                n_oversamples: int = 10,
                                n_iter: int | str = "auto",
                                random_state: int | None = None,
                                **kwargs: Any
                    ) -> None:
        """
        Parameters
        ----------
        n_components : int
            Number of singular components to retain.
        n_oversamples : int
            Additional vectors used for randomized range estimation.
        n_iter : int | str
            Number of power iterations.
        random_state : int | None
            Random seed for reproducibility.
        **kwargs : Any
            Additional arguments passed to randomized_svd.
        """

        self.n_components = n_components
        self.n_oversamples = n_oversamples
        self.n_iter = n_iter
        self.random_state = random_state
        self.kwargs = kwargs

    
    def decompose(
                        self,
                        X: NDArray[np.float64]
                  ) -> tuple[
                                NDArray[np.float64],
                                NDArray[np.float64],
                                NDArray[np.float64]
                            ]:
        """
        Parameters
        ----------
        X : ndarray
            Matrix to decompose.

        Returns
        -------
        U : ndarray
            Left singular vectors.
        sigma : ndarray
            Singular values.
        Vt : ndarray
            Transposed right singular vectors.
        """

        U, sigma, Vt = randomized_svd(
                                                X,
                                                n_components=self.n_components,
                                                n_oversamples=self.n_oversamples,
                                                n_iter=self.n_iter,
                                                random_state=self.random_state,
                                                **self.kwargs
                                        )

        return U, sigma, Vt