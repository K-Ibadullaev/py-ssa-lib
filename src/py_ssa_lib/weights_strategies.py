from typing import Any, Hashable, Sequence
from numpy.typing import NDArray

from abc import ABC, abstractmethod
import numpy as np


class WeightingStrategy(ABC):

    @abstractmethod
    def compute_weights(
                                self,
                                X: NDArray[np.float64],
                                **kwargs: Any
                        ) -> NDArray[np.float64]:
        """
        Parameters
        ----------
        X : ndarray
            Time-series matrix with shape (N, S).
        **kwargs : Any
            Strategy-specific parameters.

        Returns
        -------
        ndarray
            Weight vector with shape (S,).
        """
        pass

class UniformWeights(WeightingStrategy):

    def compute_weights(
                            self,
                            X: NDArray[np.float64],
                            **kwargs: Any
                    ) -> NDArray[np.float64]:
        """
        Parameters
        ----------
        X : ndarray
            Time-series matrix used to determine the number of series.

        Returns
        -------
        ndarray
            Unit weights for all series.
        """
        return np.ones(X.shape[1])

class PrecomputedWeights(WeightingStrategy):

    def compute_weights(
                                self,
                                X: NDArray[np.float64],
                                weights: Sequence[float],
                                **kwargs: Any
                        ) -> NDArray[np.float64]:
        """
        Parameters
        ----------
        X : ndarray
            Time-series matrix.
        weights : Sequence[float]
            User-provided weight for each series.

        Returns
        -------
        ndarray
            Precomputed weights as a NumPy array.
        """
        return np.asarray(weights, dtype=float)

class DistanceWeights(WeightingStrategy):

    def compute_weights(
                                self,
                                X: NDArray[np.float64],
                                distances: Sequence[float],
                                bandwidth: float = 1.0,
                                **kwargs: Any
                        ) -> NDArray[np.float64]:
        """
        Parameters
        ----------
        X : ndarray
            Time-series matrix.
        distances : Sequence[float]
            Distance from the focal series to each series.
        bandwidth : float
            Scale controlling distance-weight decay.

        Returns
        -------
        ndarray
            Distance-based weights.
        """
        distances = np.asarray(distances, dtype=float)

        weights = np.exp(-0.5 * (distances / bandwidth)**2  )

        return weights

class ClusterAwareWeights(WeightingStrategy):

    def compute_weights(
                                    self,
                                    X: NDArray[np.float64],
                                    distances: Sequence[float],
                                    focal_cluster: Hashable,
                                    neighbour_clusters: Sequence[Hashable],
                                    bandwidth: float = 1.0,
                                    cluster_penalty: float = 0.5,
                                    **kwargs: Any
                        ) -> NDArray[np.float64]:
        """
        Parameters
        ----------
        X : ndarray
            Time-series matrix.
        distances : Sequence[float]
            Distance from the focal series to each neighbour.
        focal_cluster : Hashable
            Cluster label of the focal series.
        neighbour_clusters : Sequence[Hashable]
            Cluster labels of neighbouring series.
        bandwidth : float
            Scale controlling distance-weight decay.
        cluster_penalty : float
            Weight multiplier applied to different-cluster neighbours.

        Returns
        -------
        ndarray
            Combined spatial and cluster-aware weights.
        """
        distances = np.asarray(distances, dtype=float)
        neighbour_clusters = np.asarray(neighbour_clusters)

        spatial = np.exp(     -0.5 * (distances / bandwidth)**2     )

        cluster = np.where(
            neighbour_clusters == focal_cluster,
            1.0,
            cluster_penalty
        )

        return spatial * cluster