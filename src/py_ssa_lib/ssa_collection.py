# type annotations
from typing import Any
from numpy.typing import NDArray


# linalg/data frames
import numpy as np
import pandas as pd

#abstract class
from abc import ABC, abstractmethod

# strategy components
from .weights_strategies import  WeightingStrategy, UniformWeights
from .svd_strategies import  SVDStrategy, FullSVD

class BaseSSA(ABC):
    """
    Base class for each SSA implementation

    """

    DEFAULT_DECOMPOSITION = FullSVD
    DEFAULT_WEIGHTING = UniformWeights

    def __init__(
                                self,
                                decomposition_strategy: SVDStrategy | None = None,
                                weighting_strategy: WeightingStrategy | None = None
                    ) -> None:
        """
        Parameters
        ----------
        decomposition_strategy : SVDStrategy | None
            Strategy used to decompose the trajectory matrix.
        weighting_strategy : WeightingStrategy | None
            Strategy used to assign weights to individual time series.
        """
        
        self.decomposition_strategy =  ( decomposition_strategy  or self.DEFAULT_DECOMPOSITION() )
       
        self.weighting_strategy = ( weighting_strategy  or self.DEFAULT_WEIGHTING() )

    def fit(
                self,
                data: pd.Series | pd.DataFrame | NDArray,
                L: int | None = None,
                **weighting_kwargs: Any
            ) -> "BaseSSA":
        """
        Parameters
        ----------
        data : Series | DataFrame | ndarray
            Input time-series data.
        L : int | None
            SSA window length.
        **weighting_kwargs : Any
            Additional arguments passed to the weighting strategy.

        Returns
        -------
        BaseSSA
            Fitted SSA/MSSA instance.
        """

        self.X_s = self._prepare_data(data)

        self.N, self.S = self.X_s.shape

        max_L = (self.N + 1) // 2

        self.L = max_L if L is None else L

        if not 2 <= self.L <= max_L:
            raise ValueError(
                "L must satisfy 2 <= L <= floor((N + 1) / 2)."
            )

        self.K = self.N - self.L + 1

        self.weights = self.weighting_strategy.compute_weights(
                                                                self.X_s,
                                                                **weighting_kwargs
                                                            )

        self.X_ss = self.construct_trajectory_matrix()

        self.U, self.Sigma, self.Vt = (
                        self.decomposition_strategy.decompose(self.X_ss)
                    )

        self.V = self.Vt.T
        self.d = len(self.Sigma)

        self.X_elem = self.elementary_matrix()

        sigma2 = self.Sigma**2
        total_energy = np.linalg.norm(self.X_ss, "fro")**2

        self.rel_contribution = sigma2 / total_energy * 100
        self.cumsum_contr = np.cumsum(self.rel_contribution)

        return self


    
    @abstractmethod
    def _prepare_data(
                            self,
                            data: pd.Series | pd.DataFrame | NDArray
                    ) -> NDArray[np.float64]:
            """
            Parameters
            ----------
            data : Series | DataFrame | ndarray
                Input data to convert to the internal (N, S) representation.

            Returns
            -------
            ndarray
                Prepared time-series matrix with shape (N, S).
            """
            pass

    def construct_trajectory_matrix(self) -> NDArray[np.float64]:
        """
        Returns
        -------
        ndarray
            Weighted trajectory matrix with shape (L, K*S).
        """

        blocks = [
            np.sqrt(self.weights[s])
            * np.column_stack([
                self.X_s[i:i + self.L, s]
                for i in range(self.K)
            ])
            for s in range(self.S)
        ]

        return np.hstack(blocks)
    
    
    def elementary_matrix(self) -> NDArray[np.float64]:
        """
        Returns
        -------
        ndarray
            Elementary trajectory matrices for all retained SVD components.
        """

        return np.array([  self.Sigma[i] * np.outer(self.U[:, i], self.V[:, i])  for i in range(self.d)     ])

    
    @staticmethod
    def diagonal_averaging(
                                        X: NDArray[np.float64]
                            ) -> NDArray[np.float64]:
                """
                Parameters
                ----------
                X : ndarray
                    Trajectory matrix to convert back to a time series.

                Returns
                -------
                ndarray
                    Reconstructed time series obtained by anti-diagonal averaging.
                """
                X_reverse = X[::-1]

                return np.array(
                                    [  
                                        X_reverse.diagonal(i).mean() for i in range(-X.shape[0] + 1, X.shape[1])    
                                    
                                    ]
                                
                                )




    def construct_hankel_weights(self) -> NDArray[np.int_]:
                """
                Returns
                -------
                ndarray
                    Hankel weights describing observation multiplicities.
                """
               

                i = np.arange(self.N)

                return np.minimum.reduce([
                                            i + 1,
                                            np.full(self.N, self.L),
                                            np.full(self.N, self.K),
                                            self.N - i
                ])


    def _reconstruct_elementary_components(
                                                self
                                            ) -> NDArray[np.float64]:
        """
        Returns
        -------
        ndarray
            Reconstructed elementary components with shape (d, N, S).
        """
        components = np.empty(
            (self.d, self.N, self.S)
        )

        for i in range(self.d):
            for s in range(self.S):

                block = self.X_elem[ i, :,  s*self.K:(s+1)*self.K
                ]

                components[i, :, s] = (
                    self.diagonal_averaging(block)
                    / np.sqrt(self.weights[s])
                )

        return components


    def compute_weighted_correlation_matrix(
                                            self
                                        ) -> NDArray[np.float64]:
                """
                Returns
                -------
                ndarray
                    Weighted component-correlation matrix with shape (d, d).
                """
                hankel_weights = self.construct_hankel_weights()

                components = self._reconstruct_elementary_components()

                gram = np.einsum(
                    "ins,n,s,jns->ij",
                    components,
                    hankel_weights,
                    self.weights,
                    components
                )

                norms = np.sqrt(np.diag(gram))

                Wcorr = np.abs( gram / np.outer(norms, norms) )

                np.fill_diagonal(Wcorr, 1.0)

                return Wcorr




    def reconstruct_components(
                                                self,
                                                idx_components: list[int] | NDArray[np.int_]
                                ) -> NDArray[np.float64]:
                """
                Parameters
                ----------
                idx_components : list[int] or np.ndarray
                    Indices of components used for reconstruction.

                Returns
                -------
                ndarray
                    Reconstructed time series with shape (N, S).
                """
                X = self.X_elem[idx_components].sum(axis=0)

                return np.column_stack(
                                        [
                                            self.diagonal_averaging( X[:, s*self.K:(s+1)*self.K] ) / np.sqrt(self.weights[s])    for s in range(self.S)
                                        ]
                
                )


    def estimate_ESPRIT(
                                    self,
                                    idx_components: list[int] | NDArray[np.int_],
                        ) -> tuple[
                            NDArray[np.complex128],
                            NDArray[np.float64],
                            NDArray[np.float64]
                        ]:
        """
        Parameters
        ----------
        idx_components :  list[int] or np.ndarray
            Components defining the signal subspace.

        Returns
        -------
        mu : ndarray
            Complex ESPRIT roots.
        rho : ndarray
            Magnitudes of the roots.
        omega : ndarray
            Angular frequencies of the roots.
        """

        P = self.U[:, idx_components]

        P_upper = P[:-1]
        P_lower = P[1:]

        Phi = np.linalg.lstsq(
            P_upper,
            P_lower,
            rcond=None
        )[0]

        mu = np.linalg.eigvals(Phi)

        order = np.argsort(np.abs(np.angle(mu)))
        mu = mu[order]

        rho = np.abs(mu)
        omega = np.angle(mu)

        return mu, rho, omega



    
    def estimate_LRR(
                                    self,
                                    idx_components: list[int] | NDArray[np.int_],
                    ) -> NDArray[np.float64]:
        """
        Parameters
        ----------
        idx_components :idx_components : list[int] or np.ndarray
            Components defining the signal subspace.
        Returns
        -------
        ndarray
            Linear recurrence coefficients.
        """
        P = self.U[:, idx_components]

        pi = P[-1, :]
        nu2 = pi @ pi

        return P[:-1, :] @ pi / (1 - nu2)




    def L_forecast(
                                self,
                                forecast_steps: int,
                                idx_components: list[int] | NDArray[np.int_],
                                return_full: bool = True
                    ) -> NDArray[np.float64]:
                """
                Parameters
                ----------
                forecast_steps : int
                    Number of future observations to predict.
                idx_components : list[int] or np.ndarray
                    Components used to estimate the recurrence relation.
                return_full : bool
                    Whether to return original and forecasted values together.

                Returns
                -------
                ndarray
                    L-recurrent forecast.
                """
                R = self.estimate_LRR(idx_components)

                y_pred = np.zeros((self.N + forecast_steps, self.S))
                y_pred[:self.N] = self.X_s

                for m in range(forecast_steps):

                    window = y_pred[
                        self.N - self.L + m + 1 :
                        self.N + m
                    ]

                    y_pred[self.N + m] = window.T @ R

                if return_full:
                    return y_pred

                return y_pred[-forecast_steps:]


    
    def K_forecast(
                                    self,
                                    forecast_steps: int,
                                   idx_components: list[int] | NDArray[np.int_],
                                    return_full: bool = True
                    ) -> NDArray[np.float64]:
                """
                Parameters
                ----------
                forecast_steps : int
                    Number of future observations to predict.
                idx_components : list[int] or np.ndarray
                    Components defining the forecasting subspace.
                return_full : bool
                    Whether to return original and forecasted values together.

                Returns
                -------
                ndarray
                    K/vector forecast.
                """

                V = self.V[:, idx_components]

                last_rows = (np.arange(1, self.S + 1) * self.K - 1   )

                W = V[last_rows, :]
                Q = np.delete(V, last_rows, axis=0)

                A = np.eye(self.S) - W @ W.T

                forecast_operator = np.linalg.solve( A, W @ Q.T    )

                # V belongs to the weighted embedding,
                # therefore work in weighted coordinates.
                sqrt_w = np.sqrt(self.weights)

                y_weighted = np.zeros(  (self.N + forecast_steps, self.S)    )

                y_weighted[:self.N] = (  self.X_s * sqrt_w  )

                for m in range(forecast_steps):

                    window = y_weighted[
                        self.N - self.K + m + 1 :
                        self.N + m
                    ]

                    # V is arranged in series blocks:
                    # series 1 values, series 2 values, ...
                    Z = window.T.reshape(-1)

                    y_weighted[self.N + m] = (
                        forecast_operator @ Z
                    )

                # Return to original physical scale
                y_pred = y_weighted / sqrt_w

                if return_full:
                    return y_pred

                return y_pred[-forecast_steps:]
                






#################################### SSA ########################
class SSA(BaseSSA):
    """
    Class for the SSA. Use for the univariate time series analysis.
     
    """

    def _prepare_data(
                        self,
                        data: pd.Series | NDArray
                    ) -> NDArray[np.float64]:
        """
        Parameters
        ----------
        data : Series | ndarray
            Single input time series.

        Returns
        -------
        ndarray
            Time series with shape (N, 1).
        """
        if isinstance(data, pd.Series):
            self.time_index = data.index
            self.series_name = data.name
            X = data.to_numpy(dtype=float)

        else:
            X = np.asarray(data, dtype=float)
            self.time_index = None
            self.series_name = None

        if X.ndim != 1:
            raise ValueError(
                "SSA expects a single time series."
            )

        return X[:, None]


#################################### MSSA ########################
class MSSA(BaseSSA):
    """
    Class for the MSSA. Use for the multivariate time series analysis.
     
    """


    def _prepare_data(
                                    self,
                                    data: pd.DataFrame | NDArray
                        ) -> NDArray[np.float64]:
        """
        Parameters
        ----------
        data : DataFrame | ndarray
            Multiple time series arranged as (N, S).

        Returns
        -------
        ndarray
            Multivariate time-series matrix with shape (N, S).
        """
        
        if isinstance(data, pd.DataFrame):
            self.time_index = data.index
            self.series_names = list(data.columns)
            X = data.to_numpy(dtype=float)

        else:
            X = np.asarray(data, dtype=float)
            self.time_index = None
            self.series_names = None

        if X.ndim != 2:
            raise ValueError(
                "MSSA expects multiple time series "
                "with shape (n_timesteps, n_series)."
            )

        return X       

