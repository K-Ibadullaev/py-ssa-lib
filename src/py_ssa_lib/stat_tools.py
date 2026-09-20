
from numpy.typing import NDArray
import numpy as np

def bootstrap_prediction_intervals(
                                    residuals: NDArray[np.float64],
                                    forecasts: NDArray[np.float64],
                                    alpha: float = 0.05,
                                    N_sims: int = 1000
                                ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Parameters
    ----------
    residuals : ndarray
        Residuals used for bootstrap resampling.
    forecasts : ndarray
        Point forecasts for future observations.
    alpha : float
        Significance level for the prediction interval.
    N_sims : int
        Number of bootstrap simulations.

    Returns
    -------
    lower_pi : ndarray
        Lower prediction bounds.
    upper_pi : ndarray
        Upper prediction bounds.
    This is a bootstrap method for the prediction intervals discucced in the online textbook of Rob J Hyndman and George Athanasopoulos https://otexts.com/fpp3/prediction-intervals.html.
    It computes lower and upper prediction intervals. This method is model free and requires only the independence of residuals and allows to get reasonable prediction intervals for lower computational costs 
    and milder conditions. This means one can use it for any other model besides the MSSA/SSA.
    
    In the case of the SSA one starts with the computation of residuals between original time series and the reconstruted one. It is crucial to check for the abscence of an autocorrelation within the obtained 
    residuals. Using predefined numbers of components one performs either L- or  K-forecast for M future steps. These values are then passed to the corresponding input parameters. Additionally, one can adjust 
    the confidence level and simulation number as well.

    In the case of MSSA, one has to pick the time series of the interest with the same index. The algorithm doesn't change.



    """
    
    y_pred = np.tile(forecasts, N_sims).reshape(N_sims,len(forecasts))
    residuals_bootstrap = np.random.choice(residuals, len(forecasts)* N_sims , replace=True).reshape(N_sims,-1) 
    sims = y_pred + residuals_bootstrap

    
    lower_pi = np.percentile(sims, (alpha/2) * 100, axis=0)
    upper_pi = np.percentile(sims, (1 - alpha/2) * 100, axis=0)
    
    return lower_pi, upper_pi