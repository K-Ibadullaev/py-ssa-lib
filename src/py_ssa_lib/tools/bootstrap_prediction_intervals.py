import numpy as np
def bootstrap_prediction_intervals( residuals, forecasts, alpha=0.05, N_sims=1000):
    """
    This is a bootstrap method for the prediction intervals discucced in the online textbook of Rob J Hyndman and George Athanasopoulos https://otexts.com/fpp3/prediction-intervals.html.
    It computes lower and upper prediction intervals. This method is model free and requires only the independence of residuals and allows to get reasonable prediction intervals for lower computational costs 
    and milder conditions. This means one can use it for any other model besides the MSSA/SSA.
    
    In the case of the SSA one starts with the computation of residuals between original time series and the reconstruted one. It is crucial to check for the abscence of an autocorrelation within the obtained 
    residuals. Using predefined numbers of components one performs either L- or  K-forecast for M future steps. These values are then passed to the corresponding input parameters. Additionally, one can adjust 
    the confidence level and simulation number as well.

    In the case of MSSA, one has to pick the time series of the interest with the same index. The algorithm doesn't change.


        Parameters
        ----------
        residuals:numpy array - this is a numpy array of residuals of a length N for a chosen time series,  N should be sufficiently large(N>30?)
        forecasts:numpy array - this is a numpy array of forecasted values of the length M(number of forecasting steps)
        alpha : float,  ranges between 0 and 1, default value is 0.05 - this is a confidence level used to compute the quantiles from the simulations
        N_sims: int, default value is 1000 - this is a number of simulations
        Returns
        -------
        lower_pi:numpy array - this is a numpy array of the lower boundary for the forecasted values of the length M(number of forecasting steps)
        upper_pit:numpy array - this is a numpy array of the  upper boundary for the forecasted values of the length M(number of forecasting steps)

    """
    
    y_pred = np.tile(forecasts, N_sims).reshape(N_sims,len(forecasts))
    residuals_bootstrap = np.random.choice(residuals, len(forecasts)* N_sims , replace=True).reshape(N_sims,-1) 
    sims = y_pred + residuals_bootstrap

    
    lower_pi = np.percentile(sims, (alpha/2) * 100, axis=0)
    upper_pi = np.percentile(sims, (1 - alpha/2) * 100, axis=0)
    
    return lower_pi, upper_pi