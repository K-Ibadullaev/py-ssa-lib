Usage
=====

Installation
------------

Install the package with:

.. code-block:: bash

   pip install py-ssa-lib


Example data
------------

The examples below use the global average temperature data set included with the
package:

.. code-block:: python

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt

   from py_ssa_lib.datasets.load_global_average_temperatures_df import (
       load_global_average_temperatures_df,
   )

   df = load_global_average_temperatures_df(rawDS=False)
   df = df.T


SSA
---

SSA operates on a single time series.

Prepare the data
~~~~~~~~~~~~~~~~

The example below selects the last 100 temperature observations for Germany and
uses the first M=80 observations for fitting and the remaining 20 for testing.

.. code-block:: python

   country_name = "Germany"

   ts = df[df["Country"] == country_name]
   subset_country_ts = ts["AverageTemperature"].iloc[-100:]

   M = 80
   train_df = subset_country_ts.iloc[:M]
   test_df = subset_country_ts.iloc[M:]
   
   steps_to_forecast = 20


Fit the model
~~~~~~~~~~~~~

Choose a window length :math:`L` and fit the SSA model:

.. code-block:: python

   from py_ssa_lib.ssa_collection import SSA

   window_size = 36

   ssa = SSA().fit(
       data=train_df,
       L=window_size,
   )


Inspect the components
~~~~~~~~~~~~~~~~~~~~~~

The package provides visual tools for inspecting the singular-value
contributions, left singular vectors, elementary matrices and weighted
correlation matrix.

.. code-block:: python

   import py_ssa_lib.vis_tools as vis_tools

   vis_tools.plot_contributions(ssa=ssa)
   vis_tools.plot_eigenvectors(ssa, components=12, ncols=4)
   vis_tools.plot_elementary_matrices(ssa=ssa, components=20, ncols=5)
   vis_tools.plot_weighted_correlation(ssa=ssa)


Reconstruct a component group
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After inspecting the decomposition, choose the elementary components to include
in the reconstruction:

.. code-block:: python

   denoised_components = [0, 1, 2, 3, 6, 7]

   denoised = ssa.reconstruct_components(
       idx_components=denoised_components
   )

   denoised_df = pd.DataFrame(
       {
           country_name + "_ts_reconstructed": denoised.flatten()
       },
       index=ssa.time_index,
   )

   denoised_df.plot()
   train_df.plot()


Forecast
~~~~~~~~

Both L-forecasting and K-forecasting are available. The same component group
used for reconstruction can be used for forecasting.

.. code-block:: python

   l_forecast = ssa.L_forecast(
       idx_components=denoised_components,
       forecast_steps=steps_to_forecast,
       return_full=False,
   )

   k_forecast = ssa.K_forecast(
       idx_components=denoised_components,
       forecast_steps=steps_to_forecast,
       return_full=False,
   )

   l_forecast_df = pd.DataFrame(
       {country_name + "_L_forecast": l_forecast.flatten()},
       index=test_df.index,
   )

   k_forecast_df = pd.DataFrame(
       {country_name + "_K_forecast": k_forecast.flatten()},
       index=test_df.index,
   )

   plt.figure(figsize=(10, 5))
   plt.plot(test_df, "-o", label="True average temperature")
   plt.plot(l_forecast_df, "--*", label="L-forecast")
   plt.plot(k_forecast_df, "--s", label="K-forecast")
   plt.legend()


ESPRIT
~~~~~~

ESPRIT estimates the complex roots :math:`\mu`, their magnitudes
:math:`\rho`, and angular frequencies :math:`\omega` from a selected signal
subspace.

.. code-block:: python

   idx_components = np.arange(ssa.L - 1)

   mu, rho, omega = ssa.estimate_ESPRIT(
       idx_components=idx_components
   )

   vis_tools.plot_esprit_roots(
       ssa=ssa,
       idx_components=idx_components,
   )


MSSA
----

MSSA operates on several time series simultaneously. Input data are arranged as
``(n_timesteps, n_series)``.


Prepare multivariate data
~~~~~~~~~~~~~~~~~~~~~~~~~

The following example extracts average-temperature time series for Italy,
France and Germany and stacks them horizontally:

.. code-block:: python

   country_names = ["Italy", "France", "Germany"]

   ts = (
       df.loc[
           df["Country"].isin(country_names),
           ["Country", "AverageTemperature"],
       ]
       .pivot(columns="Country", values="AverageTemperature")
       .dropna()
   )

   ts = ts[country_names]

   subset_country_ts = ts.iloc[-100:]

   steps_to_forecast = 80
   train_df = subset_country_ts.iloc[:steps_to_forecast]
   test_df = subset_country_ts.iloc[steps_to_forecast:]


Fit the model
~~~~~~~~~~~~~

.. code-block:: python

   from py_ssa_lib.ssa_collection import MSSA

   window_size = 36

   mssa = MSSA().fit(
       data=train_df,
       L=window_size,
   )


Inspect the components
~~~~~~~~~~~~~~~~~~~~~~

The same visualization functions can be used with MSSA:

.. code-block:: python

   vis_tools.plot_contributions(ssa=mssa)
   vis_tools.plot_eigenvectors(mssa, components=12, ncols=4)
   vis_tools.plot_elementary_matrices(ssa=mssa, components=20, ncols=5)
   vis_tools.plot_weighted_correlation(ssa=mssa)


Reconstruct several time series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A single component group is used to reconstruct all time series in the MSSA
model:

.. code-block:: python

   denoised_components = [0, 1, 2, 3]

   denoised = mssa.reconstruct_components(
       idx_components=denoised_components
   )

   denoised_df = pd.DataFrame(
       {
           country_names[i] + "_ts_reconstructed": denoised[:, i]
           for i in range(len(country_names))
       },
       index=mssa.time_index,
   )


The original and reconstructed series can be compared separately:

.. code-block:: python

   fig, ax = plt.subplots(3, 1, figsize=(16, 9))

   for i in range(len(country_names)):
       ax[i].plot(
           train_df.iloc[:, i],
           "-o",
           label="True average temperature for " + train_df.columns[i],
       )
       ax[i].plot(
           denoised_df.iloc[:, i],
           "-*",
           label=denoised_df.columns[i],
       )

       ax[i].set_title(train_df.columns[i])
       ax[i].legend()
       ax[i].grid(alpha=0.3)

   plt.tight_layout()


Forecast several time series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

L- and K-forecasting return forecasts for every series included in the MSSA
model:

.. code-block:: python

   l_forecast = mssa.L_forecast(
       idx_components=denoised_components,
       forecast_steps=steps_to_forecast,
       return_full=False,
   )

   k_forecast = mssa.K_forecast(
       idx_components=denoised_components,
       forecast_steps=steps_to_forecast,
       return_full=False,
   )

   l_forecast_df = pd.DataFrame(
       {
           country_names[i] + "_L_forecast": l_forecast[:, i]
           for i in range(len(country_names))
       },
       index=test_df.index,
   )

   k_forecast_df = pd.DataFrame(
       {
           country_names[i] + "_K_forecast": k_forecast[:, i]
           for i in range(len(country_names))
       },
       index=test_df.index,
   )


Plot the forecasts:

.. code-block:: python

   fig, ax = plt.subplots(3, 1, figsize=(16, 9))

   for i in range(len(country_names)):
       ax[i].plot(
           test_df.iloc[:, i],
           "-o",
           label="True average temperature for " + test_df.columns[i],
       )
       ax[i].plot(
           l_forecast_df.iloc[:, i],
           "--*",
           label=l_forecast_df.columns[i],
       )
       ax[i].plot(
           k_forecast_df.iloc[:, i],
           "--s",
           label=k_forecast_df.columns[i],
       )

       ax[i].tick_params(axis="x", labelrotation=90)
       ax[i].set_title(test_df.columns[i])
       ax[i].legend()
       ax[i].grid(alpha=0.3)

   plt.tight_layout()


ESPRIT with MSSA
~~~~~~~~~~~~~~~~

ESPRIT is used in the same way for MSSA:

.. code-block:: python

   idx_components = np.arange(mssa.L - 1)

   mu, rho, omega = mssa.estimate_ESPRIT(
       idx_components=idx_components
   )

   vis_tools.plot_esprit_roots(
       ssa=mssa,
       idx_components=idx_components,
   )


Component selection
-------------------

Component grouping is guided by the singular-value contributions, left singular
vectors, elementary matrices and weighted correlation matrix. Forecasting
performance on a held-out test set can also be used to compare different
component groups.
