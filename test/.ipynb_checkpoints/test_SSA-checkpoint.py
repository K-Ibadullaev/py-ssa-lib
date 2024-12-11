import sys
import py_ssa_lib

import pytest
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.utils.extmath import randomized_svd

from py_ssa_lib.SSA import SSA
from py_ssa_lib.datasets.load_energy_consumption_df import load_energy_consumption_df


def test_ssa_init():
            
            
            ssa_test = SSA(Verbose=True)
           
            assert ssa_test.Verbose == True
       

def test_ssa_fit():
            ts_index = 0
            i_start_ts = 2
            decomposition = "svd"
            #decomposition = "rand_svd"
            window_s = 7
            demo_df = load_energy_consumption_df()

            ssa_test = SSA()
            ssa_test.fit(df=demo_df,decomposition=decomposition, idx_start_ts=i_start_ts, ts=ts_index,L=window_s)
            
            assert (int(ssa_test.L) == window_s) 
            assert  (ssa_test.decomposition == decomposition )
            assert  (int(ssa_test.N) == demo_df.iloc[:,i_start_ts:].shape[1])

def test_ssa_lrr():
            ts_index = 0
            i_start_ts = 2
            decomposition = "svd"
            
            window_s = 7
            demo_df = load_energy_consumption_df()
            
            ssa_test = SSA()
            ssa_test.fit(df=demo_df,decomposition=decomposition, idx_start_ts=i_start_ts, ts=ts_index,L=window_s)
            ts_array = ssa_test.reconstruct_ts(idx_chosen_components=[0],return_as_df=False)
            M_ = 2
            fact_lrr = np.array([0.1673337, 0.1691975, 0.1712309, 0.1733025, 0.1754108, 0.1769012])

            assert np.allclose(ssa_test.estimate_LRR( idx_components=[0]) , fact_lrr)==True
            
            
            
def test_ssa_L_Forecast():
            ts_index = 0
            i_start_ts = 2
            decomposition = "svd"
            
            window_s = 7
            demo_df = load_energy_consumption_df()
       
            ssa_test = SSA()
            ssa_test.fit(df=demo_df,decomposition=decomposition, idx_start_ts=i_start_ts, ts=ts_index,L=window_s)
            ts_array = ssa_test.reconstruct_ts(idx_chosen_components=[0],return_as_df=False)
            M_ = 2

            fact_lrr  = np.array([0.1673337, 0.1691975, 0.1712309, 0.1733025, 0.1754108, 0.1769012])
            fact_L_forecast = np.array([
                                          9.663041,  9.776273,  9.896262, 10.023075, 10.160442, 10.289072, 
                                          10.405244, 10.542190, 10.677297, 10.807085, 10.922495, 11.015351,
                                          11.127770, 11.241332, 11.344905, 11.435964, 11.501652, 11.553324,
                                          11.749845, 11.857177
                                        ]) 

            assert np.allclose(ssa_test.estimate_LRR( idx_components=[0]) , fact_lrr )==True
            assert np.allclose( 
                                ssa_test.L_Forecast( ts_array, M=M_, idx_components=[0], mode='forward'), fact_L_forecast ) == True
            
def test_ssa_estimate_ESPRIT():
            ts_index = 0
            i_start_ts = 2
            decomposition = "svd"
            
            window_s = 7
            demo_df = load_energy_consumption_df()
           
            ssa_test = SSA()
            ssa_test.fit(df=demo_df,decomposition=decomposition, idx_start_ts=i_start_ts, ts=ts_index,L=window_s)
            ts_array = ssa_test.reconstruct_ts(idx_chosen_components=[0],return_as_df=False)
            M_ = 2
            fact_esprit =np.array([ 1.19055424+0.j,  1.01768198+0.j,  0.14407794+0.84729997j,  0.14407794-0.84729997j,   -0.74912593+0.6497274j , -0.74912593-0.6497274j ])
            
            assert np.allclose(ssa_test.estimate_ESPRIT(idx_components=np.arange(ssa_test.d-1), decompose_rho_omega=False),  fact_esprit) ==True
        


       
