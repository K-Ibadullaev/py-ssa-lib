import sys
import py_ssa_lib
import pytest
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.utils.extmath import randomized_svd

from py_ssa_lib.MSSA import MSSA
from py_ssa_lib.datasets.load_energy_consumption_df import load_energy_consumption_df


def test_mssa_init():
            
            
            mssa_test = MSSA(Verbose=True)
           
            assert mssa_test.Verbose == True
       

def test_mssa_fit():
            
            i_start_ts = 2
            decomposition = "svd"
            #decomposition = "rand_svd"
            window_s = 7
            demo_df = load_energy_consumption_df()

            mssa_test = MSSA(Verbose=True)
            mssa_test.fit(df=demo_df,decomposition=decomposition, idx_start_ts=i_start_ts, L=window_s)
            
            assert (int(mssa_test.L) == window_s) 
            assert  (mssa_test.decomposition == decomposition )
            assert  (int(mssa_test.N) == demo_df.iloc[:,i_start_ts:].shape[1])

def test_mssa_lrr():
           
            i_start_ts = 2
            decomposition = "svd"
            
            window_s = 7
            demo_df = load_energy_consumption_df()
            
            mssa_test = MSSA()
            mssa_test.fit(df=demo_df,decomposition=decomposition, idx_start_ts=i_start_ts, L=window_s)
            ts_array = mssa_test.reconstruct_ts(idx_chosen_components=[0],return_as_df=False)
            M_ = 2
            fact_lrr = np.array([0.1756382, 0.1845678, 0.1981957, 0.2129944, 0.2277441, 0.2433179])

            assert np.allclose(mssa_test.estimate_LRR( idx_components=[0]) , fact_lrr)==True
            
            
            
def test_mssa_L_Forecast():
            ts_index = 0
            i_start_ts = 2
            decomposition = "svd"
            
            window_s = 7
            demo_df = load_energy_consumption_df()
       
            mssa_test = MSSA()
            mssa_test.fit(df=demo_df,decomposition=decomposition, idx_start_ts=i_start_ts,L=window_s)
            ts_array = mssa_test.reconstruct_ts(idx_chosen_components=[0],return_as_df=False)
            M_ = 2

            fact_lrr = np.array([0.1756382, 0.1845678, 0.1981957, 0.2129944, 0.2277441, 0.2433179])
            fact_L_forecast = np.array([
                                        [15.38531, 16.306364],
                                        [18.77296, 19.915842],
                                        [19.89430, 21.064436],
                                        [83.41526, 89.702892],
                                        [ 5.63838, 5.986697]
                                        ])

            assert np.allclose(mssa_test.estimate_LRR( idx_components=[0]) , fact_lrr )==True
            assert np.allclose( 
                                mssa_test.L_Forecast( ts_array, M=M_, idx_components=[0], mode='forward')[:,-M_:], fact_L_forecast ) == True
            
def test_mssa_estimate_ESPRIT():
            
            i_start_ts = 2
            decomposition = "svd"
            
            window_s = 7
            demo_df = load_energy_consumption_df()
           
            mssa_test = MSSA()
            mssa_test.fit(df=demo_df,decomposition=decomposition, idx_start_ts=i_start_ts, L=window_s)
            ts_array = mssa_test.reconstruct_ts(idx_chosen_components=[0],return_as_df=False)
            M_ = 2
            fact_esprit =np.array([ 
                                   1.5373144+0.j,  1.0207511+0.j,  
                                   0.91938+0.8687121j,  0.91938-0.8687121j,
                                   -0.7997768+0.84053236j, -0.7997768-0.84053236j
                                ])
            assert np.allclose(mssa_test.estimate_ESPRIT(idx_components=np.arange(mssa_test.d-1), decompose_rho_omega=False),  fact_esprit) ==True
        


       
