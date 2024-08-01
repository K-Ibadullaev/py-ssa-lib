import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.utils.extmath import randomized_svd
from matplotlib.ticker import MaxNLocator


class SSA():
    """
        Creates an instance of the SSA object

        Parameters
        ----------
        Verbose: bool, outputs parameters used for an instance 

        Returns
        -------
        object of class SSA
    """
    
    def __init__(self,Verbose=False):
        self.Verbose = Verbose
        if self.Verbose==True:
            print("Instance is created")


    
           
    def construct_trajectory_matrix(self):
        """
            Constructs a trajectory matrix from a lagged versions of the time series

            Parameters
            ----------
            Takes arguments from the instance 

            Returns
            -------
            X:numpy.array, representing the trajectory matrix 
        """
        X = np.column_stack([self.X_s[i:i+self.L] for i in range(0,self.K)])
        return X 

    def decompose_trajectory_matrix(self,**kwargs):
        """
            Decomposes a trajectory matrix using full svd or randomized svd

            Parameters
            ----------
            Takes arguments from the instance 
            **kwargs for the svd routine

            Returns
            -------
            U,Sigma, V.T: all are numpy.arrays, representing the left-hand side and  right-hand side eigenvectors and 
            corresponding eigenvalues
        """
        if self.decomposition == "svd":
            if self.Verbose==True:
                print("Using full SVD")
            
            U, Sigma, V = np.linalg.svd(self.X_ss,**kwargs)
            self.d = np.linalg.matrix_rank(self.X_ss)
            #self.d = np.max(np.where(Sigma>0))
        elif self.decomposition == "rand_svd" :
             if self.Verbose==True:
                 print("Using randomized SVD ")
             self.d = self.L // 2 - 1
             U, Sigma, V = randomized_svd(self.X_ss,n_components=self.L - self.L//3 ,n_oversamples=100, random_state=0,power_iteration_normalizer='LU',n_iter=15)

        else:
            self.d = 1
            raise  ValueError("Wrong decomposition type")
        return U, Sigma, V.T
    
    def elementary_matrix(self):
        """
                Constructs a matrix(tensor) of all elementary components

                Parameters
                ----------
                Takes arguments from the instance 

                Returns
                -------
                X_elem:numpy.array, representing the matrix(tensor) of all elementary components
        """
        
        X_elem = np.array( [self.Sigma[i] * np.outer(self.U[:,i], self.V[:,i]) for i in range(0,self.d)] )
        return X_elem

    def X_to_TS(self, X_i):
        """
            Averages the anti-diagonals of the given elementary matrix, X_i, 
            and returns a reconstructed time series for the given component X_i.

            Parameters
            ----------
            X_i: numpy.array, elementary matrix for the given component

            Returns
            -------
            rec_ts: numpy.array, reconstructed time series for the given component X_i
        """
        # Reverse the column ordering of X_i
        X_rev = X_i[::-1]
        rec_ts = np.array([X_rev.diagonal(i).mean() for i in range(-X_i.shape[0]+1, X_i.shape[1])])
        return rec_ts
 
    def fit(self, df, L,ts, decomposition,idx_start_ts, **kwargs ):
        """
            Fits the instance of SSA to the data
            Parameters
            ----------
            df:pandas.DataFrame, data frame of the time series, it is a source of the data, used to infer the length of time series N 
                                Dataframe should have the following structure:
                                (WIDE FORMAT) 
                            
            L:int, Window Size, the most important parameter for the SSA.
            
            decomposition:str, type of decomposition of the trajectory matrix.
                                Available options are "svd" meaning full svd-decomposition, and "rand_svd" meaning its 
                                "truncated version". For SSA it is reasonable to use "rand_svd" for very large time series and high values of L
                                
            idx_start_ts:int, the first column index of the data frame where the first nummeric value occours,
                              i.e where the time series start(s). Used to cutoff irrelevant columns from the input data set.
            

            Returns
            -------
            No output
        """
       
        self.df = df
        self.idx_start_ts = idx_start_ts
        self.ts_df = self.df.iloc[ts,self.idx_start_ts:].T
        self.decomposition = decomposition
        self.X_s = self.ts_df.to_numpy().astype('float64')
       
        self.N = int(self.X_s.shape[0])
        self.L = int(self.X_s.shape[0]/1.5) if (L == None)  else L
        self.K = int(self.N - self.L + 1 )
        
        if self.Verbose==True:
            print(f' N = {self.N}, K = {self.K}, L = {self.L}')
        
        self.X_ss = self.construct_trajectory_matrix()
        self.U, self.Sigma, self.V = self.decompose_trajectory_matrix(**kwargs)
        self.X_elem = self.elementary_matrix()
        self.sigma_sumsq = (self.Sigma**2).sum()
        self.rel_contribution = self.Sigma**2 / self.sigma_sumsq * 100
        self.cumsum_contr = (self.Sigma**2).cumsum() / self.sigma_sumsq * 100
        
        
    def plot_2d(self,m,title="", **kwargs):
         """
            helper function for plotting elementary matrices 
            Parameters
            ----------
            m:numpy.array, the i-th elementary matrix
            title:str, plot title
            **kwargs some additional parameters for visualization functions of pyplot
            Returns
            -------
            No output
         """
       
         plt.imshow(m,**kwargs)
         plt.xticks([])
         plt.yticks([])
         plt.title(title)


    def plot_elem_matrices(self, i_start, i_end,**kwargs):
        """
            Plots elementary matrices with indices in the range[ i_start, i_end]
            Parameters
            ----------
            m:numpy.array, the i-th elementary matrix
            i_start: int, the first elementary matrix to plot
            i_end: int the last elementary matrix to plot
            **kwargs some additional parameters for visualization functions of pyplot
            **kwargs some additional parameters for visualization functions of pyplot
            Returns
            -------
            No output
        """
        for i in range(i_start, i_end):
             if i_end%3!=0:
                plt.subplot(3,i_end//3+1, i+1)
             else:
                 plt.subplot(3,i_end//3 , i+1) 
           
             title = r"$\mathbf{X}_{" + str(i) + "}$"
             self.plot_2d(self.X_elem[i], title,**kwargs)
        plt.tight_layout()

 
    def plot_eigenvals_contribution(self,**kwargs):
        """
            Plots the contribution of eigenvalues
            Parameters
            ----------
            **kwargs some additional parameters for visualization functions of pyplot
            Returns
            -------
            No output
        """
       
        
        fig, ax = plt.subplots(1, 2, figsize=(14,5),**kwargs)
        ax[0].plot(self.rel_contribution , lw=2.5,**kwargs)
        ax[0].set_xlim(0, self.d)
        ax[0].grid()
        ax[0].set_title(r"Relative Contribution of $\mathbf{X}_i$ to Trajectory Matrix")
        ax[0].set_xlabel(r"$i$")
        ax[0].set_ylabel("Contribution (%)")
        ax[1].plot(self.cumsum_contr, lw=2.5,**kwargs)
        ax[1].set_xlim(0,self.d)
        ax[1].set_title(r"Cumulative Contribution of $\mathbf{X}_i$ to Trajectory Matrix")
        ax[1].set_xlabel(r"$i$")
        ax[1].grid()
        ax[1].set_ylabel(r"Contribution (%)");

    def plot_eigenvectors(self,i_start, i_end,**kwargs):
        """
            Plots the several eigenvectors U with indices in [i_start, i_end]
            Parameters
            ----------
            i_start: int, the first eigenvector to plot
            i_end: int the last eigenvector to plot
            **kwargs some additional parameters for visualization functions of pyplot
            Returns
            -------
            No output
        """
        plt.figure(figsize=(25, 25))
        for i in range(i_start, i_end):
             if i_end%2!=0:
                plt.subplot(2,i_end//2+1, i+1-i_start,**kwargs)
             else:
                 plt.subplot(2,i_end//2 ,  i+1-i_start,**kwargs) 
           
             title = r" $\mathbf{U}_{" + str(i) + "} $" +f' {self.rel_contribution[i]} %'
             plt.plot(self.U[:, i],**kwargs)
             plt.title(title)
        plt.tight_layout()

    def construct_hankel_weights(self):
        """
            Constructs hankel weights for the weighted correlation matrix
            Parameters
            ----------
            
            Returns
            -------
            w: numpy.array, weights used further for the computation of the weighted correlation matrix
        """
        L_ = np.minimum(self.L, self.K)
        K_ = np.maximum(self.L, self.K)
    
        weights = []
        for i in range(self.N):
            if i <= (L_ - 1):
                weights.append(i+1)
            elif i <= K_:
                weights.append(L_)
            else:
                weights.append(self.N - i)
    
        weights = np.array(weights)
        return weights
    
    def compute_weighted_correlation_matrix(self):
        """
            Computes the weighted correlation matrix used for component grouping
            Parameters
            ----------
            
            Returns
            -------
            Wcorr: numpy.array, weighted correlation matrix used for component grouping
        """
        w = self.construct_hankel_weights()
        TS_elem = np.array([self.X_to_TS(self.X_elem[i,:,:]) for i in range(self.d)])
        TS_wnorms = np.array([w.dot((TS_elem[i]**2)) for i in range(self.d)])
        TS_wnorms = TS_wnorms**-0.5
        Wcorr = np.identity(self.d)
        print(TS_elem.shape)
        for i in range(self.d):
            for j in range(i+1,self.d):
                Wcorr[i,j] = abs(w.dot(TS_elem[i] * TS_elem[j])*TS_wnorms[i] * TS_wnorms[j]) 
                Wcorr[j,i] = Wcorr[i,j]
        return Wcorr

    def plot_weighted_correlation_matrix(self):
        """
            Plots the weighted correlation matrix used for component grouping
            Parameters
            ----------
            
            Returns
            -------
            None
        """
        W_corr = self.compute_weighted_correlation_matrix()
        ax = plt.imshow(W_corr)
        plt.xlabel(r"$\tilde{TS}_i$")
        plt.ylabel(r"$\tilde{TS}_j$")
        plt.colorbar(ax.colorbar, fraction=0.045)
        ax.colorbar.set_label("$W_{ij}$")
        plt.xlim(0,self.d-0.5)
        plt.ylim(self.d-0.5,0)
        plt.xticks(np.arange(self.d))
        plt.yticks(np.arange(self.d))
        plt.clim(0,1)
        plt.title("The Weighted Correlation Matrix for the Time Series");

    def reconstruct_ts(self, idx_chosen_components, return_as_df=False):
        """
                    Reconstructs the time series from the chosen components
                    Parameters
                    ----------
                    idx_chosen_components:list or numpy.arange of positive integer numbers, denotes the indices of elementary components used for the reconstruction 
                    return_as_df:bool, whether to return the resulting reconstructed time series as pandas.DataFrame
                    
                    Returns
                    -------
                    ts_rec: numpy.array or  pandas.DataFrame, reconstructed time series 
            """

        chosen_components = self.X_elem[0]
        chosen_components = 0
        for i in idx_chosen_components:
            chosen_components += self.X_elem[i,:,:]
         
       
        ts_rec = self.X_to_TS(chosen_components)
        if return_as_df==True:
            return pd.DataFrame(columns=self.df.columns, data=np.column_stack([self.df.iloc[:,:self.idx_start_ts].values,ts_rec.T ]))
        return ts_rec.T



    
    def estimate_LRR(self, idx_components):
        """
                    Estimates Linear Recurrence Relations(LRR) coefficients for the MSSA, which is used for forecasting
                    Parameters
                    ----------
                    idx_components:list or numpy.arange of positive integer numbers, denotes the indices of elementary components 
                    
                    Returns
                    -------
                    R: numpy.array, Linear Recurrence Relations(LRR) coefficients
        """
        # P_orth = self.U[:,idx_components]
        # P_orth = gram_schmidt(self.U[:,idx_components])
        P_orth = sp.linalg.qr(self.U[:,:self.d])[0]#self.L
        nu_sq = np.sum(P_orth[-1,idx_components]**2)
        
        if nu_sq !=1:
            R = 1/(1-nu_sq) * (P_orth[:-1,idx_components] @ P_orth[-1,idx_components] )
                
            
            return R
        else :
            print(nu_sq)
    
    def L_Forecast(self, ts, M, idx_components, mode='forward'):
        """
                    Forecasts or estimates M values for a given time series using LRR 
                    Parameters
                    ----------
                    idx_components:list or numpy.arange of positive integer numbers, denotes the indices of elementary components 
                    ts: numpy.array, input time series 
                    M:int, number of  values to forecast or estimate
                    mode:str, forecasts M future values for S time series if mode is "forward", or estimates the last M values for S input time series, if mode is 'retrospective'
                  
                    Returns
                    -------
                    y_pred: numpy.array, original time series + M forecasted values, or original time series, where the last M values are estimated
        """
        R = self.estimate_LRR(idx_components)
        R = R.reshape(-1,1)
        if M<=0:
            y_pred = ts[:self.N]
            return y_pred
        
        if mode == 'forward':
            y_pred = np.zeros((1,self.N+M))
            y_pred[0,:self.N] = ts[:self.N]
            for m in range(0,M):
                    
                    y_pred[:,self.N+m] = (y_pred[:,self.N-self.L+m+1:y_pred.shape[0]-M+m-1]) @ R
        elif mode == 'retrospective':
            y_pred = np.zeros((1,self.N))
            y_pred[0,:self.N-M] = ts[:self.N-M]
            for m in range(0,M):
                
                    y_pred[:,self.N+m-M] = (y_pred[:,self.N-self.L+m+1-M:y_pred.shape[0]-M+m-1]) @ R
    
            
        else:
            
            raise  ValueError('Wrong type of the forecasting mode')

        
            
        return y_pred
    
    def estimate_ESPRIT(self, idx_components=[0], decompose_rho_omega=False):
        """
                    Estimates polynomial roots of the signal using ESPRIT algorithm and LS
                    Parameters
                    ----------
                    idx_components:list or numpy.arange of positive integer numbers, denotes the indices of elementary components used 
                    decompose_rho_omega: bool, decompose the roots into real and imaginery part
                    Returns
                    -------
                    mu: numpy.array, complex polynomial roots
                    rho: numpy.array, real polynomial roots
                    omega: numpy.array, complex part  polynomial roots
        """
        # P_orth = self.U[:,idx_components]
    
        P_orth = sp.linalg.qr(self.U[:,:self.L])[0]
            
        P_ = P_orth[:-1,idx_components] #last row removed
        _P = P_orth[1:,idx_components] #first row removed
        
        Inv_ = np.linalg.inv(P_.T @ P_) @ P_.T 
        MM = Inv_ @ _P
        mu = np.flip(np.sort( np.linalg.eigvals(MM)))
        if decompose_rho_omega == True:
            rho = np.imag(mu)
            omega =  np.real(mu)
            return mu, rho, omega
        else:
            return mu
        
    def plot_polynomial_roots(self, idx_components):
        """
                    Plots estimated polynomial roots of the signal using ESPRIT algorithm and LS
                    Parameters
                    ----------
                    idx_components:list or numpy.arange of positive integer numbers, denotes the indices of elementary components used  
                    decompose_rho_omega: bool, decompose the roots into real and imaginery part
                    Returns
                    -------
                    None
        """
        _,rho, omega = self.estimate_ESPRIT( idx_components=idx_components, decompose_rho_omega=True )
        fig, ax = plt.subplots()
        unit_circle = plt.Circle((0, 0), 1, color='b', fill=False)
        ax.add_patch(unit_circle)
        
        plt.title("Polynomial roots on the unit circle")
        ax.plot( omega, rho, "r*")
        plt.gca().set_aspect('equal')
        ax.set_xlabel('Real Part')
        ax.set_ylabel('Imaginary Part')
        plt.show()
