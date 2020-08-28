import numpy as np
import scipy.linalg as slin

class sim_probCCA:

    def __init__(self,obs_dim1,obs_dim2,latent_dim,rand_seed=None):
        self.xDim = obs_dim1
        self.yDim = obs_dim2
        self.zDim = latent_dim

        # set random seed
        if not(rand_seed is None):
            np.random.seed(rand_seed)
        
        # generate model parameters
        mu_x = np.random.randn(obs_dim1)
        mu_y = np.random.randn(obs_dim2)
        W_x = np.random.randn(obs_dim1,latent_dim)
        W_y = np.random.randn(obs_dim2,latent_dim)
        x_sqrt = np.random.randn(obs_dim1,obs_dim1)
        y_sqrt = np.random.randn(obs_dim2,obs_dim2)
        psi_x = x_sqrt.dot(x_sqrt.T)
        psi_y = y_sqrt.dot(y_sqrt.T)

        # compute ground truth canonical correlations
        covX = W_x.dot(W_x.T) + psi_x
        covY = W_y.dot(W_y.T) + psi_y
        covXY = W_x.dot(W_y.T)
        inv_sqrt_covX = np.linalg.inv(slin.sqrtm(covX))
        inv_sqrt_covY = np.linalg.inv(slin.sqrtm(covY))
        K = inv_sqrt_covX.dot(covXY).dot(inv_sqrt_covY)
        u,d,v = np.linalg.svd(K)
        rho = d[0:latent_dim]

        # store model parameters in dict
        cca_params = {
            'mu_x':mu_x,
            'mu_y':mu_y,
            'W_x':W_x,
            'W_y':W_y,
            'psi_x':psi_x,
            'psi_y':psi_y,
            'zDim':latent_dim,
            'rho':rho
        }
        self.cca_params = cca_params


    def sim_data(self,N,rand_seed=None):
        # set random seed
        if not(rand_seed is None):
            np.random.seed(rand_seed)

        mu_x = self.cca_params['mu_x'].reshape(self.xDim,1)
        mu_y = self.cca_params['mu_y'].reshape(self.yDim,1)
        W_x,W_y = self.cca_params['W_x'], self.cca_params['W_y']
        psi_x,psi_y = self.cca_params['psi_x'], self.cca_params['psi_y']

        # generate data
        z = np.random.randn(self.zDim,N)
        ns_x = slin.sqrtm(psi_x).dot(np.random.randn(self.xDim,N))
        ns_y = slin.sqrtm(psi_y).dot(np.random.randn(self.yDim,N))
        X = (W_x.dot(z) + ns_x) + mu_x
        Y = (W_y.dot(z) + ns_y) + mu_y

        return X.T, Y.T


    def get_params(self):
        return self.cca_params


    def set_params(self,cca_params):
        self.cca_params = cca_params

