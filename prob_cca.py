import numpy as np
import cca.canon_corr as cc
import scipy.linalg as slin
import sklearn.model_selection as ms

# imports for parallelization
from joblib import Parallel, delayed
from functools import partial
from psutil import cpu_count
from tqdm import tqdm

class prob_cca:


    def __init__(self):
        self.params = []

    def train(self,X,Y,zDim,tol=1e-6,max_iter=int(1e6),verbose=False,rand_seed=None,warmstart=True):
        # set random seed
        if not(rand_seed is None):
            np.random.seed(rand_seed)

        # set some useful parameters
        N,xDim = X.shape
        yDim = Y.shape[1]
        mu_x,mu_y = X.mean(axis=0),Y.mean(axis=0)
        Xc,Yc = (X-mu_x), (Y-mu_y) 
        XcYc = np.concatenate((Xc,Yc),axis=1)
        covX = 1/N * (Xc.T).dot(Xc)
        covY = 1/N * (Yc.T).dot(Yc)
        sampleCov = 1/N * (XcYc.T).dot(XcYc)
        Iz = np.identity(zDim)
        const = (xDim+yDim)*np.log(2*np.pi)

        # check that covariance is full rank
        if np.linalg.matrix_rank(sampleCov)==(xDim+yDim):
            x_scale = np.exp(2/xDim*np.sum(np.log(np.diag(slin.cholesky(covX)))))
            y_scale = np.exp(2/yDim*np.sum(np.log(np.diag(slin.cholesky(covY)))))
        else:
            raise np.linalg.LinAlgError('Covariance matrix is low rank')

        # initialize parameters
        if warmstart:
            tmp = prob_cca()
            tmp.train_maxLL(X,Y,zDim)
            W_x = tmp.get_params()['W_x']
            W_y = tmp.get_params()['W_y']
        else:
            W_x = np.random.randn(xDim,zDim) * np.sqrt(x_scale/zDim)
            W_y = np.random.randn(yDim,zDim) * np.sqrt(y_scale/zDim)
        Ph = sampleCov
        Ph_mask = np.ones(Ph.shape)
        Ph_mask[:xDim,xDim:] = np.zeros((xDim,yDim))
        Ph_mask[xDim:,:xDim] = np.zeros((yDim,xDim))
        
        L_total = np.concatenate((W_x,W_y),axis=0) # combined loading matrix

        # em algorithm
        LL = []
        for i in range(max_iter):
            # calculate inverse of Ph
            # inv1 = np.concatenate((slin.inv(Ph[:xDim,:xDim]),np.zeros((xDim,yDim))),axis=1)
            # inv2 = np.concatenate((np.zeros((yDim,xDim)),slin.inv(Ph[xDim:,xDim:])),axis=1)
            # iPh = np.concatenate((inv1,inv2),axis=0)
            # iPh = 0.5 * (iPh + iPh.T) # Ensure symmetry
            iPh = slin.inv(Ph)

            # E-step: set q(z) = p(z,zx,zy|x,y)
            iPhL = iPh.dot(L_total)
            if zDim==0:
                iSig = iPh
            else:
                iSig = iPh - iPhL.dot(slin.inv(Iz+(L_total.T).dot(iPhL))).dot(iPhL.T)
            iSigL = iSig.dot(L_total)
            cov_iSigL = sampleCov.dot(iSigL)
            E_zz = Iz - (L_total.T).dot(iSigL) + (iSigL.T).dot(cov_iSigL)
            
            # compute log likelihood
            logDet = 2*np.sum(np.log(np.diag(slin.cholesky(iSig))))
            curr_LL = -N/2 * (const - logDet + np.trace(iSig.dot(sampleCov)))
            LL.append(curr_LL)
            if verbose: print('EM iteration ',i,', LL={:.2f}'.format(curr_LL))

            # check for convergence (training LL increases by less than tol)
            if i>1 and (LL[-1]-LL[-2])<tol:
                break

            # M-step: compute new L and Ph
            if not(zDim==0):
                L_total = cov_iSigL.dot(slin.inv(E_zz))
            Ph = sampleCov - L_total.dot(cov_iSigL.T)
            Ph = Ph * Ph_mask
            Ph = (Ph + Ph.T)/2 # enforce symmetry

        # get final parameters
        W_x, W_y = L_total[:xDim,:], L_total[xDim:,:]
        psi_x, psi_y = Ph[:xDim,:xDim], Ph[xDim:,xDim:]

        # compute canonical correlations
        est_covX = W_x.dot(W_x.T) + psi_x
        est_covY = W_y.dot(W_y.T) + psi_y
        est_covXY = W_x.dot(W_y.T)
        inv_sqrt_covX = slin.inv(slin.sqrtm(est_covX))
        inv_sqrt_covY = slin.inv(slin.sqrtm(est_covY))
        K = inv_sqrt_covX.dot(est_covXY).dot(inv_sqrt_covY)
        u,d,vt = slin.svd(K)
        rho = d[0:zDim]
        
        # order W_x, W_y by canon corrs
        pd = np.diag(np.sqrt(rho))
        W_x = slin.sqrtm(est_covX).dot(u[:,0:zDim]).dot(pd)
        W_y = slin.sqrtm(est_covY).dot(vt[0:zDim,:].T).dot(pd)

        # create parameter dict
        self.params = {
            'mu_x':mu_x,'mu_y':mu_y, # estimated mean per neuron
            'L_total': L_total, # maximum likelihood across-area loading matrix
            'W_x':W_x,'W_y':W_y, # across-area loading matrices in canonical modes (same col span as L_total)
            'psi_x':psi_x,'psi_y':psi_y, # within-area and private variance per neuron
            'zDim':zDim, # selected dimensionality
            'rho':rho # canonical correlations
        }

    def get_loading_matrices(self):
        # subdivide L_total into W_x, W_y directly from maximum likelihood estimation

        xDim = len(self.params['mu_x'])
        zDim = self.params['zDim']
        L_total = self.params['L_total']
        # get final parameters
        W_x, W_y = L_total[:xDim,:zDim], L_total[xDim:,:zDim]

        return W_x, W_y
    
    def get_canonical_directions(self):
        # get canonical directions (as returned by CCA)

        W_x, W_y = self.get_loading_matrices()
        psi_x, psi_y = self.params['psi_x'], self.params['psi_y']
        zDim = self.params['zDim']

        # compute canonical correlations
        est_covX = W_x.dot(W_x.T) + psi_x
        est_covY = W_y.dot(W_y.T) + psi_y
        est_covXY = W_x.dot(W_y.T)
        inv_sqrt_covX = slin.inv(slin.sqrtm(est_covX))
        inv_sqrt_covY = slin.inv(slin.sqrtm(est_covY))
        K = inv_sqrt_covX.dot(est_covXY).dot(inv_sqrt_covY)
        u,d,vt = slin.svd(K)
        rho = d[0:zDim]

        canonical_dirs_x = slin.inv(slin.sqrtm(est_covX)) @ u[:,:zDim]
        canonical_dirs_y = slin.inv(slin.sqrtm(est_covY)) @ vt[:zDim,:].T

        return (canonical_dirs_x, canonical_dirs_y), rho


    def train_maxLL(self,X,Y,zDim):
        N,xDim = X.shape
        N,yDim = Y.shape

        # train vanilla CCA model
        cca_model = cc.canon_corr()
        cca_model.train(X,Y,zDim)
        cca_params = cca_model.get_params()

        # center data and compute covariances
        Xc = X-cca_params['mu_x']
        Yc = Y-cca_params['mu_y']
        covX = 1/N * (Xc.T).dot(Xc)
        covY = 1/N * (Yc.T).dot(Yc)
        covXY = 1/N * (Xc.T).dot(Yc)
        XY = np.concatenate((X,Y),axis=1)
        sampleCov = np.cov(XY.T,bias=True)

        # compute pCCA parameters
        M = np.diag(np.sqrt(cca_params['rho']))
        W_x = covX.dot(cca_params['W_x']).dot(M)
        W_y = covY.dot(cca_params['W_y']).dot(M)
        psi_x = covX - W_x.dot(W_x.T)
        psi_y = covY - W_y.dot(W_y.T)

        # create pCCA parameter dict
        self.params = {
            'mu_x':cca_params['mu_x'],
            'mu_y':cca_params['mu_y'],
            'W_x':W_x,
            'W_y':W_y,
            'psi_x':psi_x,
            'psi_y':psi_y,
            'zDim':zDim,
            'rho':cca_params['rho']
        }


    def get_params(self):
        return self.params

    def set_params(self,params):
        self.params = params


    def estep(self,X,Y):
        N,xDim = X.shape
        N,yDim = Y.shape

        # get model parameters
        mu_x,mu_y = self.params['mu_x'],self.params['mu_y']
        W_x,W_y = self.params['W_x'],self.params['W_y']
        L_total = np.concatenate((W_x,W_y),axis=0)
        psi_x = self.params['psi_x']
        psi_y = self.params['psi_y']
        tmp1 = np.concatenate((psi_x,np.zeros((xDim,yDim))),axis=1)
        tmp2 = np.concatenate((np.zeros((yDim,xDim)),psi_y),axis=1)
        psi = np.concatenate((tmp1,tmp2),axis=0)

        # center data and compute covariances
        Xc = X-mu_x
        Yc = Y-mu_y
        XcYc = np.concatenate((Xc,Yc),axis=1)
        sampleCov = 1/N * (XcYc.T).dot(XcYc)

        # compute z
        Iz = np.identity(self.params['zDim'])
        C = L_total.dot(L_total.T) + psi
        invC = slin.inv(C)
        z_mu = XcYc.dot(invC).dot(L_total)
        z_cov = np.diag(np.diag(Iz - (L_total.T).dot(invC).dot(L_total)))

        # compute z_x
        C_x = C[0:xDim,0:xDim]
        invCx = np.linalg.inv(C_x)
        zx_mu = Xc.dot(invCx).dot(W_x)
        zx_cov = np.diag(np.diag(Iz - (W_x.T).dot(invCx).dot(W_x)))

        # compute z_y
        C_y = C[xDim:(xDim+yDim),xDim:(xDim+yDim)]
        invCy = np.linalg.inv(C_y)
        zy_mu = Yc.dot(invCy).dot(W_y)
        zy_cov = np.diag(np.diag(Iz - (W_y.T).dot(invCy).dot(W_y)))

        # compute LL
        const = (xDim+yDim)*np.log(2*np.pi)
        logDet = 2*np.sum(np.log(np.diag(slin.cholesky(C))))
        LL = -N/2 * (const + logDet + np.trace(invC.dot(sampleCov)))
        
        # return posterior and LL
        z = { 
            'z_mu':z_mu,
            'z_cov':z_cov,
            'zx_mu':zx_mu,
            'zx_cov':zx_cov,
            'zy_mu':zy_mu,
            'zy_cov':zy_cov
        }
        
        return z, LL


    def orthogonalize(self,zx_mu,zy_mu):
        N,zDim = zx_mu.shape

        Wx = self.params['W_x']
        Wy = self.params['W_y']

        # orthogonalize zx
        u,s,vt = np.linalg.svd(Wx,full_matrices=False)
        Wx_orth = u
        TT = np.diag(s).dot(vt)
        zx = (TT.dot(zx_mu.T)).T

        # orthogonalize zy
        u,s,vt = np.linalg.svd(Wy,full_matrices=False)
        Wy_orth = u
        TT = np.diag(s).dot(vt)
        zy = (TT.dot(zy_mu.T)).T

        # return z_orth, Lorth
        z_orth = {'zx':zx,'zy':zy}
        W_orth = {'Wx':Wx_orth,'Wy':Wy_orth}

        return z_orth, W_orth


    def crossvalidate(self,X,Y,zDim_list=np.arange(10),n_folds=10,verbose=True,max_iter=int(1e6),tol=1e-6,rand_seed=None,parallelize=False):
        # set random seed
        if not(rand_seed is None):
            np.random.seed(rand_seed)

        # make sure z dims are integers
        z_list = zDim_list.astype(int)
        LL_curves = {'z_list':z_list}

        # create k-fold iterator
        if verbose: print('Crossvalidating pCCA model to choose # of dims...')
        cv_kfold = ms.KFold(n_splits=n_folds,shuffle=True,random_state=rand_seed)

        # iterate through train/test splits
        i = 0
        LLs = np.zeros([n_folds,len(z_list)])
        for train_idx,test_idx in cv_kfold.split(X):
            if verbose: print('   Fold ',i+1,' of ',n_folds,'...')
            X_train,X_test = X[train_idx], X[test_idx]
            Y_train,Y_test = Y[train_idx], Y[test_idx]
            
            # iterate through each zDim
            func = partial(self._cv_helper,Xtrain=X_train,Ytrain=Y_train,Xtest=X_test,Ytest=Y_test,\
                           rand_seed=rand_seed,max_iter=max_iter,tol=tol)
            if parallelize:
                tmp = Parallel(n_jobs=cpu_count(logical=False),backend='loky')\
                    (delayed(func)(z_list[j]) for j in range(len(z_list)))
                LLs[i,:] = [val[0] for val in tmp]
            else:
                for j in tqdm(range(len(z_list))):
                    tmp = func(z_list[j])
                    LLs[i,j]= tmp[0]
            i = i+1
        
        sum_LLs = LLs.sum(axis=0)
        LL_curves['LLs'] = sum_LLs

        # find the best # of z dimensions and train CCA model
        max_idx = np.argmax(sum_LLs)
        zDim = z_list[max_idx]
        LL_curves['zDim']=zDim
        LL_curves['final_LL'] = sum_LLs[max_idx]
        self.train(X,Y,zDim)

        self.LL_curves = LL_curves

        return LL_curves

    def _cv_helper(self,zDim,Xtrain,Ytrain,Xtest,Ytest,rand_seed=None,max_iter=int(1e5),tol=1e-6):
        tmp = prob_cca()
        tmp.train(Xtrain,Ytrain,zDim,rand_seed=rand_seed,max_iter=max_iter,tol=tol)
        # log-likelihood
        _,LL = tmp.estep(Xtest,Ytest)
        
        return LL

    def compute_dshared(self,cutoff_thresh=0.95):
        # for area x
        Wx = self.params['W_x']
        shared_x = Wx.dot(Wx.T)
        u,s,vt = np.linalg.svd(shared_x,full_matrices=False,hermitian=True)
        var_exp = np.cumsum(s)/np.sum(s)
        dims = np.where(var_exp >= (cutoff_thresh - 1e-9))[0]
        dshared_x = dims[0]+1

        # for area y
        Wy = self.params['W_y']
        shared_y = Wy.dot(Wy.T)
        u,s,vt = np.linalg.svd(shared_y,full_matrices=False,hermitian=True)
        var_exp = np.cumsum(s)/np.sum(s)
        dims = np.where(var_exp >= (cutoff_thresh - 1e-9))[0]
        dshared_y = dims[0]+1

        # overall
        W = np.concatenate((Wx,Wy),axis=0)
        shared = W.dot(W.T)
        u,s,vt = np.linalg.svd(shared,full_matrices=False,hermitian=True)
        var_exp = np.cumsum(s)/np.sum(s)
        dims = np.where(var_exp >= (cutoff_thresh - 1e-9))[0]
        dshared_all = dims[0]+1

        return {'dshared_x':dshared_x,'dshared_y':dshared_y,'dshared_all':dshared_all}

    def compute_part_ratio(self):
        # for area x
        Wx = self.params['W_x']
        shared_x = Wx.dot(Wx.T)
        u,s,vt = np.linalg.svd(shared_x,full_matrices=False,hermitian=True)
        pr_x = np.square(s.sum()) / np.square(s).sum()

        # for area y
        Wy = self.params['W_y']
        shared_y = Wy.dot(Wy.T)
        u,s,vt = np.linalg.svd(shared_y,full_matrices=False,hermitian=True)
        pr_y = np.square(s.sum()) / np.square(s).sum()

        # overall
        W = np.concatenate((Wx,Wy),axis=0)
        shared = W.dot(W.T)
        u,s,vt = np.linalg.svd(shared,full_matrices=False,hermitian=True)
        pr = np.square(s.sum()) / np.square(s).sum()

        return {'pr':pr,'pr_x':pr_x,'pr_y':pr_y}


    def compute_psv(self):
        # for area x
        Wx = self.params['W_x']
        priv_x = np.diag(self.params['psi_x'])
        shared_x = np.diag(Wx.dot(Wx.T))
        ind_psv_x = shared_x / (shared_x+priv_x) * 100
        psv_x = np.mean(ind_psv_x)

        # for area y
        Wy = self.params['W_y']
        priv_y = np.diag(self.params['psi_y'])
        shared_y = np.diag(Wy.dot(Wy.T))
        ind_psv_y = shared_y / (shared_y+priv_y) * 100
        psv_y = np.mean(ind_psv_y)

        # overall
        psv_overall = np.mean(np.concatenate((ind_psv_x,ind_psv_y)))

        psv = {
            'ind_psv_x':ind_psv_x,
            'ind_psv_y':ind_psv_y,
            'psv_x':psv_x,
            'psv_y':psv_y,
            'psv_all':psv_overall
        }
        return psv


    def compute_metrics(self,cutoff_thresh=0.95):
        dshared = self.compute_dshared(cutoff_thresh=cutoff_thresh)
        psv = self.compute_psv()
        if 'cv_rho' in self.params:
            rho = self.params['cv_rho']
        else:
            rho = self.params['rho']
        pr = self.compute_part_ratio()

        metrics = {
            'dshared':dshared,
            'psv':psv,
            'part_ratio':pr,
            'rho':rho
        }
        return metrics