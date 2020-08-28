import numpy as np
import cca.canon_corr as cc
import sklearn.model_selection as ms

class prob_cca:


    def __init__(self):
        self.params = []


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
        mu_x = self.params['mu_x']
        mu_y = self.params['mu_y']
        mu = np.concatenate((mu_x,mu_y))
        W_x = self.params['W_x']
        W_y = self.params['W_y']
        W = np.concatenate((W_x,W_y),axis=0)
        psi_x = self.params['psi_x']
        psi_y = self.params['psi_y']
        tmp1 = np.concatenate((psi_x,np.zeros((xDim,yDim))),axis=1)
        tmp2 = np.concatenate((np.zeros((yDim,xDim)),psi_y),axis=1)
        psi = np.concatenate((tmp1,tmp2),axis=0)

        # center data and compute covariances
        Xc = X-mu_x
        Yc = Y-mu_y
        XYc = np.concatenate((Xc,Yc),axis=1)
        covX = 1/N * (Xc.T).dot(Xc)
        covY = 1/N * (Yc.T).dot(Yc)
        covXY = 1/N * (Xc.T).dot(Yc)
        sampleCov = 1/N * (XYc.T).dot(XYc)

        # compute z_xy
        Iz = np.identity(self.params['zDim'])
        C = W.dot(W.T) + psi
        invC = np.linalg.inv(C)
        z_mu = (W.T).dot(invC).dot(XYc.T)
        z_cov = np.diag(np.diag(Iz - (W.T).dot(invC).dot(W)))

        # compute z_x
        C_x = C[0:xDim,0:xDim]
        invCx = np.linalg.inv(C_x)
        zx_mu = (W_x.T).dot(invCx).dot(Xc.T)
        zx_cov = np.diag(np.diag(Iz - (W_x.T).dot(invCx).dot(W_x)))

        # compute z_y
        C_y = C[xDim:(xDim+yDim),xDim:(xDim+yDim)]
        invCy = np.linalg.inv(C_y)
        zy_mu = (W_y.T).dot(invCy).dot(Yc.T)
        zy_cov = np.diag(np.diag(Iz - (W_y.T).dot(invCy).dot(W_y)))

        # compute LL
        const = (xDim+yDim)*np.log(2*np.pi)
        logDet = 2*np.sum(np.log(np.diag(np.linalg.cholesky(C))))
        LL = -N/2 * (const + logDet + np.trace(invC.dot(sampleCov)))

        # return posterior and LL
        z = { 
            'z_mu':z_mu.T,
            'z_cov':z_cov,
            'zx_mu':zx_mu.T,
            'zx_cov':zx_cov,
            'zy_mu':zy_mu.T,
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


    def crossvalidate(self,X,Y,zDim_list=np.linspace(0,10,11),n_folds=10,verbose=True,rand_seed=None):
        N,D = X.shape

        # make sure z dims are integers
        z_list = zDim_list.astype(int)

        # create k-fold iterator
        if verbose:
            print('Crossvalidating pCCA model to choose # of dims...')
        cv_kfold = ms.KFold(n_splits=n_folds,shuffle=True,random_state=rand_seed)

        # iterate through train/test splits
        i = 0
        LLs = np.zeros([n_folds,len(z_list)])
        for train_idx,test_idx in cv_kfold.split(X):
            if verbose:
                print('   Fold ',i+1,' of ',n_folds,'...')

            X_train,X_test = X[train_idx], X[test_idx]
            Y_train,Y_test = Y[train_idx], Y[test_idx]
            # iterate through each zDim
            for j in range(len(z_list)):
                tmp = prob_cca()
                tmp.train_maxLL(X_train,Y_train,z_list[j])
                z,curr_LL = tmp.estep(X_test,Y_test)
                LLs[i,j] = curr_LL
            i = i+1
        
        sum_LLs = LLs.sum(axis=0)

        # find the best # of z dimensions and train CCA model
        max_idx = np.argmax(sum_LLs)
        zDim = z_list[max_idx]
        self.train_maxLL(X,Y,zDim)

        # cross-validate to get canonical correlations
        if verbose:
            print('Crossvalidating pCCA model to compute canon corrs...')
        zx,zy = np.zeros((2,N,zDim))
        for train_idx,test_idx in cv_kfold.split(X):
            X_train,X_test = X[train_idx], X[test_idx]
            Y_train,Y_test = Y[train_idx], Y[test_idx]

            tmp = prob_cca()
            tmp.train_maxLL(X_train,Y_train,zDim)
            z,curr_LL = tmp.estep(X_test,Y_test)

            zx[test_idx,:] = z['zx_mu']
            zy[test_idx,:] = z['zy_mu']

        cv_rho = np.zeros(zDim)
        for i in range(zDim):
            tmp = np.corrcoef(zx[:,i],zy[:,i])
            cv_rho[i] = tmp[0,1]
        
        self.params['cv_rho'] = cv_rho

        return sum_LLs, z_list, sum_LLs[max_idx], z_list[max_idx]


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