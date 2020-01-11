import numpy as np
import scipy.linalg as slin

class canon_corr:


	def __init__(self):
		self.params = []
	

	def train(self,X,Y,zDim):
		N,xDim = X.shape
		N,yDim = Y.shape

		# center data
		mu_x = np.mean(X,axis=0)
		mu_y = np.mean(Y,axis=0)
		Xc = X-mu_x
		Yc = Y-mu_y
		
		# compute covariances
		covX = (1/N) * (Xc.T).dot(Xc)
		covY = (1/N) * (Yc.T).dot(Yc)
		covXY = (1/N) * (Xc.T).dot(Yc)

		# check that data matrices are full rank
		if (np.linalg.matrix_rank(covX)!=xDim):
			raise np.linalg.LinAlgError('X is low rank')
		elif (np.linalg.matrix_rank(covY)!=yDim):
			raise np.linalg.LinAlgError('Y is low rank')

		# compute canonical vectors
		inv_sqrt_covX = np.linalg.inv(slin.sqrtm(covX))
		inv_sqrt_covY = np.linalg.inv(slin.sqrtm(covY))
		K = inv_sqrt_covX.dot(covXY).dot(inv_sqrt_covY)
		u,d,v = np.linalg.svd(K)
		W_x = inv_sqrt_covX.dot(u[:,0:zDim])
		W_y = inv_sqrt_covY.dot(v[0:zDim,:].T)
		rho = d[0:zDim]

		# create cca parameter dict
		self.params = {
			'mu_x':mu_x,
			'mu_y':mu_y,
			'W_x':W_x,
			'W_y':W_y,
			'rho':rho
		}


	def proj_data(self,X,Y):
		# center data
		Xc = X - self.params['mu_x']
		Yc = Y - self.params['mu_y']

		# project data
		canon_X = Xc.dot(self.params['W_x'])
		canon_Y = Yc.dot(self.params['W_y'])

		return canon_X, canon_Y


	def get_params(self):
		return self.params

	def set_params(self,params):
		self.params = params
