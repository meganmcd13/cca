#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sim_probCCA as sc
import canon_corr as cca
import prob_cca as pcca
import matplotlib.pyplot as plt

xDim,yDim,zDim = 30,20,5

# simulate from pCCA model
pcca_sim = sc.sim_probCCA(xDim,yDim,zDim,rand_seed=0)
X,Y = pcca_sim.sim_data(1000,rand_seed=0)
sim_model = pcca.prob_cca()
sim_model.set_params(pcca_sim.get_params())


# In[2]:


# vanilla CCA
cca_model = cca.canon_corr()
cca_model.train(X,Y,zDim)
Zx,Zy = cca_model.proj_data(X,Y)

plt.figure(0)
plt.plot(Zx[:,0],Zy[:,0],'b.')
plt.xlabel('$Zx_{1}$')
plt.ylabel('$Zy_{1}$')
plt.show()


# In[3]:


# pCCA
pcca_model = pcca.prob_cca()
pcca_model.train_maxLL(X,Y,zDim)
z_pcca,curr_LL = pcca_model.estep(X,Y)
z_orth,Worth = pcca_model.orthogonalize(z_pcca['zx_mu'],z_pcca['zy_mu'])
Z_x = z_pcca['zx_mu']
Z_y = z_pcca['zy_mu']

plt.figure(1)
plt.plot(Z_x[:,0],Z_y[:,0],'b.')
plt.xlabel('$Zx_{1}$')
plt.ylabel('$Zy_{1}$')
plt.show()


# In[4]:


# crossvalidate pCCA
pcca_model = pcca.prob_cca()
LLs,zDim_list,max_LL,zDim = pcca_model.crossvalidate(X,Y)

plt.figure(2)
plt.plot(zDim_list,LLs,'bo-')
plt.plot(zDim,max_LL,'r^')
plt.show()


# In[5]:


cv_rho = pcca_model.get_params()['cv_rho']
train_rho = pcca_model.get_params()['rho']
true_rho = sim_model.get_params()['rho']

plt.figure(3)
plt.plot(train_rho,'bo-')
plt.plot(cv_rho,'ro-')
plt.plot(true_rho,'ko-')
plt.legend(('train','cv','ground truth'))
plt.show()


# In[6]:


# compute metrics
fit_metrics = pcca_model.compute_metrics()
gt_metrics = sim_model.compute_metrics()


# In[ ]:




