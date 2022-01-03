# In[1]:


#import packages
import numpy as np
import _pickle as pk
import netCDF4 as nc
import Area_Avg as aa
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import multiprocessing as mp
import logging
from multiprocessing_logging import install_mp_handler
import time
from itertools import repeat
import sys

# In[2]:

def LinReg_alpha_wrap(predictand,predictor):
    fit = LinearRegression().fit(predictor,predictand)
    return fit.coef_

def LinReg_int_wrap(predictand,predictor):
    fit = LinearRegression().fit(predictor,predictand)
    return fit.intercept_

print('Setting up worker pools using',mp.cpu_count(),'cpu cores')
cpus = mp.cpu_count()
Pools = mp.Pool(cpus)

models = sys.argv[1]

Source = '/home/jgvirgin/Projects/CanESM/Data/'+models

#define dimensions for horizontal grid
lat = np.linspace(-87.864,87.864,64)
lon = np.linspace(0,357.1875,128)
# In[3]:

print('read in data')

q_tr_stk = pk.load(open(Source+'/Fluxes/'+models+'_TrWV_FLUX_Grid.pi','rb'))
q_sr_stk = pk.load(open(Source+'/Fluxes/'+models+'_SrWV_FLUX_Grid.pi','rb'))

tas_stk = pk.load(open(Source+'/Anomalies/'+models+'_Ajd_4xCO2_TmSrs.pi','rb'))['tas']
tas_GAM = aa.LatLonavg_Time(np.mean(tas_stk,axis=1),lat,lon)
tas_GAM_rsh = tas_GAM.reshape(-1,1)

# In[11]:


print('regress water vapour perturbations against global annual mean surface temperature change this gives a slope for each gridpoint and each month')

start_time = time.time()

q_tr_flatten_adj = dict()
q_tr_flatten_fb = dict()

q_tr_adj = dict()
q_tr_fb = dict()

q_sr_flatten_adj = dict()
q_sr_flatten_fb = dict()

q_sr_adj = dict()
q_sr_fb = dict()

for kernels in q_tr_stk.keys():
    print('on kernel -',kernels)
    q_tr_flatten_adj[kernels] = [i for i in np.swapaxes(q_tr_stk[kernels][:,:,:].reshape(len(q_tr_stk[kernels][:,0,0]),64*128),0,1)]
    q_tr_flatten_fb[kernels] = [i for i in np.swapaxes(q_tr_stk[kernels][:,:,:].reshape(len(q_tr_stk[kernels][:,0,0]),64*128),0,1)]

    q_tr_adj[kernels] = Pools.starmap(LinReg_int_wrap,zip(q_tr_flatten_adj[kernels],repeat(tas_GAM_rsh[:,:])))
    q_tr_fb[kernels] = Pools.starmap(LinReg_alpha_wrap,zip(q_tr_flatten_fb[kernels],repeat(tas_GAM_rsh[:,:])))

    q_sr_flatten_adj[kernels] = [i for i in np.swapaxes(q_sr_stk[kernels][:,:,:].reshape(len(q_tr_stk[kernels][:,0,0]),64*128),0,1)]
    q_sr_flatten_fb[kernels] = [i for i in np.swapaxes(q_sr_stk[kernels][:,:,:].reshape(len(q_tr_stk[kernels][:,0,0]),64*128),0,1)]

    q_sr_adj[kernels] = Pools.starmap(LinReg_int_wrap,zip(q_sr_flatten_adj[kernels],repeat(tas_GAM_rsh[:,:])))
    q_sr_fb[kernels] = Pools.starmap(LinReg_alpha_wrap,zip(q_sr_flatten_fb[kernels],repeat(tas_GAM_rsh[:,:])))

end_time = time.time() - start_time
print(end_time/60, 'minutes for complete regression to finish')

Pools.close()
Pools.join()

print('rebuild array shape')
q_tr_adj_rebuild = dict()
q_tr_fb_rebuild = dict()

q_sr_adj_rebuild = dict()
q_sr_fb_rebuild = dict()

for kernels in q_tr_stk.keys():
    q_tr_adj_rebuild[kernels] = np.stack(q_tr_adj[kernels][:],axis=0).reshape(64,128)
    q_tr_fb_rebuild[kernels] = np.stack(q_tr_fb[kernels][:],axis=0).reshape(64,128)

    q_sr_adj_rebuild[kernels] = np.stack(q_sr_adj[kernels][:],axis=0).reshape(64,128)
    q_sr_fb_rebuild[kernels] = np.stack(q_sr_fb[kernels][:],axis=0).reshape(64,128)


print('saving variables')

q_tr_adj_rebuild_file = open(models+'_TrWV_FLUX_ADJ_Grid.pi','wb')
pk.dump(q_tr_adj_rebuild,q_tr_adj_rebuild_file)
q_tr_adj_rebuild_file.close()

q_tr_fb_rebuild_file = open(models+'_TrWV_FLUX_FB_Grid.pi','wb')
pk.dump(q_tr_fb_rebuild,q_tr_fb_rebuild_file)
q_tr_fb_rebuild_file.close()

q_sr_adj_rebuild_file = open(models+'_SrWV_FLUX_ADJ_Grid.pi','wb')
pk.dump(q_sr_adj_rebuild,q_sr_adj_rebuild_file)
q_sr_adj_rebuild_file.close()

q_sr_fb_rebuild_file = open(models+'_SrWV_FLUX_FB_Grid.pi','wb')
pk.dump(q_sr_fb_rebuild,q_sr_fb_rebuild_file)
q_sr_fb_rebuild_file.close()

print('finished')
