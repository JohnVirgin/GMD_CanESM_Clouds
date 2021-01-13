# In[1]:


#import packages
import numpy as np
import _pickle as pk
import netCDF4 as nc
import Area_Avg as aa
import pandas as pd
from sklearn.linear_model import LinearRegression
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
plnck_stk = pk.load(open(Source+'/Fluxes/'+models+'_PLANCK_FLUX_Grid.pi','rb'))
lpse_stk = pk.load(open(Source+'/Fluxes/'+models+'_LAPSE_FLUX_Grid.pi','rb'))
ta_sr_stk = pk.load(open(Source+'/Fluxes/'+models+'_SrTEMP_FLUX_Grid.pi','rb'))

tas_stk = pk.load(open(Source+'/Anomalies/'+models+'_Ajd_4xCO2_TmSrs.pi','rb'))['tas']
tas_GAM = aa.LatLonavg_Time(np.mean(tas_stk,axis=1),lat,lon)
tas_GAM_rsh = tas_GAM.reshape(-1,1)

# In[11]:


print('regress temperature perturbations against global annual mean surface temperature change this gives a slope for each gridpoint and each month')

start_time = time.time()

plnck_flatten_adj = dict()
plnck_flatten_fb = dict()

plnck_adj = dict()
plnck_fb = dict()

lpse_flatten_adj = dict()
lpse_flatten_fb = dict()

lpse_adj = dict()
lpse_fb = dict()

ta_sr_flatten_adj = dict()
ta_sr_flatten_fb = dict()

ta_sr_adj = dict()
ta_sr_fb = dict()
for kernels in plnck_stk.keys():
    print('on kernel -',kernels)
    plnck_flatten_adj[kernels] = [i for i in np.swapaxes(plnck_stk[kernels][:,:,:].reshape(len(plnck_stk[kernels][:,0,0]),64*128),0,1)]
    plnck_flatten_fb[kernels] = [i for i in np.swapaxes(plnck_stk[kernels][:,:,:].reshape(len(plnck_stk[kernels][:,0,0]),64*128),0,1)]

    plnck_adj[kernels] = Pools.starmap(LinReg_int_wrap,zip(plnck_flatten_adj[kernels],repeat(tas_GAM_rsh[:,:])))
    plnck_fb[kernels] = Pools.starmap(LinReg_alpha_wrap,zip(plnck_flatten_fb[kernels],repeat(tas_GAM_rsh[:,:])))

    lpse_flatten_adj[kernels] = [i for i in np.swapaxes(lpse_stk[kernels][:,:,:].reshape(len(plnck_stk[kernels][:,0,0]),64*128),0,1)]
    lpse_flatten_fb[kernels] = [i for i in np.swapaxes(lpse_stk[kernels][:,:,:].reshape(len(plnck_stk[kernels][:,0,0]),64*128),0,1)]

    lpse_adj[kernels] = Pools.starmap(LinReg_int_wrap,zip(lpse_flatten_adj[kernels],repeat(tas_GAM_rsh[:,:])))
    lpse_fb[kernels] = Pools.starmap(LinReg_alpha_wrap,zip(lpse_flatten_fb[kernels],repeat(tas_GAM_rsh[:,:])))

    ta_sr_flatten_adj[kernels] = [i for i in np.swapaxes(ta_sr_stk[kernels][:,:,:].reshape(len(plnck_stk[kernels][:,0,0]),64*128),0,1)]
    ta_sr_flatten_fb[kernels] = [i for i in np.swapaxes(ta_sr_stk[kernels][:,:,:].reshape(len(plnck_stk[kernels][:,0,0]),64*128),0,1)]

    ta_sr_adj[kernels] = Pools.starmap(LinReg_int_wrap,zip(ta_sr_flatten_adj[kernels],repeat(tas_GAM_rsh[:,:])))
    ta_sr_fb[kernels] = Pools.starmap(LinReg_alpha_wrap,zip(ta_sr_flatten_fb[kernels],repeat(tas_GAM_rsh[:,:])))

end_time = time.time() - start_time
print(end_time/60, 'minutes for complete regression to finish')

Pools.close()
Pools.join()

print('rebuild array shape')
plnck_adj_rebuild = dict()
plnck_fb_rebuild = dict()

lpse_adj_rebuild = dict()
lpse_fb_rebuild = dict()

ta_sr_adj_rebuild = dict()
ta_sr_fb_rebuild = dict()
for kernels in plnck_stk.keys():
    plnck_adj_rebuild[kernels] = np.stack(plnck_adj[kernels][:],axis=0).reshape(64,128)
    plnck_fb_rebuild[kernels] = np.stack(plnck_fb[kernels][:],axis=0).reshape(64,128)

    lpse_adj_rebuild[kernels] = np.stack(lpse_adj[kernels][:],axis=0).reshape(64,128)
    lpse_fb_rebuild[kernels] = np.stack(lpse_fb[kernels][:],axis=0).reshape(64,128)

    ta_sr_adj_rebuild[kernels] = np.stack(ta_sr_adj[kernels][:],axis=0).reshape(64,128)
    ta_sr_fb_rebuild[kernels] = np.stack(ta_sr_fb[kernels][:],axis=0).reshape(64,128)

print('saving variables')

plnck_adj_rebuild_file = open(models+'_PLANCK_FLUX_ADJ_Grid.pi','wb')
pk.dump(plnck_adj_rebuild,plnck_adj_rebuild_file)
plnck_adj_rebuild_file.close()

plnck_fb_rebuild_file = open(models+'_PLANCK_FLUX_FB_Grid.pi','wb')
pk.dump(plnck_fb_rebuild,plnck_fb_rebuild_file)
plnck_fb_rebuild_file.close()

lpse_adj_rebuild_file = open(models+'_LAPSE_FLUX_ADJ_Grid.pi','wb')
pk.dump(lpse_adj_rebuild,lpse_adj_rebuild_file)
lpse_adj_rebuild_file.close()

lpse_fb_rebuild_file = open(models+'_LAPSE_FLUX_FB_Grid.pi','wb')
pk.dump(lpse_fb_rebuild,lpse_fb_rebuild_file)
lpse_fb_rebuild_file.close()

ta_sr_adj_rebuild_file = open(models+'_SrTEMP_FLUX_ADJ_Grid.pi','wb')
pk.dump(ta_sr_adj_rebuild,ta_sr_adj_rebuild_file)
ta_sr_adj_rebuild_file.close()

ta_sr_fb_rebuild_file = open(models+'_SrTEMP_FLUX_FB_Grid.pi','wb')
pk.dump(ta_sr_fb_rebuild,ta_sr_fb_rebuild_file)
ta_sr_fb_rebuild_file.close()

print('finished')
