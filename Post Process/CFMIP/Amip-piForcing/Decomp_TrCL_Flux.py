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
print('read in data')

CLlw_stk = pk.load(open(Source+'/amip/Fluxes/'+models+'_TrCLlw_FLUX_Grid.pi','rb'))
CLsw_stk = pk.load(open(Source+'/amip/Fluxes/'+models+'_TrCLsw_FLUX_Grid.pi','rb'))
CL_stk = pk.load(open(Source+'/amip/Fluxes/'+models+'_TrCL_FLUX_Grid.pi','rb'))
tas_stk = pk.load(open(Source+'/Anomalies/'+models+'_Ajd_amip-piControl_TmSrs.pi','rb'))['tas']

tas_GAM = aa.LatLonavg_Time(np.mean(tas_stk,axis=1),lat,lon)
tas_GAM_rsh = tas_GAM.reshape(-1,1)

print('regress cloud flux perturbations against global annual mean surface temperature change this gives a slope for each gridpoint and each month')

start_time = time.time()

CL_flatten_adj = dict()
CL_flatten_fb = dict()

CLlw_flatten_adj = dict()
CLlw_flatten_fb = dict()

CLsw_flatten_adj = dict()
CLsw_flatten_fb = dict()

CL_adj = dict()
CL_fb = dict()

CLlw_adj = dict()
CLlw_fb = dict()

CLsw_adj = dict()
CLsw_fb = dict()

for kernels in CL_stk.keys():
    print('on kernel -',kernels)
    CL_flatten_adj[kernels] = [i for i in np.swapaxes(CL_stk[kernels][:,:,:].reshape(len(CL_stk[kernels][:,0,0]),64*128),0,1)]
    CL_flatten_fb[kernels] = [i for i in np.swapaxes(CL_stk[kernels][:,:,:].reshape(len(CL_stk[kernels][:,0,0]),64*128),0,1)]

    CLlw_flatten_adj[kernels] = [i for i in np.swapaxes(CLlw_stk[kernels][:,:,:].reshape(len(CLlw_stk[kernels][:,0,0]),64*128),0,1)]
    CLlw_flatten_fb[kernels] = [i for i in np.swapaxes(CLlw_stk[kernels][:,:,:].reshape(len(CLlw_stk[kernels][:,0,0]),64*128),0,1)]

    CLsw_flatten_adj[kernels] = [i for i in np.swapaxes(CLsw_stk[kernels][:,:,:].reshape(len(CLsw_stk[kernels][:,0,0]),64*128),0,1)]
    CLsw_flatten_fb[kernels] = [i for i in np.swapaxes(CLsw_stk[kernels][:,:,:].reshape(len(CLsw_stk[kernels][:,0,0]),64*128),0,1)]

    CL_adj[kernels] = Pools.starmap(LinReg_int_wrap,zip(CL_flatten_adj[kernels],repeat(tas_GAM_rsh[:,:])))
    CL_fb[kernels] = Pools.starmap(LinReg_alpha_wrap,zip(CL_flatten_fb[kernels],repeat(tas_GAM_rsh[:,:])))

    CLlw_adj[kernels] = Pools.starmap(LinReg_int_wrap,zip(CLlw_flatten_adj[kernels],repeat(tas_GAM_rsh[:,:])))
    CLlw_fb[kernels] = Pools.starmap(LinReg_alpha_wrap,zip(CLlw_flatten_fb[kernels],repeat(tas_GAM_rsh[:,:])))

    CLsw_adj[kernels] = Pools.starmap(LinReg_int_wrap,zip(CLsw_flatten_adj[kernels],repeat(tas_GAM_rsh[:,:])))
    CLsw_fb[kernels] = Pools.starmap(LinReg_alpha_wrap,zip(CLsw_flatten_fb[kernels],repeat(tas_GAM_rsh[:,:])))

end_time = time.time() - start_time
print(end_time/60, 'minutes for complete regressions to finish')

Pools.close()
Pools.join()

print('rebuild array shape')
CL_adj_rebuild = dict()
CL_fb_rebuild = dict()

CLlw_adj_rebuild = dict()
CLlw_fb_rebuild = dict()

CLsw_adj_rebuild = dict()
CLsw_fb_rebuild = dict()

for kernels in CL_stk.keys():
    CL_adj_rebuild[kernels] = np.stack(CL_adj[kernels][:],axis=0).reshape(64,128)
    CL_fb_rebuild[kernels] = np.stack(CL_fb[kernels][:],axis=0).reshape(64,128)

    CLlw_adj_rebuild[kernels] = np.stack(CLlw_adj[kernels][:],axis=0).reshape(64,128)
    CLlw_fb_rebuild[kernels] = np.stack(CLlw_fb[kernels][:],axis=0).reshape(64,128)

    CLsw_adj_rebuild[kernels] = np.stack(CLsw_adj[kernels][:],axis=0).reshape(64,128)
    CLsw_fb_rebuild[kernels] = np.stack(CLsw_fb[kernels][:],axis=0).reshape(64,128)

print('saving variables')

CL_adj_rebuild_file = open(models+'_TrCL_FLUX_ADJ_Grid.pi','wb')
pk.dump(CL_adj_rebuild,CL_adj_rebuild_file)
CL_adj_rebuild_file.close()

CL_fb_rebuild_file = open(models+'_TrCL_FLUX_FB_Grid.pi','wb')
pk.dump(CL_fb_rebuild,CL_fb_rebuild_file)
CL_fb_rebuild_file.close()

CLlw_adj_rebuild_file = open(models+'_TrCLlw_FLUX_ADJ_Grid.pi','wb')
pk.dump(CLlw_adj_rebuild,CLlw_adj_rebuild_file)
CLlw_adj_rebuild_file.close()

CLlw_fb_rebuild_file = open(models+'_TrCLlw_FLUX_FB_Grid.pi','wb')
pk.dump(CLlw_fb_rebuild,CLlw_fb_rebuild_file)
CLlw_fb_rebuild_file.close()

CLsw_adj_rebuild_file = open(models+'_TrCLsw_FLUX_ADJ_Grid.pi','wb')
pk.dump(CLsw_adj_rebuild,CLsw_adj_rebuild_file)
CLsw_adj_rebuild_file.close()

CLsw_fb_rebuild_file = open(models+'_TrCLsw_FLUX_FB_Grid.pi','wb')
pk.dump(CLsw_fb_rebuild,CLsw_fb_rebuild_file)
CLsw_fb_rebuild_file.close()

print('finished')
