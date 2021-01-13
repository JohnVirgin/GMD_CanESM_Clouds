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

Alb_stk = pk.load(open(Source+'/Fluxes/'+models+'_TrALB_FLUX_Grid.pi','rb'))

tas_stk = pk.load(open(Source+'/Anomalies/'+models+'_Ajd_4xCO2_TmSrs.pi','rb'))['tas']
tas_GAM = aa.LatLonavg_Time(np.mean(tas_stk,axis=1),lat,lon)
tas_GAM_rsh = tas_GAM.reshape(-1,1)

print('regress surface albedo flux perturbations against global annual mean surface temperature change this gives a slope for each gridpoint and each month')

start_time = time.time()

Alb_flatten_adj = dict()
Alb_flatten_fb = dict()

Alb_adj = dict()
Alb_fb = dict()
for kernels in Alb_stk.keys():
    print('on kernel -',kernels)
    Alb_flatten_adj[kernels] = [i for i in np.swapaxes(Alb_stk[kernels][:,:,:].reshape(len(Alb_stk[kernels][:,0,0]),64*128),0,1)]
    Alb_flatten_fb[kernels] = [i for i in np.swapaxes(Alb_stk[kernels][:,:,:].reshape(len(Alb_stk[kernels][:,0,0]),64*128),0,1)]

    Alb_adj[kernels] = Pools.starmap(LinReg_int_wrap,zip(Alb_flatten_adj[kernels],repeat(tas_GAM_rsh[:,:])))
    Alb_fb[kernels] = Pools.starmap(LinReg_alpha_wrap,zip(Alb_flatten_fb[kernels],repeat(tas_GAM_rsh[:,:])))

end_time = time.time() - start_time
print(end_time/60, 'minutes for complete regressions to finish')

Pools.close()
Pools.join()

print('rebuild array shape')
Alb_adj_rebuild = dict()
Alb_fb_rebuild = dict()
for kernels in Alb_stk.keys():
    Alb_adj_rebuild[kernels] = np.stack(Alb_adj[kernels][:],axis=0).reshape(64,128)
    Alb_fb_rebuild[kernels] = np.stack(Alb_fb[kernels][:],axis=0).reshape(64,128)

print('saving variables')

Alb_adj_rebuild_file = open(models+'_TrALB_FLUX_ADJ_Grid.pi','wb')
pk.dump(Alb_adj_rebuild,Alb_adj_rebuild_file)
Alb_adj_rebuild_file.close()

Alb_fb_rebuild_file = open(models+'_TrALB_FLUX_FB_Grid.pi','wb')
pk.dump(Alb_fb_rebuild,Alb_fb_rebuild_file)
Alb_fb_rebuild_file.close()

print('finished')
