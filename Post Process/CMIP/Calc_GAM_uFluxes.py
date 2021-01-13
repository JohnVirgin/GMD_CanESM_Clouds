# In[1]:


#import packages
import numpy as np
import _pickle as pk
import netCDF4 as nc
import pandas as pd
from sklearn.linear_model import LinearRegression
import multiprocessing as mp
import logging
from multiprocessing_logging import install_mp_handler
import time
from itertools import repeat
import sys

def LatLonavg(var='var', lat='lat', lon='lon'):
    """Returns the area-averged variable for a certain lat-lon region"""
    y = lat*np.pi/180
    coslat = np.cos(y)
    coslat = np.tile(coslat,(lon.size,1)).T

    #area weighting (weight by cos(lat) because area of grid boxes get smaller closer to the pole)
    lat_tmp_cos = np.ma.zeros([lat.size, lon.size])
    coslat_tmp = np.ma.zeros([lat.size, lon.size])
    for i in range(lat.size):
        for j in range(lon.size):
            try:
                if var.mask[i,j] == False:
                    lat_tmp_cos[i,j] = var[i,j] * coslat[i,j]
                    coslat_tmp[i,j] = coslat[i,j]
                else:
                    lat_tmp_cos[i,j] = float('nan')
                    coslat_tmp[i,j] = float('nan')
            except:
                lat_tmp_cos[i,j] = var[i,j] * coslat[i,j]
                coslat_tmp[i,j] = coslat[i,j]
            else:
                pass

    #sum over all gridpoints
    cos_sum = np.nansum(coslat_tmp, axis=(0,1))

    lat_sum = np.nansum(lat_tmp_cos, axis=(0,1))

    #normalize by area
    LatLonavg = lat_sum / cos_sum

    return LatLonavg

print('Setting up worker pools using',mp.cpu_count(),'cpu cores')
cpus = mp.cpu_count()
Pools = mp.Pool(cpus)

#define dimensions for horizontal grid
lat = np.linspace(-87.864,87.864,64)
lon = np.linspace(0,357.1875,128)

models = sys.argv[1]

Source = '/home/jgvirgin/Projects/CanESM/Data/'+models+'/Fluxes/'


Vari = ['DIR_FLUXCS','DIR_FLUX','TrCL_FLUX','TrCLlw_FLUX','TrCLsw_FLUX','LAPSE_FLUX','PLANCK_FLUX','SrTEMP_FLUX','SrWV_FLUX','TrWV_FLUX','TrALB_FLUX',\
        'LAPSE_FLUXCS','PLANCK_FLUXCS','SrTEMP_FLUXCS','SrWV_FLUXCS','TrWV_FLUXCS','TrALB_FLUXCS']     

Kernels = ['CAM3','CAM5','ECHAM6_ctr','ERA','GFDL','HadGEM2']

print('read in data')

start_time = time.time()
Variables = dict()
for v in range(len(Vari)):
    Variables[Vari[v]] = dict()
    for k in range(len(Kernels)):
        Variables[Vari[v]][Kernels[k]] = pk.load(open(\
        Source+models+'_'+Vari[v]+'_Grid.pi','rb'))[Kernels[k]]

end_time = time.time()-start_time
print('time to read in =',end_time/60,'minutes')


start_time = time.time()
print('area averaging')
Variables_AA = dict()
Variables_Flatten = dict()
for v in range(len(Vari)):
    Variables_AA[Vari[v]] = dict()
    Variables_Flatten[Vari[v]] = dict()
    print('on variable',Vari[v])
    for k in range(len(Kernels)):
        Variables_Flatten[Vari[v]][Kernels[k]] = [i for i in Variables[Vari[v]][Kernels[k]]]

        Variables_AA[Vari[v]][Kernels[k]] = Pools.starmap(LatLonavg,
        zip(Variables_Flatten[Vari[v]][Kernels[k]],repeat(lat),repeat(lon)))

end_time = time.time()-start_time
print('that took',end_time/60,'minutes')

Pools.close()
Pools.join()

print('rebuild variables')
Variables_Rebuild = dict()
for v in range(len(Vari)):
    Variables_Rebuild[Vari[v]] = dict()
    for k in range(len(Kernels)):
        Variables_Rebuild[Vari[v]][Kernels[k]] = np.asarray(Variables_AA[Vari[v]][Kernels[k]])

print('Save variables')
Variables_file = open(models+'_ALL_uFLUXES_GAM.pi','wb')
pk.dump(Variables_Rebuild,Variables_file)
Variables_file.close()
