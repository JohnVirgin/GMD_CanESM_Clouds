#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import packages
import numpy as np
import pickle as pk
import Jacks_Functions as jf
import netCDF4 as nc
import Area_Avg as aa
import pandas as pd
import sys


# In[2]:

models = sys.argv[1]

Source = '/home/jgvirgin/Projects/CanESM/Data/'+models+'/Anomalies/'

#define dimensions for horizontal grid
lat = np.linspace(-87.864,87.864,64)
lon = np.linspace(0,357.1875,128)


print('Read in data')

Alb = pk.load(open(Source+models+'_Ajd_4xCO2_TmSrs.pi','rb'))['Alb']*100
#Albcs = pk.load(open(Source+models+'_Ajd_4xCO2_TmSrs.pi','rb'))['Albcs']*100


print('read in kernels')

#load kernels
Kernels_interpolated = pk.load(open('/praid/users/jgvirgin/Radiative_Kernels/Interpolated_Kernels.pickle','rb'))

Kernel_interp_mask = dict()
for model, kernel in Kernels_interpolated.items():
    Kernel_interp_mask[model] = dict()
    for kernel, data in Kernels_interpolated[model].items():
        if kernel == 'Alb_TOA' or kernel == 'Alb_TOA_CLR':
            if Kernels_interpolated[model][kernel].ndim == 4:
                Kernel_interp_mask[model][kernel] = np.tile(np.transpose(\
                Kernels_interpolated[model][kernel][None,:,:,:,:],(0,1,2,3,4)),(150,1,1,1,1))
                Kernel_interp_mask[model][kernel][Kernel_interp_mask[model][kernel] > 1e5] = np.nan
            elif Kernels_interpolated[model][kernel].ndim == 3:
                Kernel_interp_mask[model][kernel] = np.tile(np.transpose(\
                Kernels_interpolated[model][kernel][None,:,:,:],(0,1,2,3)),(150,1,1,1))
                Kernel_interp_mask[model][kernel][Kernel_interp_mask[model][kernel] > 1e5] = np.nan
            else:
                pass
        else:
            pass

Kernel_interp_mask.pop('CloudSat')

Alb_fb_tot = dict()
Alb_fb_clr = dict()
for kernel in Kernel_interp_mask.keys():

    Alb_fb_tot[kernel] = np.nanmean(Alb*Kernel_interp_mask[kernel]['Alb_TOA'],axis=1)

    Alb_fb_clr[kernel] = np.nanmean(Alb*Kernel_interp_mask[kernel]['Alb_TOA_CLR'],axis=1)


print('saving')

Alb_fb_tot_file = open(models+'_TrALB_FLUX_Grid.pi','wb')
pk.dump(Alb_fb_tot,Alb_fb_tot_file)
Alb_fb_tot_file.close()

Alb_fb_clr_file = open(models+'_TrALB_FLUXCS_Grid.pi','wb')
pk.dump(Alb_fb_clr,Alb_fb_clr_file)
Alb_fb_clr_file.close()

print('finished')
