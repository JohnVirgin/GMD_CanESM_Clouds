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
import glob


# In[2]:

models = sys.argv[1]

Source = '/home/jgvirgin/Projects/CanESM/Data/'+models+'/Anomalies/'

#define dimensions for horizontal grid
lat = np.linspace(-87.864,87.864,64)
lon = np.linspace(0,357.1875,128)


# In[3]:
print('read in data')

ta = pk.load(open(Source+models+'_Ajd_amip-piControl_TmSrs.pi', 'rb'))['ta']
tas = pk.load(open(Source+models+'_Ajd_amip-piControl_TmSrs.pi', 'rb'))['tas']


# In[4]:


print('create isothermal atmosphere temperature change')
ta_iso = np.tile(np.transpose(tas[:,:,None,:,:],(0,1,2,3,4)),(1,1,17,1,1))


# In[5]:
print('create arrays for CMIP pressure levels and approximate tropopause height')

CMIP_File = nc.Dataset('/praid/users/jgvirgin/Radiative_Kernels/CAM3_Kernels.nc')
CMIP_plevs_scalar = np.squeeze(CMIP_File.variables['lev'])

CMIP_plevs = np.tile(CMIP_plevs_scalar[None,None,:,None,None],(145,12,1,lat.size,lon.size))

p_tropo_linear_NH = np.linspace(300,100,32)
p_tropo_linear_SH = np.linspace(100,300,32)
p_tropo_linear = np.concatenate((p_tropo_linear_NH,p_tropo_linear_SH))
p_tropopause = np.tile(p_tropo_linear[None,None,None,:,None],(145,12,17,1,lon.size))


# In[7]:


print('Calculate departure of temperature change from an isothermal atmos change + mask stratosphere')
#create air temperature metrics for the troposphere and stratosphere separately
#for the troposphere, separate temperature perturbations in to planck and lapse rate
    #iso and departure

ta_dep_tropo = (ta-ta_iso)*(CMIP_plevs>=p_tropopause)
ta_iso_tropo = ta_iso*(CMIP_plevs>=p_tropopause)

# In[8]:


print('read in kernels')
kernel_models = ['CAM3','CAM5','CloudSat','GFDL','ECHAM6_ctr','HadGEM2','ERA']
ind_kernels = ['Ta_TOA','Ta_TOA_CLR','Ts_TOA','Ts_TOA_CLR']

Kernel_interp_mask = dict()
for m in range(len(kernel_models)):
    Kernel_interp_mask[kernel_models[m]] = dict()
    for k in range(len(ind_kernels)):
        Kernel_interp_mask[kernel_models[m]][ind_kernels[k]] = pk.load(open(\
        '/praid/users/jgvirgin/Radiative_Kernels/Interpolated_Kernels.pickle','rb'))[kernel_models[m]][ind_kernels[k]]
        Kernel_interp_mask[kernel_models[m]][ind_kernels[k]][Kernel_interp_mask[kernel_models[m]][ind_kernels[k]] > 1e5] = np.nan


Kernel_interp_mask.pop('CloudSat')

# In[9]:

ps_path = '/praid/users/jgvirgin/CanESM_Data/CanESM2/Raw/vars_interp_Amon_CanESM2_piControl_r1i1p1_201501-241012.nc'
PS = np.squeeze(nc.Dataset(ps_path).variables['ps'])[:12, :, :]

#Calculate pressure level thickness using CMIP pressure levels and surface pressure
dp = np.zeros([12,17,64,128])

for i in range(12):
    for j in range(64):
        for k in range(128):
            dp[i,:,j,k] = jf.PlevThck(PS = PS[i,j,k]/100,plevs=CMIP_plevs_scalar,p_top = min(CMIP_plevs_scalar))

dp = dp/100 #Kernel units are per 100 hPa


# In[11]:

print('calculate lapse rate & planck related fluxes')

lpse_fb_tot = dict()
lpse_fb_clr = dict()

plnck_fb_tot = dict()
plnck_fb_clr = dict()

for kernel in Kernel_interp_mask.keys():

    plnck_fb_tot[kernel] = np.zeros([145,64,128])
    plnck_fb_clr[kernel] = np.zeros([145,64,128])

    lpse_fb_tot[kernel] = np.zeros([145,64,128])
    lpse_fb_clr[kernel] = np.zeros([145,64,128])

    for i in range(145):

        lpse_fb_tot[kernel][i,:,:] = np.nanmean(np.nansum(\
        ta_dep_tropo[i,:,:,:,:]*Kernel_interp_mask[kernel]['Ta_TOA']*dp,axis=1),axis=0)

        lpse_fb_clr[kernel][i,:,:] = np.nanmean(np.nansum(\
        ta_dep_tropo[i,:,:,:,:]*Kernel_interp_mask[kernel]['Ta_TOA_CLR']*dp,axis=1),axis=0)

        plnck_fb_tot[kernel][i,:,:] = np.nanmean(np.nansum(\
        ta_iso_tropo[i,:,:,:,:]*Kernel_interp_mask[kernel]['Ta_TOA']*dp,axis=1)+\
        tas[i,:,:,:]*Kernel_interp_mask[kernel]['Ts_TOA'],axis=0)

        plnck_fb_clr[kernel][i,:,:] = np.nanmean(np.nansum(\
        ta_iso_tropo[i,:,:,:,:]*Kernel_interp_mask[kernel]['Ta_TOA_CLR']*dp,axis=1)+\
        tas[i,:,:,:]*Kernel_interp_mask[kernel]['Ts_TOA_CLR'],axis=0)

print('saving')

#save lapse vars
lpse_fb_tot_file = open(models+'_LAPSE_FLUX_Grid.pi','wb')
pk.dump(lpse_fb_tot,lpse_fb_tot_file)
lpse_fb_tot_file.close()

lpse_fb_clr_file = open(models+'_LAPSE_FLUXCS_Grid.pi','wb')
pk.dump(lpse_fb_clr,lpse_fb_clr_file)
lpse_fb_clr_file.close()

plnck_fb_tot_file = open(models+'_PLANCK_FLUX_Grid.pi','wb')
pk.dump(plnck_fb_tot,plnck_fb_tot_file)
plnck_fb_tot_file.close()

plnck_fb_clr_file = open(models+'_PLANCK_FLUXCS_Grid.pi','wb')
pk.dump(plnck_fb_clr,plnck_fb_clr_file)
plnck_fb_clr_file.close()

print('finished')
# In[ ]:
