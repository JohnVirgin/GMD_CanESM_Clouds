#!/usr/bin/env python
# coding: utf-8

# In[3]:


#import packages
import numpy as np
import pickle as pk
import Jacks_Functions as jf
import netCDF4 as nc
import Area_Avg as aa
import pandas as pd
import sys
import glob


models = sys.argv[1]

Source = '/home/jgvirgin/Projects/CanESM/Data/'+models+'/Anomalies/'

#define dimensions for horizontal grid
lat = np.linspace(-87.864,87.864,64)
lon = np.linspace(0,357.1875,128)


# In[11]:
print('read in data')

#logarathmic change in specific humidity with respect to temperature
dlnqdt = pk.load(open(Source+'dlnqdt_'+models+'.pi','rb'))
hus_base = pk.load(open(Source+models+'_hus_climo_picon.pi','rb'))
dhus = pk.load(open(Source+models+'_Ajd_4xCO2_TmSrs.pi','rb'))['hus']
dloghus = (dhus/hus_base)/dlnqdt

# In[24]:
print('create arrays for CMIP pressure levels and approximate tropopause height')

CMIP_File = nc.Dataset('/praid/users/jgvirgin/Radiative_Kernels/CAM3_Kernels.nc')
CMIP_plevs_scalar = np.squeeze(CMIP_File.variables['lev'])

CMIP_plevs = np.tile(CMIP_plevs_scalar[None,None,:,None,None],(150,12,1,lat.size,lon.size))

p_tropo_linear_NH = np.linspace(300,100,32)
p_tropo_linear_SH = np.linspace(100,300,32)
p_tropo_linear = np.concatenate((p_tropo_linear_NH,p_tropo_linear_SH))
p_tropopause = np.tile(p_tropo_linear[None,None,None,:,None],(150,12,17,1,lon.size))

# In[26]:


print('separate into stratosphere and troposphere')

dloghus_strato = dloghus*(CMIP_plevs<p_tropopause)
dloghus_tropo = dloghus*(CMIP_plevs>=p_tropopause)

# In[28]:


print('read in kernels')

#load kernels
Kernels_interpolated = pk.load(open('/praid/users/jgvirgin/Radiative_Kernels/Interpolated_Kernels.pickle','rb'))

for model, kernel in Kernels_interpolated.items():
    for kernel, data in Kernels_interpolated[model].items():
        Kernels_interpolated[model][kernel][Kernels_interpolated[model][kernel] > 1e5] = np.nan

Kernels_interpolated.pop('CloudSat')

# In[11]:

ps_path = '/praid/users/jgvirgin/CanESM_Data/CanESM2/Raw/vars_interp_Amon_CanESM2_piControl_r1i1p1_201501-241012.nc'
PS = np.squeeze(nc.Dataset(ps_path).variables['ps'])[:12,:,:]

#Calculate pressure level thickness using CMIP pressure levels and surface pressure
dp = np.zeros([12,17,64,128])

for i in range(12):
    for j in range(64):
        for k in range(128):
            dp[i,:,j,k] = jf.PlevThck( PS = PS[i,j,k]/100,plevs=CMIP_plevs_scalar,p_top = min(CMIP_plevs_scalar))

dp = dp/100 #Kernel units are per 100 hPa


# In[36]:


print('calculate stratosphere & troposphere flux perturbations separately')
#multiply all water vapour perturbations by the proper kernels
    #integrate throughout the stratosphere & troposphere
    #take the annual mean

WV_fb_strato_tot = dict()
WV_fb_strato_clr = dict()

WV_fb_tropo_tot = dict()
WV_fb_tropo_clr = dict()

for kernel in Kernels_interpolated.keys():

    WV_fb_strato_tot[kernel] = np.zeros([150,64,128])
    WV_fb_strato_clr[kernel] = np.zeros([150,64,128])

    WV_fb_tropo_tot[kernel] = np.zeros([150,64,128])
    WV_fb_tropo_clr[kernel] = np.zeros([150,64,128])

    for i in range(150):

        WV_fb_strato_tot[kernel][i,:,:] = np.nanmean(np.nansum(dloghus_strato[i,:,:,:,:]*(Kernels_interpolated[kernel]['WVlw_TOA']*dp),axis=1)+\
        np.nansum(dloghus_strato[i,:,:,:,:]*Kernels_interpolated[kernel]['WVsw_TOA']*dp,axis=1),axis=0)

        WV_fb_strato_clr[kernel][i,:,:] = np.nanmean(np.nansum(dloghus_strato[i,:,:,:,:]*(Kernels_interpolated[kernel]['WVlw_TOA_CLR']*dp),axis=1)+\
        np.nansum(dloghus_strato[i,:,:,:,:]*Kernels_interpolated[kernel]['WVsw_TOA_CLR']*dp,axis=1),axis=0)

        WV_fb_tropo_tot[kernel][i,:,:] = np.nanmean(np.nansum(dloghus_tropo[i,:,:,:,:]*(Kernels_interpolated[kernel]['WVlw_TOA']*dp),axis=1)+\
        np.nansum(dloghus_tropo[i,:,:,:,:]*Kernels_interpolated[kernel]['WVsw_TOA']*dp,axis=1),axis=0)

        WV_fb_tropo_clr[kernel][i,:,:] = np.nanmean(np.nansum(dloghus_tropo[i,:,:,:,:]*(Kernels_interpolated[kernel]['WVlw_TOA_CLR']*dp),axis=1)+\
        np.nansum(dloghus_tropo[i,:,:,:,:]*Kernels_interpolated[kernel]['WVsw_TOA_CLR']*dp,axis=1),axis=0)



print('save variables')

#save strato vars
WV_fb_strato_tot_file = open(models+'_SrWV_FLUX_Grid.pi','wb')
pk.dump(WV_fb_strato_tot,WV_fb_strato_tot_file)
WV_fb_strato_tot_file.close()

WV_fb_tropo_tot_file = open(models+'_TrWV_FLUX_Grid.pi','wb')
pk.dump(WV_fb_tropo_tot,WV_fb_tropo_tot_file)
WV_fb_tropo_tot_file.close()

WV_fb_strato_clr_file = open(models+'_SrWV_FLUXCS_Grid.pi','wb')
pk.dump(WV_fb_strato_clr,WV_fb_strato_clr_file)
WV_fb_strato_clr_file.close()

WV_fb_tropo_clr_file = open(models+'_TrWV_FLUXCS_Grid.pi','wb')
pk.dump(WV_fb_tropo_clr,WV_fb_tropo_clr_file)
WV_fb_tropo_clr_file.close()

print('finished')
