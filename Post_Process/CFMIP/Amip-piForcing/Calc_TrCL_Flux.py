#!/usr/bin/env python
# coding: utf-8

#import packages
import numpy as np
import pickle as pk
import netCDF4 as nc
import pandas as pd
import gc
import glob
import sys

models = sys.argv[1]

Source = '/mnt/data/users/jgvirgin/GMD_CanESM_p1/CanESM5_p2/CFMIP/Anomalies/'

def PlevThck(PS='PS', plevs='plevs',p_top='p_top'):

    if min(plevs) == plevs[0]:
        plevs = plevs
    else:
        plevs = plevs[::-1]

    Upper = 0
    Mid = 1
    Lower = 2

    dp = np.empty([plevs.size])
    dp[:] = np.nan

    def lower_boundary():
        for i,value in enumerate(plevs):
            if value > PS:
                    return i

        return plevs.size

    for i in range(lower_boundary()):

        if i == 0:
            dp[i] = -(p_top-(((plevs[1]+plevs[2])/2)-plevs[0]))

        elif Lower < lower_boundary():
            dp[i] = ((plevs[Mid]+plevs[Lower])/2)-((plevs[Upper]+plevs[Mid])/2)

            Upper += 1
            Mid += 1
            Lower += 1

        else:
            dp[i] = PS-(((plevs[lower_boundary()-1]+plevs[lower_boundary()-2])/2))


    return dp

#define dimensions for horizontal grid
lat = np.linspace(-87.864,87.864,64)
lon = np.linspace(0,357.1875,128)

print('read in data')

fb_var = ['hus','ta','tas','Alb']

var = dict()
for j in range(len(fb_var)):
    var[fb_var[j]] = pk.load(open(Source+models+'_Ajd_amip-piControl_TmSrs.pi','rb'))[fb_var[j]]
    if fb_var[j] == 'Alb':
        var[fb_var[j]] = var[fb_var[j]]*100
    else:
        pass


print('Calculating logarathmic change in specific humidity')

#logarathmic change in specific humidity with respect to temperature
dlnqdt = pk.load(open(Source+'dlnqdt_amip_'+models+'.pi','rb'))
hus_base = pk.load(open(Source+models+'_hus_climo_amip.pi','rb'))

var['dloghus'] = (var['hus']/hus_base)/dlnqdt
var.pop('hus')


print('check new keys after log q calculation')
print(var.keys())

print('create pressure based CMIP levs and tropopause height arrays')
CMIP_plevs_scalar = np.asarray([  10.,   20.,   30.,   50.,   70.,  100.,  150.,  200.,  250.,\
        300.,  400.,  500.,  600.,  700.,  850.,  925., 1000.])

CMIP_plevs = np.tile(CMIP_plevs_scalar[None,None,:,None,None],(145,12,1,lat.size,lon.size))

p_tropo_linear_NH = np.linspace(300,100,32)
p_tropo_linear_SH = np.linspace(100,300,32)
p_tropo_linear = np.concatenate((p_tropo_linear_NH,p_tropo_linear_SH))
p_tropopause = np.tile(p_tropo_linear[None,None,None,:,None],(145,12,17,1,lon.size))

gc.collect()

print('mask the stratosphere for 4 dimensional variables')

hus_tropo = (var['dloghus'])*(CMIP_plevs>=p_tropopause)
ta_tropo = var['ta']*(CMIP_plevs>=p_tropopause)


print('read in kernels')

#load kernels
kernels = ['CAM3','CAM5','GFDL','ECHAM6_ctr','ERA','HadGEM2']
data = ['Alb_TOA','Alb_TOA_CLR','Ts_TOA','Ts_TOA_CLR','Ta_TOA','Ta_TOA_CLR','WVlw_TOA','WVlw_TOA_CLR','WVsw_TOA','WVsw_TOA_CLR']

Kernel_interp_mask = dict()
for k in range(len(kernels)):
    Kernel_interp_mask[kernels[k]] = dict()
    for d in range(len(data)):
        Kernel_interp_mask[kernels[k]][data[d]] = pk.load(open('/mnt/data/users/jgvirgin/Kernels/Interpolated_Kernels.pickle','rb'))[kernels[k]][data[d]]
        Kernel_interp_mask[kernels[k]][data[d]][Kernel_interp_mask[kernels[k]][data[d]]>1e5] = np.nan

# In[11]:

ps_path = '/mnt/data/users/jgvirgin/GMD_CanESM_p1/CanESM5_p2/CMIP/raw/Amon/vars_interp_Amon_CanESM5_piControl_r1i1p2f1_gn_555001-580012.nc'
PS = np.squeeze(nc.Dataset(ps_path).variables['ps'])[:12, :, :]

#Calculate pressure level thickness using CMIP pressure levels and surface pressure
dp = np.zeros([12,17,64,128])

for i in range(12):
    for j in range(64):
        for k in range(128):
            dp[i,:,j,k] = PlevThck( PS = PS[i,j,k]/100,plevs=CMIP_plevs_scalar,p_top = min(CMIP_plevs_scalar))

dp = dp/100 #Kernel units are per 100 hPa

print('calculate corrections')
alb_corr = dict()
ta_corr = dict()
tas_corr = dict()
hus_lw_corr = dict()
hus_sw_corr = dict()
for kernel in Kernel_interp_mask.keys():
    alb_corr[kernel] = np.zeros([145,64,128])
    ta_corr[kernel] = np.zeros([145,64,128])
    tas_corr[kernel] = np.zeros([145,64,128])
    hus_lw_corr[kernel] = np.zeros([145,64,128])
    hus_sw_corr[kernel] = np.zeros([145,64,128])
    for i in range(145):
        alb_corr[kernel][i,:,:] = np.nanmean((\
        Kernel_interp_mask[kernel]['Alb_TOA_CLR']-Kernel_interp_mask[kernel]['Alb_TOA'])\
        *(var['Alb'][i,:,:,:]),axis=0)

        tas_corr[kernel][i,:,:] = np.nanmean((\
        Kernel_interp_mask[kernel]['Ts_TOA_CLR']-Kernel_interp_mask[kernel]['Ts_TOA'])\
        *(var['tas'][i,:,:,:]),axis=0)

        ta_corr[kernel][i,:,:] = np.nanmean(np.nansum(\
        ta_tropo[i,:,:,:,:]*(Kernel_interp_mask[kernel]['Ta_TOA_CLR']-Kernel_interp_mask[kernel]['Ta_TOA'])*dp,axis=1),axis=0)

        hus_lw_corr[kernel][i,:,:] = np.nanmean(np.nansum(\
        hus_tropo[i,:,:,:,:]*(Kernel_interp_mask[kernel]['WVlw_TOA_CLR']-Kernel_interp_mask[kernel]['WVlw_TOA'])*dp,axis=1),axis=0)

        hus_sw_corr[kernel][i,:,:] = np.nanmean(np.nansum(\
        hus_tropo[i,:,:,:,:]*(Kernel_interp_mask[kernel]['WVsw_TOA_CLR']-Kernel_interp_mask[kernel]['WVsw_TOA'])*dp,axis=1),axis=0)

gc.collect()

print('Adjust the CRE by the feedback and forcing corrections')

CRE_lw = pk.load(open('/mnt/data/users/jgvirgin/GMD_CanESM_p1/CanESM5_p2/CFMIP/Fluxes/CanESM5_p2_TrCRElw_FLUX_Grid.pi','rb'))
CRE_sw = pk.load(open('/mnt/data/users/jgvirgin/GMD_CanESM_p1/CanESM5_p2/CFMIP/Fluxes/CanESM5_p2_TrCREsw_FLUX_Grid.pi','rb'))

Cloud_lw = dict()
Cloud_sw = dict()
Cloud = dict()
for kernel in Kernel_interp_mask.keys():
    Cloud_lw[kernel] = CRE_lw+ta_corr[kernel]+tas_corr[kernel]+hus_lw_corr[kernel]
    Cloud_sw[kernel] = CRE_sw+alb_corr[kernel]+hus_sw_corr[kernel]

    Cloud[kernel] = Cloud_sw[kernel]+Cloud_lw[kernel]

print('Save variable')

Cloud_file = open(models+'_TrCL_FLUX_Grid.pi','wb')
pk.dump(Cloud,Cloud_file)
Cloud_file.close()

Cloud_lw_file = open(models+'_TrCLlw_FLUX_Grid.pi','wb')
pk.dump(Cloud_lw,Cloud_lw_file)
Cloud_lw_file.close()

Cloud_sw_file = open(models+'_TrCLsw_FLUX_Grid.pi','wb')
pk.dump(Cloud_sw,Cloud_sw_file)
Cloud_sw_file.close()

print('done')
