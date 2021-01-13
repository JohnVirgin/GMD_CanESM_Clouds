#!/usr/bin/env python
## Import Packages

import numpy as np
import pandas as pd
import os
import _pickle as pk
import netCDF4 as nc
import Jacks_Functions as jf
import glob
import scipy as sci
import sys
from natsort import natsorted

print('load in anomalies')

models = sys.argv[1]

Source = '/praid/users/jgvirgin/CanESM_Data/'+models

#define dimensions for horizontal grid
lat = np.linspace(-87.864, 87.864, 64)
lon = np.linspace(0, 357.1875, 128)

Anoms = pk.load(open(Source+'/Anomalies/'+models+'_Ajd_sstClim4xCO2.pi', 'rb'))

print('read in kernels')
#load kernels
Kernel_interp_mask = pk.load(open('/praid/users/jgvirgin/Radiative_Kernels/Interpolated_Kernels.pickle','rb'))

for model, kernel in Kernel_interp_mask.items():
    for kernel, data in Kernel_interp_mask[model].items():
        Kernel_interp_mask[model][kernel][Kernel_interp_mask[model][kernel] > 1e5] = np.nan

Kernel_interp_mask.pop('CloudSat')

print('Calculate surface albedo forcing...')

Alb_fb = dict()
for kernel in Kernel_interp_mask.keys():
    Alb_fb[kernel] = np.nanmean(Anoms['Alb']*Kernel_interp_mask[kernel]['Alb_TOA'],axis=0)

print('check the shape?', Alb_fb['CAM3'].shape)

print('calculate water vapour forcing...')

#logarathmic change in specific humidity with respect to temperature
dlnqdt = pk.load(open(Source+'/Anomalies/dlnqdt_'+models+'_sstClim.pi', 'rb'))
hus_base = pk.load(open(Source+'/Anomalies/'+models+'_hus_climo_sstClim_picon.pi', 'rb'))
dloghus = (Anoms['hus']/hus_base)/dlnqdt

print('create arrays for CMIP pressure levels and approximate tropopause height')

CMIP_File = nc.Dataset('/praid/users/jgvirgin/Radiative_Kernels/CAM3_Kernels.nc')
CMIP_plevs_scalar = np.squeeze(CMIP_File.variables['lev'])

CMIP_plevs = np.tile(CMIP_plevs_scalar[None,:,None,None],(12,1,lat.size,lon.size))

p_tropo_linear_NH = np.linspace(300,100,32)
p_tropo_linear_SH = np.linspace(100,300,32)
p_tropo_linear = np.concatenate((p_tropo_linear_NH,p_tropo_linear_SH))
p_tropopause = np.tile(p_tropo_linear[None,None,:,None],(12,17,1,lon.size))

print('separate into stratosphere and troposphere')

dloghus_strato = dloghus*(CMIP_plevs < p_tropopause)
dloghus_tropo = dloghus*(CMIP_plevs >= p_tropopause)

ps_path = '/praid/users/jgvirgin/CanESM_Data/CanESM2/Raw/vars_interp_Amon_CanESM2_piControl_r1i1p1_201501-241012.nc'
PS = np.squeeze(nc.Dataset(ps_path).variables['ps'])[:12,:,:]

#Calculate pressure level thickness using CMIP pressure levels and surface pressure
dp = np.zeros([12,17,64,128])

for i in range(12):
    for j in range(64):
        for k in range(128):
            dp[i,:,j,k] = jf.PlevThck( PS = PS[i,j,k]/100,plevs=CMIP_plevs_scalar,p_top = min(CMIP_plevs_scalar))

dp = dp/100 #Kernel units are per 100 hPa


print('calculate stratosphere & troposphere flux perturbations separately')
#multiply all water vapour perturbations by the proper kernels
    #integrate throughout the stratosphere & troposphere
    #take the annual mean

WV_fb_strato = dict()
WV_fb_tropo = dict()

for kernel in Kernel_interp_mask.keys():

    WV_fb_strato[kernel] = np.nanmean(np.nansum(dloghus_tropo*(Kernel_interp_mask[kernel]['WVlw_TOA']*dp),axis=1)+\
    np.nansum(dloghus_strato*Kernel_interp_mask[kernel]['WVsw_TOA']*dp,axis=1),axis=0)

    WV_fb_tropo[kernel] = np.nanmean(np.nansum(dloghus_tropo*(Kernel_interp_mask[kernel]['WVlw_TOA']*dp), axis=1) +
    np.nansum(dloghus_tropo*Kernel_interp_mask[kernel]['WVsw_TOA']*dp,axis=1),axis=0)


print('check shape? \n', WV_fb_tropo['CAM3'].shape)

print('Calculate temperature forcing...')


ta_tropo = Anoms['ta']*(CMIP_plevs>=p_tropopause)
ta_strato = Anoms['ta']*(CMIP_plevs<p_tropopause)

#multiply all nonlinear & linear strato temperature perturbations by the proper kernels
    #integrate throughout the stratosphere
    #take the annual mean

temp_fb_strato = dict()
temp_fb_tropo = dict()
for kernel in Kernel_interp_mask.keys():

    temp_fb_strato[kernel] = np.nanmean(np.nansum(ta_strato*Kernel_interp_mask[kernel]['Ta_TOA']*dp,axis=1),axis=0)

    temp_fb_tropo[kernel] = np.nanmean(np.nansum(\
        ta_tropo*Kernel_interp_mask[kernel]['Ta_TOA']*dp,axis=1)+Anoms['tas']*Kernel_interp_mask[kernel]['Ts_TOA'],axis=0)


print('check shape? \n',temp_fb_tropo['CAM3'].shape)

print('calculating cloud forcing...')
from scipy.interpolate import interp1d
k_source = '/praid/users/jgvirgin/Radiative_Kernels/Cloud/'

#load in cloud kernels
LWkernel = np.squeeze(nc.Dataset(k_source+'cloud_kernels2.nc').variables['LWkernel'])
SWkernel = np.squeeze(nc.Dataset(k_source+'cloud_kernels2.nc').variables['SWkernel'])
print(LWkernel.shape,'\n month, optical depth, cloud top pressure, latitude, clear sky surface albedo')

lons = np.arange(1.25, 360, 2.5)
lats = np.squeeze(nc.Dataset(k_source+'cloud_kernels2.nc').variables['lat'])
# the clear-sky albedos over which the kernel is computed
albcs = np.arange(0.0, 1.1, 0.5)

print('interpolate the kernels along their latitude dimension to match CanESM output')
#interpolate the kernels down to the CanESM2 resolution

LWkernel_func = interp1d(lats, LWkernel, axis=3, kind='nearest')
SWkernel_func = interp1d(lats, SWkernel, axis=3, kind='nearest')
LWkernel_interp = LWkernel_func(lat)
SWkernel_interp = SWkernel_func(lat)

# LW kernel does not depend on albcs, just repeat the final dimension over longitudes:
LWkernel_map = np.tile(LWkernel_interp[:,:,:,:,0,None],(1,1,1,1,128))

print('need baseline cloud fraction and clear sky albedos for forcing decomposition.... ')

if models == 'CanESM2':
    clouds_picon_file = natsorted(glob.glob(Source+'/Raw/clisccp_*'))[1]
    print(models, '- picon files - ', clouds_picon_file)
    clouds_picon_data = np.squeeze(nc.Dataset(clouds_picon_file).variables['clisccp'])[:360,:,:,:,:]
else:
    clouds_picon_file = natsorted(glob.glob(Source+'/Raw/clisccp_*'))[2]
    print(models, '- picon files - ', clouds_picon_file)
    clouds_picon_data = np.squeeze(nc.Dataset(clouds_picon_file).variables['clisccp'])[:360,:,:,:,:]

print('shape check for cloud fraction read in? \n', clouds_picon_data.shape)

print('stack and take a 30 year mean')

clouds_picon_stk = np.zeros([30,12,7,7,64,128])

s = 0
f = 12

for i in range(30):
    clouds_picon_stk[i,:,:,:,:,:] = np.stack(clouds_picon_data[s:f,:,:,:,:],axis=0)

    s+=12
    f+=12

clouds_picon_stk = np.mean(clouds_picon_stk,axis=0)
clouds_picon_stk[clouds_picon_stk > 1e5] = np.nan


print('Deocomposing cloud fraction anomalies and radiative kernels into contributions form amount, altitude, and optical depth feedbacks')
sect = ['Low', 'Hi', 'All']
sections_ind = [slice(0, 2), slice(2, 7), slice(0, 7)]
sections_length = [2, 5, 7]

c1_sum = dict()
dc_sum = dict()
dc_prop = dict()
dc_star = dict()
for i in range(len(sect)):

    #sum total cloud fraction and project it across all CTPs and optical depths
    c1_sum[sect[i]] = np.tile(np.nansum(clouds_picon_stk[:, :, sections_ind[i], :, :], axis=(
        1, 2))[:, None, None, :, :], (1, 7, sections_length[i], 1, 1))

    #some the change in cloud fractions over CTP and tau
    dc_sum[sect[i]] = np.tile(np.nansum(Anoms['clisccp'][:, :, sections_ind[i], :, :], axis=(
        1, 2))[:, None, None, :, :], (1, 7, sections_length[i], 1, 1))

    #change in cloud fraction due to proportional changes in clouds, but not due to CTP or optical depth
    dc_prop[sect[i]] = clouds_picon_stk[:, :, sections_ind[i], :, :]*(dc_sum[sect[i]]/c1_sum[sect[i]])

    #changes in optical depth and CTP, but with proportional changes in cloud fraction fixed
    dc_star[sect[i]] = Anoms['clisccp'][:, :, sections_ind[i], :, :] - dc_prop[sect[i]]


print('read in clear sky surface albedo output')
#read in clear sky shortwave fluxes at the surface for the same time period

if models == 'CanESM2':
    albcs_source = natsorted(glob.glob(Source+'/Raw/vars_interp_Amon_*'))[1]
else:
    albcs_source = natsorted(glob.glob(Source+'/Raw/vars_interp_Amon_*'))[2]

print('albedo source file in use: \n', albcs_source)

rsdscs1 = np.squeeze(nc.Dataset(albcs_source).variables['rsdscs'])[:360, :, :]
rsuscs1 = np.squeeze(nc.Dataset(albcs_source).variables['rsuscs'])[:360, :, :]
albcs1 = rsuscs1/rsdscs1

albcs1_stacked = np.zeros([30, 12, 64, 128])

s = 0
f = 12

for i in range(30):
    albcs1_stacked[i, :, :, :] = np.stack(albcs1[s:f, :, :], axis=0)
    s += 12
    f += 12

avgalbcs1 = np.mean(albcs1_stacked, axis=0)

print('Remapping the SW kernel to appropriate values based on control albedo meridional climatology')
# ## Remap Shortwave Cloud Kernel to clear sky surface albedo bins from Model output

SWkernel_map = np.zeros([12, 7, 7, 64, 128])
for m in range(12):  # loop through months
    for la in range(64):  # loop through longitudes

        # pluck out a zonal slice of clear sky surface albedo
        alb_lon = avgalbcs1[m, la, :]

        #remap the kernel onto the same grid as the model output
        function = interp1d(albcs, SWkernel_interp[m, :, :, la, :], axis=2, kind='linear')
        new_kernel_lon = function(alb_lon)
        SWkernel_map[m, :, :, la, :] = new_kernel_lon

sundown = np.nansum(SWkernel_map, axis=(1, 2))
#set the SW feedbacks to zero in the polar night
night = np.where(sundown == 0)
print('decompose the longwave kernel')

Klw0 = dict()
Klw_prime = dict()
lw_this = dict()
lw_that = dict()
Klw_p_prime = dict()
Klw_t_prime = dict()
Klw_resid_prime = dict()
for i in range(len(sect)):

    print('on section - ',sect[i])

    Klw0[sect[i]] = np.zeros([12, 7, sections_length[i], 64, 128])
    Klw_prime[sect[i]] = np.zeros([12, 7, sections_length[i], 64, 128])


    #sum the kernels CTP and optical depth bins and weight it by total cloud cover fraction
    #this not term is a kernel to estimate feedbacks from changes in cloud cover alone
    Klw0[sect[i]] = np.tile(np.nansum(
        LWkernel_map[:, :, sections_ind[i], :, :]*(clouds_picon_stk[:, :, sections_ind[i], :, :]/c1_sum[sect[i]]), axis=(1, 2))[:, None, None, :, :], (1, 7, sections_length[i], 1, 1))

    #prime represents a kernel to estimate feedbacks from changes in CTP and tau
    Klw_prime[sect[i]]= LWkernel_map[:, :,sections_ind[i], :, :] - Klw0[sect[i]]

    #we further decompose the CTP & tau kernel into separate kernels for both
    #CTP
    lw_this[sect[i]] = np.nansum(
        Klw_prime[sect[i]]*np.tile(np.nansum(clouds_picon_stk[:, :, sections_ind[i], :, :]/c1_sum[sect[i]], axis=2)[:, :, None, :, :], (1, 1, sections_length[i], 1, 1)), axis=1)

    #tau
    lw_that[sect[i]] = np.nansum(
        Klw_prime[sect[i]]*np.tile(np.nansum(clouds_picon_stk[:, :, sections_ind[i], :, :]/c1_sum[sect[i]], axis=1)[:, None, :, :, :], (1, 7, 1, 1, 1)), axis=2)

    Klw_p_prime[sect[i]] = np.tile(lw_this[sect[i]][:, None, :, :, :], (1, 7, 1, 1, 1))
    Klw_t_prime[sect[i]] = np.tile(lw_that[sect[i]][:, :, None, :, :], (1, 1, sections_length[i], 1, 1))

    #residual
    Klw_resid_prime[sect[i]] = Klw_prime[sect[i]] - Klw_p_prime[sect[i]] - Klw_t_prime[sect[i]]

print('longwave cloud forcings')

#now compute decomposed fluxes
dRlw_true = dict()
dRlw_prop = dict()
dRlw_dctp = dict()
dRlw_dtau = dict()
dRlw_resid = dict()
dRlw_sum = dict()
for i in range(len(sect)):

    # lw total
    dRlw_true[sect[i]] = np.nansum(LWkernel_map[:, :, sections_ind[i], :, :]*\
        Anoms['clisccp'][:, :, sections_ind[i], :, :], axis=(1, 2))

    # lw amount component
    dRlw_prop[sect[i]] = (Klw0[sect[i]][:, 0, 0, :, :]*dc_sum[sect[i]][:, 0, 0, :, :])
    # lw altitude component
    dRlw_dctp[sect[i]] = np.nansum(Klw_p_prime[sect[i]]*dc_star[sect[i]], axis=(1, 2))
    # lw optical depth component
    dRlw_dtau[sect[i]] = np.nansum(Klw_t_prime[sect[i]]*dc_star[sect[i]], axis=(1, 2))
    # lw residual
    dRlw_resid[sect[i]] = np.nansum(Klw_resid_prime[sect[i]]*dc_star[sect[i]], axis=(1, 2))
    # sum should equal true
    dRlw_sum[sect[i]] = dRlw_prop[sect[i]] + dRlw_dctp[sect[i]] + dRlw_dtau[sect[i]] + dRlw_resid[sect[i]]


print('longwave done, moving onto shortwave')
#decompose the sw kernel now

Ksw0 = dict()
Ksw_prime = dict()
sw_this = dict()
sw_that = dict()
Ksw_p_prime = dict()
Ksw_t_prime = dict()
Ksw_resid_prime = dict()
for i in range(len(sect)):

    Ksw0[sect[i]] = np.zeros([12, 7, sections_length[i], 64, 128])
    Ksw_prime[sect[i]] = np.zeros([12, 7, sections_length[i], 64, 128])


    #sum the kernels CTP and optical depth bins and weight it by total cloud cover fraction
    #this not term is a kernel to estimate feedbacks from changes in cloud cover alone
    Ksw0[sect[i]] = np.tile(np.nansum(
            SWkernel_map[:, :, sections_ind[i], :, :]*(clouds_picon_stk[:, :, sections_ind[i], :, :]/c1_sum[sect[i]]), axis=(1, 2))[:, None, None, :, :], (1, 7, sections_length[i], 1, 1))

    #prime represents a kernel to estimate feedbacks from changes in CTP and tau
    Ksw_prime[sect[i]] = SWkernel_map[:, :,sections_ind[i], :, :] - Ksw0[sect[i]]

    #we further decompose the CTP & tau kernel into separate kernels for both
    #CTP
    sw_this[sect[i]] = np.nansum(
        Ksw_prime[sect[i]]*np.tile(np.nansum(clouds_picon_stk[:, :, sections_ind[i], :, :]/c1_sum[sect[i]], axis=2)[:, :, None, :, :], (1, 1, sections_length[i], 1, 1)), axis=1)

    #tau
    sw_that[sect[i]] = np.nansum(
        Ksw_prime[sect[i]]*np.tile(np.nansum(clouds_picon_stk[:, :, sections_ind[i], :, :]/c1_sum[sect[i]], axis=1)[:, None, :, :, :], (1, 7, 1, 1, 1)), axis=2)

    Ksw_p_prime[sect[i]] = np.tile(sw_this[sect[i]][:, None, :, :, :], (1, 7, 1, 1, 1))
    Ksw_t_prime[sect[i]] = np.tile(sw_that[sect[i]][:, :, None, :, :], (1, 1, sections_length[i], 1, 1))

    #residual
    Ksw_resid_prime[sect[i]] = Ksw_prime[sect[i]] - Ksw_p_prime[sect[i]] - Ksw_t_prime[sect[i]]


print('shortwave feedbacks...')
#now compute decomposed fluxes
dRsw_true = dict()
dRsw_prop = dict()
dRsw_dctp = dict()
dRsw_dtau = dict()
dRsw_resid = dict()
dRsw_sum = dict()
for i in range(len(sect)):

    # sw total
    dRsw_true[sect[i]] = np.nansum(SWkernel_map[:, :, sections_ind[i], :, :]*\
        Anoms['clisccp'][:, :, sections_ind[i], :, :], axis=(1, 2))

    # sw amount component
    dRsw_prop[sect[i]] = (Ksw0[sect[i]][:, 0, 0, :, :]*dc_sum[sect[i]][:, 0, 0, :, :])
    # sw altitude component
    dRsw_dctp[sect[i]] = np.nansum(Ksw_p_prime[sect[i]]*dc_star[sect[i]], axis=(1, 2))
    # sw optical depth component
    dRsw_dtau[sect[i]] = np.nansum(Ksw_t_prime[sect[i]]*dc_star[sect[i]], axis=(1, 2))
    # sw residual
    dRsw_resid[sect[i]] = np.nansum(Ksw_resid_prime[sect[i]]*dc_star[sect[i]], axis=(1, 2))
    # sum should equal true
    dRsw_sum[sect[i]] = dRsw_prop[sect[i]] + dRsw_dctp[sect[i]] + dRsw_dtau[sect[i]] + dRsw_resid[sect[i]]

    dRsw_true[sect[i]][night]=0
    dRsw_prop[sect[i]][night]=0
    dRsw_dctp[sect[i]][night]=0
    dRsw_dtau[sect[i]][night]=0
    dRsw_resid[sect[i]][night]=0
    dRsw_sum[sect[i]][night]=0


print('check final shape? \n',dRsw_sum['All'].shape)

#first, create another dictionary for storing\
SW_fluxes = dict()
SW_fluxes['Standard'] = dRsw_true
SW_fluxes['Amount'] = dRsw_prop
SW_fluxes['Altitude'] = dRsw_dctp
SW_fluxes['Optical Depth'] = dRsw_dtau
SW_fluxes['Residual'] = dRsw_resid
SW_fluxes['Sum'] = dRsw_sum

LW_fluxes = dict()
LW_fluxes['Standard'] = dRlw_true
LW_fluxes['Amount'] = dRlw_prop
LW_fluxes['Altitude'] = dRlw_dctp
LW_fluxes['Optical Depth'] = dRlw_dtau
LW_fluxes['Residual'] = dRlw_resid
LW_fluxes['Sum'] = dRlw_sum

SW_fluxes_ann = dict()
LW_fluxes_ann = dict()
for decomp in SW_fluxes.keys():
    SW_fluxes_ann[decomp] = dict()
    LW_fluxes_ann[decomp] = dict()
    for sections in SW_fluxes[decomp].keys():
        SW_fluxes_ann[decomp][sections] = np.nanmean(SW_fluxes[decomp][sections],axis=0)
        LW_fluxes_ann[decomp][sections] = np.nanmean(LW_fluxes[decomp][sections],axis=0)

print('check cloud flux shape one more time - \n',SW_fluxes_ann['Sum']['All'].shape)

print('lastly, calculate CO2 forcing as a residual of adjustments and total ERF...')

CO2 = dict()
for kernel in Kernel_interp_mask.keys():
    CO2[kernel] = np.nanmean(Anoms['fnet'],axis=0)-\
        (Alb_fb[kernel]+WV_fb_strato[kernel]+WV_fb_tropo[kernel]+temp_fb_strato[kernel]+temp_fb_tropo[kernel]+
            SW_fluxes_ann['Standard']['All']+LW_fluxes_ann['Standard']['All'])

print('checking CO2 forcing shape - \n',CO2['CAM3'].shape)

Rap_adj = dict()
Rap_adj['Alb'] = Alb_fb
Rap_adj['Temp_S'] = temp_fb_strato
Rap_adj['Temp_T'] = temp_fb_tropo
Rap_adj['WV_S'] = WV_fb_strato
Rap_adj['WV_T'] = WV_fb_tropo
Rap_adj['CO2'] = CO2
Rap_adj['CLD_SW'] = SW_fluxes_ann
Rap_adj['CLD_LW'] = LW_fluxes_ann
Rap_adj['ERFh'] = np.nanmean(Anoms['fnet'],axis=0)

print('saving...')

save_file = open(models+'_ERFh_Decomp.pi','wb')
pk.dump(Rap_adj,save_file)
save_file.close()

print('done!')
