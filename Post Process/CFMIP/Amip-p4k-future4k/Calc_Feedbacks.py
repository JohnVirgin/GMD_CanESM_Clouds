#!/usr/bin/env python

import numpy as np
import _pickle as pk
import netCDF4 as nc
import Jacks_Functions as jf
import sys
import glob

version = sys.argv[1]

print('reading in Normalized Anomalies')

anom = pk.load(open(version+'_dVARS.pi','rb'))
dlnqdt = pk.load(open(version+'_dlnqdt.pi','rb'))
hus_base = pk.load(open(version+'_hus_climo_amip.pi', 'rb'))
for keys in anom.keys():
    anom[keys]['dlogq'] = (anom[keys]['hus']/hus_base)/dlnqdt[keys]

print('read in radiative kernels and pressure level data')

#define grid and weights
lat = np.linspace(-87.864, 87.864, 64)
lon = np.linspace(0, 357.1875, 128)

y = lat*np.pi/180
coslat = np.cos(y)
coslat = np.tile(coslat[:, None], (1, lon.size))

CMIP_File = nc.Dataset('/praid/users/jgvirgin/Radiative_Kernels/CAM3_Kernels.nc')
CMIP_plevs_scalar = np.squeeze(CMIP_File.variables['lev'])

CMIP_plevs = np.tile(CMIP_plevs_scalar[None,:,None,None],(12,1,lat.size,lon.size))

p_tropo_linear_NH = np.linspace(300,100,32)
p_tropo_linear_SH = np.linspace(100,300,32)
p_tropo_linear = np.concatenate((p_tropo_linear_NH,p_tropo_linear_SH))
p_tropopause = np.tile(p_tropo_linear[None,None,:,None],(12,17,1,lon.size))


print('calculate dp')
ps_path = glob.glob('/praid/users/jgvirgin/CanESM_Data/CanESM5_p2/Raw/vars_interp_Amon_CanESM5_piControl_*')[0]
PS = np.squeeze(nc.Dataset(ps_path).variables['ps'])[:12, :, :]

#Calculate pressure level thickness using CMIP pressure levels and surface pressure
dp = np.zeros([12,17,64,128])

for i in range(12):
    for j in range(64):
        for k in range(128):
            dp[i,:,j,k] = jf.PlevThck(PS = PS[i,j,k]/100,plevs=CMIP_plevs_scalar,p_top = min(CMIP_plevs_scalar))

dp = dp/100 #Kernel units are per 100 hPa

Kernels_interpolated = pk.load(open('/praid/users/jgvirgin/Radiative_Kernels/Interpolated_Kernels.pickle','rb'))

Kernel_interp_mask = {}
for model, kernel in Kernels_interpolated.items():
    Kernel_interp_mask[model] = {}
    for kernel, data in Kernels_interpolated[model].items():
        if Kernels_interpolated[model][kernel].ndim >= 2:
            Kernel_interp_mask[model][kernel] = Kernels_interpolated[model][kernel]
            Kernel_interp_mask[model][kernel][Kernel_interp_mask[model][kernel] > 1e5] = np.nan
        else:
            pass
Kernel_interp_mask.pop('CloudSat')

for keys in anom.keys():

    print('\non experiment - ',keys,'\n')
    print('create isothermal temperature change')
    anom[keys]['ta_iso'] = np.tile(anom[keys]['tas'][:,None,:,:],(1,17,1,1))

    print('mask troposhere/stratosphere where relevent')
    anom[keys]['ta_dep_tropo'] = (anom[keys]['ta']-anom[keys]['ta_iso'])*(CMIP_plevs >= p_tropopause)
    anom[keys]['ta_iso_tropo'] = anom[keys]['ta_iso']*(CMIP_plevs >= p_tropopause)
    anom[keys]['q_tropo'] = anom[keys]['dlogq']*(CMIP_plevs >= p_tropopause)


    print('Calculate feedbacks')
    anom[keys]['planck'] = {}
    anom[keys]['saf'] = {}
    anom[keys]['lapse'] = {}
    anom[keys]['wv'] = {}

    for k in Kernel_interp_mask.keys():
        anom[keys]['saf'][k] = anom[keys]['alb']*Kernel_interp_mask[k]['Alb_TOA']
        anom[keys]['planck'][k] = np.nansum(anom[keys]['ta_iso_tropo']*Kernel_interp_mask[k]['Ta_TOA']*dp,axis=1)+\
                                    anom[keys]['tas']*Kernel_interp_mask[k]['Ts_TOA']
        anom[keys]['lapse'][k] = np.nansum(anom[keys]['ta_dep_tropo']*Kernel_interp_mask[k]['Ta_TOA']*dp,axis=1)
        anom[keys]['wv'][k] = np.nansum(anom[keys]['q_tropo']*Kernel_interp_mask[k]['WVlw_TOA']*dp,axis=1)+\
                              np.nansum(anom[keys]['q_tropo']*Kernel_interp_mask[k]['WVsw_TOA']*dp,axis=1)

    print('global mean values\nsaf - ',np.average(np.nanmean(anom[keys]['saf']['GFDL'],axis=0),weights=coslat),\
        '\nplanck - ',np.average(np.nanmean(anom[keys]['planck']['GFDL'],axis=0),weights=coslat),\
        '\nlapse - ',np.average(np.nanmean(anom[keys]['lapse']['GFDL'],axis=0),weights=coslat),   
        '\nwv - ',np.average(np.nanmean(anom[keys]['wv']['GFDL'],axis=0),weights=coslat))


print('saving')
pk.dump(anom, open(version+'_FB_Grid.pi', 'wb'))
