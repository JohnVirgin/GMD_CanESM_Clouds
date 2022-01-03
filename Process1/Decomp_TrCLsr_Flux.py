#!/usr/bin/env python
# coding: utf-8

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
from scipy.interpolate import interp1d
from netCDF4 import Dataset
import sys
import glob
from natsort import natsorted


def LinReg_alpha_wrap(predictand,predictor):
    fit = LinearRegression().fit(predictor,predictand)
    return fit.coef_

def LinReg_int_wrap(predictand,predictor):
    fit = LinearRegression().fit(predictor,predictand)
    return fit.intercept_

print('Setting up worker pools using',mp.cpu_count(),'cpu cores')
cpus = mp.cpu_count()
Pools = mp.Pool(cpus)

model = sys.argv[1]

CanESM_lat = np.linspace(-87.864, 87.864, 64)
CanESM_lon = np.linspace(0, 357.1875, 128)

Source = '/praid/users/jgvirgin/CanESM_Data/'+model

print('read in data')

CLlw = pk.load(open(Source+'/Fluxes/'+model+'_TrCLlw_FLUX_SRdecomp_Grid.pi', 'rb'))
CLsw = pk.load(open(Source+'/Fluxes/'+model+'_TrCLsw_FLUX_SRdecomp_Grid.pi', 'rb'))

tas_stk = pk.load(open(Source+'/Anomalies/'+model+'_Ajd_4xCO2_TmSrs.pi','rb'))['tas']

tas_GAM = aa.LatLonavg_Time(np.mean(tas_stk,axis=1),CanESM_lat,CanESM_lon)

tas_gam_1 = tas_GAM[:20]
tas_gam_2 = tas_GAM[120:141]

tas_GAM_rsh = np.concatenate((tas_gam_1, tas_gam_2),axis=0).reshape(-1,1)
#print('reshaped surface temperature shape - ',tas_GAM_rsh.shape)
#tas_GAM_rsh = tas_GAM.reshape(-1,1)

k_source = '/praid/users/jgvirgin/Radiative_Kernels/Cloud/'

SWkernel = np.squeeze(
    Dataset(k_source+'cloud_kernels2.nc').variables['SWkernel'])

lons = np.arange(1.25, 360, 2.5)
lats = np.squeeze(Dataset(k_source+'cloud_kernels2.nc').variables['lat'])
# the clear-sky albedos over which the kernel is computed
albcs = np.arange(0.0, 1.1, 0.5)

print('interpolate the kernel along its latitude dimension to match CanESM output')
#interpolate the kernels down to the CanESM2 resolution

SWkernel_func = interp1d(lats, SWkernel, axis=3, kind='nearest')
SWkernel_interp = SWkernel_func(CanESM_lat)

print('read in surface albedo output')
#read in clear sky shortwave fluxes at the surface for the same time period

albcs_source = natsorted(glob.glob(\
    '/praid/users/jgvirgin/CanESM_Data/'+model+'/Raw/vars_interp_Amon_CanESM2_piControl_*'))[0]


print('albedo source file in use: ')
print(albcs_source)

rsdscs1 = np.squeeze(Dataset(albcs_source).variables['rsdscs'])[:1800, :, :]
rsuscs1 = np.squeeze(Dataset(albcs_source).variables['rsuscs'])[:1800, :, :]
albcs1 = rsuscs1/rsdscs1

albcs1_stacked = np.zeros([150, 12, 64, 128])

s = 0
f = 12

for i in range(150):
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
        function = interp1d(
            albcs, SWkernel_interp[m, :, :, la, :], axis=2, kind='linear')
        new_kernel_lon = function(alb_lon)
        SWkernel_map[m, :, :, la, :] = new_kernel_lon

sundown = np.nansum(SWkernel_map, axis=(1, 2))
#set the SW feedbacks to zero in the polar night
night = np.where(sundown == 0)

print('regress cloud flux perturbations against global annual mean surface temperature change this gives a slope for each gridpoint and each month')

start_time = time.time()

CLlw_flatten_fb = dict()
CLsw_flatten_fb = dict()

CLlw_fb = dict()
CLsw_fb = dict()

for prop in CLlw.keys():

    print('on cloud property - ',prop)

    CLlw_flatten_fb[prop] = [i for i in np.swapaxes(CLlw[prop][:,:,:,:].reshape(len(CLlw[prop][:,:,0,0]),12*64*128),0,1)]
    CLsw_flatten_fb[prop] = [i for i in np.swapaxes(CLsw[prop][:,:,:,:].reshape(len(CLsw[prop][:,:,0,0]),12*64*128),0,1)]

    CLlw_fb[prop] = Pools.starmap(LinReg_alpha_wrap,zip(CLlw_flatten_fb[prop],repeat(tas_GAM_rsh[:,:])))
    CLsw_fb[prop] = Pools.starmap(LinReg_alpha_wrap,zip(CLsw_flatten_fb[prop],repeat(tas_GAM_rsh[:,:])))

end_time = time.time() - start_time
print(end_time/60, 'minutes for complete regressions to finish')

Pools.close()
Pools.join()

print('rebuild array shape and set SW feedbacks to zero in the polar night')

CLlw_fb_rebuild = dict()
CLsw_fb_rebuild = dict()

for prop in CLlw.keys():

    CLlw_fb_rebuild[prop] = np.stack(CLlw_fb[prop][:],axis=0).reshape(12,64,128)
    CLsw_fb_rebuild[prop] = np.stack(CLsw_fb[prop][:],axis=0).reshape(12,64,128)

    CLsw_fb_rebuild[prop][night] = 0


print('take the annual means')

CLlw_fb_AM = dict()
CLsw_fb_AM = dict()
for prop in CLlw.keys():
    CLsw_fb_AM[prop] = np.mean(CLsw_fb_rebuild[prop], axis=0)
    CLlw_fb_AM[prop] = np.mean(CLlw_fb_rebuild[prop], axis=0)
print('saving variables')

print('global mean values... ')
print('SW Low - ', aa.LatLonavg_Time(CLsw_fb_AM['Low'][None,:,:],CanESM_lat,CanESM_lon)[0])
print('SW Low unobscured - ', aa.LatLonavg_Time(CLsw_fb_AM['Low_unobscured'][None,:,:],CanESM_lat,CanESM_lon)[0])


print('saving...')
CLlw_fb_rebuild_file = open(model+'_TrCLlw_FLUX_FB_SRdecomp_Grid.pi', 'wb')
pk.dump(CLlw_fb_AM, CLlw_fb_rebuild_file)
CLlw_fb_rebuild_file.close()

CLsw_fb_rebuild_file = open(model+'_TrCLsw_FLUX_FB_SRdecomp_Grid.pi','wb')
pk.dump(CLsw_fb_AM, CLsw_fb_rebuild_file)
CLsw_fb_rebuild_file.close()

print('finished')