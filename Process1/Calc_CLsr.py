#!/usr/bin/env python
# coding: utf-8

## Import Packages
from scipy.interpolate import interp1d
import numpy as np
import _pickle as pk
from netCDF4 import Dataset
import sys
import glob
from natsort import natsorted

print('read in cloud kernels')

k_source = '/praid/users/jgvirgin/Radiative_Kernels/Cloud/'

model = sys.argv[1]
step_n = sys.argv[2]
step_n_int = int(step_n)

#load in cloud kernels
LWkernel = np.squeeze(
    Dataset(k_source+'cloud_kernels2.nc').variables['LWkernel'])
SWkernel = np.squeeze(
    Dataset(k_source+'cloud_kernels2.nc').variables['SWkernel'])
print(LWkernel.shape,
      '\n month, optical depth, cloud top pressure, latitude, clear sky surface albedo')


lons = np.arange(1.25, 360, 2.5)
lats = np.squeeze(Dataset(k_source+'cloud_kernels2.nc').variables['lat'])
# the clear-sky albedos over which the kernel is computed
albcs = np.arange(0.0, 1.1, 0.5)
CanESM_lat = np.linspace(-87.864, 87.864, 64)
CanESM_lon = np.linspace(0, 357.1875, 128)

print('interpolate the kernels along their latitude dimension to match CanESM output')
#interpolate the kernels down to the CanESM2 resolution

LWkernel_func = interp1d(lats, LWkernel, axis=3, kind='nearest')
SWkernel_func = interp1d(lats, SWkernel, axis=3, kind='nearest')
LWkernel_interp = LWkernel_func(CanESM_lat)
SWkernel_interp = SWkernel_func(CanESM_lat)

# LW kernel does not depend on albcs, just repeat the final dimension over longitudes:
LWkernel_map = np.tile(LWkernel_interp[:,:,:,:,0,None],(1,1,1,1,128))


if model == 'CanESM2':
    mn_range_s = [0, 300]
    mn_range_f = [300, 492]

    if step_n == '1':
        nyr_4xco2 = 16
    else:
        nyr_4xco2 = 25
else:
    mn_range_s = [0, 300, 600, 900, 1200, 1500]
    mn_range_f = [300, 600, 900, 1200, 1500, 1800]

    nyr_4xco2 = 25

print('Read in cloud fraction output')

cl_source = '/praid/users/jgvirgin/CanESM_Data/'+model+'/Raw/'


clissp2_path = natsorted(glob.glob(cl_source+'*_abrupt-4xCO2_*'))[0]
clissp1_path = natsorted(glob.glob(cl_source+'*_piControl_*'))[0]

print('picon path - ',clissp1_path)
print('4xco2 path - ',clissp2_path)

clisccp2 = np.squeeze(Dataset(clissp2_path).variables['clisccp'])[
    mn_range_s[step_n_int]:mn_range_f[step_n_int], :, :, :, :]
clisccp2[clisccp2 > 1e10] = np.nan
clisccp1 = np.squeeze(Dataset(clissp1_path).variables['clisccp'])
clisccp1[clisccp1 > 1e10] = np.nan

nmon = clisccp1[:, 0, 0, 0, 0].size
nyr = int(nmon/12)
print('number of years read in of picon data - ',nyr)

print('anomaly shapes - picontrol - ', clisccp1.shape)
print('anomaly shapes - 4xCO2 - ', clisccp2.shape)

print('Calculate anomalies')
#convert time dimensions to months/years, then take the climatological mean
clisccp1_stacked = np.zeros([nyr, 12, 7, 7, 64, 128])
clisccp2_stacked = np.zeros([nyr_4xco2, 12, 7, 7, 64, 128])

print('Calculate anomalies')
#convert time dimensions to months/years, then take the climatological mean
clisccp1_stacked = np.zeros([nyr, 12, 7, 7, 64, 128])
clisccp2_stacked = np.zeros([nyr_4xco2, 12, 7, 7, 64, 128])

s = 0
f = 12

for i in range(nyr):
    clisccp1_stacked[i, :, :, :, :, :] = np.stack(
        clisccp1[s:f, :, :, :, :], axis=0)

    s += 12
    f += 12

s = 0
f = 12

for i in range(nyr_4xco2):
    clisccp2_stacked[i, :, :, :, :, :] = np.stack(
        clisccp2[s:f, :, :, :, :], axis=0)

    s += 12
    f += 12

avgclisccp1 = np.tile(np.mean(clisccp1_stacked, axis=0)[
                      None, :, :, :, :, :], (nyr_4xco2, 1, 1, 1, 1, 1))

# Compute clisccp anomalies
dc = clisccp2_stacked - avgclisccp1

print('Partitioning cloud fraction response into obscured low cloud versus non obscured anomalies')

climo_cl = avgclisccp1
pert_cl = clisccp2_stacked
climo_cl_hi_sum = np.tile(\
    np.nansum(climo_cl[:,:,:,2:,:,:],axis=(2,3))[:,:,None,None,:,:],(1,1,7,2,1,1))

climo_cl_lo_n = climo_cl[:,:,:,:2,:,:]/(100-climo_cl_hi_sum)
    
pert_cl_hi_sum = np.tile(\
    np.nansum(pert_cl[:,:,:,2:,:,:],axis=(2,3))[:,:,None,None,:,:],(1,1,7,2,1,1))
    
pert_cl_lo_n = pert_cl[:,:,:,:2,:,:]/(100-pert_cl_hi_sum)

dc_lo_n = (pert_cl_lo_n-climo_cl_lo_n)*(100-climo_cl_hi_sum)
dc_lo = dc[:,:,:,:2,:,:]
dc_hi = dc[:,:,:,2:,:,:]

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

print('calculate cloud feedbacks')

dRsw_lo = np.zeros([nyr_4xco2, 12, 64, 128])
dRsw_lo_n = np.zeros([nyr_4xco2, 12, 64, 128])
dRsw_hi = np.zeros([nyr_4xco2, 12, 64, 128])

dRlw_lo = np.zeros([nyr_4xco2, 12, 64, 128])
dRlw_lo_n = np.zeros([nyr_4xco2, 12, 64, 128])
dRlw_hi = np.zeros([nyr_4xco2, 12, 64, 128])

for i in range(nyr_4xco2):
    dRsw_lo[i,:,:,:] = np.nansum(SWkernel_map[:,:,:2,:,:]*dc_lo[i,:,:,:,:,:],axis=(1,2))
    dRsw_lo_n[i,:,:,:] = np.nansum(SWkernel_map[:,:,:2,:,:]*dc_lo_n[i,:,:,:,:,:],axis=(1,2))
    dRsw_hi[i,:,:,:] = np.nansum(SWkernel_map[:,:,2:,:,:]*dc_hi[i,:,:,:,:,:],axis=(1,2))

    dRlw_lo[i,:,:,:] = np.nansum(LWkernel_map[:,:,:2,:,:]*dc_lo[i,:,:,:,:,:],axis=(1,2))
    dRlw_lo_n[i,:,:,:] = np.nansum(LWkernel_map[:,:,:2,:,:]*dc_lo_n[i,:,:,:,:,:],axis=(1,2))
    dRlw_hi[i,:,:,:] = np.nansum(LWkernel_map[:,:,2:,:,:]*dc_hi[i,:,:,:,:,:],axis=(1,2))


print('Saving')

#first, create another dictionary for storing\
SW_fluxes = {}
SW_fluxes['Low'] = dRsw_lo
SW_fluxes['Low_unobscured'] = dRsw_lo_n
SW_fluxes['Hi'] = dRsw_hi

pk.dump(SW_fluxes, open(model+'_'+step_n+'_TrCLsw_FLUX_SRdecomp_Grid.pi', 'wb'))

LW_fluxes = {}
LW_fluxes['Low'] = dRlw_lo
LW_fluxes['Low_unobscured'] = dRlw_lo_n
LW_fluxes['Hi'] = dRlw_hi

pk.dump(LW_fluxes, open(model+'_'+step_n+'_TrCLlw_FLUX_SRdecomp_Grid.pi', 'wb'))