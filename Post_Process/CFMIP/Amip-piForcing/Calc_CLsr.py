#!/usr/bin/env python
# coding: utf-8

## Import Packages
from scipy.interpolate import interp1d
import numpy as np
import _pickle as pk
from netCDF4 import Dataset
import xarray as xr
import sys
import glob
from natsort import natsorted

print('read in cloud kernels')

k_source = '/mnt/data/users/jgvirgin/Kernels/Cloud/'

model = sys.argv[1]

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

print('Read in cloud fraction output')

cl_source = '/mnt/data/users/jgvirgin/GMD_CanESM_p1/CanESM5_p2/CFMIP/raw/CFmon/'

clissp_path = natsorted(glob.glob(cl_source+'*_amip-piForcing_*'))[0]
data = xr.open_dataset(clissp_path)

print('piForcing path - ',clissp_path)

clisccp = data['clisccp']

#take the first 30 years as the control
clisccp_control = np.nanmean(clisccp.values[1320:1680],axis=0)
clisccp_con_vals = np.tile(clisccp_control[None,None,:,:,:,:],(145,12,1,1,1,1))
clisccp_con_vals[clisccp_con_vals > 1e10] = np.nan

clisccp_resp = clisccp.values
clisccp_resp_vals = np.zeros([145,12,7,7,64,128])

print('stacking')

s = 0
f = 12
for i in range(145):
    clisccp_resp_vals[i] = np.stack(clisccp_resp[s:f,:,:,:,:],axis=0)

    s+=12
    f+=12

clisccp_resp_vals[clisccp_resp_vals > 1e10] = np.nan

print('Calculate anomlies')
dc = clisccp_resp_vals - clisccp_con_vals

print('Partitioning cloud fraction response into obscured low cloud versus non obscured anomalies')

climo_cl = clisccp_con_vals
pert_cl = clisccp_resp_vals

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

albcs_source = natsorted(glob.glob('/mnt/data/users/jgvirgin/GMD_CanESM_p1/CanESM5_p2/CFMIP/raw/Amon/*_amip-piForcing_*'))[0]
albcs_data = xr.open_dataset(albcs_source)
print('albedo source file in use: ')
print(albcs_source)

rsdscs1 = albcs_data['rsdscs'][1320:1680, :, :]
rsuscs1 = albcs_data['rsuscs'][1320:1680, :, :]
albcs1 = rsuscs1/rsdscs1
avgalbcs1 = albcs1.groupby('time.month').mean().values

print('final control albedo shape - ',avgalbcs1.shape)

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

dRsw_lo = np.zeros([145, 12, 64, 128])
dRsw_lo_n = np.zeros([145, 12, 64, 128])
dRsw_hi = np.zeros([145, 12, 64, 128])

dRlw_lo = np.zeros([145, 12, 64, 128])
dRlw_lo_n = np.zeros([145, 12, 64, 128])
dRlw_hi = np.zeros([145, 12, 64, 128])

for i in range(145):
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

pk.dump(SW_fluxes, open(model+'_TrCLsw_FLUX_SRdecomp_Grid.pi', 'wb'))

LW_fluxes = {}
LW_fluxes['Low'] = dRlw_lo
LW_fluxes['Low_unobscured'] = dRlw_lo_n
LW_fluxes['Hi'] = dRlw_hi

pk.dump(LW_fluxes, open(model+'_TrCLlw_FLUX_SRdecomp_Grid.pi', 'wb'))