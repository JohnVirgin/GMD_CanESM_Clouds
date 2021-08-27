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

#load in cloud kernels
LWkernel = np.squeeze(Dataset(k_source+'cloud_kernels2.nc').variables['LWkernel'])
SWkernel = np.squeeze(Dataset(k_source+'cloud_kernels2.nc').variables['SWkernel'])
print(LWkernel.shape,'\n month, optical depth, cloud top pressure, latitude, clear sky surface albedo')

lons = np.arange(1.25, 360, 2.5)
lats = np.squeeze(Dataset(k_source+'cloud_kernels2.nc').variables['lat'])
# the clear-sky albedos over which the kernel is computed
albcs = np.arange(0.0, 1.1, 0.5)
CanESM_lat = np.linspace(-87.864, 87.864, 64)
CanESM_lon = np.linspace(0, 357.1875, 128)

y = CanESM_lat*np.pi/180
coslat = np.cos(y)
coslat = np.tile(coslat, (CanESM_lon.size, 1)).T

print('interpolate the kernels along their latitude dimension to match CanESM output')
#interpolate the kernels down to the CanESM2 resolution

LWkernel_func = interp1d(lats, LWkernel, axis=3, kind='nearest')
SWkernel_func = interp1d(lats, SWkernel, axis=3, kind='nearest')
LWkernel_interp = LWkernel_func(CanESM_lat)
SWkernel_interp = SWkernel_func(CanESM_lat)

# LW kernel does not depend on albcs, just repeat the final dimension over longitudes:
LWkernel_map = np.tile(LWkernel_interp[:,:,:,:,0,None],(1,1,1,1,128))

print('Read in cloud fraction output')

cl_source = '/praid/users/jgvirgin/CanESM_Data/'+model+'/CFMIP/Climatology/'
exps = ['amip','amip-future4K','amip-p4K']
style = ['amip-future4K','amip-p4K']

cl = dict()
tas = dict()
for i in range(len(exps)):
    cl[exps[i]] = pk.load(open(cl_source+model+'_'+exps[i]+'_clisccp_Climo.pi','rb'))
    tas[exps[i]] = pk.load(open(cl_source+model+'_'+exps[i]+'_tas_Climo.pi', 'rb'))

print('cloud fraction area output shape? - ',cl[exps[0]].shape)
print('calculate anomalies')

dtas = dict()
for i in range(len(style)):
    dtas[style[i]] = tas[style[i]]-tas['amip']   

dtas_GAM = dict()
for keys in dtas.keys():
    print(keys, '- Global mean TAS change')
    dtas_GAM[keys] = np.average(np.nanmean(dtas[keys],axis=0), weights=coslat)
    print(dtas_GAM[keys])


print('Partitioning cloud fraction response into obscured low cloud versus non obscured anomalies')
dc = {}
climo_cl = {}
climo_cl_hi_sum = {}
climo_cl_lo_n = {}

pert_cl_hi_sum = {}
pert_cl_lo_n = {}

dc_lo = {}
dc_lo_n = {}
dc_hi = {}

dc_lo_fb = {}
dc_lo_n_fb = {}
dc_hi_fb = {}
for keys in dtas.keys():
    dc[keys] = cl[keys]-cl['amip']
    climo_cl[keys] = cl['amip']
    climo_cl_hi_sum[keys] = np.tile(\
        np.nansum(climo_cl[keys][:,:,2:,:,:],axis=(1,2))[:,None,None,:,:],(1,7,2,1,1))

    climo_cl_lo_n[keys] = climo_cl[keys][:,:,:2,:,:]/(100-climo_cl_hi_sum[keys])
    
    pert_cl_hi_sum[keys] = np.tile(\
        np.nansum(cl[keys][:,:,2:,:,:],axis=(1,2))[:,None,None,:,:],(1,7,2,1,1))
    
    pert_cl_lo_n[keys] = cl[keys][:,:,:2,:,:]/(100-pert_cl_hi_sum[keys])

    dc_lo_n[keys] = (pert_cl_lo_n[keys]-climo_cl_lo_n[keys])*(100-climo_cl_hi_sum[keys])
    dc_lo[keys] = dc[keys][:,:,:2,:,:]
    dc_hi[keys] = dc[keys][:,:,2:,:,:]

    dc_lo_n_fb[keys] = dc_lo_n[keys]/dtas_GAM[keys]
    dc_lo_fb[keys] = dc_lo[keys]/dtas_GAM[keys]
    dc_hi_fb[keys] = dc_hi[keys]/dtas_GAM[keys]

print('read in surface albedo output')
#read in clear sky shortwave fluxes at the surface for the same time period

avgalbcs1 = pk.load(open(cl_source+model+'_amip_albcs_Climo.pi', 'rb'))

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

print('calculate cloud feedbacks')

dRsw_lo = {}
dRsw_lo_n = {}
dRsw_hi = {}

dRlw_lo = {}
dRlw_lo_n = {}
dRlw_hi = {}

for keys in dtas_GAM.keys():
    dRsw_lo[keys] = np.nansum(SWkernel_map[:,:,:2,:,:]*dc_lo_fb[keys],axis=(1,2))
    dRsw_lo_n[keys] = np.nansum(SWkernel_map[:,:,:2,:,:]*dc_lo_n_fb[keys],axis=(1,2))
    dRsw_hi[keys] = np.nansum(SWkernel_map[:,:,2:,:,:]*dc_hi_fb[keys],axis=(1,2))

    dRsw_lo[keys][night] = 0
    dRsw_lo_n[keys][night] = 0
    dRsw_hi[keys][night] = 0

    dRlw_lo[keys] = np.nansum(LWkernel_map[:,:,:2,:,:]*dc_lo_fb[keys],axis=(1,2))
    dRlw_lo_n[keys] = np.nansum(LWkernel_map[:,:,:2,:,:]*dc_lo_n_fb[keys],axis=(1,2))
    dRlw_hi[keys] = np.nansum(LWkernel_map[:,:,2:,:,:]*dc_hi_fb[keys],axis=(1,2))


print('Saving')

#first, create another dictionary for storing\
SW_fluxes = {}
SW_fluxes['Low'] = dRsw_lo
SW_fluxes['Low_unobscured'] = dRsw_lo_n
SW_fluxes['Hi'] = dRsw_hi

pk.dump(SW_fluxes, open(model+'_CFMIP_TrCLsw_FB_SRdecomp_Grid.pi', 'wb'))

LW_fluxes = {}
LW_fluxes['Low'] = dRlw_lo
LW_fluxes['Low_unobscured'] = dRlw_lo_n
LW_fluxes['Hi'] = dRlw_hi

pk.dump(LW_fluxes, open(model+'_CFMIP_TrCLlw_FB_SRdecomp_Grid.pi', 'wb'))




