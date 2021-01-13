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

y = CanESM_lat*np.pi/180
coslat = np.cos(y)
coslat = np.tile(coslat, (CanESM_lon.size, 1)).T
coslat = np.tile(coslat[None,:,:], (12,1,1))

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

cl = dict()
tas = dict()
for i in range(len(exps)):
    cl[exps[i]] = pk.load(open(cl_source+model+'_'+exps[i]+'_clisccp_Climo.pi','rb'))
    tas[exps[i]] = pk.load(open(cl_source+model+'_'+exps[i]+'_tas_Climo.pi', 'rb'))

print('cloud fraction area output shape? - ',cl[exps[0]].shape)
print('calculate anomalies')

dtas = dict()
dtas['Uniform'] = tas['amip-p4K']-tas['amip']
dtas['Pattern'] = tas['amip-future4K']-tas['amip']

dtas_GAM = dict()
for keys in dtas.keys():
    print(keys, '- Global mean TAS change')
    print(np.nanmean(np.average(dtas[keys],weights=coslat)))
    dtas_GAM[keys] = np.nanmean(np.average(dtas[keys], weights=coslat))

dc = dict()
avgclisccp1 = dict()
dc['Uniform'] = (cl['amip-p4K']-cl['amip'])/dtas_GAM['Uniform']
dc['Pattern'] = (cl['amip-future4K']-cl['amip'])/dtas_GAM['Pattern']

avgclisccp1['Uniform'] = cl['amip']
avgclisccp1['Pattern'] = cl['amip']

print('Deocomposing cloud fraction anomalies and radiative kernels into contributions from amount, altitude, and optical depth feedbacks')
sect = ['Low', 'Hi', 'All']
sections_ind = [slice(0, 2), slice(2, 7), slice(0, 7)]
sections_length = [2, 5, 7]

c1_sum = dict()
dc_sum = dict()
dc_prop = dict()
dc_star = dict()

for keys in dc.keys():
    c1_sum[keys] = dict()
    dc_sum[keys] = dict()
    dc_prop[keys] = dict()
    dc_star[keys] = dict()

    for i in range(len(sect)):

        #sum total cloud fraction and project it across all CTPs and optical depths
        c1_sum[keys][sect[i]] = np.tile(np.nansum(avgclisccp1[keys][:, :, sections_ind[i], :, :], axis=(
            1, 2))[:, None, None, :, :], (1, 7, sections_length[i], 1, 1))

    #some the change in cloud fractions over CTP and tau
        dc_sum[keys][sect[i]] = np.tile(np.nansum(dc[keys][:, :, sections_ind[i], :, :], axis=(
            1, 2))[:, None, None, :, :], (1, 7, sections_length[i], 1, 1))

    #change in cloud fraction due to proportional changes in clouds, but not due to CTP or optical depth
        dc_prop[keys][sect[i]] = avgclisccp1[keys][:, :, sections_ind[i], :, :] * \
            (dc_sum[keys][sect[i]]/c1_sum[keys][sect[i]])

    #changes in optical depth and CTP, but with proportional changes in cloud fraction fixed
        dc_star[keys][sect[i]] = dc[keys][:, :, sections_ind[i], :, :] - dc_prop[keys][sect[i]]


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


print('decompose the longwave kernel')

Klw0 = dict()
Klw_prime = dict()
lw_this = dict()
lw_that = dict()
Klw_p_prime = dict()
Klw_t_prime = dict()
Klw_resid_prime = dict()

for keys in c1_sum.keys():

    print('on experiment - ',keys)
    Klw0[keys] = dict()
    Klw_prime[keys] = dict()
    lw_this[keys] = dict()
    lw_that[keys] = dict()
    Klw_p_prime[keys] = dict()
    Klw_t_prime[keys] = dict()
    Klw_resid_prime[keys] = dict()
    for i in range(len(sect)):

        print('on section - ',sect[i])

        #sum the kernels CTP and optical depth bins and weight it by total cloud cover fraction
        #this not term is a kernel to estimate feedbacks from changes in cloud cover alone
        Klw0[keys][sect[i]] = np.tile(np.nansum(
            LWkernel_map[:, :, sections_ind[i], :, :]*(avgclisccp1[keys][:, :, sections_ind[i], :, :]/c1_sum[keys][sect[i]]), \
            axis=(1, 2))[:, None, None, :, :], (1, 7, sections_length[i], 1, 1))

        #prime represents a kernel to estimate feedbacks from changes in CTP and tau
        Klw_prime[keys][sect[i]] = LWkernel_map[:, :,sections_ind[i], :, :] - Klw0[keys][sect[i]]

        #we further decompose the CTP & tau kernel into separate kernels for both
        #CTP
        lw_this[keys][sect[i]] = np.nansum(
            Klw_prime[keys][sect[i]]*np.tile(np.nansum(avgclisccp1[keys][:, :, sections_ind[i], :, :]/c1_sum[keys][sect[i]], axis=2)[:, :, None, :, :], (1, 1, sections_length[i], 1, 1)), axis=1)

        #tau
        lw_that[keys][sect[i]] = np.nansum(
            Klw_prime[keys][sect[i]]*np.tile(np.nansum(avgclisccp1[keys][:, :, sections_ind[i], :, :]/c1_sum[keys][sect[i]], axis=1)[:, None, :, :, :], (1, 7, 1, 1, 1)), axis=2)

        Klw_p_prime[keys][sect[i]] = np.tile(lw_this[keys][sect[i]][:, None, :, :, :], (1, 7, 1, 1, 1))
        Klw_t_prime[keys][sect[i]] = np.tile(lw_that[keys][sect[i]][:, :, None, :, :], (1, 1, sections_length[i], 1, 1))

        #residual
        Klw_resid_prime[keys][sect[i]] = Klw_prime[keys][sect[i]] - Klw_p_prime[keys][sect[i]] - Klw_t_prime[keys][sect[i]]

print('longwave feedbacks')


#now compute decomposed fluxes
dRlw_true = dict()
dRlw_prop = dict()
dRlw_dctp = dict()
dRlw_dtau = dict()
dRlw_resid = dict()
dRlw_sum = dict()

for keys in c1_sum.keys():
    dRlw_true[keys] = dict()
    dRlw_prop[keys] = dict()
    dRlw_dctp[keys] = dict()
    dRlw_dtau[keys] = dict()
    dRlw_resid[keys] = dict()
    dRlw_sum[keys] = dict()

    for i in range(len(sect)):

        # lw total
        dRlw_true[keys][sect[i]] = np.nansum(
            LWkernel_map[:, :, sections_ind[i], :, :]*dc[keys][:, :, sections_ind[i], :, :], axis=(1, 2))

        # lw amount component
        dRlw_prop[keys][sect[i]] = (Klw0[keys][sect[i]][:, 0, 0, :, :]*dc_sum[keys][sect[i]][:, 0, 0, :, :])
        # lw altitude component
        dRlw_dctp[keys][sect[i]] = np.nansum(
            Klw_p_prime[keys][sect[i]]*dc_star[keys][sect[i]], axis=(1, 2))
        # lw optical depth component
        dRlw_dtau[keys][sect[i]] = np.nansum(
            Klw_t_prime[keys][sect[i]]*dc_star[keys][sect[i]], axis=(1, 2))
        # lw residual
        dRlw_resid[keys][sect[i]] = np.nansum(
            Klw_resid_prime[keys][sect[i]]*dc_star[keys][sect[i]], axis=(1, 2))
        # sum should equal true
        dRlw_sum[keys][sect[i]] = dRlw_prop[keys][sect[i]] + dRlw_dctp[keys][sect[i]] + \
            dRlw_dtau[keys][sect[i]] + dRlw_resid[keys][sect[i]]

print('longwave done, moving onto shortwave')
#decompose the sw kernel now
#now compute decomposed fluxes
dRlw_true = dict()
dRlw_prop = dict()
dRlw_dctp = dict()
dRlw_dtau = dict()
dRlw_resid = dict()
dRlw_sum = dict()

for keys in c1_sum.keys():
    dRlw_true[keys] = dict()
    dRlw_prop[keys] = dict()
    dRlw_dctp[keys] = dict()
    dRlw_dtau[keys] = dict()
    dRlw_resid[keys] = dict()
    dRlw_sum[keys] = dict()

    for i in range(len(sect)):

        # lw total
        dRlw_true[keys][sect[i]] = np.nansum(
            LWkernel_map[:, :, sections_ind[i], :, :]*dc[keys][:, :, sections_ind[i], :, :], axis=(1, 2))

        # lw amount component
        dRlw_prop[keys][sect[i]] = (Klw0[keys][sect[i]][:, 0, 0, :, :]*dc_sum[keys][sect[i]][:, 0, 0, :, :])
        # lw altitude component
        dRlw_dctp[keys][sect[i]] = np.nansum(
            Klw_p_prime[keys][sect[i]]*dc_star[keys][sect[i]], axis=(1, 2))
        # lw optical depth component
        dRlw_dtau[keys][sect[i]] = np.nansum(
            Klw_t_prime[keys][sect[i]]*dc_star[keys][sect[i]], axis=(1, 2))
        # lw residual
        dRlw_resid[keys][sect[i]] = np.nansum(
            Klw_resid_prime[keys][sect[i]]*dc_star[keys][sect[i]], axis=(1, 2))
        # sum should equal true
        dRlw_sum[keys][sect[i]] = dRlw_prop[keys][sect[i]] + dRlw_dctp[keys][sect[i]] + \
            dRlw_dtau[keys][sect[i]] + dRlw_resid[keys][sect[i]]

print('longwave done, moving onto shortwave')
#decompose the sw kernel now

Ksw0 = dict()
Ksw_prime = dict()
sw_this = dict()
sw_that = dict()
Ksw_p_prime = dict()
Ksw_t_prime = dict()
Ksw_resid_prime = dict()

for keys in c1_sum.keys():

    print('on experiment - ',keys)
    Ksw0[keys] = dict()
    Ksw_prime[keys] = dict()
    sw_this[keys] = dict()
    sw_that[keys] = dict()
    Ksw_p_prime[keys] = dict()
    Ksw_t_prime[keys] = dict()
    Ksw_resid_prime[keys] = dict()
    for i in range(len(sect)):

        print('on section - ',sect[i])

        #sum the kernels CTP and optical depth bins and weight it by total cloud cover fraction
        #this not term is a kernel to estimate feedbacks from changes in cloud cover alone
        Ksw0[keys][sect[i]] = np.tile(np.nansum(
            SWkernel_map[:, :, sections_ind[i], :, :]*(avgclisccp1[keys][:, :, sections_ind[i], :, :]/c1_sum[keys][sect[i]]), \
            axis=(1, 2))[:, None, None, :, :], (1, 7, sections_length[i], 1, 1))

        #prime represents a kernel to estimate feedbacks from changes in CTP and tau
        Ksw_prime[keys][sect[i]] = SWkernel_map[:, :,sections_ind[i], :, :] - Ksw0[keys][sect[i]]

        #we further decompose the CTP & tau kernel into separate kernels for both
        #CTP
        sw_this[keys][sect[i]] = np.nansum(
            Ksw_prime[keys][sect[i]]*np.tile(np.nansum(avgclisccp1[keys][:, :, sections_ind[i], :, :]/c1_sum[keys][sect[i]], axis=2)[:, :, None, :, :], (1, 1, sections_length[i], 1, 1)), axis=1)

        #tau
        sw_that[keys][sect[i]] = np.nansum(
            Ksw_prime[keys][sect[i]]*np.tile(np.nansum(avgclisccp1[keys][:, :, sections_ind[i], :, :]/c1_sum[keys][sect[i]], axis=1)[:, None, :, :, :], (1, 7, 1, 1, 1)), axis=2)

        Ksw_p_prime[keys][sect[i]] = np.tile(sw_this[keys][sect[i]][:, None, :, :, :], (1, 7, 1, 1, 1))
        Ksw_t_prime[keys][sect[i]] = np.tile(sw_that[keys][sect[i]][:, :, None, :, :], (1, 1, sections_length[i], 1, 1))

        #residual
        Ksw_resid_prime[keys][sect[i]] = Ksw_prime[keys][sect[i]] - Ksw_p_prime[keys][sect[i]] - Ksw_t_prime[keys][sect[i]]

print('shortwave feedbacks')

#now compute decomposed fluxes
dRsw_true = dict()
dRsw_prop = dict()
dRsw_dctp = dict()
dRsw_dtau = dict()
dRsw_resid = dict()
dRsw_sum = dict()

for keys in c1_sum.keys():
    dRsw_true[keys] = dict()
    dRsw_prop[keys] = dict()
    dRsw_dctp[keys] = dict()
    dRsw_dtau[keys] = dict()
    dRsw_resid[keys] = dict()
    dRsw_sum[keys] = dict()

    for i in range(len(sect)):

        # sw total
        dRsw_true[keys][sect[i]] = np.nansum(
            SWkernel_map[:, :, sections_ind[i], :, :]*dc[keys][:, :, sections_ind[i], :, :], axis=(1, 2))

        # sw amount component
        dRsw_prop[keys][sect[i]] = (Ksw0[keys][sect[i]][:, 0, 0, :, :]
                          * dc_sum[keys][sect[i]][:, 0, 0, :, :])
        # sw altitude component
        dRsw_dctp[keys][sect[i]] = np.nansum(
            Ksw_p_prime[keys][sect[i]]*dc_star[keys][sect[i]], axis=(1, 2))
        # sw optical depth component
        dRsw_dtau[keys][sect[i]] = np.nansum(
            Ksw_t_prime[keys][sect[i]]*dc_star[keys][sect[i]], axis=(1, 2))
        # sw residual
        dRsw_resid[keys][sect[i]] = np.nansum(
            Ksw_resid_prime[keys][sect[i]]*dc_star[keys][sect[i]], axis=(1, 2))
        # sum should equal true
        dRsw_sum[keys][sect[i]] = dRsw_prop[keys][sect[i]] + dRsw_dctp[keys][sect[i]] + \
            dRsw_dtau[keys][sect[i]] + dRsw_resid[keys][sect[i]]

        dRsw_true[keys][sect[i]][night] = 0
        dRsw_prop[keys][sect[i]][night] = 0
        dRsw_dctp[keys][sect[i]][night] = 0
        dRsw_dtau[keys][sect[i]][night] = 0
        dRsw_resid[keys][sect[i]][night] = 0
        dRsw_sum[keys][sect[i]][night] = 0


print('saving variables')

#save files!

#first, create another dictionary for storing\
SW_fluxes = dict()
SW_fluxes['Standard'] = dRsw_true
SW_fluxes['Amount'] = dRsw_prop
SW_fluxes['Altitude'] = dRsw_dctp
SW_fluxes['Optical Depth'] = dRsw_dtau
SW_fluxes['Residual'] = dRsw_resid
SW_fluxes['Sum'] = dRsw_sum

pk.dump(SW_fluxes, open(model+'_CFMIP_TrCLsw_FB_MZdecomp_Grid.pi', 'wb'))

LW_fluxes = dict()
LW_fluxes['Standard'] = dRlw_true
LW_fluxes['Amount'] = dRlw_prop
LW_fluxes['Altitude'] = dRlw_dctp
LW_fluxes['Optical Depth'] = dRlw_dtau
LW_fluxes['Residual'] = dRlw_resid
LW_fluxes['Sum'] = dRlw_sum

pk.dump(LW_fluxes, open(model+'_CFMIP_TrCLlw_FB_MZdecomp_Grid.pi', 'wb'))
