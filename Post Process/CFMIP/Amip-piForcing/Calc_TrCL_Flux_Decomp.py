#!/usr/bin/env python
# coding: utf-8

## Import Packages
from scipy.interpolate import interp1d
import numpy as np
import _pickle as pk
import Area_Avg as aa
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
    mn_range_f = [300, 600, 900, 1200, 1500, 1740]

    yr_range = int((mn_range_f[step_n_int] - mn_range_s[step_n_int])/12)

print('Read in cloud fraction output')

cl_source = '/praid/users/jgvirgin/CanESM_Data/'+model+'/Raw/'
clissp_path = natsorted(glob.glob(cl_source+'*_amip-piForcing_*'))[0]

print('amip-piForcing path - ',clissp_path)

clisccp = np.squeeze(Dataset(clissp_path).variables['clisccp'])
clisccp[clisccp > 1e10] = np.nan

nmon = clisccp[:, 0, 0, 0, 0].size
nyr = int(nmon/12)
print('number of total years read in - ',nyr)
print('number of chunked years read in - ', yr_range)

print(' clisccp shape - ', clisccp.shape)

print('Calculate anomalies')
#convert time dimensions to months/years, then take the climatological mean
clisccp1_stacked = np.zeros([nyr,12,7,7,64,128])
clisccp2_stacked = np.zeros([yr_range,12,7,7,64,128])

s = 0
f = 12

for i in range(nyr):
    clisccp1_stacked[i, :, :, :, :, :] = np.stack(clisccp[s:f, :, :, :, :], axis=0)

    s += 12
    f += 12

s = mn_range_s[step_n_int]
f = mn_range_s[step_n_int]+12

for i in range(yr_range):
    clisccp2_stacked[i, :, :, :, :, :] = np.stack(clisccp[s:f, :, :, :, :], axis=0)

    s += 12
    f += 12

avgclisccp1 = np.tile(np.mean(clisccp1_stacked, axis=0)[None, :, :, :, :, :], (yr_range, 1, 1, 1, 1, 1))

# Compute clisccp anomalies
dc = clisccp2_stacked - avgclisccp1


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
    c1_sum[sect[i]] = np.tile(np.nansum(avgclisccp1[:, :, :, sections_ind[i], :, :], axis=(
        2, 3))[:, :, None, None, :, :], (1, 1, 7, sections_length[i], 1, 1))

    #some the change in cloud fractions over CTP and tau
    dc_sum[sect[i]] = np.tile(np.nansum(dc[:, :, :, sections_ind[i], :, :], axis=(
        2, 3))[:, :, None, None, :, :], (1, 1, 7, sections_length[i], 1, 1))

    #change in cloud fraction due to proportional changes in clouds, but not due to CTP or optical depth
    dc_prop[sect[i]] = avgclisccp1[:, :, :, sections_ind[i], :, :] * \
        (dc_sum[sect[i]]/c1_sum[sect[i]])

    #changes in optical depth and CTP, but with proportional changes in cloud fraction fixed
    dc_star[sect[i]] = dc[:, :, :, sections_ind[i], :, :] - dc_prop[sect[i]]


#decompose the LW kernel now


print('read in surface albedo output')
#read in clear sky shortwave fluxes at the surface for the same time period

raw_file = '/praid/users/jgvirgin/CanESM_Data/'+model+'/Raw/'
albcs_source = natsorted(glob.glob(raw_file+'*amip-piForcing_*'))[1]

print('albedo source output file - ', albcs_source)

rsdscs1 = np.squeeze(Dataset(albcs_source).variables['rsdscs'])[:360, :, :]
rsuscs1 = np.squeeze(Dataset(albcs_source).variables['rsuscs'])[:360, :, :]
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
        function = interp1d(
            albcs, SWkernel_interp[m, :, :, la, :], axis=2, kind='linear')
        new_kernel_lon = function(alb_lon)
        SWkernel_map[m, :, :, la, :] = new_kernel_lon

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

    Klw0[sect[i]] = np.zeros([yr_range, 12, 7, sections_length[i], 64, 128])
    Klw_prime[sect[i]] = np.zeros([yr_range, 12, 7, sections_length[i], 64, 128])

    for j in range(yr_range):

        #sum the kernels CTP and optical depth bins and weight it by total cloud cover fraction
        #this not term is a kernel to estimate feedbacks from changes in cloud cover alone
        Klw0[sect[i]][j, :, :, :, :, :] = np.tile(np.nansum(
            LWkernel_map[:, :, sections_ind[i], :, :]*(avgclisccp1[j, :, :, sections_ind[i], :, :]/c1_sum[sect[i]][j, :, :, :, :, :]), axis=(1, 2))[:, None, None, :, :], (1, 7, sections_length[i], 1, 1))

    #prime represents a kernel to estimate feedbacks from changes in CTP and tau
        Klw_prime[sect[i]][j, :, :, :, :, :] = LWkernel_map[:, :,
                                                            sections_ind[i], :, :] - Klw0[sect[i]][j, :, :, :, :, :]

    #we further decompose the CTP & tau kernel into separate kernels for both
    #CTP
    lw_this[sect[i]] = np.nansum(
        Klw_prime[sect[i]]*np.tile(np.nansum(avgclisccp1[:, :, :, sections_ind[i], :, :]/c1_sum[sect[i]], axis=3)[:, :, :, None, :, :], (1, 1, 1, sections_length[i], 1, 1)), axis=2)

    #tau
    lw_that[sect[i]] = np.nansum(
        Klw_prime[sect[i]]*np.tile(np.nansum(avgclisccp1[:, :, :, sections_ind[i], :, :]/c1_sum[sect[i]], axis=2)[:, :, None, :, :, :], (1, 1, 7, 1, 1, 1)), axis=3)

    Klw_p_prime[sect[i]] = np.tile(
        lw_this[sect[i]][:, :, None, :, :, :], (1, 1, 7, 1, 1, 1))
    Klw_t_prime[sect[i]] = np.tile(
        lw_that[sect[i]][:, :, :, None, :, :], (1, 1, 1, sections_length[i], 1, 1))

    #residual
    Klw_resid_prime[sect[i]] = Klw_prime[sect[i]] - \
        Klw_p_prime[sect[i]] - Klw_t_prime[sect[i]]


# In[29]:

print('longwave feedbacks')

#now compute decomposed fluxes
dRlw_true = dict()
dRlw_prop = dict()
dRlw_dctp = dict()
dRlw_dtau = dict()
dRlw_resid = dict()
dRlw_sum = dict()
for i in range(len(sect)):

    dRlw_true[sect[i]] = np.zeros([yr_range, 12, 64, 128])

    for j in range(yr_range):

        # lw total
        dRlw_true[sect[i]][j, :, :, :] = np.nansum(
            LWkernel_map[:, :, sections_ind[i], :, :]*dc[j, :, :, sections_ind[i], :, :], axis=(1, 2))

    # lw amount component
    dRlw_prop[sect[i]] = (Klw0[sect[i]][:, :, 0, 0, :, :]
                          * dc_sum[sect[i]][:, :, 0, 0, :, :])
    # lw altitude component
    dRlw_dctp[sect[i]] = np.nansum(
        Klw_p_prime[sect[i]]*dc_star[sect[i]], axis=(2, 3))
    # lw optical depth component
    dRlw_dtau[sect[i]] = np.nansum(
        Klw_t_prime[sect[i]]*dc_star[sect[i]], axis=(2, 3))
    # lw residual
    dRlw_resid[sect[i]] = np.nansum(
        Klw_resid_prime[sect[i]]*dc_star[sect[i]], axis=(2, 3))
    # sum should equal true
    dRlw_sum[sect[i]] = dRlw_prop[sect[i]] + dRlw_dctp[sect[i]] + \
        dRlw_dtau[sect[i]] + dRlw_resid[sect[i]]


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

    Ksw0[sect[i]] = np.zeros([yr_range, 12, 7, sections_length[i], 64, 128])
    Ksw_prime[sect[i]] = np.zeros([yr_range, 12, 7, sections_length[i], 64, 128])

    for j in range(yr_range):

        #sum the kernels CTP and optical depth bins and weight it by total cloud cover fraction
        #this not term is a kernel to estimate feedbacks from changes in cloud cover alone
        Ksw0[sect[i]][j, :, :, :, :, :] = np.tile(np.nansum(
            SWkernel_map[:, :, sections_ind[i], :, :]*(avgclisccp1[j, :, :, sections_ind[i], :, :]/c1_sum[sect[i]][j, :, :, :, :, :]), axis=(1, 2))[:, None, None, :, :], (1, 7, sections_length[i], 1, 1))

    #prime represents a kernel to estimate feedbacks from changes in CTP and tau
        Ksw_prime[sect[i]][j, :, :, :, :, :] = SWkernel_map[:, :,
                                                            sections_ind[i], :, :] - Ksw0[sect[i]][j, :, :, :, :, :]

    #we further decompose the CTP & tau kernel into separate kernels for both
    #CTP
    sw_this[sect[i]] = np.nansum(
        Ksw_prime[sect[i]]*np.tile(np.nansum(avgclisccp1[:, :, :, sections_ind[i], :, :]/c1_sum[sect[i]], axis=3)[:, :, :, None, :, :], (1, 1, 1, sections_length[i], 1, 1)), axis=2)

    #tau
    sw_that[sect[i]] = np.nansum(
        Ksw_prime[sect[i]]*np.tile(np.nansum(avgclisccp1[:, :, :, sections_ind[i], :, :]/c1_sum[sect[i]], axis=2)[:, :, None, :, :, :], (1, 1, 7, 1, 1, 1)), axis=3)

    Ksw_p_prime[sect[i]] = np.tile(
        sw_this[sect[i]][:, :, None, :, :, :], (1, 1, 7, 1, 1, 1))
    Ksw_t_prime[sect[i]] = np.tile(
        sw_that[sect[i]][:, :, :, None, :, :], (1, 1, 1, sections_length[i], 1, 1))

    #residual
    Ksw_resid_prime[sect[i]] = Ksw_prime[sect[i]] - \
        Ksw_p_prime[sect[i]] - Ksw_t_prime[sect[i]]


print('shortwave feedbacks...')
#now compute decomposed fluxes
dRsw_true = dict()
dRsw_prop = dict()
dRsw_dctp = dict()
dRsw_dtau = dict()
dRsw_resid = dict()
dRsw_sum = dict()
for i in range(len(sect)):

    dRsw_true[sect[i]] = np.zeros([yr_range, 12, 64, 128])

    for j in range(yr_range):

        # sw total
        dRsw_true[sect[i]][j, :, :, :] = np.nansum(
            SWkernel_map[:, :, sections_ind[i], :, :]*dc[j, :, :, sections_ind[i], :, :], axis=(1, 2))

    # sw amount component
    dRsw_prop[sect[i]] = (Ksw0[sect[i]][:, :, 0, 0, :, :]
                          * dc_sum[sect[i]][:, :, 0, 0, :, :])
    # sw altitude component
    dRsw_dctp[sect[i]] = np.nansum(
        Ksw_p_prime[sect[i]]*dc_star[sect[i]], axis=(2, 3))
    # sw optical depth component
    dRsw_dtau[sect[i]] = np.nansum(
        Ksw_t_prime[sect[i]]*dc_star[sect[i]], axis=(2, 3))
    # sw residual
    dRsw_resid[sect[i]] = np.nansum(
        Ksw_resid_prime[sect[i]]*dc_star[sect[i]], axis=(2, 3))
    # sum should equal true
    dRsw_sum[sect[i]] = dRsw_prop[sect[i]] + dRsw_dctp[sect[i]] + \
        dRsw_dtau[sect[i]] + dRsw_resid[sect[i]]

    #dRsw_true[sect[i]][night]=0
    #dRsw_prop[sect[i]][night]=0
    #dRsw_dctp[sect[i]][night]=0
    #dRsw_dtau[sect[i]][night]=0
    #dRsw_resid[sect[i]][night]=0
    #dRsw_sum[sect[i]][night]=0


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

SW_fluxes_file = open(model+'_'+step_n+'_TrCLsw_FLUX_MZdecomp_Grid.pi', 'wb')
pk.dump(SW_fluxes, SW_fluxes_file)
SW_fluxes_file.close()

LW_fluxes = dict()
LW_fluxes['Standard'] = dRlw_true
LW_fluxes['Amount'] = dRlw_prop
LW_fluxes['Altitude'] = dRlw_dctp
LW_fluxes['Optical Depth'] = dRlw_dtau
LW_fluxes['Residual'] = dRlw_resid
LW_fluxes['Sum'] = dRlw_sum

LW_fluxes_file = open(model+'_'+step_n+'_TrCLlw_FLUX_MZdecomp_Grid.pi', 'wb')
pk.dump(LW_fluxes, LW_fluxes_file)
LW_fluxes_file.close()
