#!/usr/bin/env python
## Import Packages
import numpy as np
import pandas as pd
import os
import _pickle as pk
import Area_Avg
from netCDF4 import Dataset
import glob
import scipy as sci
import Jacks_Functions as jf
import sys
from natsort import natsorted

models = sys.argv[1]

Source = '/praid/users/jgvirgin/CanESM_Data/'+models+'/Raw/'

vars = ['hus','ta']

print(Source)

files_amip = natsorted(glob.glob(Source+'*amip-piForcing_*'))[0]

print('files - ', files_amip)

nc = Dataset(files_amip)

print('filing variables into dictionaries')
control = dict()
response = dict()
ajd_amip = dict()

for v in range(len(vars)):
    print('On ', vars[v])
    if np.squeeze(nc.variables[vars[v]]).ndim == 4:

        control[vars[v]] = np.zeros([145, 12, 17, 64, 128])

        s = 0
        f = 12

        for i in range(145):
            control[vars[v]][i, :, :, :, :] = np.stack(
                np.squeeze(nc.variables[vars[v]][:1740, :17, :, :])[s:f, :, :, :], axis=0)
            s += 12
            f += 12

        print('flipping pressure dimension to match kernels')
        control[vars[v]] = control[vars[v]][:, :, ::-1, :, :]

        print('Calculate amip 1980-2010 climatology')
        response[vars[v]] = np.tile(np.mean(control[vars[v]][110:140, :, :, :, :], axis=0)[
                                    None, :, :, :, :], (145, 1, 1, 1, 1))

    else:

        control[vars[v]] = np.zeros([145, 12, 64, 128])

        s = 0
        f = 12

        for i in range(145):
            control[vars[v]][i, :, :, :] = np.stack(
                np.squeeze(nc.variables[vars[v]][:1740, :, :])[s:f, :, :], axis=0)
            s += 12
            f += 12

        print('Calculate amip 1980-2010 climatology')
        response[vars[v]] = np.tile(np.mean(control[vars[v]][110:140, :, :, :], axis=0)[
                                    None, :, :, :], (145, 1, 1, 1))

    print('Switch fill values to NaNs')
    control[vars[v]][control[vars[v]] > 1e5] = np.nan
    response[vars[v]][response[vars[v]] > 1e5] = np.nan


    print('done')

nc.close()

print('calculate dta')
dta = control['ta']-response['ta']

CMIP_File = Dataset('/praid/users/jgvirgin/Radiative_Kernels/CAM3_Kernels.nc')

lat = np.squeeze(CMIP_File.variables['lat'])
lon = np.squeeze(CMIP_File.variables['lon'])

CMIP_plevs_scalar = np.squeeze(CMIP_File.variables['lev'])
CMIP_plevs = np.tile(CMIP_plevs_scalar[None,None,:,None,None],(145,12,1,lat.size,lon.size))

print('Calculating...')
print('saturdation specific humidity for the baseline')
Sat_Hum_base = jf.Calc_SatSpec_Hum(Ta=response['ta'],P = CMIP_plevs)

print('saturation specific humidity for the response')
Sat_Hum_resp = jf.Calc_SatSpec_Hum(Ta=control['ta'],P = CMIP_plevs)

print('rate of change of saturation specific humidity with respect to temperature')
dqsdt = (Sat_Hum_resp-Sat_Hum_base)/dta

print('relative humidity for the model')
Rel_Hum = (1000*response['hus'])/Sat_Hum_base #swap specific humidity units to g/kg to match specific humidity

print('rate of change of specific humidity with respect to temperature')
dqdt = dqsdt*Rel_Hum

print('logarathmic rate of change of specific humidity with respect to temperature')
dlnqdt = dqdt/(1000*response['hus'])

print('saving')

file_save = open('dlnqdt_amip_'+models+'.pi','wb')
pk.dump(dlnqdt,file_save,protocol=-1)
file_save.close()

picon_save = open(models+'_hus_climo_amip.pi','wb')
pk.dump(response['hus'],picon_save,protocol=-1)
picon_save.close()

print('done!')
