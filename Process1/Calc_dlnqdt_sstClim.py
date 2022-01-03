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

if models == 'CanESM2':
    files_4xCO2 = natsorted(glob.glob(Source+'vars_interp_Amon_*'))[0]
    files_PiCon = natsorted(glob.glob(Source+'vars_interp_Amon_*'))[1]
else:
    files_4xCO2 = natsorted(glob.glob(Source+'vars_interp_Amon_*'))[1]
    files_PiCon = natsorted(glob.glob(Source+'vars_interp_Amon_*'))[2]

print(files_4xCO2)
print(files_PiCon)

nc_4xCO2 = Dataset(files_4xCO2)
nc_PiCon = Dataset(files_PiCon)

print('filing variables into dictionaries')
PiCon= dict()
fourxCO2 = dict()
ajd_4xCO2 = dict()

for v in range(len(vars)):
    print('On ',vars[v])
    if np.squeeze(nc_4xCO2.variables[vars[v]]).ndim == 4:

        fourxCO2[vars[v]] = np.zeros([30,12,17,64,128])
        PiCon[vars[v]] = np.zeros([30,12,17,64,128])

        s=0
        f=12

        for i in range(30):
            fourxCO2[vars[v]][i,:,:,:,:] = np.stack(\
            np.squeeze(nc_4xCO2.variables[vars[v]][:360,:17,:,:])[s:f,:,:,:],axis=0)

            PiCon[vars[v]][i,:,:,:,:] = np.stack(\
            np.squeeze(nc_PiCon.variables[vars[v]][:360,:17,:,:])[s:f,:,:,:],axis=0)

            s+=12
            f+=12

    print('flipping pressure dimension to match kernels')
    fourxCO2[vars[v]] = fourxCO2[vars[v]][:,:,::-1,:,:]
    PiCon[vars[v]] = PiCon[vars[v]][:,:,::-1,:,:]

    print('Switch fill values to NaNs')
    fourxCO2[vars[v]][fourxCO2[vars[v]] > 1e5] = np.nan
    PiCon[vars[v]][PiCon[vars[v]] > 1e5] = np.nan

    print('Calculate climatologies')
    PiCon[vars[v]] = np.mean(PiCon[vars[v]], axis=0)
    fourxCO2[vars[v]] = np.mean(fourxCO2[vars[v]], axis=0)


    print('done')

nc_4xCO2.close()
nc_PiCon.close()

print('calculate dta')
dta = fourxCO2['ta']-PiCon['ta']

CMIP_File = Dataset('/praid/users/jgvirgin/Radiative_Kernels/CAM3_Kernels.nc')

lat = np.squeeze(CMIP_File.variables['lat'])
lon = np.squeeze(CMIP_File.variables['lon'])

CMIP_plevs_scalar = np.squeeze(CMIP_File.variables['lev'])
CMIP_plevs = np.tile(CMIP_plevs_scalar[None,:,None,None],(12,1,lat.size,lon.size))

print('Calculating...')
print('saturdation specific humidity for the baseline')
Sat_Hum_base = jf.Calc_SatSpec_Hum(Ta=PiCon['ta'],P = CMIP_plevs)

print('saturation specific humidity for the response')
Sat_Hum_resp = jf.Calc_SatSpec_Hum(Ta=fourxCO2['ta'],P = CMIP_plevs)

print('rate of change of saturation specific humidity with respect to temperature')
dqsdt = (Sat_Hum_resp-Sat_Hum_base)/dta

print('relative humidity for the model')
Rel_Hum = (1000*PiCon['hus'])/Sat_Hum_base #swap specific humidity units to g/kg to match specific humidity

print('rate of change of specific humidity with respect to temperature')
dqdt = dqsdt*Rel_Hum

print('logarathmic rate of change of specific humidity with respect to temperature')
dlnqdt = dqdt/(1000*PiCon['hus'])

print('saving')

file_save = open('dlnqdt_'+models+'_sstClim.pi','wb')
pk.dump(dlnqdt,file_save,protocol=-1)
file_save.close()

picon_save = open(models+'_hus_climo_sstClim_picon.pi','wb')
pk.dump(PiCon['hus'],picon_save,protocol=-1)
picon_save.close()

print('done!')
