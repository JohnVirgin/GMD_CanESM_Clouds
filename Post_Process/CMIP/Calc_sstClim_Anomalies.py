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
import sys
from natsort import natsorted

models = sys.argv[1]

Source = '/praid/users/jgvirgin/CanESM_Data/'+models+'/Raw/'

vars = ['tas','rlut','rsds','rsdt','rsus','rsut','hus','ta']

print(Source)

if models == 'CanESM2':
    files_4xCO2 = natsorted(glob.glob(Source+'vars_interp_Amon_*'))[0]
    files_PiCon = natsorted(glob.glob(Source+'vars_interp_Amon_*'))[1]
else:
    files_4xCO2 = natsorted(glob.glob(Source+'vars_interp_Amon_*'))[1]
    files_PiCon = natsorted(glob.glob(Source+'vars_interp_Amon_*'))[2]

print('4xco2 files - ',files_4xCO2)
print('picon files - ',files_PiCon)

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

        print('Calculate 30 year average')
        PiCon[vars[v]] = np.mean(PiCon[vars[v]], axis=0)
        fourxCO2[vars[v]] = np.mean(fourxCO2[vars[v]], axis=0)

    else:

        fourxCO2[vars[v]] = np.zeros([30,12,64,128])
        PiCon[vars[v]] = np.zeros([30,12,64,128])

        s=0
        f=12

        for i in range(30):
            fourxCO2[vars[v]][i,:,:,:] = np.stack(\
            np.squeeze(nc_4xCO2.variables[vars[v]][:360,:,:])[s:f,:,:],axis=0)

            PiCon[vars[v]][i,:,:,:] = np.stack(\
            np.squeeze(nc_PiCon.variables[vars[v]][:360,:,:])[s:f,:,:],axis=0)

            s+=12
            f+=12

        print('Calculate 30 year average')
        PiCon[vars[v]] = np.mean(PiCon[vars[v]], axis=0)
        fourxCO2[vars[v]] = np.mean(fourxCO2[vars[v]], axis=0)

    print('Switch fill values to NaNs')
    fourxCO2[vars[v]][fourxCO2[vars[v]] > 1e5] = np.nan
    PiCon[vars[v]][PiCon[vars[v]] > 1e5] = np.nan

    print('adjusting 4xco2 data as anomalies relative to preindustrial control climatology')

    ajd_4xCO2[vars[v]] = fourxCO2[vars[v]]-PiCon[vars[v]]

    print('done')

nc_4xCO2.close()
nc_PiCon.close()

print('Calculate surface albedo from downwelling and upwelling short wave fluxes')
PiCon['Alb'] = PiCon['rsus']/PiCon['rsds']
fourxCO2['Alb'] = fourxCO2['rsus']/fourxCO2['rsds']

ajd_4xCO2['Alb'] = fourxCO2['Alb']-PiCon['Alb']

ajd_4xCO2.pop('rsus')
ajd_4xCO2.pop('rsds')

print('check keys')
print(ajd_4xCO2.keys())

print('saving')

file_save = open(models+'_Ajd_sstClim4xCO2.pi','wb')
pk.dump(ajd_4xCO2,file_save,protocol=-1)
file_save.close()

print('done!')
