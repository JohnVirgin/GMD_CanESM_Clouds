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

vars = ['tas','rlut','rlutcs','rsds','rsdt','rsus','rsut','rsutcs','hus','ta']

print(Source)

files_amip = natsorted(glob.glob(Source+'*amip-piForcing_*'))[0]

print('files - ',files_amip)

nc = Dataset(files_amip)

print('filing variables into dictionaries')
control= dict()
response = dict()
ajd_amip = dict()

for v in range(len(vars)):
    print('On ',vars[v])
    if np.squeeze(nc.variables[vars[v]]).ndim == 4:

        control[vars[v]] = np.zeros([145,12,17,64,128])

        s=0
        f=12

        for i in range(145):
            control[vars[v]][i,:,:,:,:] = np.stack(\
            np.squeeze(nc.variables[vars[v]][:1740,:17,:,:])[s:f,:,:,:],axis=0)


            s+=12
            f+=12

        print('flipping pressure dimension to match kernels')
        control[vars[v]] = control[vars[v]][:,:,::-1,:,:]

        print('Calculate amip 1980-2010 climatology')
        response[vars[v]] = np.tile(np.mean(control[vars[v]][110:140,:,:,:,:], axis=0)[None,:,:,:,:],(145,1,1,1,1))

    else:

        control[vars[v]] = np.zeros([145, 12, 64, 128])

        s=0
        f=12

        for i in range(145):
            control[vars[v]][i,:,:,:] = np.stack(\
            np.squeeze(nc.variables[vars[v]][:1740,:,:])[s:f,:,:],axis=0)


            s+=12
            f+=12

        print('Calculate pre-industrial control climatology')
        response[vars[v]] = np.tile(np.mean(control[vars[v]][110:140,:,:,:], axis=0)[None,:,:,:],(145,1,1,1))

    print('Switch fill values to NaNs')
    control[vars[v]][control[vars[v]] > 1e5] = np.nan
    response[vars[v]][response[vars[v]] > 1e5] = np.nan

    print('adjusting amip-picontrol data relative to 1980-2010 mean')

    ajd_amip[vars[v]] = control[vars[v]]-response[vars[v]]

    print('done')

nc.close()

print('Calculate surface albedo from downwelling and upwelling short wave fluxes')
response['Alb'] = response['rsus']/response['rsds']
control['Alb'] = control['rsus']/control['rsds']

ajd_amip['Alb'] = control['Alb']-response['Alb']

ajd_amip.pop('rsus')
ajd_amip.pop('rsds')

print('check keys')
print(ajd_amip.keys())

print('saving')

file_save = open(models+'_Ajd_amip-piControl_TmSrs.pi','wb')
pk.dump(ajd_amip,file_save,protocol=-1)
file_save.close()

print('done!')
