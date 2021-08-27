#!/usr/bin/env python

import numpy as np
import _pickle as pk
import netCDF4 as nc
import sys
import glob

version = sys.argv[1]

exps = ['amip','amip-future4K','amip-p4K']
var = 'ts'
Source = '/praid/users/jgvirgin/CanESM_Data/'

data = {}
data_stk = {}
for i in range(len(exps)):

    print('reading in data on experiment ',exps[i])
    file = glob.glob(Source+version+'/CFMIP/'+exps[i]+'/'+var+'_*')[0]

    if version == 'CanESM2':
        data[exps[i]] = np.squeeze(nc.Dataset(file).variables[var][360:])
    else:
        data[exps[i]] = np.squeeze(nc.Dataset(file).variables[var][360:720])

    nyr = int(len(data[exps[i]][:])/12)

    print('Checking shape of variable ',var,'\narray shape - ',data[exps[i]].shape)
    data[exps[i]][data[exps[i]] > 1e5] = np.nan


    print('Stacking')
    data_stk[exps[i]] = np.zeros([nyr,12,64,128])
    s = 0
    f = 12

    for j in range(nyr):
        data_stk[exps[i]][j] = np.stack(data[exps[i]][s:f],axis=0)

        s+=12
        f+=12

    print('take climatological average')
    data_stk[exps[i]] = np.nanmean(data_stk[exps[i]],axis=0)

    print('Checking shape of variable ',var,'\narray shape - ',data_stk[exps[i]].shape)

print('Calculating Anomalies')
delta = {}
delta['Uniform'] = data_stk['amip-p4K']-data_stk['amip']
delta['Pattern'] = data_stk['amip-future4K']-data_stk['amip']

print('saving')
pk.dump(delta,open(version+'_dSST.pi','wb'))
