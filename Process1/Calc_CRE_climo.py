#!/usr/bin/env python
# coding: utf-8

## Import Packages

import numpy as np
import _pickle as pk
import netCDF4 as nc
import sys
import glob
from natsort import natsorted

models = sys.argv[1]

Source = '/praid/users/jgvirgin/CanESM_Data/'+models+'/Raw/'

if models == 'CanESM2':
    picon_file = natsorted(glob.glob(Source+'vars_interp_*'))[3:]
    print(models, '- files - ', picon_file)
    picon_data = nc.MFDataset(picon_file)
else:
    picon_file = natsorted(glob.glob(Source+'vars_interp_*'))[3]
    print(models, '- files - ', picon_file)
    picon_data = nc.Dataset(picon_file)


var = ['tas','rlut','rlutcs','rsdt','rsut','rsutcs']

data = dict()
data_stk = dict()

print('reading in data... ')

for i in range(len(var)):

    print('on variable,',var[i])
    data[var[i]] = picon_data.variables[var[i]][:]

    print('time length (in years) = ', len(data[var[i]][:, 0, 0])/12)

    nyrs = int(len(data[var[i]][:, 0, 0])/12)

    data_stk[var[i]] = np.zeros([nyrs, 12, 64, 128])

    s = 0
    f = 12

    for j in range(nyrs):
        data_stk[var[i]][j, :, :, :] = np.stack(data[var[i]][s:f, :, :], axis=0)

        s += 12
        f += 12

    data_stk[var[i]][data_stk[var[i]] > 1e5] = np.nan

    data_stk[var[i]] = np.nanmean(data_stk[var[i]],axis=0)

    print('final array shape - ',data_stk[var[i]].shape)

data_stk['rsnt'] = data_stk['rsdt']-data_stk['rsut']
data_stk['rsntcs'] = data_stk['rsdt']-data_stk['rsutcs']


data_stk['CRE_lw'] = data_stk['rlut']-data_stk['rlutcs']
data_stk['CRE_sw'] = data_stk['rsnt']-data_stk['rsntcs']

data_stk.pop('rlut')
data_stk.pop('rlutcs')
data_stk.pop('rsdt')
data_stk.pop('rsut')
data_stk.pop('rsutcs')
data_stk.pop('rsnt')
data_stk.pop('rsntcs')

print('saving...')

save_file = open(models+'_CRE_Climo.pi', 'wb')
pk.dump(data_stk, save_file)
save_file.close()

print('done')
