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
    picon_file = glob.glob(Source+'tos_interp_Omon_*')[1]
    x4_file = glob.glob(Source+'tos_interp_Omon_*')[0]
    print(models, 'picon - file - ', picon_file)
    print(models, '4x - file - ', x4_file)
    picon_data = nc.Dataset(picon_file)
    x4_data = nc.Dataset(x4_file)
else:
    picon_file = glob.glob(Source+'tos_interp_remap_Omon_*')[0]
    x4_file = glob.glob(Source+'tos_interp_remap_Omon_*')[1]
    print(models, 'picon - file - ', picon_file)
    print(models, '4x - file - ', x4_file)
    picon_data = nc.Dataset(picon_file)
    x4_data = nc.Dataset(x4_file)


lat = picon_data.variables['lat'][:]
lon = picon_data.variables['lon'][:]

var = ['tos']

data_picon = dict()
data_picon_stk = dict()

data_x4 = dict()
data_x4_stk = dict()

print('reading in data... ')

for i in range(len(var)):

    print('on variable,',var[i])

    if models == 'CanESM2':
        data_x4[var[i]] = x4_data.variables[var[i]][:1800,:,:]-273.15
        data_picon[var[i]] = picon_data.variables[var[i]][:1800,:,:]-273.15
    else:
        data_x4[var[i]] = x4_data.variables[var[i]][:1800,:,:]
        data_picon[var[i]] = picon_data.variables[var[i]][:1800,:,:]



    data_picon_stk[var[i]] = np.zeros([150, 12, len(lat), len(lon)])
    data_x4_stk[var[i]] = np.zeros([150, 12, len(lat), len(lon)])

    s = 0
    f = 12

    for j in range(150):
        data_picon_stk[var[i]][j, :, :, :] = np.stack(data_picon[var[i]][s:f, :, :], axis=0)
        data_x4_stk[var[i]][j, :, :, :] = np.stack(data_x4[var[i]][s:f, :, :], axis=0)

        s += 12
        f += 12

    data_picon_stk[var[i]][data_picon_stk[var[i]] > 1e5] = np.nan
    data_x4_stk[var[i]][data_x4_stk[var[i]] > 1e5] = np.nan

    data_picon_stk[var[i]] = np.nanmean(data_picon_stk[var[i]],axis=1)
    data_x4_stk[var[i]] = np.nanmean(data_x4_stk[var[i]],axis=1)

    print('final array shape - ',data_picon_stk[var[i]].shape)


print('saving...')

save_file = open(models+'_SST_picon_TmSrs.pi', 'wb')
pk.dump(data_picon_stk['tos'], save_file)
save_file.close()

save_x4file = open(models+'_SST_x4_TmSrs.pi', 'wb')
pk.dump(data_x4_stk['tos'], save_x4file)
save_x4file.close()

print('done')