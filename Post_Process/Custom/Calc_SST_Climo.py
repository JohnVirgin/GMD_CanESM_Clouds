#!/usr/bin/env python
# coding: utf-8

## Import Packages

import numpy as np
import _pickle as pk
import netCDF4 as nc
import sys
import glob
from natsort import natsorted

exps = ['iga-sst-1xco2','iga-sst-4xco2','idj-sst-4xco2','iga-dsst-4xco2','5pi-sst-1xco2','5a4-sst-4xco2','5pi-dsst-4xco2']
var = ['ts']

print('reading in data')
for i in range(len(exps)):

    print('on experiment ',exps[i])
    Source = '/praid/users/jgvirgin/CanESM_Data/CanESM5_p2/Custom/'+exps[i]
    file = glob.glob(Source+'/ts_Amon_*')[0]
    data = nc.Dataset(file)

    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]

    ts = np.squeeze(data.variables['ts'])

    yrs = int(len(ts[:,0,0])/12)
    ts_stack = np.zeros([yrs, 12, len(lat), len(lon)])

    s = 0
    f = 12

    for j in range(yrs):
        ts_stack[j, :, :, :] = np.stack(ts[s:f, :, :], axis=0)

        s += 12
        f += 12

    ts_stack[ts_stack > 1e5] = np.nan

    ts_stack = np.nanmean(ts_stack, axis=0)

    print('final array shape - ', ts_stack.shape)


    print('saving...')

    save_file = open('CanESM5_p2_'+exps[i]+'_SST_Climo.pi', 'wb')
    pk.dump(ts_stack, save_file)
    save_file.close()

    print('done')

print('all done')