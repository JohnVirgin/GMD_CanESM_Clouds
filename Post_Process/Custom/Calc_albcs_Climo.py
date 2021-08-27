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

print('reading in data')
for i in range(len(exps)):

    print('on experiment ',exps[i])
    Source = '/praid/users/jgvirgin/CanESM_Data/CanESM5_p2/Custom/'+exps[i]
    file_up = glob.glob(Source+'/rsuscs_Amon_*')[0]
    file_down = glob.glob(Source+'/rsdscs_Amon_*')[0]
    data_up = nc.Dataset(file_up)
    data_down = nc.Dataset(file_down)

    lat = data_up.variables['lat'][:]
    lon = data_up.variables['lon'][:]

    rsuscs = np.squeeze(data_up.variables['rsuscs'])
    rsdscs = np.squeeze(data_down.variables['rsdscs'])

    yrs = int(len(rsuscs[:,0,0])/12)
    rsuscs_stack = np.zeros([yrs, 12, len(lat), len(lon)])
    rsdscs_stack = np.zeros([yrs, 12, len(lat), len(lon)])

    s = 0
    f = 12

    for j in range(yrs):
        rsuscs_stack[j, :, :, :] = np.stack(rsuscs[s:f, :, :], axis=0)
        rsdscs_stack[j, :, :, :] = np.stack(rsdscs[s:f, :, :], axis=0)

        s += 12
        f += 12

    rsuscs_stack[rsuscs_stack > 1e5] = np.nan
    rsdscs_stack[rsdscs_stack > 1e5] = np.nan

    rsuscs_stack = np.nanmean(rsuscs_stack, axis=0)
    rsdscs_stack = np.nanmean(rsdscs_stack, axis=0)

    albcs = rsuscs_stack/rsdscs_stack
    print('final array shape - ', albcs.shape)


    print('saving...')

    save_file = open('CanESM5_p2_'+exps[i]+'_albcs_Climo.pi', 'wb')
    pk.dump(albcs, save_file)
    save_file.close()

    print('done')

print('all done')
