#!/usr/bin/env python

import numpy as np
import _pickle as pk
import netCDF4 as nc
import sys
import glob

version = sys.argv[1]

exps = ['amip']
Source = '/praid/users/jgvirgin/CanESM_Data/'

print('reading in data')
for i in range(len(exps)):

    print('on experiment ', exps[i])
    file_up = glob.glob(Source+version+'/CFMIP/'+exps[i]+'/rsuscs_Amon_*')[0]
    file_down = glob.glob(Source+version+'/CFMIP/'+exps[i]+'/rsdscs_Amon_*')[0]
    data_up = nc.Dataset(file_up)
    data_down = nc.Dataset(file_down)

    lat = data_up.variables['lat'][:]
    lon = data_up.variables['lon'][:]

    rsuscs = np.squeeze(data_up.variables['rsuscs'][360:720])
    rsdscs = np.squeeze(data_down.variables['rsdscs'][360:720])

    yrs = int(len(rsuscs[:, 0, 0])/12)
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

    pk.dump(albcs, open(version+'_'+exps[i]+'_albcs_Climo.pi', 'wb'))

    print('done')
