#!/usr/bin/env python

import numpy as np
import _pickle as pk
import netCDF4 as nc
import sys
import glob

version = sys.argv[1]

exps = ['amip','amip-future4K','amip-p4K']
Source = '/praid/users/jgvirgin/CanESM_Data/'

print('reading in data')
for i in range(len(exps)):

    print('on experiment ', exps[i])
    file = glob.glob(Source+version+'/CFMIP/'+exps[i]+'/tas_*')[0]
    data = nc.Dataset(file)

    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]

    tas = np.squeeze(data.variables['tas'][360:720])

    yrs = int(len(tas[:, 0, 0])/12)
    tas_stack = np.zeros([yrs, 12, len(lat), len(lon)])

    s = 0
    f = 12

    for j in range(yrs):
        tas_stack[j, :, :, :] = np.stack(tas[s:f, :, :], axis=0)

        s += 12
        f += 12

    tas_stack[tas_stack > 1e5] = np.nan
    tas_stack = np.nanmean(tas_stack, axis=0)

    print('final array shape - ', tas_stack.shape)

    print('saving...')
    pk.dump(tas_stack, open(version+'_'+exps[i]+'_tas_Climo.pi', 'wb'))
    print('done')