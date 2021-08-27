#!/usr/bin/env python

import numpy as np
import _pickle as pk
import netCDF4 as nc
import sys
import glob

version = sys.argv[1]

exps = ['amip', 'amip-future4K', 'amip-p4K']
Source = '/praid/users/jgvirgin/CanESM_Data/'

print('reading in data')
for i in range(len(exps)):

    print('on experiment ', exps[i])
    file = glob.glob(Source+version+'/CFMIP/'+exps[i]+'/clisccp_*')[0]
    data = nc.Dataset(file)

    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]

    clisccp = np.squeeze(data.variables['clisccp'][360:720])

    yrs = int(len(clisccp[:, 0, 0, 0, 0])/12)
    clisccp_stack = np.zeros([yrs, 12, 7, 7, len(lat), len(lon)])

    s = 0
    f = 12

    for j in range(yrs):
        clisccp_stack[j, :, :, :, :, :] = np.stack(clisccp[s:f, :, :, :, :], axis=0)

        s += 12
        f += 12

    clisccp_stack[clisccp_stack > 1e5] = np.nan

    clisccp_stack = np.nanmean(clisccp_stack, axis=0)

    print('final array shape - ', clisccp_stack.shape)


    print('saving...')

    pk.dump(clisccp_stack, open(version+'_'+exps[i]+'_clisccp_Climo.pi', 'wb'))

    print('done')

print('all done')
