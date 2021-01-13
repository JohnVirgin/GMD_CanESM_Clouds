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
    picon_file = natsorted(glob.glob(Source+'vars_interp_Omon*'))[1]
    print(models, '- files - ', picon_file)
    picon_data = nc.MFDataset(picon_file)
else:
    picon_file = natsorted(glob.glob(Source+'vars_interp_Omon*'))[2]
    print(models, '- files - ', picon_file)
    picon_data = nc.MFDataset(picon_file)

lat = picon_data.variables['lat'][:]
lon = picon_data.variables['lon'][:]

var = ['tos']

data = dict()
data_stk = dict()

print('reading in data... ')

for i in range(len(var)):

    print('on variable,',var[i])
    data[var[i]] = picon_data.variables[var[i]][:]

    print('time length (in years) = ', len(data[var[i]][:, 0, 0])/12)

    nyrs = int(len(data[var[i]][:, 0, 0])/12)

    data_stk[var[i]] = np.zeros([nyrs, 12, len(lat), len(lon)])

    s = 0
    f = 12

    for j in range(nyrs):
        data_stk[var[i]][j, :, :, :] = np.stack(data[var[i]][s:f, :, :], axis=0)

        s += 12
        f += 12

    data_stk[var[i]][data_stk[var[i]] > 1e5] = np.nan

    data_stk[var[i]] = np.nanmean(data_stk[var[i]],axis=0)

    print('final array shape - ',data_stk[var[i]].shape)


print('saving...')

save_file = open(models+'_SST_Climo.pi', 'wb')
pk.dump(data_stk['tos'], save_file)
save_file.close()

print('done')
