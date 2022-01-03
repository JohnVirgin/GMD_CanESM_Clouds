#!/usr/bin/env python
## Import Packages

import numpy as np
import pandas as pd
import os
import _pickle as pk
import Area_Avg
import netCDF4 as nc
import glob
import scipy as sci
import sys
from natsort import natsorted

model = sys.argv[1]

path = '/praid/users/jgvirgin/CanESM_Data/'
var  = ['clivi','clwvi']
print('read in output')

CanESM_climo = dict()

for i in range(2):
    if model == 'CanESM2':
        nc_picon_2 = nc.Dataset(glob.glob(path+model+'/Raw/'+var[i]+'_Amon_CanESM2_piControl_*')[0])
    else:
        nc_picon_2 = nc.Dataset(glob.glob(path+model+'/Raw/'+var[i]+'_Amon_CanESM5_piControl_*')[0])

    CanESM_climo[var[i]] = np.squeeze(nc_picon_2.variables[var[i]])

print('check keys - ',CanESM_climo.keys())
lat = np.squeeze(nc_picon_2.variables['lat'])
lon = np.squeeze(nc_picon_2.variables['lon'])

CanESM_climo_stk = dict()

for keys in CanESM_climo.keys():
    print('on variable,',keys)

    print('time length (in years) = ', len(CanESM_climo[keys][:, 0, 0])/12)

    nyrs = int(len(CanESM_climo[keys][:, 0, 0])/12)

    CanESM_climo_stk[keys] = np.zeros([nyrs, 12, len(lat), len(lon)])

    s = 0
    f = 12

    for j in range(nyrs):
        CanESM_climo_stk[keys][j, :, :, :] = np.stack(CanESM_climo[keys][s:f, :, :], axis=0)

        s += 12
        f += 12

    CanESM_climo_stk[keys][CanESM_climo_stk[keys] > 1e5] = np.nan

    CanESM_climo_stk[keys] = np.nanmean(CanESM_climo_stk[keys], axis=0)

    print('final array shape - ', CanESM_climo_stk[keys].shape)

CanESM_LWP = CanESM_climo_stk['clwvi']-CanESM_climo_stk['clivi']
CanESM_IWP = CanESM_climo_stk['clivi']

print('saving')

CanESM_LWP_file = open(model+'_LWP_Climo.pi', 'wb')
pk.dump(CanESM_LWP, CanESM_LWP_file)
CanESM_LWP_file.close()

CanESM_IWP_file = open(model+'_IWP_Climo.pi', 'wb')
pk.dump(CanESM_IWP, CanESM_IWP_file)
CanESM_IWP_file.close()

