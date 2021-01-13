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
CanESM_4x = dict()
for i in range(2):
    if model == 'CanESM2':
        nc_picon = nc.Dataset(glob.glob(path+model+'/Raw/'+var[i]+'_Amon_CanESM2_piControl_*')[0])
        nc_4x = nc.Dataset(glob.glob(path+model+'/Raw/'+var[i]+'_Amon_CanESM2_abrupt4xCO2_*')[0])
    else:
        nc_picon = nc.Dataset(glob.glob(path+model+'/Raw/'+var[i]+'_Amon_CanESM5_piControl_*')[0])
        nc_4x = nc.Dataset(glob.glob(path+model+'/Raw/'+var[i]+'_Amon_CanESM5_abrupt-4xCO2_*')[0])

    CanESM_climo[var[i]] = np.squeeze(nc_picon.variables[var[i]])
    CanESM_4x[var[i]] = np.squeeze(nc_4x.variables[var[i]])

print('check keys - ',CanESM_climo.keys())
lat = np.squeeze(nc_picon.variables['lat'])
lon = np.squeeze(nc_picon.variables['lon'])

CanESM_climo_stk = dict()
CanESM_resp_stk = dict()
CanESM_anom = dict()

for keys in CanESM_climo.keys():
    print('on variable,',keys)
    print('time length for piControl (in years) = ', len(CanESM_climo[keys][:, 0, 0])/12)
    print('time length for 4xCO2 (in years) = ', len(CanESM_4x[keys][:, 0, 0])/12)

    nyrs = int(len(CanESM_climo[keys][:, 0, 0])/12)

    CanESM_climo_stk[keys] = np.zeros([nyrs, 12, len(lat), len(lon)])
    CanESM_resp_stk[keys] = np.zeros([150, 12, len(lat), len(lon)])

    s = 0
    f = 12

    for j in range(nyrs):
        CanESM_climo_stk[keys][j, :, :, :] = np.stack(CanESM_climo[keys][s:f, :, :], axis=0)

        s += 12
        f += 12

    s2 = 0
    f2 = 12

    for i in range(150):
        CanESM_resp_stk[keys][i, :, :, :] = np.stack(CanESM_4x[keys][s2:f2, :, :], axis=0)

        s2 += 12
        f2 += 12


    CanESM_climo_stk[keys][CanESM_climo_stk[keys] > 1e5] = np.nan
    CanESM_resp_stk[keys][CanESM_resp_stk[keys] > 1e5] = np.nan

    CanESM_climo_stk[keys] = np.tile(np.nanmean(CanESM_climo_stk[keys], axis=0)[None,:,:,:],(150,1,1,1))
    CanESM_anom[keys] = CanESM_resp_stk[keys] - CanESM_climo_stk[keys]

    print('final array shape - ', CanESM_anom[keys].shape)

CanESM_LWP = CanESM_anom['clwvi']-CanESM_anom['clivi']
CanESM_IWP = CanESM_anom['clivi']

print('take a 20 year mean...')
dLWP = np.mean(CanESM_LWP[130:,:,:,:],axis=0)
dIWP = np.mean(CanESM_IWP[130:,:,:,:], axis=0)

print('saving...')
CanESM_LWP_file = open(model+'_dLWP.pi', 'wb')
pk.dump(dLWP, CanESM_LWP_file)
CanESM_LWP_file.close()

CanESM_IWP_file = open(model+'_dIWP.pi', 'wb')
pk.dump(dIWP, CanESM_IWP_file)
CanESM_IWP_file.close()
