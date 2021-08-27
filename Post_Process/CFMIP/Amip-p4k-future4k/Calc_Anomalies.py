#!/usr/bin/env python

import numpy as np
import _pickle as pk
import netCDF4 as nc
import sys
import glob

version = sys.argv[1]

exps = ['amip','amip-future4K','amip-p4K']
var = ['tas','hus','ta','rsus','rsds']
Source = '/praid/users/jgvirgin/CanESM_Data/'


data = {}
data_stk = {}

#define grid and weights
lat = np.linspace(-87.864, 87.864, 64)
lon = np.linspace(0, 357.1875, 128)

y = lat*np.pi/180
coslat = np.cos(y)
coslat = np.tile(coslat[:,None], (1,lon.size))

for i in range(len(exps)):
    data[exps[i]] = {}
    data_stk[exps[i]] = {}
    print('\nreading in data on experiment ', exps[i],'\n')
    for v in range(len(var)):
        print('on variable ',var[v])
        file = glob.glob(Source+version+'/CFMIP/'+exps[i]+'/'+var[v]+'_*')[0]

        print('using file - \n',file)
        if version == 'CanESM2':
            data[exps[i]][var[v]] = np.squeeze(nc.Dataset(file).variables[var[v]][360:])
        else:
            data[exps[i]][var[v]] = np.squeeze(nc.Dataset(file).variables[var[v]][360:720])

        nyr = int(len(data[exps[i]][var[v]][:])/12)

        print('Checking shape of variable ',var[v],'\narray shape - ',data[exps[i]][var[v]].shape)
        data[exps[i]][var[v]][data[exps[i]][var[v]] > 1e5] = np.nan


        print('Stacking')
        if data[exps[i]][var[v]].ndim == 3:
            data_stk[exps[i]][var[v]] = np.zeros([nyr, 12, 64, 128])
        else:
            data[exps[i]][var[v]] = data[exps[i]][var[v]][:,:17,:,:]
            data[exps[i]][var[v]] = data[exps[i]][var[v]][:,::-1,:,:]
            data_stk[exps[i]][var[v]] = np.zeros([nyr, 12, 17, 64, 128])
        s = 0
        f = 12

        for j in range(nyr):
            data_stk[exps[i]][var[v]][j] = np.stack(data[exps[i]][var[v]][s:f], axis=0)

            s+=12
            f+=12

        print('take climatological average')
        data_stk[exps[i]][var[v]] = np.nanmean(data_stk[exps[i]][var[v]], axis=0)

        print('Checking shape of variable ',var[v],'\narray shape - ',data_stk[exps[i]][var[v]].shape)

print('Calculating surface albedo metric')

for i in range(len(exps)):
    data_stk[exps[i]]['alb'] = data_stk[exps[i]]['rsus']/data_stk[exps[i]]['rsds']

    data_stk[exps[i]].pop('rsus')
    data_stk[exps[i]].pop('rsds')


print('Calculate Anomalies')

anom = {}
anom['Uniform'] = {}
anom['Pattern'] = {}
for keys in data_stk[exps[0]].keys():
    anom['Uniform'][keys] = data_stk['amip-p4K'][keys]-data_stk['amip'][keys]
    anom['Pattern'][keys] = data_stk['amip-future4K'][keys]-data_stk['amip'][keys]

anom['Uniform']['tas_gam'] = np.average(np.nanmean(anom['Uniform']['tas'],axis=0),weights=coslat)
anom['Pattern']['tas_gam'] = np.average(np.nanmean(anom['Pattern']['tas'],axis=0),weights=coslat)

print('global, annual mean surface temperature response?\n',\
    anom['Uniform']['tas_gam'],'\n',anom['Pattern']['tas_gam'])

print('normalize anomlies by global mean tas change')
for keys in anom['Pattern'].keys():
    if keys == 'tas_gam':
        pass
    else:
        anom['Uniform'][keys] = anom['Uniform'][keys]/anom['Uniform']['tas_gam']
        anom['Pattern'][keys] = anom['Pattern'][keys]/anom['Pattern']['tas_gam']
 
print('saving')
pk.dump(anom, open(version+'_dVARS.pi', 'wb'))

