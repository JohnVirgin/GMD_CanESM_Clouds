#!/usr/bin/env python

import numpy as np
import _pickle as pk
import netCDF4 as nc
import sys
import glob
from natsort import natsorted

exps = ['iga-sst-1xco2', 'iga-sst-4xco2', 'idj-sst-4xco2',
        'iga-dsst-4xco2', '5pi-sst-1xco2', '5a4-sst-4xco2', \
            '5pi-dsst-4xco2']

lat = np.linspace(-87.864,87.864,64)
lon = np.linspace(0,357.1875,128)

y = lat*np.pi/180
coslat = np.cos(y)
coslat = np.tile(coslat, (lon.size, 1)).T

proc_fnet = {}
proc_ts = {}

print('reading in data')
for i in range(len(exps)):

    print('on experiment ', exps[i])
    Source = '/praid/users/jgvirgin/CanESM_Data/CanESM5_p2/Custom/'+exps[i]
    rlut = np.squeeze(nc.Dataset(\
        glob.glob(Source+'/rlut_*')[0]).variables['rlut'])

    rsut = np.squeeze(nc.Dataset(
        glob.glob(Source+'/rsut_*')[0]).variables['rsut'])

    rsdt = np.squeeze(nc.Dataset(
        glob.glob(Source+'/rsdt_*')[0]).variables['rsdt'])

    ts = np.squeeze(nc.Dataset(
        glob.glob(Source+'/tas_*')[0]).variables['tas'])

    rlut[rlut > 1e5] = np.nan
    rsut[rsut > 1e5] = np.nan
    rsdt[rsdt > 1e5] = np.nan
    ts[ts > 1e5] = np.nan

    rlut = np.nanmean(rlut, axis=0)
    rsut = np.nanmean(rsut, axis=0)
    rsdt = np.nanmean(rsdt, axis=0)
    ts = np.average(np.nanmean(ts, axis=0), weights=coslat)

    rsnt = rsdt-rsut
    fnet = rsnt-rlut

    proc_fnet[exps[i]] = fnet
    proc_ts[exps[i]] = ts

 
dtas = {}
dtas['2SST'] = proc_ts['idj-sst-4xco2']-proc_ts['iga-sst-1xco2']
dtas['2SSTu'] = proc_ts['iga-dsst-4xco2']-proc_ts['iga-sst-1xco2']
dtas['5SST'] = proc_ts['5a4-sst-4xco2']-proc_ts['5pi-sst-1xco2']
dtas['5SSTu'] = proc_ts['5pi-dsst-4xco2']-proc_ts['iga-sst-1xco2']

print('dTAS?')
for keys in dtas.keys():
    print(keys,round(dtas[keys],3))

dFNET = {}
dFNET['2SST'] = proc_fnet['idj-sst-4xco2']-proc_fnet['iga-sst-1xco2']
dFNET['2SSTu'] = proc_fnet['iga-dsst-4xco2']-proc_fnet['iga-sst-1xco2']
dFNET['5SST'] = proc_fnet['5a4-sst-4xco2']-proc_fnet['5pi-sst-1xco2']
dFNET['5SSTu'] = proc_fnet['5pi-dsst-4xco2']-proc_fnet['iga-sst-1xco2']

CF = {}
CF['2SST'] = -dFNET['2SST']/dtas['2SST']
CF['2SSTu'] = -dFNET['2SSTu']/dtas['2SSTu']
CF['5SST'] = -dFNET['5SST']/dtas['5SST']
CF['5SSTu'] = -dFNET['5SSTu']/dtas['5SSTu']

for keys in CF.keys():
    print(keys)
    print('CF value = ', np.average(CF[keys],weights=coslat), '\n')

print('saving')
pk.dump(CF,open('CanESM5_p2_Custom_CFP.pi','wb'))