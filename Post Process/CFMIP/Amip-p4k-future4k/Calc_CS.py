#!/usr/bin/env python

import numpy as np
import _pickle as pk
import netCDF4 as nc
import sys
import glob

version = sys.argv[1]

exps = ['amip','amip-future4K','amip-p4K']
Source = '/praid/users/jgvirgin/CanESM_Data/'

ts = {}
ts_stk = {}
rlut = {}
rlutcs = {}
rsdt = {}
rsut = {}
rsutcs = {}
fnet = {}
fnetcs = {}
fnet_stk = {}
fnetcs_stk = {}
for i in range(len(exps)):

    print('reading in data on experiment ', exps[i])
    ts_file = glob.glob(Source+version+'/CFMIP/'+exps[i]+'/tas_*')[0]
    rlut_file = glob.glob(Source+version+'/CFMIP/'+exps[i]+'/rlut_*')[0]
    rlutcs_file = glob.glob(Source+version+'/CFMIP/'+exps[i]+'/rlutcs_*')[0]
    rsdt_file = glob.glob(Source+version+'/CFMIP/'+exps[i]+'/rsdt_*')[0]
    rsut_file = glob.glob(Source+version+'/CFMIP/'+exps[i]+'/rsut_*')[0]
    rsutcs_file = glob.glob(Source+version+'/CFMIP/'+exps[i]+'/rsutcs_*')[0]


    ts[exps[i]] = np.squeeze(nc.Dataset(ts_file).variables['tas'][360:720])
    rlut[exps[i]] = np.squeeze(nc.Dataset(rlut_file).variables['rlut'][360:720])
    rlutcs[exps[i]] = np.squeeze(nc.Dataset(rlutcs_file).variables['rlutcs'][360:720])
    rsdt[exps[i]] = np.squeeze(nc.Dataset(rsdt_file).variables['rsdt'][360:720])
    rsut[exps[i]] = np.squeeze(nc.Dataset(rsut_file).variables['rsut'][360:720])
    rsutcs[exps[i]] = np.squeeze(nc.Dataset(rsutcs_file).variables['rsutcs'][360:720])

    nyr = int(len(ts[exps[i]][:])/12)

    print('Checking shape of variable TS\narray shape - ', ts[exps[i]].shape)
    print('Calculate FNET')

    fnet[exps[i]] = (rsdt[exps[i]]-rsut[exps[i]])-rlut[exps[i]]
    fnetcs[exps[i]] = (rsdt[exps[i]]-rsutcs[exps[i]])-rlutcs[exps[i]]
    fnet[exps[i]][fnet[exps[i]] > 1e5] = np.nan
    fnetcs[exps[i]][fnetcs[exps[i]] > 1e5] = np.nan

    print('Stacking')
    fnet_stk[exps[i]] = np.zeros([nyr, 12, 64, 128])
    fnetcs_stk[exps[i]] = np.zeros([nyr, 12, 64, 128])
    ts_stk[exps[i]] = np.zeros([nyr, 12, 64, 128])
    s = 0
    f = 12

    for j in range(nyr):
        fnet_stk[exps[i]][j] = np.stack(fnet[exps[i]][s:f], axis=0)
        fnetcs_stk[exps[i]][j] = np.stack(fnetcs[exps[i]][s:f], axis=0)
        ts_stk[exps[i]][j] = np.stack(ts[exps[i]][s:f], axis=0)

        s += 12
        f += 12

    print('take climatological average')
    fnet_stk[exps[i]] = np.nanmean(fnet_stk[exps[i]], axis=0)
    fnetcs_stk[exps[i]] = np.nanmean(fnetcs_stk[exps[i]], axis=0)
    ts_stk[exps[i]] = np.nanmean(ts_stk[exps[i]], axis=0)

    print('Checking shape of variable FNET\narray shape - ',fnet_stk[exps[i]].shape)

print('Calculating Anomalies')
dts = {}
dts['Uniform'] = ts_stk['amip-p4K']-ts_stk['amip']
dts['Pattern'] = ts_stk['amip-future4K']-ts_stk['amip']

dfnet = {}
dfnet['Uniform'] = fnet_stk['amip-p4K']-fnet_stk['amip']
dfnet['Pattern'] = fnet_stk['amip-future4K']-fnet_stk['amip']

dfnetcs = {}
dfnetcs['Uniform'] = fnetcs_stk['amip-p4K']-fnetcs_stk['amip']
dfnetcs['Pattern'] = fnetcs_stk['amip-future4K']-fnetcs_stk['amip']

print('Calculating Climate Sensitivity')

CS = {}
CS['Uniform'] = dfnet['Uniform']/dts['Uniform']
CS['Pattern'] = dfnet['Pattern']/dts['Pattern']

CScs = {}
CScs['Uniform'] = dfnetcs['Uniform']/dts['Uniform']
CScs['Pattern'] = dfnetcs['Pattern']/dts['Pattern']

print('saving')
pk.dump(CS, open(version+'_CS.pi', 'wb'))
pk.dump(CScs, open(version+'_CScs.pi', 'wb'))
