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

models = sys.argv[1]

Source = '/praid/users/jgvirgin/CanESM_Data/'+models+'/Raw/'

if models == 'CanESM2':
    files_4xCO2 = natsorted(glob.glob(Source+'vars_interp_Amon_*'))[0]
    files_PiCon = natsorted(glob.glob(Source+'vars_interp_Amon_*'))[1]
else:
    files_4xCO2 = natsorted(glob.glob(Source+'vars_interp_Amon_*'))[1]
    files_PiCon = natsorted(glob.glob(Source+'vars_interp_Amon_*'))[2]

print('4xCO2 files - ',files_4xCO2)
print('PiCon files - ',files_PiCon)

var = ['rsdt','rlut','rsut','tas','hus','ta','rsds','rsus']

var_control = dict()
var_4xco2 = dict()

print('read in data')
for v in range(len(var)):
    print('in variable - ',var[v])
    if var[v] == 'ta' or var[v] == 'hus':
        var_control[var[v]] = np.squeeze(nc.Dataset(files_PiCon).variables[var[v]][:360,:17,:,:])
        var_4xco2[var[v]] = np.squeeze(nc.Dataset(files_4xCO2).variables[var[v]][:360,:17,:,:])
    else:
        var_control[var[v]] = np.squeeze(nc.Dataset(files_PiCon).variables[var[v]][:360,:,:])
        var_4xco2[var[v]] = np.squeeze(nc.Dataset(files_4xCO2).variables[var[v]][:360,:,:])

var_control_climo = dict()
var_4xco2_climo = dict()

print('stack and take 30 year mean')
for var in var_control.keys():
    print('on variable - ',var)
    if var_control[var].ndim == 3:
        var_control_climo[var] = np.zeros([30,12,64,128])
        var_4xco2_climo[var] = np.zeros([30,12,64,128])

        s=0
        f=12

        for i in range(30):
            var_control_climo[var][i,:,:,:] = np.stack(var_control[var][s:f,:,:],axis=0)
            var_4xco2_climo[var][i,:,:,:] = np.stack(var_4xco2[var][s:f,:,:],axis=0)

            s+=12
            f+=12

        var_control_climo[var] = np.mean(var_control_climo[var],axis=0)
        var_4xco2_climo[var] = np.mean(var_4xco2_climo[var],axis=0)

    else:
        var_control_climo[var] = np.zeros([30,12,17,64,128])
        var_4xco2_climo[var] = np.zeros([30,12,17,64,128])

        s=0
        f=12

        for i in range(30):
            var_control_climo[var][i,:,:,:,:] = np.stack(var_control[var][s:f,:,:,:],axis=0)
            var_4xco2_climo[var][i,:,:,:,:] = np.stack(var_4xco2[var][s:f,:,:,:],axis=0)

            s+=12
            f+=12

        print('flipping pressure dimension to match the kernels')
        var_control_climo[var] = np.mean(var_control_climo[var],axis=0)[:,::-1,:,:]
        var_4xco2_climo[var] = np.mean(var_4xco2_climo[var],axis=0)[:,::-1,:,:]

    print('switch fill values to NaNs')
    var_control_climo[var][var_control_climo[var] > 1e5] = np.nan
    var_4xco2_climo[var][var_4xco2_climo[var] > 1e5] = np.nan

    print('Control minimum value? - ',np.nanmean(var_control_climo[var]))
    print('Control maximum value? - ',np.nanmean(var_control_climo[var]))

    print('4xCO2 minimum value? - ',np.nanmean(var_4xco2_climo[var]))
    print('4xCO2 maximum value? - ',np.nanmean(var_4xco2_climo[var]))

print('check the shape of processed output? \n', var_control_climo['tas'].shape, '\n', var_control_climo['ta'].shape)

print('calculate FNET')
var_control_climo['rsnt'] = var_control_climo['rsdt']-var_control_climo['rsut']
var_control_climo['fnet'] = var_control_climo['rsnt']-var_control_climo['rlut']

var_4xco2_climo['rsnt'] = var_4xco2_climo['rsdt']-var_4xco2_climo['rsut']
var_4xco2_climo['fnet'] = var_4xco2_climo['rsnt']-var_4xco2_climo['rlut']

print('Calculate surface albedo')
var_control_climo['Alb'] = var_control_climo['rsus']/var_control_climo['rsds']
var_4xco2_climo['Alb'] = var_4xco2_climo['rsus']/var_4xco2_climo['rsds']

print('take the difference')
var_adjusted = dict()
for var in var_control_climo.keys():
    var_adjusted[var] = var_4xco2_climo[var]-var_control_climo[var]


print('moving on to clouds....')
print('read in data')
if models == 'CanESM2':
    clouds_picon_file = natsorted(glob.glob(Source+'clisccp_*'))[1]
    clouds_4x_file = natsorted(glob.glob(Source+'clisccp_*'))[0]
    print(models, '- picon files - ', clouds_picon_file)
    print(models, '- 4x files - ', clouds_4x_file)
    clouds_picon_data = np.squeeze(nc.Dataset(clouds_picon_file).variables['clisccp'])[:360,:,:,:,:]
    clouds_4x_data = np.squeeze(nc.Dataset(clouds_4x_file).variables['clisccp'])[:360,:,:,:,:]
else:
    clouds_picon_file = natsorted(glob.glob(Source+'clisccp_*'))[2]
    clouds_4x_file = natsorted(glob.glob(Source+'clisccp_*'))[1]
    print(models, '- picon files - ', clouds_picon_file)
    print(models, '- 4x files - ', clouds_4x_file)
    clouds_picon_data = np.squeeze(nc.Dataset(clouds_picon_file).variables['clisccp'])[:360,:,:,:,:]
    clouds_4x_data = np.squeeze(nc.Dataset(clouds_4x_file).variables['clisccp'])[:360,:,:,:,:]

print('shape check? \n', clouds_picon_data.shape)

print('stack and take a 30 year mean')

clouds_picon_stk = np.zeros([30,12,7,7,64,128])
clouds_4x_stk = np.zeros([30,12,7,7,64,128])

s = 0
f = 12

for i in range(30):
    clouds_picon_stk[i,:,:,:,:,:] = np.stack(clouds_picon_data[s:f,:,:,:,:],axis=0)
    clouds_4x_stk[i,:,:,:,:,:] = np.stack(clouds_4x_data[s:f,:,:,:,:],axis=0)

    s+=12
    f+=12

clouds_picon_stk = np.mean(clouds_picon_stk,axis=0)
clouds_4x_stk = np.mean(clouds_4x_stk,axis=0)

clouds_picon_stk[clouds_picon_stk > 1e5] = np.nan
clouds_4x_stk[clouds_4x_stk > 1e5] = np.nan

print('take the difference')
clouds_adj = clouds_4x_stk-clouds_picon_stk

var_adjusted['clisccp'] = clouds_adj

var_adjusted.pop('rsus')
var_adjusted.pop('rsds')
var_adjusted.pop('rsdt')
var_adjusted.pop('rsut')

print('check final list of keys - ',var_adjusted.keys())
print('saving...')

file_save = open(models+'_Ajd_sstClim4xCO2.pi','wb')
pk.dump(var_adjusted,file_save,protocol=-1)
file_save.close()

print('done')
