#!/usr/bin/env python
## Import Packages

import numpy as np
import pandas as pd
import os
import _pickle as pk
import netCDF4 as nc
import glob
import scipy as sci
import sys
from natsort import natsorted
from scipy.interpolate import interp1d

models = sys.argv[1]

Source = '/praid/users/jgvirgin/CanESM_Data/'+models+'/Raw/'

if models == 'CanESM2':
    files_4xCO2 = natsorted(glob.glob(Source+'vars_interp_Amon_*'))[2]
    files_PiCon = natsorted(glob.glob(Source+'vars_interp_Amon_*'))[3]
else:
    files_4xCO2 = natsorted(glob.glob(Source+'vars_interp_Amon_*'))[0]
    files_PiCon = natsorted(glob.glob(Source+'vars_interp_Amon_*'))[3]  

print('4xCO2 files - ', files_4xCO2)
print('PiCon files - ', files_PiCon)

var = ['ta','clw','cli']

var_control = dict()
var_4xco2 = dict()

print('read in data')
for v in range(len(var)):
    print('on variable - ',var[v])
    var_control[var[v]] = np.squeeze(nc.Dataset(files_PiCon).variables[var[v]][:1800,:,:,:])
    var_4xco2[var[v]] = np.squeeze(nc.Dataset(files_4xCO2).variables[var[v]][:1800,:,:,:])


print('check shapes? \n clw - ', var_control['clw'].shape,'\n ta - ',var_control['ta'].shape)

cords = dict()

cord_var = ['lat','lon','plev','lev','ap','b','ps']

print('read in coordinate data...')
for v in range(len(cord_var)):
    if cord_var == 'ps':
        cords[cord_var[v]] = np.squeeze(nc.Dataset(files_PiCon).variables[cord_var[v]][:12,:,:,:])
    else:
        cords[cord_var[v]] = np.squeeze(nc.Dataset(files_PiCon).variables[cord_var[v]])

print('calculate pressure coordinates from hybrid sigma pressure levels...')

plevs_c = np.zeros([12,35,64,128])

for i in range(12):
    for j in range(64):
        for k in range(128):
            plevs_c[i,:,j,k] = cords['ap'] + cords['b']*cords['ps'][i,j,k]

print('pressure level slice sample \n',plevs_c[5,:,33,0])

cords['plevs_c'] = plevs_c

print('stack variables and take climatological means... ')

var_control_climo = dict()
var_4xco2_climo = dict()

print('stack')
for var in var_control.keys():
    print('on variable - ',var)
    if var == 'ta':
        var_control_climo[var] = np.zeros([150,12,22,64,128])
        var_4xco2_climo[var] = np.zeros([150,12,22,64,128])

        s=0
        f=12

        for i in range(150):
            var_control_climo[var][i,:,:,:,:] = np.stack(var_control[var][s:f,:,:,:],axis=0)
            var_4xco2_climo[var][i,:,:,:,:] = np.stack(var_4xco2[var][s:f,:,:,:],axis=0)

            s+=12
            f+=12

    else:
        var_control_climo[var] = np.zeros([150,12,35,64,128])
        var_4xco2_climo[var] = np.zeros([150,12,35,64,128])

        s=0
        f=12

        for i in range(150):
            var_control_climo[var][i,:,:,:,:] = np.stack(var_control[var][s:f,:,:,:],axis=0)
            var_4xco2_climo[var][i,:,:,:,:] = np.stack(var_4xco2[var][s:f,:,:,:],axis=0)

            s+=12
            f+=12


    print('switch fill values to NaNs')
    var_control_climo[var][var_control_climo[var] > 1e5] = np.nan
    var_4xco2_climo[var][var_4xco2_climo[var] > 1e5] = np.nan

print('check the shape of processed output? \n', var_control_climo['clw'].shape, '\n', var_control_climo['ta'].shape)
print('take means')

for var in var_control_climo.keys():
    var_control_climo[var] = np.nanmean(var_control_climo[var], axis=0)
    var_4xco2_climo[var] = np.nanmean(var_4xco2_climo[var][130:],axis=0)

print('check the final shape one more time \n',var_control_climo['clw'].shape, '\n', var_4xco2_climo['ta'].shape)

print('saving')

cord_file = open(models+'_dims.pi','wb')
pk.dump(cords,cord_file)
cord_file.close()

vars_file = open(models+'_LCFvars_piControl.pi','wb')
pk.dump(var_control_climo,vars_file)
vars_file.close()

vars_4x_file = open(models+'_LCFvars_4xCO2.pi','wb')
pk.dump(var_4xco2_climo,vars_4x_file)
vars_4x_file.close()

