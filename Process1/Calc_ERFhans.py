#!/usr/bin/env python

## Import Packages
import numpy as np
import pickle as pk
import Area_Avg as aa
import warnings
import Jacks_Functions as jf
import glob
from natsort import natsorted
from netCDF4 import Dataset
import sys


models = sys.argv[1]

Source = '/praid/users/jgvirgin/CanESM_Data/'+models+'/Raw/'

files_4xCO2 = natsorted(glob.glob(Source+'vars_interp_Amon_*'))[1]
files_PiCon = natsorted(glob.glob(Source+'vars_interp_Amon_*'))[2]

print('4xCO2 files - ',files_4xCO2)
print('PiCon files - ',files_PiCon)

var = ['rsdt','rlut','rsut']

var_control = dict()
var_4xco2 = dict()

for v in range(len(var)):
        var_control[var[v]] = np.squeeze(Dataset(files_PiCon).variables[var[v]][:360,:,:])
        var_4xco2[var[v]] = np.squeeze(Dataset(files_4xCO2).variables[var[v]][:360,:,:])


print('reshape data and take 30 year mean')

var_control_climo = dict()
var_4xco2_climo = dict()
for var in var_control.keys():
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

print('Calculate ERF')

var_control_climo['rsnt'] = var_control_climo['rsdt']-var_control_climo['rsut']
var_control_climo['fnet'] = var_control_climo['rsnt']-var_control_climo['rlut']

var_4xco2_climo['rsnt'] = var_4xco2_climo['rsdt']-var_4xco2_climo['rsut']
var_4xco2_climo['fnet'] = var_4xco2_climo['rsnt']-var_4xco2_climo['rlut']

ERF = var_4xco2_climo['fnet']-var_control_climo['fnet']


ERF_file = open(models+'_ERF_HANS_Grid.pi','wb')
pk.dump(ERF,ERF_file)
ERF_file.close()

print('done!')
