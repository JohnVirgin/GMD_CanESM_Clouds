#!/usr/bin/env python
## Import Packages

import numpy as np
import pandas as pd
import os
import multiprocessing as mp
import _pickle as pk
from itertools import repeat
import netCDF4 as nc
import Jacks_Functions as jf
import glob
from sklearn.linear_model import LinearRegression
import sys
import time
from natsort import natsorted

models = sys.argv[1]

Source = '/praid/users/jgvirgin/CanESM_Data/'+models+'/Time/'

print('read in data')

var = ['EIS','LCC','SST']

length = np.arange(0,150,1)

data_c = dict()
data_4 = dict()
for v in range(len(var)):
    data_c[var[v]] = pk.load(open(Source+models+'_'+var[v]+'_picon_TmSrs.pi','rb'))
    data_4[var[v]] = pk.load(open(Source+models+'_'+var[v]+'_x4_TmSrs.pi','rb'))

data_c['SST'] = np.nan_to_num(data_c['SST'])
data_4['SST'] = np.nan_to_num(data_4['SST'])


if models == 'CanESM2':
    print('checking variable shapes for CanESM2')

    print('SST')
    print(data_c['SST'].shape)
    print(data_4['SST'].shape)

    print('LCC')
    print(data_c['LCC'].shape)
    print(data_4['LCC'].shape)


delta = dict()
for keys in data_c.keys():

    print('on variable - ',keys)

    if models == 'CanESM2':
        if keys == 'LCC':
            delta[keys] = np.mean(data_4[keys][20:,:,:],axis=0)-np.mean(data_4[keys][:20,:,:],axis=0)
        else:
            delta[keys] = np.mean(data_4[keys][120:140, :, :], axis=0)-np.mean(data_4[keys][:20, :, :], axis=0)

    else:
        delta[keys] = np.mean(data_4[keys][120:140,:,:],axis=0)-np.mean(data_4[keys][:20,:,:],axis=0)

print('saving')

file_4x = open(models+'_dVARS_x4.pi', 'wb')
pk.dump(delta, file_4x)
file_4x.close()
