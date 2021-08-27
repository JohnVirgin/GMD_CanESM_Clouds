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

def predict_wrap(predictand, predictor):
    fit = LinearRegression().fit(predictor, predictand)
    prediction = fit.predict(np.arange(0, 150).reshape(-1,1))
    return prediction


print('Setting up worker pools using', mp.cpu_count(), 'cpu cores')
cpus = mp.cpu_count()
Pools = mp.Pool(cpus)

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

print(np.max(data_c['SST']))
print(np.min(data_c['SST']))

print('check keys and shapes?')

for keys in data_c.keys():
    print(keys)
    print(data_c[keys].shape)

if models == 'CanESM2':
    print('Using CanESM2 output - selecting 40 year range from SST and EIS output...')

    data_c['EIS'] = np.concatenate((data_c['EIS'][:20],data_c['EIS'][120:140]),axis=0)
    data_c['SST'] = np.concatenate((data_c['SST'][:20],data_c['SST'][120:140]),axis=0)

    data_4['EIS'] = np.concatenate((data_4['EIS'][:20],data_4['EIS'][120:140]),axis=0)
    data_4['SST'] = np.concatenate((data_4['SST'][:20],data_4['SST'][120:140]),axis=0)

    length = np.concatenate((length[:20], length[120:140]))

    print('check keys and shapes one more time?')

    for keys in data_c.keys():
        print(keys)
        print(data_c[keys].shape)
else:
    print('Using CanESM5 output, no restructuring needed...')


print('get trend line for variables')

start_time = time.time()

data_c_flatten = dict()
data_4_flatten = dict()

data_c_pred = dict()
data_4_pred = dict()

data_c_rebuild = dict()
data_4_rebuild = dict()

for keys in data_c.keys():

    print('\non Variable - ',keys)

    print('flatten')
    data_c_flatten[keys] = [i for i in np.swapaxes(data_c[keys][:,:,:].reshape(len(data_c[keys][:,0,0]),64*128),0,1)]
    data_4_flatten[keys] = [i for i in np.swapaxes(data_4[keys][:,:,:].reshape(len(data_4[keys][:,0,0]),64*128),0,1)]

    print('fit and predict')

    data_c_pred[keys] = Pools.starmap(predict_wrap,zip(data_c_flatten[keys],repeat(length.reshape(-1,1))))
    data_4_pred[keys] = Pools.starmap(predict_wrap,zip(data_4_flatten[keys],repeat(length.reshape(-1,1))))

    print('rebuild original shape')

    data_c_rebuild[keys] = np.stack(data_c_pred[keys][:],axis=0).reshape(150,64,128)
    data_4_rebuild[keys] = np.stack(data_4_pred[keys][:],axis=0).reshape(150,64,128)
    
end_time = time.time() - start_time
print(end_time/60, 'minutes for complete regressions to finish')

Pools.close()
Pools.join()

print('remove trend from raw output')

data_c_detrend = dict()
data_4_detrend = dict()

for keys in data_c_rebuild.keys():
    if models == 'CanESM2':
        data_c_detrend[keys] = data_c[keys] - np.concatenate((data_c_rebuild[keys][:20], data_c_rebuild[keys][120:140]), axis=0)
        data_4_detrend[keys] = data_4[keys] - np.concatenate((data_4_rebuild[keys][:20], data_4_rebuild[keys][120:140]), axis=0)
    else:
        data_c_detrend[keys] = data_c[keys] - data_c_rebuild[keys]
        data_4_detrend[keys] = data_4[keys] - data_4_rebuild[keys]


print('Saving...')

file_control = open(models+'_vars_picon_Detrend.pi', 'wb')
pk.dump(data_c_detrend, file_control)
file_control.close()

file_4x = open(models+'_vars_x4_Detrend.pi', 'wb')
pk.dump(data_4_detrend, file_4x)
file_4x.close()

print('done')
