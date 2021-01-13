#!/usr/bin/env python

import numpy as np
import _pickle as pk
import netCDF4 as nc
import sys
import glob
import multiprocessing as mp
import logging
from multiprocessing_logging import install_mp_handler
from sklearn.linear_model import LinearRegression
import time
from itertools import repeat
import sys

def LinReg_alpha_wrap(predictand, predictor):
    fit = LinearRegression().fit(predictor, predictand)
    return fit.coef_


def LinReg_int_wrap(predictand, predictor):
    fit = LinearRegression().fit(predictor, predictand)
    return fit.intercept_


print('Setting up worker pools using', mp.cpu_count(), 'cpu cores')
cpus = mp.cpu_count()
Pools = mp.Pool(cpus)

version = 'CanESM5_p2'

exps = 'amip-piForcing'
var = 'ts'
Source = '/praid/users/jgvirgin/CanESM_Data/'


print('reading in data on experiment ', exps)
file = glob.glob(Source+version+'/CFMIP/'+exps+'/'+var+'_*')[0]
data = np.squeeze(nc.Dataset(file).variables[var])

nyr = int(len(data[:])/12)
print('Checking shape of variable ', var,'\narray shape - ', data.shape)
data[data > 1e5] = np.nan

print('Stacking')
data_stk = np.zeros([nyr, 12, 64, 128])
s = 0
f = 12
for j in range(nyr):
    data_stk[j] = np.stack(data[s:f], axis=0)

    s += 12
    f += 12

print('Checking shape of variable ',var,'\narray shape - ',data_stk.shape)


sim_len = np.arange(145)

print('Calculate Trend using regression')

flatten = [i for i in np.swapaxes(data_stk.reshape(len(data_stk[:, 0, 0, 0]), 12*64*128), 0, 1)]
alpha = Pools.starmap(LinReg_alpha_wrap,zip(flatten,repeat(sim_len.reshape(-1,1)[:,:])))
rebuild = np.stack(alpha[:], axis=0).reshape(12, 64, 128)

Pools.close()
Pools.join()

print('saving')
pk.dump(rebuild,open(version+'_dSST_piForcing.pi','wb'))

