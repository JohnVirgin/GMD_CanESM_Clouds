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


def alpha_wrap(predictand, predictor):
    fit = LinearRegression().fit(predictor.reshape(-1, 1), predictand)
    return fit.coef_

print('Setting up worker pools using', mp.cpu_count(), 'cpu cores')
cpus = mp.cpu_count()
Pools = mp.Pool(cpus)

models = sys.argv[1]

Source = '/praid/users/jgvirgin/CanESM_Data/'+models+'/Time/'
length = np.arange(0, 150, 1)

if models == 'CanESM2':
    length = np.concatenate((length[:20], length[120:140]))


var = ['EIS', 'LCC', 'SST']

data_c = dict()
data_4 = dict()
for v in range(len(var)):
    data_c[var[v]] = pk.load(open(Source+models+'_vars_picon_Detrend.pi','rb'))[var[v]]
    data_4[var[v]] = pk.load(open(Source+models+'_vars_x4_Detrend.pi','rb'))[var[v]]

print('Calculate EIS sensitivity first')

EIS_c_flatten = [i for i in np.swapaxes(data_c['EIS'][:,:,:].reshape(len(length),64*128),0,1)]
EIS_4_flatten = [i for i in np.swapaxes(data_4['EIS'][:,:,:].reshape(len(length),64*128),0,1)]

SST_c_flatten = [i for i in np.swapaxes(data_c['SST'][:,:,:].reshape(len(length),64*128),0,1)]
SST_4_flatten = [i for i in np.swapaxes(data_4['SST'][:,:,:].reshape(len(length),64*128),0,1)]

LCC_c_flatten = [i for i in np.swapaxes(data_c['LCC'][:,:,:].reshape(len(length),64*128),0,1)]
LCC_4_flatten = [i for i in np.swapaxes(data_4['LCC'][:,:,:].reshape(len(length),64*128),0,1)]


print('Regress EIS onto SST for alpha 1')

alpha1_c_flat = Pools.starmap(alpha_wrap,zip(EIS_c_flatten,SST_c_flatten))
alpha1_4_flat = Pools.starmap(alpha_wrap,zip(EIS_4_flatten,SST_4_flatten))

print('rebuild alpha 1')

alpha1_c = np.stack(alpha1_c_flat[:], axis=0).reshape(64,128)
alpha1_4 = np.stack(alpha1_4_flat[:], axis=0).reshape(64,128)
alpha1_c_T = np.tile(np.transpose(alpha1_c[None,:,:],(0,1,2)),(len(length),1,1))
alpha1_4_T = np.tile(np.transpose(alpha1_4[None,:,:],(0,1,2)),(len(length),1,1))

print('Separate EIS components into correlated/uncorrelated with SST')

EIS_c_clean = data_c['EIS'] - (alpha1_c_T*data_c['SST'])
EIS_4_clean = data_4['EIS'] - (alpha1_4_T*data_4['SST'])

print('Regress LCC onto SST for beta 1')

beta1_c_flat = Pools.starmap(alpha_wrap,zip(LCC_c_flatten,SST_c_flatten))
beta1_4_flat = Pools.starmap(alpha_wrap,zip(LCC_4_flatten,SST_4_flatten))

print('rebuild beta 1')

beta1_c = np.stack(beta1_c_flat[:], axis=0).reshape(64,128)
beta1_4 = np.stack(beta1_4_flat[:], axis=0).reshape(64,128)
beta1_c_T = np.tile(np.transpose(beta1_c[None,:,:],(0,1,2)),(len(length),1,1))
beta1_4_T = np.tile(np.transpose(beta1_4[None,:,:],(0,1,2)),(len(length),1,1))

print('Separate LCC components into correlated/uncorrelated with SST')

LCC_c_clean = data_c['LCC'] - (beta1_c_T*data_c['SST'])
LCC_4_clean = data_4['LCC'] - (beta1_4_T*data_4['SST'])

print('flatten our detrended, separated LCC and EIS variables')

LCC_c_clean_flat = [i for i in np.swapaxes(LCC_c_clean[:,:,:].reshape(len(length),64*128),0,1)]
LCC_4_clean_flat = [i for i in np.swapaxes(LCC_4_clean[:,:,:].reshape(len(length),64*128),0,1)]

EIS_c_clean_flat = [i for i in np.swapaxes(EIS_c_clean[:,:,:].reshape(len(length),64*128),0,1)]
EIS_4_clean_flat = [i for i in np.swapaxes(EIS_4_clean[:,:,:].reshape(len(length),64*128),0,1)]

print('finally, Regress the uncorrelated, detrended LCC onto EIS for gamma 1')

gamma1_c_flat = Pools.starmap(alpha_wrap,zip(LCC_c_clean_flat,EIS_c_clean_flat))
gamma1_4_flat = Pools.starmap(alpha_wrap,zip(LCC_4_clean_flat,EIS_4_clean_flat))

print('rebuild gamma 1')

gamma1_c = np.stack(gamma1_c_flat[:], axis=0).reshape(64,128)
gamma1_4 = np.stack(gamma1_4_flat[:], axis=0).reshape(64,128)

print('Saving...')

h_mod = dict()
h_mod['Alpha_c'] = alpha1_c
h_mod['Alpha_4'] = alpha1_4
h_mod['Beta_c'] = beta1_c
h_mod['Beta_4'] = beta1_4
h_mod['Gamma_c'] = gamma1_c
h_mod['Gamma_4'] = gamma1_4
h_mod['LCC_c_clean'] = LCC_c_clean
h_mod['LCC_4_clean'] = LCC_4_clean
h_mod['EIS_c_clean'] = EIS_c_clean
h_mod['EIS_4_clean'] = EIS_4_clean

file_hmod = open(models+'_Hmod_EIS.pi','wb')
pk.dump(h_mod,file_hmod)
file_hmod.close()

print('done')
