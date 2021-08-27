#!/usr/bin/env python
# coding: utf-8

## Import Packages

import numpy as np
import _pickle as pk
import netCDF4 as nc
import sys
import glob
from natsort import natsorted


exps = ['iga-sst-1xco2', 'iga-sst-4xco2', 'idj-sst-4xco2',
        'iga-dsst-4xco2', '5pi-sst-1xco2', '5a4-sst-4xco2', '5pi-dsst-4xco2']

print('reading in data')
for i in range(len(exps)):

    print('on experiment ', exps[i])
    Source = '/praid/users/jgvirgin/CanESM_Data/CanESM5_p2/Custom/'+exps[i]
    file_wap = glob.glob(Source+'/wap_Amon_*')[0]
    data_wap = nc.Dataset(file_wap)

    lat = data_wap.variables['lat'][:]
    lon = data_wap.variables['lon'][:]
    plevs = data_wap.variables['plev'][:]/100
    print('pressure levels? - ',plevs)

    wap = np.squeeze(data_wap.variables['wap'])
    wap[wap > 1e5] = np.nan

    wap_climo = np.nanmean(wap,axis=0)

    print('Climatology array shape - ', wap_climo.shape)

    print('saving...')
    file_save = open('CanESM5_'+exps[i]+'_wap_Climo.pi','wb')
    pk.dump(wap_climo,file_save)
    file_save.close()
