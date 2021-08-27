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
var = ['tas','tas']

g = 9.8  # gravitational acceleration m s^-1
c_p = 1004  # specific heat of air at constant pressure #J K^-1 Kg ^-1
R_a = 287  # dry air ideal gas constant #J K^-1 Kg ^-1
R_v = 461  # water vapour ideal gas constant #J K^-1 Kg ^-1
P = 850  # desired pressure #hPa
L_v = 2.5e6  # latent heat of vaporization for liquid water #J kg ^-1

z700 = 3250  # 700 hpa height #meters
LCL = 430  # lifting condensation level #meters

print('reading in data')
for i in range(len(exps)):

    print('on experiment ', exps[i])
    Source = '/praid/users/jgvirgin/CanESM_Data/CanESM5_p2/Custom/'+exps[i]
    file_tas = glob.glob(Source+'/tas_Amon_*')[0]
    file_ta = glob.glob(Source+'/ta_Amon_*')[0]
    data_tas = nc.Dataset(file_tas)
    data_ta = nc.Dataset(file_ta)

    lat = data_ta.variables['lat'][:]
    lon = data_ta.variables['lon'][:]
    plevs = data_ta.variables['plev'][:]/100
    print('pressure levels? - ',plevs)

    tas = np.squeeze(data_tas.variables['tas'])
    tas[tas > 1e5] = np.nan
    ta = np.squeeze(data_ta.variables['ta'])
    ta[ta > 1e5] = np.nan

    tas_climo = np.nanmean(tas,axis=0)
    ta_climo = np.nanmean(ta, axis=0)

    print('Climatology array shape - ', ta_climo.shape)
    print('get 700 hpa air temp')

    ind_700 = np.where(plevs == 700)
    ind_850 = np.where(plevs == 850)

    ta700 = np.squeeze(ta_climo[ind_700,:,:])
    ta850 = ta_climo[ind_850,:,:]

    print('calculate potential temperatures at 700 & 850 hpa...')
    pta700 = ta700*((1000/700)**(2/7))
    pta850 = ta850*((1000/850)**(2/7))

    print('calculate Lower tropospheric stability... ')
    LTS = pta700-tas_climo

    print('calculate moist adiabatic potential temperature gradient')
    W_s = (1.0007+(3.46e-6*P))*6.1121*(np.exp((17.502)*(ta850-273.15)/(240.97+(ta850-273.15))))  # saturation vapour pressure for liquid water
    Q_s = 0.622*W_s/(P-W_s) #saturation mixing ratio for liquid water #in g/kg

    gamma850 = (g/c_p)*(1-(1+(L_v*Q_s)/\
                                  (R_a*ta850))/\
                                (1+((L_v**2)*Q_s)/\
                                  (R_v*(ta850**2)*c_p)\
                                ))

    print('finally, calculate the EIS...')
    EIS = LTS-gamma850*(z700-LCL)

    print('saving...')
    file_save = open('CanESM5_'+exps[i]+'_EIS_Climo.pi','wb')
    pk.dump(EIS,file_save)
    file_save.close()
