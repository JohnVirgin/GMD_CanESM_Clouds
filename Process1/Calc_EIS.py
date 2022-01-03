#!/usr/bin/env python
# coding: utf-8

## Import Packages

import numpy as np
import _pickle as pk
import netCDF4 as nc
import sys
import glob
from natsort import natsorted


models = sys.argv[1]

Source = '/praid/users/jgvirgin/CanESM_Data/'+models+'/Raw/'

if models == 'CanESM2':
    files_4xCO2 = natsorted(glob.glob(Source+'vars_interp_Amon_*'))[2]
    files_PiCon = natsorted(glob.glob(Source+'vars_interp_Amon_*'))[4]
else:
    files_4xCO2 = natsorted(glob.glob(Source+'vars_interp_Amon_*'))[0]
    files_PiCon = natsorted(glob.glob(Source+'vars_interp_Amon_*'))[3]  

print('4xCO2 files - ', files_4xCO2)
print('PiCon files - ', files_PiCon)

var = ['ta','tas','ps']

var_control = dict()
var_4xco2 = dict()

print('read in data')
for v in range(len(var)):
    print('on variable - ',var[v])
    var_control[var[v]] = np.squeeze(nc.Dataset(files_PiCon).variables[var[v]][:1800])
    var_control[var[v]] = np.mean(var_control[var[v]], axis=0)

    var_4xco2[var[v]] = np.squeeze(nc.Dataset(files_4xCO2).variables[var[v]][:1800])
    var_4xco2[var[v]] = np.mean(var_4xco2[var[v]][1560:], axis=0)

print('check shapes? \n', var_control['tas'].shape, '\n', var_control['ta'].shape)


print('getting pressure levels... ')

plevs = np.squeeze(nc.Dataset(files_PiCon).variables['plev'])/100
print(plevs)
print('get 700 hpa air temp')

ind_700 = np.where(plevs == 700)
ind_850 = np.where(plevs == 850)

var_control['ta700'] = np.squeeze(var_control['ta'][ind_700,:,:])
var_4xco2['ta700'] = np.squeeze(var_4xco2['ta'][ind_700,:,:])

print(var_control['ta700'].shape)

var_control['ta850'] = var_control['ta'][ind_850,:,:]
var_4xco2['ta850'] = var_4xco2['ta'][ind_850,:,:]

print('calculate potential temperatures at 700 & 850 hpa...')

var_control['pta700'] = var_control['ta700']*((1000/700)**(2/7))
var_4xco2['pta700'] = var_4xco2['ta700']*((1000/700)**(2/7))

var_control['pta850'] = var_control['ta850']*((1000/850)**(2/7))
var_4xco2['pta850'] = var_4xco2['ta850']*((1000/850)**(2/7))

print('calculate Lower tropospheric stability... ')

var_control['LTS'] = var_control['pta700']-var_control['tas']
var_4xco2['LTS'] = var_4xco2['pta700']-var_4xco2['tas']

print('calculate moist adiabatic potential temperature gradient')

g = 9.8 #gravitational acceleration m s^-1
c_p = 1004  # specific heat of air at constant pressure #J K^-1 Kg ^-1
R_a = 287  # dry air ideal gas constant #J K^-1 Kg ^-1
R_v = 461  # water vapour ideal gas constant #J K^-1 Kg ^-1
P = 850  # desired pressure #hPa
L_v = 2.5e6 #latent heat of vaporization for liquid water #J kg ^-1

z700 = 3250 #700 hpa height #meters
LCL = 430 #lifting condensation level #meters
 
var_control['W_s'] = (1.0007+(3.46e-6*P))*6.1121*(np.exp((17.502)*(var_control['ta850']-273.15)/(240.97+(var_control['ta850']-273.15))))  # saturation vapour pressure for liquid water
var_control['Q_s'] = 0.622*var_control['W_s']/(P-var_control['W_s']) #saturation mixing ratio for liquid water #in g/kg

var_4xco2['W_s'] = (1.0007+(3.46e-6*P))*6.1121*(np.exp((17.502)*(var_4xco2['ta850']-273.15)/(240.97+(var_4xco2['ta850']-273.15))))  # saturation vapour pressure for liquid water
var_4xco2['Q_s'] = 0.622*var_4xco2['W_s']/(P-var_4xco2['W_s']) #saturation mixing ratio for liquid water #in g/kg


var_control['gamma850'] = (g/c_p)*(1-(1+(L_v*var_control['Q_s'])/\
                                  (R_a*var_control['ta850']))/\
                                (1+((L_v**2)*var_control['Q_s'])/\
                                  (R_v*(var_control['ta850']**2)*c_p)\
                                ))

var_4xco2['gamma850'] = (g/c_p)*(1-(1+(L_v*var_4xco2['Q_s'])/\
                                  (R_a*var_4xco2['ta850']))/\
                                (1+((L_v**2)*var_4xco2['Q_s'])/\
                                  (R_v*(var_4xco2['ta850']**2)*c_p)\
                                ))


print('finally, calculate the EIS...')

var_control['EIS'] = var_control['LTS']-var_control['gamma850']*(z700-LCL)
var_4xco2['EIS'] = var_4xco2['LTS']-var_4xco2['gamma850']*(z700-LCL)

print('saving...')

file_control = open(models+'_EIS_Climo.pi','wb')
pk.dump(var_control,file_control)
file_control.close()

file_4x = open(models+'_EIS_x4.pi','wb')
pk.dump(var_4xco2,file_4x)
file_4x.close()
