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

var = ['ta','tas']

var_control = dict()
var_4xco2 = dict()

print('read in data')
for v in range(len(var)):
    print('on variable - ',var[v])
    var_control[var[v]] = np.squeeze(nc.Dataset(files_PiCon).variables[var[v]][:1800])

    var_4xco2[var[v]] = np.squeeze(nc.Dataset(files_4xCO2).variables[var[v]][:1800])


print('check shapes? \n', var_control['tas'].shape, '\n', var_control['ta'].shape)


print('getting pressure levels... ')

plevs = np.squeeze(nc.Dataset(files_PiCon).variables['plev'])/100
print(plevs)
print('get 700 hpa air temp')

ind_700 = np.where(plevs == 700)
ind_850 = np.where(plevs == 850)

var_control['ta700'] = np.squeeze(var_control['ta'][:,ind_700,:,:])
var_4xco2['ta700'] = np.squeeze(var_4xco2['ta'][:,ind_700,:,:])

print(var_control['ta700'].shape)

var_control['ta850'] = np.squeeze(var_control['ta'][:,ind_850,:,:])
var_4xco2['ta850'] = np.squeeze(var_4xco2['ta'][:,ind_850,:,:])

print(var_control['ta850'].shape)

new_v = ['tas','ta850','ta700']
print('stacking... ')


var_control_stk = dict()
var_4xco2_stk = dict()

for v in range(len(new_v)):
    print('on variable - ', new_v[v])

    var_control_stk[new_v[v]] = np.zeros([150,12,64,128])
    var_4xco2_stk[new_v[v]] = np.zeros([150,12,64,128])
    
    s = 0
    f = 12

    for i in range(150):
        var_control_stk[new_v[v]][i,:,:,:] = np.stack(var_control[new_v[v]][s:f,:,:])
        var_4xco2_stk[new_v[v]][i,:,:,:] = np.stack(var_4xco2[new_v[v]][s:f,:,:])

        s+=12
        f+=12
    
    print('take the annual mean...')

    var_control_stk[new_v[v]] = np.mean(var_control_stk[new_v[v]],axis=1)
    var_4xco2_stk[new_v[v]] = np.mean(var_4xco2_stk[new_v[v]], axis=1)

    print('check shape? - ', var_control_stk[new_v[v]].shape)


print('calculate potential temperatures at 700 & 850 hpa...')

var_control_stk['pta700'] = var_control_stk['ta700']*((1000/700)**(2/7))
var_4xco2_stk['pta700'] = var_4xco2_stk['ta700']*((1000/700)**(2/7))

var_control_stk['pta850'] = var_control_stk['ta850']*((1000/850)**(2/7))
var_4xco2_stk['pta850'] = var_4xco2_stk['ta850']*((1000/850)**(2/7))

print('calculate Lower tropospheric stability... ')

var_control_stk['LTS'] = var_control_stk['pta700']-var_control_stk['tas']
var_4xco2_stk['LTS'] = var_4xco2_stk['pta700']-var_4xco2_stk['tas']

print('calculate moist adiabatic potential temperature gradient')

g = 9.8 #gravitational acceleration m s^-1
c_p = 1004  # specific heat of air at constant pressure #J K^-1 Kg ^-1
R_a = 287  # dry air ideal gas constant #J K^-1 Kg ^-1
R_v = 461  # water vapour ideal gas constant #J K^-1 Kg ^-1
P = 850  # desired pressure #hPa
L_v = 2.5e6 #latent heat of vaporization for liquid water #J kg ^-1

z700 = 3250 #700 hpa height #meters
LCL = 430 #lifting condensation level #meters
 
var_control_stk['W_s'] = (1.0007+(3.46e-6*P))*6.1121*(np.exp((17.502)*(var_control_stk['ta850']-273.15)/(240.97+(var_control_stk['ta850']-273.15))))  # saturation vapour pressure for liquid water
var_control_stk['Q_s'] = 0.622*var_control_stk['W_s']/(P-var_control_stk['W_s']) #saturation mixing ratio for liquid water #in g/kg

var_4xco2_stk['W_s'] = (1.0007+(3.46e-6*P))*6.1121*(np.exp((17.502)*(var_4xco2_stk['ta850']-273.15)/(240.97+(var_4xco2_stk['ta850']-273.15))))  # saturation vapour pressure for liquid water
var_4xco2_stk['Q_s'] = 0.622*var_4xco2_stk['W_s']/(P-var_4xco2_stk['W_s']) #saturation mixing ratio for liquid water #in g/kg


var_control_stk['gamma850'] = (g/c_p)*(1-(1+(L_v*var_control_stk['Q_s']) /
                                  (R_a*var_control_stk['ta850']))/\
                                (1+((L_v**2)*var_control_stk['Q_s'])/\
                                  (R_v*(var_control_stk['ta850']**2)*c_p)\
                                ))

var_4xco2_stk['gamma850'] = (g/c_p)*(1-(1+(L_v*var_4xco2_stk['Q_s']) /
                                  (R_a*var_4xco2_stk['ta850']))/\
                                (1+((L_v**2)*var_4xco2_stk['Q_s'])/\
                                  (R_v*(var_4xco2_stk['ta850']**2)*c_p)\
                                ))


print('finally, calculate the EIS...')

var_control_stk['EIS'] = var_control_stk['LTS']-var_control_stk['gamma850']*(z700-LCL)
var_4xco2_stk['EIS'] = var_4xco2_stk['LTS']-var_4xco2_stk['gamma850']*(z700-LCL)

print('saving...')

file_control = open(models+'_EIS_picon_TmSrs.pi','wb')
pk.dump(var_control_stk['EIS'],file_control)
file_control.close()

file_4x = open(models+'_EIS_x4_TmSrs.pi','wb')
pk.dump(var_4xco2_stk['EIS'],file_4x)
file_4x.close()




