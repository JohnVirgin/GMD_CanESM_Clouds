#!/usr/bin/env python

import numpy as np
import _pickle as pk
import netCDF4 as nc
import sys
import Jacks_Functions as jf
import glob

version = sys.argv[1]

exps = ['amip','amip-future4K','amip-p4K']
var = ['hus','ta']
Source = '/praid/users/jgvirgin/CanESM_Data/'

data = {}
data_stk = {}

#define grid and weights
lat = np.linspace(-87.864, 87.864, 64)
lon = np.linspace(0, 357.1875, 128)

y = lat*np.pi/180
coslat = np.cos(y)
coslat = np.tile(coslat[:,None], (1,lon.size))

for i in range(len(exps)):
    data[exps[i]] = {}
    data_stk[exps[i]] = {}
    print('\nreading in data on experiment ', exps[i],'\n')
    for v in range(len(var)):
        print('on variable ',var[v])
        file = glob.glob(Source+version+'/CFMIP/'+exps[i]+'/'+var[v]+'_*')[0]

        if version == 'CanESM2':
            data[exps[i]][var[v]] = np.squeeze(nc.Dataset(file).variables[var[v]][360:])
        else:
            data[exps[i]][var[v]] = np.squeeze(nc.Dataset(file).variables[var[v]][360:720])

        nyr = int(len(data[exps[i]][var[v]][:])/12)

        print('Checking shape of variable ',var[v],'\narray shape - ',data[exps[i]][var[v]].shape)
        data[exps[i]][var[v]][data[exps[i]][var[v]] > 1e5] = np.nan


        print('Stacking')
        if data[exps[i]][var[v]].ndim == 3:
            data_stk[exps[i]][var[v]] = np.zeros([nyr, 12, 64, 128])
        else:
            data[exps[i]][var[v]] = data[exps[i]][var[v]][:,:17,:,:]
            data[exps[i]][var[v]] = data[exps[i]][var[v]][:,::-1,:,:]
            data_stk[exps[i]][var[v]] = np.zeros([nyr, 12, 17, 64, 128])
        s = 0
        f = 12

        for j in range(nyr):
            data_stk[exps[i]][var[v]][j] = np.stack(data[exps[i]][var[v]][s:f], axis=0)

            s+=12
            f+=12

        print('take climatological average')
        data_stk[exps[i]][var[v]] = np.nanmean(data_stk[exps[i]][var[v]], axis=0)

        print('Checking shape of variable ',var[v],'\narray shape - ',data_stk[exps[i]][var[v]].shape)

dta = {}
dta['Uniform'] = data_stk['amip-p4K']['ta']-data_stk['amip']['ta']
dta['Pattern'] = data_stk['amip-future4K']['ta']-data_stk['amip']['ta']

CMIP_File = nc.Dataset('/praid/users/jgvirgin/Radiative_Kernels/CAM3_Kernels.nc')

lat = np.squeeze(CMIP_File.variables['lat'])
lon = np.squeeze(CMIP_File.variables['lon'])

CMIP_plevs_scalar = np.squeeze(CMIP_File.variables['lev'])
CMIP_plevs = np.tile(CMIP_plevs_scalar[None,:,None,None],(12,1,lat.size,lon.size))

print('Calculating...')
print('saturdation specific humidity for the baseline')
Sat_Hum_base = jf.Calc_SatSpec_Hum(Ta=data_stk['amip']['ta'], P=CMIP_plevs)

print('saturation specific humidity for the response')
Sat_Hum_resp_p4K = jf.Calc_SatSpec_Hum(Ta=data_stk['amip-p4K']['ta'], P=CMIP_plevs)
Sat_Hum_resp_f4K = jf.Calc_SatSpec_Hum(Ta=data_stk['amip-future4K']['ta'], P=CMIP_plevs)

print('rate of change of saturation specific humidity with respect to temperature')
dqsdt_p4K = (Sat_Hum_resp_p4K-Sat_Hum_base)/dta['Uniform']
dqsdt_f4K = (Sat_Hum_resp_f4K-Sat_Hum_base)/dta['Pattern']

print('relative humidity for the model')
# swap specific humidity units to g/kg to match specific humidity
Rel_Hum_p4K = (1000*data_stk['amip-p4K']['hus'])/Sat_Hum_base
Rel_Hum_f4K = (1000*data_stk['amip-future4K']['hus'])/Sat_Hum_base

print('rate of change of specific humidity with respect to temperature')
dqdt_p4K = dqsdt_p4K*Rel_Hum_p4K
dqdt_f4K = dqsdt_f4K*Rel_Hum_f4K

print('logarathmic rate of change of specific humidity with respect to temperature')
dlnqdt_p4K = dqdt_p4K/(1000*data_stk['amip-p4K']['hus'])
dlnqdt_f4K = dqdt_f4K/(1000*data_stk['amip-future4K']['hus'])

dlnqdt = {}
dlnqdt['Uniform'] = dlnqdt_p4K
dlnqdt['Pattern'] = dlnqdt_f4K

print('saving')
pk.dump(dlnqdt, open(version+'_dlnqdt.pi', 'wb'))
