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
    picon_file = natsorted(glob.glob(Source+'clisccp_*'))[3:]
    co2_file = natsorted(glob.glob(Source+'clisccp_*'))[2]
    print(models, '- picon files - ', picon_file)
    print(models, '- 4xco2 files - ', co2_file)
    picon_data = nc.MFDataset(picon_file)
    co2_data = nc.Dataset(co2_file)
else:
    picon_file = natsorted(glob.glob(Source+'clisccp_*'))[3]
    co2_file = natsorted(glob.glob(Source+'clisccp_*'))[0]
    print(models, '- picon files - ', picon_file)
    print(models, '- 4xco2 files - ', co2_file)
    picon_data = nc.Dataset(picon_file)
    co2_data = nc.Dataset(co2_file)


print('reading in data... ')
control_data = picon_data.variables['clisccp']
co2_data = co2_data.variables['clisccp']

print('Picon time length (in years) = ',len(control_data[:,0,0,0,0])/12)
print('4xCO2 time length (in years) = ',len(co2_data[:,0,0,0,0])/12)

print('sum over low cloud bins and all optical depth bins...')

control_data = np.sum(control_data[:,:,:2,:,:],axis=(1,2))
co2_data = np.sum(co2_data[:,:,:2,:,:],axis=(1,2))

print('check shapes??? - ', control_data.shape)

if models == 'CanESM2':
    nyrs = int(len(control_data[:,0,0])/12)
else:
    nyrs = 150

control_data_stk = np.zeros([nyrs,12,64,128])
co2_data_stk = np.zeros([nyrs,12,64,128])

s=0
f=12

for i in range(nyrs):
    control_data_stk[i,:,:,:] = np.stack(control_data[s:f,:,:],axis=0)
    co2_data_stk[i,:,:,:] = np.stack(co2_data[s:f,:,:],axis=0)

    s+=12
    f+=12

control_data_stk[control_data_stk > 1e5] = np.nan
co2_data_stk[co2_data_stk > 1e5] = np.nan

print('picon max values? - ', np.nanmax(control_data_stk))
print('4xco2 max values? - ', np.nanmax(co2_data_stk))

control_data_time = np.nanmean(control_data_stk, axis=1)
co2_data_time = np.nanmean(co2_data_stk, axis=1)

print('final array shape = ', control_data_time.shape)

print('saving...')

save_file = open(models+'_LCC_picon_TmSrs.pi','wb')
pk.dump(control_data_time, save_file)
save_file.close()

save_file2 = open(models+'_LCC_x4_TmSrs.pi','wb')
pk.dump(co2_data_time, save_file2)
save_file2.close()

print('done')
