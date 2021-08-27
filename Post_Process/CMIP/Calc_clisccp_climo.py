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
    print(models, '- files - ', picon_file)
    picon_data = nc.MFDataset(picon_file)
else:
    picon_file = natsorted(glob.glob(Source+'clisccp_*'))[3]
    print(models, '- files - ', picon_file)
    picon_data = nc.Dataset(picon_file)


print('reading in data... ')
data = picon_data.variables['clisccp'][:]

print('time length (in years) = ',len(data[:,0,0,0,0])/12)

nyrs = int(len(data[:,0,0,0,0])/12)

data_stk = np.zeros([nyrs,12,7,7,64,128])

s=0
f=12

for i in range(nyrs):
    data_stk[i,:,:,:,:,:] = np.stack(data[s:f,:,:,:,:],axis=0)

    s+=12
    f+=12

data_stk[data_stk > 1e5] = np.nan

print('max values? - ',np.nanmax(data_stk))

data_climo = np.nanmean(data_stk,axis=0)

print('final array shape = ',data_climo.shape)

print('saving...')

save_file = open(models+'_CLDFRAC_Climo.pi','wb')
pk.dump(data_climo,save_file)
save_file.close()

print('done')
