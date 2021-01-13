#!/usr/bin/env python
# coding: utf-8

## Import Packages

import numpy as np
import _pickle as pk
import netCDF4 as nc
import sys
import glob
from natsort import natsorted

Source = '/praid/users/jgvirgin/CanESM_Data/CanESM2/Raw/'

x4 = nc.Dataset(glob.glob(Source+'wap_Amon_CanESM2_abrupt*')[0])
picon = nc.Dataset(glob.glob(Source+'wap_Amon_CanESM2_piC*')[0])

plevs = x4.variables['plev'][:]/100
print('pressure levels? - ', plevs)

x4_wap = np.squeeze(x4.variables['wap'])[1440:,:,:,:]
x4_wap[x4_wap > 1e5] = np.nan
x4_wap = np.nanmean(x4_wap,axis=0)

picon_wap = np.squeeze(picon.variables['wap'])
picon_wap[picon_wap > 1e5] = np.nan
picon_wap = np.nanmean(picon_wap,axis=0)

pk.dump(picon_wap, open('CanESM2_wap_Climo_picon.pi', 'wb'))
pk.dump(x4_wap, open('CanESM2_wap_Climo_x4.pi', 'wb'))
