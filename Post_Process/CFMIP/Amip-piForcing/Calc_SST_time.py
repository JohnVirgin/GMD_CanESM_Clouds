#!/usr/bin/env python3

import numpy as np 
import netCDF4 as nc 
import xarray as xr
import glob 
import sys 
import pickle as pk 

source = '/mnt/data/users/jgvirgin/GMD_CanESM_p1/CanESM5_p2/CFMIP/raw/Amon/'

nc_file = glob.glob(source+'*piForcing*')[0]
print(nc_file)
data = xr.open_dataset(nc_file)

tas = data['tas'][:]
tas_tmsrs = tas.groupby('time.year').mean('time')
tas_arr = tas_tmsrs.values

print('done')
pk.dump(tas_arr,open('CanESM5_p2_SST_Tmsrs.pi','wb'))