#!/usr/bin/env python3

import numpy as np 
import xarray as xr
import glob 
import sys 
import pickle as pk
import metpy.calc as mp

source = '/mnt/data/users/jgvirgin/GMD_CanESM_p1/'
mod = sys.argv[1]
timeframe = sys.argv[2]
file_atmos = glob.glob(source+mod+'/CFMIP/raw/Amon/*'+timeframe+'*')[0]

print('Read in data')
Amon = xr.open_dataset(file_atmos)
lat = Amon['lat']
lon = Amon['lon']

print('calculate inversion strength')
print('\tDefining Constants')
g = 9.8  # gravitational acceleration m s^-1
c_p = 1004  # specific heat of air at constant pressure #J K^-1 Kg ^-1
R_a = 287  # dry air ideal gas constant #J K^-1 Kg ^-1
R_v = 461  # water vapour ideal gas constant #J K^-1 Kg ^-1
P = 850  # desired pressure #hPa
L_v = 2.5e6  # latent heat of vaporization for liquid water #J kg ^-1

z700 = 3250  # 700 hpa height #meters
lcl = 430  # lifting condensation level #meters

plevs = Amon['plev']/100
plev_val = plevs.values
hpa700 = np.where(plev_val==700)
hpa850 = np.where(plev_val==850)

tas = Amon['tas'][:]
tas = tas.groupby('time.month').mean('time')

ta = Amon['ta'][:]
ta = ta.groupby('time.month').mean('time')

tas_ref = tas.values
ta_ref = ta.values

ta_700 = np.squeeze(ta_ref[:,hpa700,:,:])
ta_850 = np.squeeze(ta_ref[:,hpa850,:,:])

print('\tcalculate potential temperature at 700 & 850 hpa')
pta700 = ta_700*((1000/700)**(2/7))
pta850 = ta_850*((1000/850)**(2/7))

print('\tcalculate Lower tropospheric stability... ')
lts = pta700-tas_ref

print('\tcalculate moist adiabatic potential temperature gradient')
W_s = (1.0007+(3.46e-6*P))*6.1121*(np.exp((17.502)*(ta_850-273.15)/(240.97+(ta_850-273.15))))  # saturation vapour pressure for liquid water
Q_s = 0.622*W_s/(P-W_s) #saturation mixing ratio for liquid water #in g/kg

gamma850 = (g/c_p)*(1-(1+(L_v*Q_s)/\
                                  (R_a*ta_850))/\
                                (1+((L_v**2)*Q_s)/\
                                  (R_v*(ta_850**2)*c_p)\
                                ))

print('\tfinally, calculate the EIS')
eis = lts-gamma850*(z700-lcl)

pk.dump(eis,open('EIS_'+mod+'_'+timeframe+'_climo.pi','wb'))