#!/usr/bin/env python

#import packages
import numpy as np
import _pickle as pk
import netCDF4 as nc
import Area_Avg as aa
import pandas as pd
from sklearn.linear_model import LinearRegression
import multiprocessing as mp
import logging
from multiprocessing_logging import install_mp_handler
import time
from itertools import repeat
import sys

print('defining regression wrapper functions')

def LinReg_int_wrap(predictand,predictor):
    fit = LinearRegression().fit(predictor,predictand)
    return fit.intercept_

def LinReg_alpha_wrap(predictand,predictor):
    fit = LinearRegression().fit(predictor,predictand)
    return -fit.coef_

print('Setting up worker pools using',mp.cpu_count(),'cpu cores')
cpus = mp.cpu_count()
Pools = mp.Pool(cpus)

#define dimensions for horizontal grid
lat = np.linspace(-87.864,87.864,64)
lon = np.linspace(0,357.1875,128)

Models = sys.argv[1]

Source = '/home/jgvirgin/Projects/CanESM/Data/'+Models+'/Anomalies/'


print('read in data and calculate FNET/FNETCS')
rlut = pk.load(open(Source+Models+'_Ajd_amip-piControl_TmSrs.pi','rb'))['rlut']
rlutcs = pk.load(open(Source+Models+'_Ajd_amip-piControl_TmSrs.pi','rb'))['rlutcs']

rsut = pk.load(open(Source+Models+'_Ajd_amip-piControl_TmSrs.pi','rb'))['rsut']
rsutcs = pk.load(open(Source+Models+'_Ajd_amip-piControl_TmSrs.pi','rb'))['rsutcs']

rsdt = pk.load(open(Source+Models+'_Ajd_amip-piControl_TmSrs.pi','rb'))['rsdt']

rsnt = rsdt-rsut
rsntcs = rsdt-rsutcs

FNET_stk = rsnt-rlut
FNETCS_stk = rsntcs-rlutcs


tas_stk = pk.load(open(Source+Models+'_Ajd_amip-piControl_TmSrs.pi','rb'))['tas']
tas_grid = np.mean(tas_stk[135:,:,:,:],axis=(0,1))
tas_season = aa.LatLonavg_Time(np.ma.mean(tas_stk[135:,:,:,:],axis=0),lat,lon)
tas_GAM = aa.LatLonavg_Time(np.ma.mean(tas_stk,axis=1),lat,lon)
FNET_GAM = aa.LatLonavg_Time(np.ma.mean(FNET_stk,axis=1),lat,lon)
FNETCS_GAM = aa.LatLonavg_Time(np.ma.mean(FNETCS_stk,axis=1),lat,lon)
tas_GAM_rsh = tas_GAM.reshape(-1,1)


print('regress Net fluxes against global annual mean surface temperature change this gives a slope for each gridpoint and each month')


start_time = time.time()

FNET_Flatten = [i for i in np.swapaxes(FNET_stk.reshape(len(FNET_stk[:,0,0,0]),12*64*128),0,1)]
#FNETCS_Flatten = [i for i in np.swapaxes(FNETCS_stk.reshape(len(FNET_stk[:,0,0,0]),12*64*128),0,1)]

FNET_int = Pools.starmap(LinReg_int_wrap,zip(FNET_Flatten,repeat(tas_GAM_rsh)))
FNET_alpha = Pools.starmap(LinReg_alpha_wrap,zip(FNET_Flatten,repeat(tas_GAM_rsh)))

#FNETCS_int = Pools.starmap(LinReg_int_wrap,zip(FNETCS_Flatten,repeat(tas_GAM_rsh)))
#FNETCS_alpha = Pools.starmap(LinReg_alpha_wrap,zip(FNETCS_Flatten,repeat(tas_GAM_rsh)))

end_time = time.time() - start_time
print(end_time/60, 'minutes for complete regression to finish')

print('rebuilding arrays with original dimensions')

FNET_int_rebuild = np.stack(FNET_int[:],axis=0).reshape(12,64,128)
FNET_alpha_rebuild = np.stack(FNET_alpha[:],axis=0).reshape(12,64,128)

#FNETCS_int_rebuild = np.stack(FNETCS_int[:],axis=0).reshape(12,64,128)
#FNETCS_alpha_rebuild = np.stack(FNETCS_alpha[:],axis=0).reshape(12,64,128)


print('calculate ECS')

ECS = (FNET_int_rebuild/FNET_alpha_rebuild)/2
#ECSCS = (FNETCS_int_rebuild/FNETCS_alpha_rebuild)/2

print('saving variables')

FNET_int_file = open(Models+'_ERF_GREG_Grid.pi','wb')
pk.dump(FNET_int_rebuild,FNET_int_file)
FNET_int_file.close()

#FNETCS_int_file = open(Models+'_ERFCS_GREG_Grid.pi','wb')
#pk.dump(FNETCS_int_rebuild,FNETCS_int_file)
#FNETCS_int_file.close()

FNET_alpha_file = open(Models+'_CFP_GREG_Grid.pi','wb')
pk.dump(FNET_alpha_rebuild,FNET_alpha_file)
FNET_alpha_file.close()

#FNETCS_alpha_file = open(Models+'_CFPCS_GREG_Grid.pi','wb')
#pk.dump(FNETCS_alpha_rebuild,FNETCS_alpha_file)
#FNETCS_alpha_file.close()

tas_GAM_file = open(Models+'_TAS_GAM.pi','wb')
pk.dump(tas_GAM,tas_GAM_file)
tas_GAM_file.close()

tas_grid_file = open(Models+'_TAS_Grid.pi','wb')
pk.dump(tas_grid,tas_grid_file)
tas_grid_file.close()

tas_season_file = open(Models+'_TAS_Months.pi','wb')
pk.dump(tas_season,tas_season_file)
tas_season_file.close()

FNET_GAM_file = open(Models+'_FNET_GAM.pi','wb')
pk.dump(FNET_GAM,FNET_GAM_file)
FNET_GAM_file.close()

FNETCS_GAM_file = open(Models+'_FNETCS_GAM.pi','wb')
pk.dump(FNETCS_GAM,FNETCS_GAM_file)
FNETCS_GAM_file.close()

ECS_file = open(Models+'_ECS_GREG_Grid.pi','wb')
pk.dump(ECS,ECS_file)
ECS_file.close()

#ECSCS_file = open(Models+'_ECSCS_GREG_Grid.pi','wb')
#pk.dump(ECSCS,ECSCS_file)
#ECSCS_file.close()
