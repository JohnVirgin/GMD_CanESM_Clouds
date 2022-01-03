#!/usr/bin/env python
# coding: utf-8

#import packages
import numpy as np
import pickle as pk
import Jacks_Functions as jf
import netCDF4 as nc
import Area_Avg as aa
import pandas as pd
import sys

models = sys.argv[1]

Source = '/home/jgvirgin/Projects/CanESM/Data/'+models+'/Anomalies/'

#define dimensions for horizontal grid
lat = np.linspace(-87.864,87.864,64)
lon = np.linspace(0,357.1875,128)


print('read in TOA fluxes from model output')
rlutcs = pk.load(open(Source+models+'_Ajd_4xCO2_TmSrs.pi','rb'))['rlutcs']
rsutcs = pk.load(open(Source+models+'_Ajd_4xCO2_TmSrs.pi','rb'))['rsutcs']
rsdt = pk.load(open(Source+models+'_Ajd_4xCO2_TmSrs.pi','rb'))['rsdt']

rsntcs = rsdt-rsutcs
FNETCS_stk = rsntcs-rlutcs


print('read in alb/WV/temp related fluxes')

vari = ['TrWV','SrWV','LAPSE','PLANCK','SrTEMP','TrALB']

vars_data = dict()
for i in range(len(vari)):
    vars_data[vari[i]] = pk.load(open(\
    '/home/jgvirgin/Projects/CanESM/Data/'+models+'/Fluxes/'+models+'_'+vari[i]+'_FLUXCS_Grid.pi','rb'))



print('calculate clear sky pure radiative forcing')

DIRCS = dict()
DIR = dict()
flux_SUM = dict()
for kernels in vars_data['PLANCK'].keys():
    flux_SUM[kernels] = np.nansum(\
    (vars_data['TrWV'][kernels],vars_data['SrWV'][kernels],vars_data['SrTEMP'][kernels],\
    vars_data['TrALB'][kernels],vars_data['LAPSE'][kernels],vars_data['PLANCK'][kernels]),axis=0)

    DIRCS[kernels] = np.nanmean(FNETCS_stk,axis=1)-flux_SUM[kernels]
    DIR[kernels] = DIRCS[kernels]/1.16


#print('create global, annual mean array for validation')

#DIRCS_GAM = aa.LatLonavg_Time(DIRCS[Models[0]]['CAM3'],lat,lon)

print('saving vars')

DIRCS_file = open(models+'_DIR_FLUXCS_Grid.pi','wb')
pk.dump(DIRCS,DIRCS_file)
DIRCS_file.close()

DIR_file = open(models+'_DIR_FLUX_Grid.pi','wb')
pk.dump(DIR,DIR_file)
DIR_file.close()

#DIRCS_GAM_file = open(Models[0]+'_DIRCS_test.pickle','wb')
#pk.dump(DIRCS_GAM,DIRCS_GAM_file)
#DIRCS_GAM_file.close()

print('done')
