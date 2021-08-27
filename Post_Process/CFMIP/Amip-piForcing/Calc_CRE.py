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

print('read in data')
var = pk.load(open(Source+models+'_Ajd_amip-piControl_TmSrs.pi', 'rb'))


var.pop('hus')
var.pop('ta')
var.pop('Alb')

print('CRE calculation')

rsnt = var['rsdt']-var['rsut']
rsntcs = var['rsdt']-var['rsutcs']
CRE_lw = -(np.nanmean((var['rlut']-var['rlutcs']),axis=1))
CRE_sw = np.nanmean((rsnt-rsntcs),axis=1)

print('Saving the CRE separately')

CRElw_file = open(models+'_TrCRElw_FLUX_Grid.pi','wb')
pk.dump(CRE_lw,CRElw_file)
CRElw_file.close()

CREsw_file = open(models+'_TrCREsw_FLUX_Grid.pi','wb')
pk.dump(CRE_sw,CREsw_file)
CREsw_file.close()
