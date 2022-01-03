#!/usr/bin/env python
# coding: utf-8

## Import Packages
from scipy.interpolate import interp1d
import numpy as np
import _pickle as pk
import Area_Avg as aa
from netCDF4 import Dataset
import sys
import glob
from natsort import natsorted

model = sys.argv[1]

Source = '/praid/users/jgvirgin/CanESM_Data/'+model+'/Fluxes/'
print('reading in model cloud flux data')

chunk0_lw = pk.load(open(Source+model+'_0_TrCLlw_FLUX_MZdecomp_Grid.pi','rb'))
chunk1_lw = pk.load(open(Source+model+'_1_TrCLlw_FLUX_MZdecomp_Grid.pi', 'rb'))
chunk2_lw = pk.load(open(Source+model+'_2_TrCLlw_FLUX_MZdecomp_Grid.pi', 'rb'))
chunk3_lw = pk.load(open(Source+model+'_3_TrCLlw_FLUX_MZdecomp_Grid.pi', 'rb'))
chunk4_lw = pk.load(open(Source+model+'_4_TrCLlw_FLUX_MZdecomp_Grid.pi', 'rb'))
chunk5_lw = pk.load(open(Source+model+'_5_TrCLlw_FLUX_MZdecomp_Grid.pi', 'rb'))

chunk0_sw = pk.load(open(Source+model+'_0_TrCLsw_FLUX_MZdecomp_Grid.pi', 'rb'))
chunk1_sw = pk.load(open(Source+model+'_1_TrCLsw_FLUX_MZdecomp_Grid.pi', 'rb'))
chunk2_sw = pk.load(open(Source+model+'_2_TrCLsw_FLUX_MZdecomp_Grid.pi', 'rb'))
chunk3_sw = pk.load(open(Source+model+'_3_TrCLsw_FLUX_MZdecomp_Grid.pi', 'rb'))
chunk4_sw = pk.load(open(Source+model+'_4_TrCLsw_FLUX_MZdecomp_Grid.pi', 'rb'))
chunk5_sw = pk.load(open(Source+model+'_5_TrCLsw_FLUX_MZdecomp_Grid.pi', 'rb'))


print('merging along the time dimension and taking the annual mean')
merged_lw = dict()
merged_sw = dict()
for cld in chunk1_lw.keys():
    merged_lw[cld] = dict()
    merged_sw[cld] = dict()
    for hieght in chunk1_lw[cld].keys():
        merged_lw[cld][hieght] = np.concatenate(\
            (chunk0_lw[cld][hieght],
             chunk1_lw[cld][hieght],
             chunk2_lw[cld][hieght],
             chunk3_lw[cld][hieght],
             chunk4_lw[cld][hieght],
             chunk5_lw[cld][hieght]), axis = 0)

        merged_sw[cld][hieght] = np.concatenate(
            (chunk0_sw[cld][hieght],
             chunk1_sw[cld][hieght],
             chunk2_sw[cld][hieght],
             chunk3_sw[cld][hieght],
             chunk4_sw[cld][hieght],
             chunk5_sw[cld][hieght]), axis=0)


print('check shape of final array - ', merged_lw['Standard']['All'].shape)

print('saving')

file_lw = open(model+'_TrCLlw_FLUX_SRdecomp_Grid.pi','wb')
pk.dump(merged_lw,file_lw)
file_lw.close()

file_sw = open(model+'_TrCLsw_FLUX_SRdecomp_Grid.pi', 'wb')
pk.dump(merged_sw, file_sw)
file_sw.close()
