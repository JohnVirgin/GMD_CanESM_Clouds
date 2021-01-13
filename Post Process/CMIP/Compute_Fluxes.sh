#!/bin/bash

#Model='CanESM2 CanESM5_p1 CanESM5_p2'
#Model='CanESM5_p3 CanESM5_p4 CanESM5_p5 CanESM5_p6'
Model='CanESM5_p2'
for ESM in $Model
do

  Source=/home/jgvirgin/Projects/CanESM/Data/$ESM

  echo -e 'NOW ON ESM - ' $ESM '\n'
  echo 'Calculating Surface albedo fluxes'
  python Calc_TrAlb_Flux.py $ESM
  mv *$ESM* $Source/Fluxes/

  echo 'Now, Decomposing the Surface albedo fluxes'
  python Decomp_TrAlb_Flux.py $ESM
  mv *$ESM* $Source/Fluxes/

  echo 'Calculating Water Vapour fluxes'
  python Calc_WV_Flux.py $ESM
  mv *$ESM* $Source/Fluxes/

  echo 'Now, Decomposing Water Vapour fluxes'
  python Decomp_WV_Flux.py $ESM
  mv *$ESM* $Source/Fluxes/

  echo 'Calculating Stratospheric temperature fluxes'
  python Calc_SrTemp_Flux.py $ESM
  mv *$ESM* $Source/Fluxes/

  echo 'Calculating Tropospheric temperature fluxes'
  python Calc_TrTemp_Flux.py $ESM
  mv *$ESM* $Source/Fluxes/

  echo 'Now, Decomposing Temperature fluxes'
  python Decomp_TEMP_Flux.py $ESM
  mv *$ESM* $Source/Fluxes/

  echo 'Calculating Cloud Radiative Effect'
  python Calc_CRE.py $ESM
  mv *$ESM* $Source/Fluxes/

  echo 'Calculating CO2 Forcing'
  python Calc_DIR.py $ESM
  mv *$ESM* $Source/Fluxes/

  echo 'Calculating Cloud feedback fluxes'
  python Calc_TrCL_Flux.py $ESM
  mv *$ESM* $Source/Fluxes/

  echo 'Now, Decomposing Cloud fluxes'
  python Decomp_TrCL_Flux.py $ESM
  mv *$ESM* $Source/Fluxes/

  echo 'Area averaging both total and decomposed fluxes'
  python Calc_GAM_Fluxes.py $ESM
  python Calc_GAM_uFluxes.py $ESM
  mv *$ESM* $Source/Fluxes/

  echo 'Calculate Effective Radiative Forcing using the Regression technique'
  python Calc_ERFgreg.py $ESM
  mv *$ESM* $Source/ERF/

  #echo 'Lastly, Calculate Effective Radiative Forcing using the Hansen technique'
  #python Calc_ERFhans.py $ESM
  #mv *$ESM* $Source/ERF/

  echo -e '\nMOVING ONTO THE NEXT MODEL\n'

done
echo 'done!'
