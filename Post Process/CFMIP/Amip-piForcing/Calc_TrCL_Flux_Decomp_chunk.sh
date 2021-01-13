#!/bin/bash

#Model='CanESM5_p1 CanESM5_p2 CanESM5_p3 CanESM5_p4 CanESM5_p5 CanESM5_p6'
Model='CanESM5_p2'
for ESM in $Model
do
    Source=/praid/users/jgvirgin/CanESM_Data/$ESM/amip

    echo -e 'NOW ON ESM - ' $ESM '\n'
    step='0 1 2 3 4 5'
    for number in $step
    do
        echo -e 'Now on Step - ' $number '\n'
        ./Calc_TrCL_Flux_Decomp.py $Model $number
        mv *$Model* $Source/Fluxes/
        echo -e 'Finished with step - ' $number '\n'
    done
    echo -e 'Finished with ESM - ' $Model '\n'
    echo 'Merging chunked fluxes...'
    ./Cat_TrCL_Flux_Decomp.py $Model
    mv *$Model* $Source/Fluxes/
    
done
