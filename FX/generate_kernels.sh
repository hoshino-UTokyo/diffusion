#!/bin/sh

for args in `seq -f %02g 1 40`
do  
    cp diffusion_ker00.h diffusion_ker${args}.h
    sed -i -e "s/ker00/ker${args}/g" diffusion_ker${args}.h
    sed -i -e "s/KER00/KER${args}/g" diffusion_ker${args}.h
done
