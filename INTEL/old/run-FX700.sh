#!/bin/bash
#SBATCH --partition=fx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
##SBATCH -J test

module load fujitsu
export FLIB_FASTOMP=TRUE
export FLIB_HPCFUNC=TRUE
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand

export OMP_NUM_THREADS=48
./diffusion.out

