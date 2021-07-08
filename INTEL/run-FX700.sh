#!/bin/bash
#SBATCH --partition=fx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
##SBATCH -J test

module load fujitsu
module unload fujitsu
module load fujitsu
export FLIB_FASTOMP=TRUE
export FLIB_HPCFUNC=TRUE
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand

export OMP_NUM_THREADS=48
#./diffusion.out -s 5
#./diffusion.out -s 5

for args in `seq 1 17`
do 
    fapp -C -d ./rep_26_s${args} -Hevent=pa${args} ./diffusion.out -nx 128 -ny 384 -nz 128 -s 26 -e 26
done

# fapp -C -d ./rep_s1 -Hevent=pa1 ./diffusion.out -s 6 -e 6
# fapp -C -d ./rep_s2 -Hevent=pa2 ./diffusion.out -s 6 -e 6
# fapp -C -d ./rep_s3 -Hevent=pa3 ./diffusion.out -s 6 -e 6
# fapp -C -d ./rep_s4 -Hevent=pa4 ./diffusion.out -s 6 -e 6
# fapp -C -d ./rep_s5 -Hevent=pa5 ./diffusion.out -s 6 -e 6
# fapp -C -d ./rep_s6 -Hevent=pa6 ./diffusion.out -s 6 -e 6
# fapp -C -d ./rep_s7 -Hevent=pa7 ./diffusion.out -s 6 -e 6
# fapp -C -d ./rep_s8 -Hevent=pa8 ./diffusion.out -s 6 -e 6
# fapp -C -d ./rep_s9 -Hevent=pa9 ./diffusion.out -s 6 -e 6
# fapp -C -d ./rep_s10 -Hevent=pa10 ./diffusion.out -s 6 -e 6
# fapp -C -d ./rep_s11 -Hevent=pa11 ./diffusion.out -s 6 -e 6
# fapp -C -d ./rep_s12 -Hevent=pa12 ./diffusion.out -s 6 -e 6
# fapp -C -d ./rep_s13 -Hevent=pa13 ./diffusion.out -s 6 -e 6
# fapp -C -d ./rep_s14 -Hevent=pa14 ./diffusion.out -s 6 -e 6
# fapp -C -d ./rep_s15 -Hevent=pa15 ./diffusion.out -s 6 -e 6
# fapp -C -d ./rep_s16 -Hevent=pa16 ./diffusion.out -s 6 -e 6
# fapp -C -d ./rep_s17 -Hevent=pa17 ./diffusion.out -s 6 -e 6
