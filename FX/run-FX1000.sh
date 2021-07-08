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
# for args in `seq 1 11`
# do
# #     ./diffusion.clang.out -s 0 -e 0 >> 0.baseline.csv
# #     ./diffusion.clang.out -s 1 -e 1 >> 1.first.csv
# #     ./diffusion.clang.out -s 2 -e 2 >> 2.peeling.csv
# #     ./diffusion.clang.out -s 27 -e 27 >> 3.y.csv
#      # ./diffusion.clang.out -s 28 -e 28 >> 4.yz.csv
# #     ./diffusion.clang.out -s 5 -e 5 >> 5.intrin.csv
# #     ./diffusion.clang.out -s 13 -e 13 >> 6.woFMA.csv
# #     ./diffusion.clang.out -s 11 -e 11 >> 7.unroll.csv
#     # ./diffusion.clang.out -s 19 -e 19 >> 8.register.csv
#     # ./diffusion.clang.out -s 20 -e 20 >> 9.blocking.csv
#     # ./diffusion.out -s 38 -e 38 >> 10.tb.csv
# done

STR=39
END=39
TAG=01
# for args in `seq 1 17`
# do 
# #    fapp -C -d ./rep_${STR}_${END}_s${args} -Hevent=pa${args} ./diffusion.out -nx 256 -ny 256 -nz 256 -s ${STR} -e ${END}
#     fapp -C -d ./rep_${STR}_${END}_${TAG}_s${args} -Hevent=pa${args} ./diffusion.out -s ${STR} -e ${END}
# done

# fapp -C -d ./rep_s1 -Hevent=pa1 ./diffusion.out -s 3 -e 6
# fapp -C -d ./rep_s2 -Hevent=pa2 ./diffusion.out -s 10 -e 15
# fapp -C -d ./rep_s3 -Hevent=pa3 ./diffusion.out -s 37 
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
