#!/bin/sh 
#PJM -g gk57
#PJM -L rscgrp=debug-flat
#PJM -L node=1
#PJM --mpi proc=1
#PJM --omp thread=64
#PJM -L elapse=00:30:00
#PJM -N diffusion
#PJM -X
#PJM -j
#PJM -s

module switch intel/2017.4.196 intel/2018.0.128
export OMP_STACKSIZE=1G
ulimit -s 1000000
# HTX=1
# source /work/gc26/share/sample-scripts/core-setting.sh
# numactl --membind=1 ./diffusion
# source /work/gc26/share/sample-scripts/core-setting2.sh
# numactl --membind=1 ./diffusion
# source /work/gc26/share/sample-scripts/core-setting3.sh
# numactl --membind=1 ./diffusion

# source /work/gc26/share/sample-scripts/core-setting.sh
export OMP_NUM_THREADS=64
export KMP_HW_SUBSET=64C@2,1T
export KMP_AFFINITY=scatter,verbose
numactl --membind=1 ./diffusion

# #export OMP_NUM_THREADS=64
# echo ${OMP_NUM_THREADS}
# #export KMP_AFFINITY=granularity=fine,proclist=[2-67],explicit


# HTX=1
# source /work/gc26/share/sample-scripts/core-setting2.sh
# # export OMP_STACKSIZE=1G
# # ulimit -s 1000000
# # export OMP_NUM_THREADS=64
# # export KMP_HW_SUBSET=64C@2,1T
# # export KMP_AFFINITY=scatter,verbose
# # #export OMP_NUM_THREADS=64
# # echo ${OMP_NUM_THREADS}
# #export KMP_AFFINITY=granularity=fine,proclist=[2-67],explicit

# numactl --membind=1 ./diffusion
