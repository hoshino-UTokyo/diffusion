#! /bin/bash -x                                                                          
################################################################################         
#                                                                                        
# ------ FOR Oakforest-PACS -----                                                        
#                                                                                        
################################################################################         
#PJM -g pz0108
##PJM -L rscgrp=regular-flat                                                              
#PJM -L rscgrp=debug
#PJM -L node=1
#PJM --mpi proc=1
#PJM --omp thread=28
#PJM -L elapse=00:30:00
#PJM -X  
#PJM -j 
#PJM -s

module purge
module load intel impi advisor

#source /usr/local/bin/hybrid_core_setting.sh 2

#mpiexec.hydra -n ${PJM_MPI_PROC} numactl --membind=1 ./bem-bb-SCM0.out ./bbi/input_200ts.pbf >> opt0.200ts.txt.${PJM_JOBID}.${PJM_MPI_PROC}.${OMP_NUM_THREADS}


export KMP_AFFINITY=compact
# export OMP_NUM_THREADS=1
# numactl --cpubind=0 --membind=0 ./L2test_c

# export OMP_NUM_THREADS=2
# numactl --cpubind=0 --membind=0 ./L2test_c

export OMP_NUM_THREADS=28
ulimit -s unlimited
for args in `seq 1 11`
do 
    # numactl --cpubind=0 --membind=0 ./diffusion.out -s 0 -e 0 >> 0.base.csv
    # numactl --cpubind=0 --membind=0 ./diffusion.out -s 1 -e 1 >> 1.first.csv
    # numactl --cpubind=0 --membind=0 ./diffusion.out -s 2 -e 2 >> 2.peel.csv
    # numactl --cpubind=0 --membind=0 ./diffusion.out -s 12 -e 12 >> 3.y.csv
    # numactl --cpubind=0 --membind=0 ./diffusion.out -s 13 -e 13 >> 4.yz.csv
    # numactl --cpubind=0 --membind=0 ./diffusion.out -s 6 -e 6 >> 6.woFMA.csv
    # numactl --cpubind=0 --membind=0 ./diffusion.out -s 5 -e 5 >> 5.intrin2.csv
    # numactl --cpubind=0 --membind=0 ./diffusion.out -s 7 -e 7 >> 7.unroll.csv
    # numactl --cpubind=0 --membind=0 ./diffusion.out -s 8 -e 8 >> 8.reg.csv
    # numactl --cpubind=0 --membind=0 ./diffusion.out -s 9 -e 9 >> 9.block.csv
    numactl --cpubind=0 --membind=0 ./diffusion.out -s 14 -e 14 >> 10.tb.csv
done
