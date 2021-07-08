#! /bin/bash -x                                                                          
################################################################################         
#                                                                                        
# ------ FOR Oakforest-PACS -----                                                        
#                                                                                        
################################################################################         
#PJM -g gz00
#PJM -L rscgrp=regular-o
#PJM -L node=1
#PJM --mpi proc=1
#PJM --omp thread=48
#PJM -L elapse=01:00:00
##PJM -X  
##PJM -j 
##PJM -s

cd $PJM_O_WORKDIR
module purge
module load fj/1.2.31

export OMP_STACKSIZE=1G
ulimit -s 1000000
echo ${OMP_NUM_THREADS}

export FLIB_FASTOMP=TRUE
export FLIB_HPCFUNC=TRUE
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand

./diffusion.out -s 3 -e 6
./diffusion.out -s 10 -e 15
./diffusion.out -s 37 

# OPT=H4
# INPUT=200ts
# # mpiexec -n ${PJM_MPI_PROC} numactl -l ./bem-bb-SCM_${OPT}.out ./bbi/input_${INPUT}.pbf >> ${OPT}.${INPUT}.${PJM_JOBID}.${OMP_NUM_THREADS}
# for args in `seq 1 17`
# do
#     fapp -C -d ./rep_${OPT}_${INPUT}_s${args}_${PJM_JOBID} -Hevent=pa${args} mpiexec -n ${PJM_MPI_PROC} ./bem-bb-SCM_${OPT}.out ./bbi/input_${INPUT}.pbf >> ${OPT}.${INPUT}.${PJM_JOBID}.${OMP_NUM_THREADS}
# done

# for OPT in H0 H1 H2 H3 H4 H5 H6 H7
# #for OPT in H0 H7
# do 
#     for INPUT in 200ts 320th
#     do 
# 	for TIMES in `seq 1 5`
# 	do 
# 	    mpiexec -n ${PJM_MPI_PROC} ./bem-bb-SCM_${OPT}.out ./bbi/input_${INPUT}.pbf >> ${OPT}.${INPUT}.${PJM_JOBID}.${OMP_NUM_THREADS}
# 	done
#     done
# done

