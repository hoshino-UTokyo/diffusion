SYSTEM = INTEL
#SYSTEM = KNL
#SYSTEM = FX

#intel
ifeq ($(SYSTEM),INTEL)
OPTFLAGS = -align array64byte -xCORE-AVX512 -qopenmp -O3 -lm -DAVX512
OPTFLAGC = -xCORE-AVX512 -qopenmp -O3 -lm -DAVX512 -std:c99 
CC=icc
F90=ifort
CCFLAGS = $(OPTFLAGC)
#F90FLAGS = $(OPTFLAGS) -fpp -assume nounderscore -names uppercase
F90FLAGS = $(OPTFLAGS) -fpp
#F90FLAGS = $(OPTFLAGS) -fpp -check all
#F90FLAGS = -fpe0 -traceback -g -CB -assume nounderscore -names lowercase -fpp -check all
#LDFLAGS = -mkl -trace
LDFLAGS = -qopenmp -mkl
endif

#intel-knl
ifeq ($(SYSTEM),KNL)
OPTFLAGS = -align array64byte -xMIC-AVX512 -qopenmp -O3 -lm -DAVX512
OPTFLAGC = -xMIC-AVX512 -qopenmp -O3 -lm -DAVX512
CC=icc
F90=ifort
CCFLAGS = $(OPTFLAGC)
#F90FLAGS = $(OPTFLAGS) -fpp -assume nounderscore -names uppercase
F90FLAGS = $(OPTFLAGS) -fpp
#F90FLAGS = $(OPTFLAGS) -fpp -check all
#F90FLAGS = -fpe0 -traceback -g -CB -assume nounderscore -names lowercase -fpp -check all
#LDFLAGS = -mkl -trace
LDFLAGS = -qopenmp -mkl
endif

#fx700
ifeq ($(SYSTEM),FX)
OPTFLAGS = -Kfast,openmp,zfill -O3 -DSVE
OPTFLAGC = -Kfast,openmp -O3 -DSVE -DBLAS -Khpctag -Nfjomplib -Nfjprof #-Kzfill=18 -Kprefetch_line=9 -Kprefetch_line_L2=70 -Kprefetch_sequential=soft 
#OPTFLAGC = -g -DSVE -Kopenmp #-Kzfill=18 -Kprefetch_line=9 -Kprefetch_line_L2=70 -Kprefetch_sequential=soft 
#OPTFLAGC = -Kfast,openmp -O3 -DSVE -Nclang #-Kzfill=18 -Kprefetch_line=9 -Kprefetch_line_L2=70 -Kprefetch_sequential=soft 
#OPTFLAGC += -Kprefetch_sequential=soft,prefetch_strong,prefetch_line=9 -Kprefetch_line_L2=70 
CC=fccpx
F90=frtpx
CCFLAGS = $(OPTFLAGC)
F90FLAGS = $(OPTFLAGS) -Cpp
#LDFLAGS = -Kfast,openmp -lfjlapack
LDFLAGS = $(OPTFLAGC)
endif

LINK= $(CC)

# OBJS =  diffusion_ker00.o diffusion_ker01.o diffusion_ker02.o diffusion_ker03.o diffusion_ker04.o diffusion_ker05.o diffusion_ker06.o diffusion_ker07.o diffusion_ker08.o diffusion_ker09.o
# OBJS += diffusion_ker10.o diffusion_ker11.o diffusion_ker12.o diffusion_ker13.o diffusion_ker14.o diffusion_ker15.o diffusion_ker16.o diffusion_ker17.o diffusion_ker18.o diffusion_ker19.o
# OBJS += diffusion_ker20.o diffusion_ker21.o diffusion_ker22.o diffusion_ker23.o diffusion_ker24.o diffusion_ker25.o diffusion_ker26.o
# OBJS += main.o 

OBJS =  diffusion_ker00.o diffusion_ker01.o diffusion_ker02.o diffusion_ker03.o diffusion_ker04.o diffusion_ker05.o
OBJS += diffusion_ker11.o diffusion_ker13.o diffusion_ker19.o 
OBJS += diffusion_ker20.o diffusion_ker24.o diffusion_ker25.o diffusion_ker27.o diffusion_ker28.o diffusion_ker38.o
OBJS += main.o 

#OBJS= matvec.o

TARGET=diffusion.out
#TARGET=matvec.out

.SUFFIXES: .o .c .out

$(TARGET): $(OBJS)
			$(LINK) -o $@ $(OBJS) $(LDFLAGS)

.c.o: *.c
			$(CC) -c $(CCFLAGS) $<

.f90.o: *.f90
			$(F90) -c $< $(F90FLAGS)

clean:
	rm -f *.o *.mod $(TARGET) *.optrpt *~

rmod:
	rm -f m_*.o *.mod

