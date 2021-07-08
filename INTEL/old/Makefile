#SYSTEM = INTEL
#SYSTEM = KNL
SYSTEM = FX

#intel
ifeq ($(SYSTEM),INTEL)
OPTFLAGS = -align array64byte -xCORE-AVX512 -qopenmp -O3 -lm -DAVX512
OPTFLAGC = -xCORE-AVX512 -qopenmp -O3 -lm -DAVX512
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
OPTFLAGC = -Kfast,openmp,zfill -O3 -DSVE -DBLAS -Kocl,restp=all -Khpctag -Kopt prefetch_iteration=20
CC=fccpx
F90=frtpx
CCFLAGS = $(OPTFLAGC)
F90FLAGS = $(OPTFLAGS) -Cpp
#LDFLAGS = -Kfast,openmp -lfjlapack
LDFLAGS = -Kfast,openmp -SSL2BLAMP
endif

LINK= $(CC)

OBJS= diffusion.o
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

