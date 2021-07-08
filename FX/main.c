#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#ifdef INTEL
#include "immintrin.h"
#endif
#ifdef _OPENACC
#include <openacc.h>
#endif

#include "fj_tool/fapp.h"
#include "diffusion.h"


#ifndef REAL
#define REAL double
#endif

#ifndef NX
#define NX (256)
#endif
#ifndef NY
#define NY (384)
#endif
#ifndef NZ
#define NZ (384)
#endif

#ifndef M_PI
#define M_PI (3.1415926535897932384626)
#endif

REAL accuracy(const REAL *b1, REAL *b2, const int len) {
  REAL err = 0.0;
  int i;
  for (i = 0; i < len; i++) {
    err += (b1[i] - b2[i]) * (b1[i] - b2[i]);
  }
  return (REAL)sqrt(err/len);
}

int main(int argc, char *argv[]) 
{

  struct timeval time_begin, time_end;

  int nx    = NX;
  int ny    = NY;
  int nz    = NZ;

  int num_kers = sizeof(diffusion)/sizeof(*diffusion);
  int str = 0;
  int end = num_kers-1;


  
  int i;
  for(i = 1; i < argc; i++){
    if(strcmp(argv[i],"-nx") == 0 || strcmp(argv[i],"-NX") == 0){
      i++;
      nx = atoi(argv[i]);
    }
    if(strcmp(argv[i],"-ny") == 0 || strcmp(argv[i],"-NY") == 0){
      i++;
      ny = atoi(argv[i]);
    }
    if(strcmp(argv[i],"-nz") == 0 || strcmp(argv[i],"-NZ") == 0){
      i++;
      nz = atoi(argv[i]);
    }
    if(strcmp(argv[i],"-n") == 0 || strcmp(argv[i],"-N") == 0){
      i++;
      nx = atoi(argv[i]);
      ny = atoi(argv[i]);
      nz = atoi(argv[i]);
    }
    if(strcmp(argv[i],"-s") == 0 || strcmp(argv[i],"-str") == 0 || strcmp(argv[i],"-start") == 0){
      i++;
      str = atoi(argv[i]);
    }
    if(strcmp(argv[i],"-e") == 0 || strcmp(argv[i],"-end") == 0){
      i++;
      int tmp = atoi(argv[i]);
      if(tmp > num_kers-1){
	fprintf(stderr,"warning: the end number shoud be smaller than the number of registered kernels (%d)\n", num_kers);
      }else{
	end = tmp;
      }
    }
  }
  
  /* REAL   time  = 0.0; */
  /* int    count = 0;   */
  
  REAL l, dx, dy, dz, kx, ky, kz, kappa, dt;
  REAL ce, cw, cn, cs, ct, cb, cc;

  l = 1.0;
  kappa = 0.1;
  dx = dy = dz = l / nx;
  kx = ky = kz = 2.0 * M_PI;
  dt = 0.1*dx*dx / kappa;
  ce = cw = kappa*dt/(dx*dx);
  cn = cs = kappa*dt/(dy*dy);
  ct = cb = kappa*dt/(dz*dz);
  cc = 1.0 - (ce + cw + cn + cs + ct + cb);

#ifdef DEBUG
  fprintf(stderr,"debug mode. answer making\n");
  REAL *answer = NULL;
  REAL *answer2 = NULL;
  (*allocate[1])(&answer ,nx,ny,nz);
  (*allocate[1])(&answer2,nx,ny,nz);
  REAL   time  = 0.0;
  int    count = 0;
  (*init[1])(answer , nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);
  (*init[1])(answer2, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);
  (*diffusion[1])(answer, answer2, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt,
		  &answer, &answer2, &time, &count);
#endif

  printf("(nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
  int ker_id;
  for(ker_id = str;ker_id <= end;ker_id++){

    char name[300] = "diffusion";
    char ker_id_string[6];
    if(ker_id < 10) {
      sprintf(ker_id_string,"%s%d%d","_ker",0,ker_id);
    }else{
      sprintf(ker_id_string,"%s%d","_ker",ker_id);
    }
    strcat(name,ker_id_string);
    int num_opt = sizeof(opt_list)/sizeof(*opt_list);
    //int num_opt = sizeof(opt_flags[ker_id])/sizeof(*opt_flags[ker_id]);
    int i;
    for(i = 0;i < num_opt;i++){
      //      fprintf(stdout, "%d/%d,%s,%d,%d\n", i,num_opt,opt_list[i],opt_flags[ker_id][i],ker_id);
      if(opt_flags[ker_id][i]) {
	char bar[] = "_";
	strcat(name,bar);
	strcat(name,opt_list[i]);
      }
    }
    
    REAL *f1 = NULL;
    REAL *f2 = NULL;

    (*allocate[ker_id])(&f1,nx,ny,nz);
    (*allocate[ker_id])(&f2,nx,ny,nz);

    REAL   time  = 0.0;
    int    count = 0;
    (*init[ker_id])(f1, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);
    (*init[ker_id])(f2, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);
    gettimeofday(&time_begin, NULL);
    fapp_start(name,ker_id,0);
    (*diffusion[ker_id])(f1, f2, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt,
			 &f1, &f2, &time, &count);
    fapp_stop(name,ker_id,0);
    gettimeofday(&time_end, NULL);
    // print data

#ifndef DEBUG
    REAL *answer = (REAL *)malloc(sizeof(REAL) * nx*ny*nz);
    (*init[1])(answer, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);
#endif
    REAL err = accuracy(f1, answer, nx*ny*nz);

    double elapsed_time = (time_end.tv_sec - time_begin.tv_sec)
      + (time_end.tv_usec - time_begin.tv_usec)*1.0e-6;
    REAL gflops = (nx*ny*nz)*13.0*count/elapsed_time * 1.0e-09;
    double thput = (nx * ny * nz) * sizeof(REAL) * 2.0 * count
      / elapsed_time * 1.0e-09;
    fprintf(stdout, "\n %s\n", name);
    fprintf(stdout, "elapsed time : %.3f (s)\n", elapsed_time);
    fprintf(stdout, "flops        : %.3f (GFlops)\n", gflops);
    fprintf(stdout, "throughput   : %.3f (GB/s)\n", thput);
    fprintf(stdout, "accuracy     : %e\n", err);
    fprintf(stdout, "count        : %.3d\n", count);
    free(f1);
    free(f2);
#ifndef DEBUG
    free(answer);
#endif
    
  }
#ifdef DEBUG    
  free(answer);
#endif

  return 0;
}
