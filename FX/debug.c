#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#include "diffusion.h"


#ifndef REAL
#define REAL double
#endif

#ifndef NX
#define NX (256)
#endif
#ifndef NY
#define NY (128)
#endif
#ifndef NZ
#define NZ (128)
#endif

#ifndef M_PI
#define M_PI (3.1415926535897932384626)
#endif

void allocate_ker00(REAL **buff_ret, const int nx, const int ny, const int nz) {

  posix_memalign((void**)buff_ret, 64, sizeof(REAL)*nx*ny*nz);

}

void init_ker00(REAL *buff1, const int nx, const int ny, const int nz,
		const REAL kx, const REAL ky, const REAL kz,
		const REAL dx, const REAL dy, const REAL dz,
		const REAL kappa, const REAL time) {

  REAL ax, ay, az;
  int jz, jy, jx;
  ax = exp(-kappa*time*(kx*kx));
  ay = exp(-kappa*time*(ky*ky));
  az = exp(-kappa*time*(kz*kz));
  for (jz = 0; jz < nz; jz++) {
    for (jy = 0; jy < ny; jy++) {
      for (jx = 0; jx < nx; jx++) {
        int j = jz*nx*ny + jy*nx + jx;
        REAL x = dx*((REAL)(jx + 0.5));
        REAL y = dy*((REAL)(jy + 0.5));
        REAL z = dz*((REAL)(jz + 0.5));
        REAL f0 = (REAL)0.125
          *(1.0 - ax*cos(kx*x))
          *(1.0 - ay*cos(ky*y))
          *(1.0 - az*cos(kz*z));
        buff1[j] = f0;
      }
    }
  }
}

void diffusion_ker00(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		     REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		     REAL cb, REAL cc, REAL dt,
		     REAL **f1_ret, REAL **f2_ret, REAL *time_ret, int *count_ret) {
  
  {
    REAL time = 0.0;
    int count = 0;
    REAL *restrict f1_t = f1;
    REAL *restrict f2_t = f2;
    int c, w, e, n, s, b, t;
    int z, y, x;
    
    do {
/* #pragma omp parallel for private(x,y,z,c,w,e,n,s,b,t) */
/*       for (z = 0; z < nz; z++) { */
/* 	for (y = 0; y < ny; y++) { */
/* 	  for (x = 0; x < nx; x++) { */
/* 	    c =  x + y * nx + z * nx * ny; */
/* 	    w = (x == 0)    ? c : c - 1; */
/* 	    e = (x == nx-1) ? c : c + 1; */
/* 	    n = (y == 0)    ? c : c - nx; */
/* 	    s = (y == ny-1) ? c : c + nx; */
/* 	    b = (z == 0)    ? c : c - nx * ny; */
/* 	    t = (z == nz-1) ? c : c + nx * ny; */
/* 	    f2_t[c] = cc * f1_t[c] + cw * f1_t[w] + ce * f1_t[e] */
/* 	      + cs * f1_t[s] + cn * f1_t[n] + cb * f1_t[b] + ct * f1_t[t]; */
/* 	  } */
/* 	} */
/*       } */

      REAL *tmp = f1_t;
      f1_t = f2_t;
      f2_t = tmp;
            
      //      time += dt;
      time += 4.0*dt;
      //time += 1;
      count++;
    } while (time + 0.5*dt < 0.1);
    *f1_ret = f1_t;
    *f2_ret = f2_t;
    *time_ret = time;      
    *count_ret = count;        
  }
  return;
}


/* REAL accuracy(const REAL *b1, REAL *b2, const int len) { */
/*   REAL err = 0.0; */
/*   int i; */
/*   for (i = 0; i < len; i++) { */
/*     err += (b1[i] - b2[i]) * (b1[i] - b2[i]); */
/*   } */
/*   return (REAL)sqrt(err/len); */
/* } */

int main(int argc, char *argv[]) 
{

  struct timeval time_begin, time_end;

  int nx    = NX;
  int ny    = NY;
  int nz    = NZ;

  /* int num_kers = sizeof(diffusion)/sizeof(*diffusion); */
  /* int str = 0; */
  /* int end = num_kers-1; */
  
  /* int i; */
  /* for(i = 1; i < argc; i++){ */
  /*   if(strcmp(argv[i],"-nx") == 0 || strcmp(argv[i],"-NX") == 0){ */
  /*     i++; */
  /*     nx = atoi(argv[i]); */
  /*   } */
  /*   if(strcmp(argv[i],"-ny") == 0 || strcmp(argv[i],"-NY") == 0){ */
  /*     i++; */
  /*     ny = atoi(argv[i]); */
  /*   } */
  /*   if(strcmp(argv[i],"-nz") == 0 || strcmp(argv[i],"-NZ") == 0){ */
  /*     i++; */
  /*     nz = atoi(argv[i]); */
  /*   } */
  /*   if(strcmp(argv[i],"-n") == 0 || strcmp(argv[i],"-N") == 0){ */
  /*     i++; */
  /*     nx = atoi(argv[i]); */
  /*     ny = atoi(argv[i]); */
  /*     nz = atoi(argv[i]); */
  /*   } */
  /*   if(strcmp(argv[i],"-s") == 0 || strcmp(argv[i],"-str") == 0 || strcmp(argv[i],"-start") == 0){ */
  /*     i++; */
  /*     str = atoi(argv[i]); */
  /*   } */
  /*   if(strcmp(argv[i],"-e") == 0 || strcmp(argv[i],"-end") == 0){ */
  /*     i++; */
  /*     int tmp = atoi(argv[i]); */
  /*     if(tmp > num_kers-1){ */
  /* 	fprintf(stderr,"warning: the end number shoud be smaller than the number of registered kernels (%d)\n", num_kers); */
  /*     }else{ */
  /* 	end = tmp; */
  /*     } */
  /*   } */
  /* } */
    
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

  printf("(nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
  /* int ker_id; */
  /* for(ker_id = str;ker_id <= end;ker_id++){ */

  /*   char name[200] = "diffusion"; */
  /*   char ker_id_string[6]; */
  /*   if(ker_id < 10) { */
  /*     sprintf(ker_id_string,"%s%d%d","_ker",0,ker_id); */
  /*   }else{ */
  /*     sprintf(ker_id_string,"%s%d","_ker",ker_id); */
  /*   } */
  /*   strcat(name,ker_id_string); */
  /*   int num_opt = sizeof(opt_list)/sizeof(*opt_list); */
  /*   //int num_opt = sizeof(opt_flags[ker_id])/sizeof(*opt_flags[ker_id]); */
  /*   int i; */
  /*   for(i = 0;i < num_opt;i++){ */
  /*     if(opt_flags[ker_id][i]) { */
  /* 	char bar[] = "_"; */
  /* 	strcat(name,bar); */
  /* 	strcat(name,opt_list[i]); */
  /*     } */
  /*   } */
    
    REAL *f1 = NULL;
    REAL *f2 = NULL;

    allocate_ker00(&f1,nx,ny,nz);
    allocate_ker00(&f2,nx,ny,nz);
    
    REAL   time  = 0.0;
    int    count = 0;
    init_ker00(f1, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);
    init_ker00(f2, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);
    /* gettimeofday(&time_begin, NULL); */



    REAL *restrict f1_t = f1;
    REAL *restrict f2_t = f2;
    int c, w, e, n, s, b, t;
    int z, y, x;
    
    diffusion_ker00(f1, f2, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt,
		    &f1, &f2, &time, &count);
    /* gettimeofday(&time_end, NULL); */
    // print data
    /* REAL *answer = (REAL *)malloc(sizeof(REAL) * nx*ny*nz); */
    /* init_ker00(answer, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time); */
    /* REAL err = accuracy(f1, answer, nx*ny*nz); */

    /* double elapsed_time = (time_end.tv_sec - time_begin.tv_sec) */
    /*   + (time_end.tv_usec - time_begin.tv_usec)*1.0e-6; */
    /* REAL gflops = (nx*ny*nz)*13.0*count/elapsed_time * 1.0e-09; */
    /* double thput = (nx * ny * nz) * sizeof(REAL) * 2.0 * count */
    /*   / elapsed_time * 1.0e-09; */
    /* fprintf(stdout, "\n %s\n", name); */
    /* fprintf(stdout, "elapsed time : %.3f (s)\n", elapsed_time); */
    /* fprintf(stdout, "flops        : %.3f (GFlops)\n", gflops); */
    /* fprintf(stdout, "throughput   : %.3f (GB/s)\n", thput); */
    /* fprintf(stdout, "accuracy     : %e\n", err); */
    /* fprintf(stdout, "count        : %.3d\n", count); */
    /* free(answer);  */
    free(f1);
    free(f2);
    
  /* } */

  return 0;
}
