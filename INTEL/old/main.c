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

#include "diffusion_kernel.h"


#ifndef REAL
#define REAL double
#endif
#define NX (256)
#ifndef M_PI
#define M_PI (3.1415926535897932384626)
#endif


int main(int argc, char *argv[]) 
{

  struct timeval time_begin, time_end;

  int i = 1;
  int    nx    = NX;
  int    ny    = NX;
  int    nz    = NX;

  /* REAL   time  = 0.0; */
  /* int    count = 0;   */
  
  REAL l, dx, dy, dz, kx, ky, kz, kappa, dt;
  REAL ce, cw, cn, cs, ct, cb, cc;

#ifdef _OPENACC
  acc_init(0);
#endif

  l = 1.0;
  kappa = 0.1;
  dx = dy = dz = l / nx;
  kx = ky = kz = 2.0 * M_PI;
  dt = 0.1*dx*dx / kappa;
  ce = cw = kappa*dt/(dx*dx);
  cn = cs = kappa*dt/(dy*dy);
  ct = cb = kappa*dt/(dz*dz);
  cc = 1.0 - (ce + cw + cn + cs + ct + cb);

  /* REAL *f1 = (REAL *)malloc(sizeof(REAL)*nx*ny*nz); */
  /* REAL *f2 = (REAL *)malloc(sizeof(REAL)*nx*ny*nz); */
  REAL *f1, *f2;
  posix_memalign((void**)&f1, 64, sizeof(REAL)*nx*ny*nz);
  posix_memalign((void**)&f2, 64, sizeof(REAL)*nx*ny*nz);


  // print data
  printf("(nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);

  REAL *answer = (REAL *)malloc(sizeof(REAL) * nx*ny*nz);

  void (*diffusion[])(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		      REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		      REAL cb, REAL cc, REAL dt,
		      REAL **f_ret, REAL *time_ret, int *count_ret) 
    = {diffusion_openmp
       //       diffusion_openmp_declare_simd
       ,diffusion_openmp_y
       ,diffusion_openmp_y_nowait
       /* ,diffusion_openmp_y_nowait_simdwrong */
       /* ,diffusion_openmp_y_nowait_simd */
       /* ,diffusion_openmp_y_nowait_simd_peel */
       /* ,diffusion_openmp_y_nowait_simd_peel_aligned */
       /* ,diffusion_openmp_y_nowait_simd_peel_aligned_mvparallel */
       /* ,diffusion_openmp_tiled_nowait_simd_peel_aligned */
       /* ,diffusion_openmp_tiled_nowait_simd_peel_aligned_mvparallel */
       /* ,diffusion_openmp_tiled_nowait_simd_peel_aligned_mvparallel_mvfor */
       /* ,diffusion_openmp_tiled_nowait_simd_peel_aligned_mvparallel_mvfor2 */
       /* //       ,diffusion_openmp_tiled2_nowait_simd_peel_aligned */
       /* ,diffusion_openmp_KNC_book_asis */
       /* ,diffusion_openmp_intrin */
       /* ,diffusion_like_stream_scale */
       /* //       ,diffusion_openmp_intrin_independent */
       /* ,diffusion_openmp_z_simd_peel_aligned */
  };
  char *name[] = {"diffusion_openmp",
		  //		  "diffusion_openmp_declare_simd",
		  "diffusion_openmp_y",
		  "diffusion_openmp_y_nowait",
		  "diffusion_openmp_y_nowait_simdwrong",
		  "diffusion_openmp_y_nowait_simd",
		  "diffusion_openmp_y_nowait_simd_peel",
		  "diffusion_openmp_y_nowait_simd_peel_aligned",
		  "diffusion_openmp_y_nowait_simd_peel_aligned_mvparallel",
		  "diffusion_openmp_tiled_nowait_simd_peel_aligned",
		  "diffusion_openmp_tiled_nowait_simd_peel_aligned_mvparallel",
		  "diffusion_openmp_tiled_nowait_simd_peel_aligned_mvparallel_mvfor",
		  "diffusion_openmp_tiled_nowait_simd_peel_aligned_mvparallel_mvfor2",
		  //		  "diffusion_openmp_tiled2_nowait_simd_peel_aligned",
		  "diffusion_openmp_KNC_book_asis",
		  "diffusion_openmp_intrin",
		  "diffusion_like_stream_scale",
		  //		  "diffusion_openmp_intrin_independent",
		  "diffusion_openmp_z_simd_peel_aligned"};

  int args;
  for(args = 0;args < 15;args++){
    REAL   time  = 0.0;
    int    count = 0;  
    init(f1, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);
    gettimeofday(&time_begin, NULL);
    /* diffusion_openacc(f1, f2, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt, */
    /* 		    &f1, &time, &count); */
    (*diffusion[args])(f1, f2, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt,
		       &f1, &time, &count);
    gettimeofday(&time_end, NULL);
    
    init(answer, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);
    REAL err = accuracy(f1, answer, nx*ny*nz);
    double elapsed_time = (time_end.tv_sec - time_begin.tv_sec)
      + (time_end.tv_usec - time_begin.tv_usec)*1.0e-6;
    REAL gflops = (nx*ny*nz)*13.0*count/elapsed_time * 1.0e-09;
    double thput = (nx * ny * nz) * sizeof(REAL) * 2.0 * count
      / elapsed_time * 1.0e-09;

    
    fprintf(stdout, "\n %s\n", name[args]);
    fprintf(stdout, "elapsed time : %.3f (s)\n", elapsed_time);
    fprintf(stdout, "flops        : %.3f (GFlops)\n", gflops);
    fprintf(stdout, "throughput   : %.3f (GB/s)\n", thput);  
    fprintf(stdout, "accuracy     : %e\n", err);
    fprintf(stdout, "count        : %.3d\n", count);
  }
  REAL   time  = 0.0;
  int    count = 0;  
  init(f1, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);
  REAL *f1_dis, *f2_dis;
  posix_memalign((void**)&f1_dis, 64, sizeof(REAL)*nx*ny*nz);
  posix_memalign((void**)&f2_dis, 64, sizeof(REAL)*nx*ny*nz);
  transform(f1_dis, f2_dis, f1, f2, nx, ny, nz);
  gettimeofday(&time_begin, NULL);
  /* diffusion_openmp_intrin_independent(f1_dis, f2_dis, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt, */
  /* 		     &f1_dis, &time, &count); */
  diffusion_openmp_intrin_independent2(f1_dis, f2_dis, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt,
  		     &f1_dis, &time, &count);
  /* diffusion_like_stream_scale(f1_dis, f2_dis, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt, */
  /* 		     &f1_dis, &time, &count); */
  gettimeofday(&time_end, NULL);
  retransform(f1, f2, f1_dis, f2_dis, nx, ny, nz);
  
  init(answer, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);
  REAL err = accuracy(f1, answer, nx*ny*nz);
  double elapsed_time = (time_end.tv_sec - time_begin.tv_sec)
    + (time_end.tv_usec - time_begin.tv_usec)*1.0e-6;
  REAL gflops = (nx*ny*nz)*13.0*count/elapsed_time * 1.0e-09;
  double thput = (nx * ny * nz) * sizeof(REAL) * 2.0 * count
    / elapsed_time * 1.0e-09;
  
  fprintf(stdout, "\n independent\n");
  fprintf(stdout, "elapsed time : %.3f (s)\n", elapsed_time);
  fprintf(stdout, "flops        : %.3f (GFlops)\n", gflops);
  fprintf(stdout, "throughput   : %.3f (GB/s)\n", thput);  
  fprintf(stdout, "accuracy     : %e\n", err);
  fprintf(stdout, "count        : %.3d\n", count);
  
  free(answer);
  free(f1);
  free(f2);
  return 0;
}
