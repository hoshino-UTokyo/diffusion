#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#ifdef SVE
#include <arm_sve.h>
#endif /* SVE */
#include "diffusion_ker38.h"

#ifndef REAL
#define REAL double
#endif

#ifndef M_PI
#define M_PI (3.1415926535897932384626)
#endif

#ifndef MIN
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#endif
#ifndef MAX
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#endif

#define YBF 8
#define HALO 1
#define ZB (3+HALO-1)
#define STEP 3
#define TB (HALO+STEP-2)
#define SIMDLENGTH 8

void allocate_ker38(REAL **buff_ret, const int nx, const int ny, const int nz) {

  posix_memalign((void**)buff_ret, 64, sizeof(REAL)*nx*ny*nz);

}

void init_ker38(REAL *buff1, const int nx, const int ny, const int nz,
		const REAL kx, const REAL ky, const REAL kz,
		const REAL dx, const REAL dy, const REAL dz,
		const REAL kappa, const REAL time) {

  REAL ax, ay, az;
  int jz, jy, jx;
  ax = exp(-kappa*time*(kx*kx));
  ay = exp(-kappa*time*(ky*ky));
  az = exp(-kappa*time*(kz*kz));
#pragma omp parallel private(jx,jy,jz)
  {
/*     int tid = omp_get_thread_num(); */
/*     int nth = omp_get_num_threads(); */
/*     int tz = tid/12; */
/*     int ty = tid%12; */
/*     int zchunk = nz/((nth-1)/12+1); */
/*     int yblock = YBF; */
/*     int ychunk = yblock * 12; */
    int xx,yy;
    int tid = omp_get_thread_num();
    int nth = omp_get_num_threads();
    int ty = tid;
    int ychunk = YBF*nth;
    int yystr = ty*YBF;
/*     int yystr = ty*yblock; */
    /* for (yy = yystr; yy < ny; yy+= ychunk) { */
    /*   for (jz = tz*zchunk; jz < MIN((tz+1)*zchunk,nz); jz++) { */
    /* 	for (jy = yy; jy < MIN(yy+yblock,ny); jy++) { */
    /* 	  for (jx = 0; jx < nx; jx++) { */
    //#pragma omp parallel for private(yy,jx,jy,jz)
    for (yy = yystr; yy < ny; yy+= ychunk) {
      for (jz = 0; jz < nz; jz++) {
	for (jy = yy; jy < MIN(yy+YBF,ny); jy++) {
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
  }
}


void diffusion_ker38(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		     REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		     REAL cb, REAL cc, REAL dt,
		     REAL **f1_ret, REAL **f2_ret, REAL *time_ret, int *count_ret) {

#pragma omp parallel
  {
    REAL time = 0.0;
    int count = 0;
    REAL *restrict f1_t = f1;
    REAL *restrict f2_t = f2;
    int c, w, e, n, s, b, t,b0, t0, n0, s0;//, nt0, st0, w0, e0;
    int z, y, x, xx, yy, zz, h;
    int izm,izc,izp,iy,ix,tmp,halo;
    int step;
    int xstr,xend,ystr,yend;
    int id2,id1;

    int tid = omp_get_thread_num();
    int nth = omp_get_num_threads();
    int ty = tid;
    int ychunk = YBF*nth;
    int yystr = ty*YBF;

    REAL *temporal;
    temporal = (REAL*)malloc(sizeof(double)*(STEP-1)*(2*HALO+1)*(YBF+2*TB)*nx);
    int tbx = nx;
    int tby = (YBF+2*TB);
    int tbz = (2*HALO+1);

    const svfloat64_t cc_vec = svdup_f64(cc);
    const svfloat64_t cw_vec = svdup_f64(cw);
    const svfloat64_t ce_vec = svdup_f64(ce);
    const svfloat64_t cs_vec = svdup_f64(cs);
    const svfloat64_t cn_vec = svdup_f64(cn);
    const svfloat64_t cb_vec = svdup_f64(cb);
    const svfloat64_t ct_vec = svdup_f64(ct);
    const svbool_t pg = svptrue_b64();
#pragma statement cache_sector_size 4 0 2 14
#pragma statement cache_subsector_assign temporal
    do {
      for (yy = yystr; yy < ny; yy+=ychunk) {

	if(yy == 0){
#define YY 0
#include "ker38.inc"
#undef YY 
	}else if(yy >= ny-YBF){
#define YY 2
#include "ker38.inc"
#undef YY 
	}else{
#define YY 1
#include "ker38.inc"
#undef YY 
	}
      }
 
#pragma omp barrier
      REAL *tmp = f1_t;
      f1_t = f2_t;
      f2_t = tmp;
      time += STEP*dt;
      //time += 1;
      count+=STEP;
      
    } while (time + 0.5*dt < 0.1);
#pragma omp master
    {
      *f1_ret = f1_t; *f2_ret = f2_t;
      *time_ret = time;      
      *count_ret = count;        
    }
#pragma statement end_cache_subsector
#pragma statement end_cache_sector_size
    free(temporal);
  }
  
}

#undef YBF
#undef HALO 
#undef ZB 
#undef STEP 
#undef TB 
#undef SIMDLENGTH
