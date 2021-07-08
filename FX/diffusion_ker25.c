#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#ifdef SVE
#include <arm_sve.h>
#endif /* SVE */
#include "diffusion_ker25.h"

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

#define XBF 32
#define YBF 16
#define HALO 1
#define ZB (3+HALO-1)
#define STEP 3
#define TB (HALO+STEP-2)

void allocate_ker25(REAL **buff_ret, const int nx, const int ny, const int nz) {

  posix_memalign((void**)buff_ret, 64, sizeof(REAL)*nx*ny*nz);

}

void init_ker25(REAL *buff1, const int nx, const int ny, const int nz,
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
    int tx = tid%4;
    int xchunk = XBF*4;
    int xxstr = tx*XBF;
    int ty = tid/4;
    int ychunk = YBF*12;
    int yystr = ty*YBF;
/*     int yystr = ty*yblock; */
    /* for (yy = yystr; yy < ny; yy+= ychunk) { */
    /*   for (jz = tz*zchunk; jz < MIN((tz+1)*zchunk,nz); jz++) { */
    /* 	for (jy = yy; jy < MIN(yy+yblock,ny); jy++) { */
    /* 	  for (jx = 0; jx < nx; jx++) { */
    //#pragma omp parallel for private(yy,jx,jy,jz)
    for (yy = yystr; yy < ny; yy+= ychunk) {
      for (xx = xxstr; xx < nx; xx+= xchunk){
	for (jz = 0; jz < nz; jz++) {
	  for (jy = yy; jy < MIN(yy+YBF,ny); jy++) {
	    for (jx = xx; jx < MIN(xx+XBF,nx); jx++) {
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
}


void diffusion_ker25(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		     REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		     REAL cb, REAL cc, REAL dt,
		     REAL **f1_ret, REAL **f2_ret, REAL *time_ret, int *count_ret) {

#pragma omp parallel
  {
    REAL time = 0.0;
    int count = 0;
    REAL *restrict f1_t = f1;
    REAL *restrict f2_t = f2;
    int c, w, e, n, s, b, t,b0, t0, n0, s0, nt0, st0, w0, e0;
    int z, y, x, xx, yy, zz, h;
    int izm,izc,izp,iy,ix,tmp,halo;
    int step;
    int xstr,xend,ystr,yend;
    int id2,id1;

    int tid = omp_get_thread_num();
    int nth = omp_get_num_threads();
    int tx = tid%4;
    int xchunk = XBF*4;
    int xxstr = tx*XBF;
    int ty = tid/4;
    int ychunk = YBF*12;
    int yystr = ty*YBF;
    //    int xblock = XBF;
    //    int yblock = YBF;
    //    int hst = TB;
    /* int zchunk = nz/((nth-1)/12+1); */
    /* int zstr = tz*zchunk; */
    /* int zend = MIN((tz+1)*zchunk,nz); */
    //    int hight = 3;
    
    //    REAL temporal[STEP-1][3][YBF+2*TB][XBF+2*TB] = {0};
    REAL temporal[(STEP-1)*(2*HALO+1)*(YBF+2*TB)*(XBF+2*TB)] = {0};
    int tbx = (XBF+2*TB);
    int tby = (YBF+2*TB);
    int tbz = (2*HALO+1);

/* #pragma statement cache_sector_size 1 3 12 4 */
/* #pragma statement cache_subsector_assign temporal */
    //    int prefetch_flag; 
    /* const svfloat64_t cc_vec = svdup_f64(cc); */
    /* const svfloat64_t cw_vec = svdup_f64(cw); */
    /* const svfloat64_t ce_vec = svdup_f64(ce); */
    /* const svfloat64_t cs_vec = svdup_f64(cs); */
    /* const svfloat64_t cn_vec = svdup_f64(cn); */
    /* const svfloat64_t cb_vec = svdup_f64(cb); */
    /* const svfloat64_t ct_vec = svdup_f64(ct); */
    /* const svbool_t pg = svptrue_b64(); */
    do {
/* #pragma omp for */
/*       for (int yy = 0; yy < ny; yy+=YBF) { */
/* 	for (int xx = 0; xx < nx; xx+=XBF) { */
      for (int yy = yystr; yy < ny; yy+=ychunk) {
	for (int xx = xxstr; xx < nx; xx+=xchunk) {

	  if ((xx == 0 && xx >= nx-XBF) || (yy == 0 && yy >= ny-YBF)) {
#define XX 3
#define YY 3
#include "ker25.inc"
#undef YY 
#undef XX
	  }else if(xx == 0     && yy == 0){
#define XX 0
#define YY 0
#include "ker25.inc"
#undef YY 
#undef XX
	  }else if(xx == 0     && yy >= ny-YBF){
#define XX 0
#define YY 2
#include "ker25.inc"
#undef YY 
#undef XX
	  }else if(xx >=nx-XBF && yy == 0){
#define XX 2
#define YY 0
#include "ker25.inc"
#undef YY 
#undef XX
	  }else if(xx >=nx-XBF && yy >= ny-YBF){
#define XX 2
#define YY 2
#include "ker25.inc"
#undef YY 
#undef XX
	  }else if(xx == 0){
#define XX 0
#define YY 1
#include "ker25.inc"
#undef YY 
#undef XX
	  }else if(xx >= nx-XBF){
#define XX 2
#define YY 1
#include "ker25.inc"
#undef YY 
#undef XX
	  }else if(yy == 0){
#define XX 1
#define YY 0
#include "ker25.inc"
#undef YY 
#undef XX
	  }else if(yy >= ny-YBF){
#define XX 1
#define YY 2
#include "ker25.inc"
#undef YY 
#undef XX
	  }else{
#define XX 1
#define YY 1
#include "ker25.inc"
#undef YY 
#undef XX
	  }
	}
      }
 
      /* printf("temporal(:,:,:)\n"); */
      /* for(step = 0; step < STEP-1; step++){ */
      /* 	printf("step = %d\n",step); */
      /* 	for(z = 0; z < HALO+2; z++){ */
      /* 	  printf("z = %d\n",z); */
      /* 	  for(y = 0; y < YBF+2*TB; y++){ */
      /* 	    for(x = 0; x < XBF+2*TB; x++){ */
      /* 	      printf("%f ",temporal[step][z][y][x]); */
      /* 	    } */
      /* 	    printf("\n"); */
      /* 	  } */
      /* 	} */
      /* } */
      /* printf("f1(:,:,:)\n"); */
      /* for(z = 0; z < nz; z++){ */
      /* 	printf("z = %d\n",z); */
      /* 	for(y = 0; y < ny; y++){ */
      /* 	  for(x = 0; x < nx; x++){ */
      /* 	    printf("%f ",f1_t[x+y*nx+z*nx*ny]); */
      /* 	  } */
      /* 	  printf("\n"); */
      /* 	} */
      /* } */
      /* printf("f2(:,:,:)\n"); */
      /* for(z = 0; z < nz; z++){ */
      /* 	printf("z = %d\n",z); */
      /* 	for(y = 0; y < ny; y++){ */
      /* 	  for(x = 0; x < nx; x++){ */
      /* 	    printf("%f ",f2_t[x+y*nx+z*nx*ny]); */
      /* 	  } */
      /* 	  printf("\n"); */
      /* 	} */
      /* } */
      /* exit(0); */
      
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
/* #pragma statement end_cache_subsector */
/* #pragma statement end_cache_sector_size */
  }
  
}

#undef YBF
#undef XBF
#undef HALO 
#undef ZB 
#undef STEP 
#undef TB 
