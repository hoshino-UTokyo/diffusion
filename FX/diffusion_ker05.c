#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#ifdef SVE
#include <arm_sve.h>
#endif /* SVE */
#include "diffusion_ker05.h"

#ifndef REAL
#define REAL double
#endif

#ifndef M_PI
#define M_PI (3.1415926535897932384626)
#endif

#ifndef MIN
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#endif

void allocate_ker05(REAL **buff_ret, const int nx, const int ny, const int nz) {

  posix_memalign((void**)buff_ret, 64, sizeof(REAL)*nx*ny*nz);

}

void init_ker05(REAL *buff1, const int nx, const int ny, const int nz,
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
    int tid = omp_get_thread_num();
    int nth = omp_get_num_threads();
    int tz = tid/12;
    int ty = tid%12;
    int ychunk = (ny-1)/12 + 1;
    int zchunk = nz/((nth-1)/12+1);
    for (jz = tz*zchunk; jz < MIN((tz+1)*zchunk,nz); jz++) {
      for (jy = ty*ychunk; jy < MIN((ty+1)*ychunk,ny); jy++) {
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


void diffusion_ker05(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		     REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		     REAL cb, REAL cc, REAL dt,
		     REAL **f1_ret, REAL **f2_ret, REAL *time_ret, int *count_ret) {

#pragma omp parallel 
  {
    REAL time = 0.0;
    int count = 0;
    REAL *restrict f1_t = f1;
    REAL *restrict f2_t = f2;
    int c, n, s, b, t, b0, t0, w, e;
    int z, y, x;
    int tid = omp_get_thread_num();
    int nth = omp_get_num_threads();
/* #pragma omp master */
/*     if(nth%12 != 0){ */
/*       fprintf(stderr,"omp_get_num_threads()%12 should be 0, nth = %d\n",nth); */
/*       return; */
/*     } */
    int tz = tid/12;
    int ty = tid%12;
    int ychunk = (ny-1)/12 + 1;
    int zchunk = nz/((nth-1)/12+1);
    int zstr = tz*zchunk;
    int zend = MIN((tz+1)*zchunk,nz);
    int ystr = ty*ychunk;
    int yend = MIN((ty+1)*ychunk,ny);
    const svfloat64_t cc_vec = svdup_f64(cc);
    const svfloat64_t cw_vec = svdup_f64(cw);
    const svfloat64_t ce_vec = svdup_f64(ce);
    const svfloat64_t cs_vec = svdup_f64(cs);
    const svfloat64_t cn_vec = svdup_f64(cn);
    const svfloat64_t cb_vec = svdup_f64(cb);
    const svfloat64_t ct_vec = svdup_f64(ct);
    const svbool_t pg = svptrue_b64();
    do {
      for (z = zstr; z < zend; z++) {
	b0 = (z == 0)    ? 0 : - nx * ny;
	t0 = (z == nz-1) ? 0 :   nx * ny;
	for (y = ystr; y < yend; y++) {
	  n = (y == 0)    ? 0 : - nx;
	  s = (y == ny-1) ? 0 :   nx;
	  c =  y * nx + z * nx * ny;

	  e = c + 1;
	  w = c - 1;
	  n = c + n;
	  s = c + s;
	  t = c + t0;
	  b = c + b0;
	  
	  float64_t fcm1_arr[8] = {f1_t[c],f1_t[c],f1_t[c+1],f1_t[c+2],f1_t[c+3],f1_t[c+4],f1_t[c+5],f1_t[c+6]};
	  svfloat64_t fc_vec  = svld1(pg,(float64_t*)&f1_t[c]);
	  svfloat64_t fce_vec = svld1(pg,(float64_t*)&f1_t[e]);
	  svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
	  svfloat64_t fcs_vec = svld1(pg,(float64_t*)&f1_t[s]);
	  svfloat64_t fcn_vec = svld1(pg,(float64_t*)&f1_t[n]);
	  svfloat64_t fcb_vec = svld1(pg,(float64_t*)&f1_t[b]);
	  svfloat64_t fct_vec = svld1(pg,(float64_t*)&f1_t[t]);
	  svfloat64_t tmp = svmul_x(pg,cc_vec,fc_vec);
	  tmp = svmad_x(pg,cw_vec,fcw_vec,tmp);
	  tmp = svmad_x(pg,ce_vec,fce_vec,tmp);
	  tmp = svmad_x(pg,cs_vec,fcs_vec,tmp);
	  tmp = svmad_x(pg,cn_vec,fcn_vec,tmp);
	  tmp = svmad_x(pg,cb_vec,fcb_vec,tmp);
	  tmp = svmad_x(pg,ct_vec,fct_vec,tmp);
	  svst1(pg,(float64_t*)&f2_t[c],tmp);

	  c += 8;
	  e += 8;
	  w += 8;
	  n += 8;
	  s += 8;
	  t += 8;
	  b += 8;
	  for (x = 8; x < nx-8; x+=8) {
	    fc_vec  = svld1(pg,(float64_t*)&f1_t[c]);
	    fce_vec = svld1(pg,(float64_t*)&f1_t[e]);
	    fcw_vec = svld1(pg,(float64_t*)&f1_t[w]);
	    fcs_vec = svld1(pg,(float64_t*)&f1_t[s]);
	    fcn_vec = svld1(pg,(float64_t*)&f1_t[n]);
	    fcb_vec = svld1(pg,(float64_t*)&f1_t[b]);
	    fct_vec = svld1(pg,(float64_t*)&f1_t[t]);
	    tmp = svmul_x(pg,cc_vec,fc_vec);
	    tmp = svmad_x(pg,cw_vec,fcw_vec,tmp);
	    tmp = svmad_x(pg,ce_vec,fce_vec,tmp);
	    tmp = svmad_x(pg,cs_vec,fcs_vec,tmp);
	    tmp = svmad_x(pg,cn_vec,fcn_vec,tmp);
	    tmp = svmad_x(pg,cb_vec,fcb_vec,tmp);
	    tmp = svmad_x(pg,ct_vec,fct_vec,tmp);
	    svst1(pg,(float64_t*)&f2_t[c],tmp);
	    c += 8;
	    e += 8;
	    w += 8;
	    n += 8;
	    s += 8;
	    t += 8;
	    b += 8;
	  }
	  float64_t fcp1_arr[8] = {f1_t[c+1],f1_t[c+2],f1_t[c+3],f1_t[c+4],f1_t[c+5],f1_t[c+6],f1_t[c+7],f1_t[c+7]};
	  fc_vec  = svld1(pg,(float64_t*)&f1_t[c]);
	  fce_vec = svld1(pg,(float64_t*)&fcp1_arr[0]);
	  fcw_vec = svld1(pg,(float64_t*)&f1_t[w]);
	  fcs_vec = svld1(pg,(float64_t*)&f1_t[s]);
	  fcn_vec = svld1(pg,(float64_t*)&f1_t[n]);
	  fcb_vec = svld1(pg,(float64_t*)&f1_t[b]);
	  fct_vec = svld1(pg,(float64_t*)&f1_t[t]);
	  tmp = svmul_x(pg,cc_vec,fc_vec);
	  tmp = svmad_x(pg,cw_vec,fcw_vec,tmp);
	  tmp = svmad_x(pg,ce_vec,fce_vec,tmp);
	  tmp = svmad_x(pg,cs_vec,fcs_vec,tmp);
	  tmp = svmad_x(pg,cn_vec,fcn_vec,tmp);
	  tmp = svmad_x(pg,cb_vec,fcb_vec,tmp);
	  tmp = svmad_x(pg,ct_vec,fct_vec,tmp);
	  svst1(pg,(float64_t*)&f2_t[c],tmp);
	}
      }
#pragma omp barrier

/*     int tz = tid/12; */
/*     int ty = tid%12; */
/*     int ychunk = (ny-1)/12 + 1; */
/*     int zchunk = nz/((nth-1)/12+1); */
/*     const svfloat64_t cc_vec = svdup_f64(cc); */
/*     const svfloat64_t cw_vec = svdup_f64(cw); */
/*     const svfloat64_t ce_vec = svdup_f64(ce); */
/*     const svfloat64_t cs_vec = svdup_f64(cs); */
/*     const svfloat64_t cn_vec = svdup_f64(cn); */
/*     const svfloat64_t cb_vec = svdup_f64(cb); */
/*     const svfloat64_t ct_vec = svdup_f64(ct); */
/*     const svbool_t pg = svptrue_b64(); */
/*     do { */
/*       for (z = tz*zchunk; z < MIN((tz+1)*zchunk,nz); z++) { */
/* 	b = (z == 0)    ? 0 : - nx * ny; */
/* 	t = (z == nz-1) ? 0 :   nx * ny; */
/* 	for (y = ty*ychunk; y < MIN((ty+1)*ychunk,ny); y++) { */
/* 	  n = (y == 0)    ? 0 : - nx; */
/* 	  s = (y == ny-1) ? 0 :   nx; */
/* 	  c =  y * nx + z * nx * ny; */
/* 	  float64_t fcm1_arr[8] = {f1_t[c],f1_t[c],f1_t[c+1],f1_t[c+2],f1_t[c+3],f1_t[c+4],f1_t[c+5],f1_t[c+6]}; */
/* 	  svfloat64_t fc_vec  = svld1(pg,(float64_t*)&f1_t[c]); */
/* 	  svfloat64_t fce_vec = svld1(pg,(float64_t*)&f1_t[c+1]); */
/* 	  svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]); */
/* 	  svfloat64_t fcs_vec = svld1(pg,(float64_t*)&f1_t[c+s]); */
/* 	  svfloat64_t fcn_vec = svld1(pg,(float64_t*)&f1_t[c+n]); */
/* 	  svfloat64_t fcb_vec = svld1(pg,(float64_t*)&f1_t[c+b]); */
/* 	  svfloat64_t fct_vec = svld1(pg,(float64_t*)&f1_t[c+t]); */
/* 	  svfloat64_t tmp = svmul_x(pg,cc_vec,fc_vec); */
/* 	  tmp = svmad_x(pg,cw_vec,fcw_vec,tmp); */
/* 	  tmp = svmad_x(pg,ce_vec,fce_vec,tmp); */
/* 	  tmp = svmad_x(pg,cs_vec,fcs_vec,tmp); */
/* 	  tmp = svmad_x(pg,cn_vec,fcn_vec,tmp); */
/* 	  tmp = svmad_x(pg,cb_vec,fcb_vec,tmp); */
/* 	  tmp = svmad_x(pg,ct_vec,fct_vec,tmp); */
/* 	  svst1(pg,(float64_t*)&f2_t[c],tmp); */
/* 	  for (x = 8; x < nx-8; x+=8) { */
/* 	    fc_vec  = svld1(pg,(float64_t*)&f1_t[c+x]); */
/* 	    fce_vec = svld1(pg,(float64_t*)&f1_t[c+x+1]); */
/* 	    fcw_vec = svld1(pg,(float64_t*)&f1_t[c+x-1]); */
/* 	    fcs_vec = svld1(pg,(float64_t*)&f1_t[c+x+s]); */
/* 	    fcn_vec = svld1(pg,(float64_t*)&f1_t[c+x+n]); */
/* 	    fcb_vec = svld1(pg,(float64_t*)&f1_t[c+x+b]); */
/* 	    fct_vec = svld1(pg,(float64_t*)&f1_t[c+x+t]); */
/* 	    tmp = svmul_x(pg,cc_vec,fc_vec); */
/* 	    tmp = svmad_x(pg,cw_vec,fcw_vec,tmp); */
/* 	    tmp = svmad_x(pg,ce_vec,fce_vec,tmp); */
/* 	    tmp = svmad_x(pg,cs_vec,fcs_vec,tmp); */
/* 	    tmp = svmad_x(pg,cn_vec,fcn_vec,tmp); */
/* 	    tmp = svmad_x(pg,cb_vec,fcb_vec,tmp); */
/* 	    tmp = svmad_x(pg,ct_vec,fct_vec,tmp); */
/* 	    svst1(pg,(float64_t*)&f2_t[c+x],tmp); */
/* 	  } */
/* 	  float64_t fcp1_arr[8] = {f1_t[c+x+1],f1_t[c+x+2],f1_t[c+x+3],f1_t[c+x+4],f1_t[c+x+5],f1_t[c+x+6],f1_t[c+x+7],f1_t[c+x+7]}; */
/* 	  fc_vec  = svld1(pg,(float64_t*)&f1_t[c+x]); */
/* 	  fce_vec = svld1(pg,(float64_t*)&fcp1_arr[0]); */
/* 	  fcw_vec = svld1(pg,(float64_t*)&f1_t[c+x-1]); */
/* 	  fcs_vec = svld1(pg,(float64_t*)&f1_t[c+x+s]); */
/* 	  fcn_vec = svld1(pg,(float64_t*)&f1_t[c+x+n]); */
/* 	  fcb_vec = svld1(pg,(float64_t*)&f1_t[c+x+b]); */
/* 	  fct_vec = svld1(pg,(float64_t*)&f1_t[c+x+t]); */
/* 	  tmp = svmul_x(pg,cc_vec,fc_vec); */
/* 	  tmp = svmad_x(pg,cw_vec,fcw_vec,tmp); */
/* 	  tmp = svmad_x(pg,ce_vec,fce_vec,tmp); */
/* 	  tmp = svmad_x(pg,cs_vec,fcs_vec,tmp); */
/* 	  tmp = svmad_x(pg,cn_vec,fcn_vec,tmp); */
/* 	  tmp = svmad_x(pg,cb_vec,fcb_vec,tmp); */
/* 	  tmp = svmad_x(pg,ct_vec,fct_vec,tmp); */
/* 	  svst1(pg,(float64_t*)&f2_t[c+x],tmp); */
/* 	} */
/*       } */
/* #pragma omp barrier */
      REAL *tmp = f1_t;
      f1_t = f2_t;
      f2_t = tmp;
      time += dt;
      //time += 1;
      count++;
      
    } while (time + 0.5*dt < 0.1);
#pragma omp master
    {
      *f1_ret = f1_t; *f2_ret = f2_t;
      *time_ret = time;      
      *count_ret = count;        
    }
    
  }
}
  
