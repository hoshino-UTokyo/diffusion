#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#ifdef SVE
#include <arm_sve.h>
#endif /* SVE */
#include "diffusion_ker31.h"

#ifndef REAL
#define REAL double
#endif

#ifndef M_PI
#define M_PI (3.1415926535897932384626)
#endif

#ifndef MIN
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#endif

void allocate_ker31(REAL **buff_ret, const int nx, const int ny, const int nz) {

  posix_memalign((void**)buff_ret, 64, sizeof(REAL)*nx*ny*nz);

}

void init_ker31(REAL *buff1, const int nx, const int ny, const int nz,
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


void diffusion_ker31(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		     REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		     REAL cb, REAL cc, REAL dt,
		     REAL **f1_ret, REAL **f2_ret, REAL *time_ret, int *count_ret) {

#pragma omp parallel
  {
    REAL time = 0.0;
    int count = 0;
    REAL *restrict f1_t = f1;
    REAL *restrict f2_t = f2;
    int c, w, e, n, s, b, t, b0, t0;
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

	  /* w = c - 1; */
	  /* e = c + 1; */
	  n = c + n;
	  s = c + s;
	  b = c + b0;
	  t = c + t0;

#if 0	  
	  svfloat64_t fc_vec0,fc_vec1;
	  svfloat64_t tmp0,tmp1;
	  
	  fc_vec0  = svld1(pg,(float64_t*)&f1_t[c]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[c+8]);
	  tmp0     = svmul_x(pg,cc_vec,fc_vec0);      tmp1     = svmul_x(pg,cc_vec,fc_vec1);
	  fc_vec0  = svld1(pg,(float64_t*)&f1_t[e]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[e+8]);
	  tmp0     = svmad_x(pg,ce_vec,fc_vec0,tmp0); tmp1     = svmad_x(pg,ce_vec,fc_vec1,tmp1);
	  float64_t fcm1[8] = {f1_t[c],f1_t[c],f1_t[c+1],f1_t[c+2],f1_t[c+3],f1_t[c+4],f1_t[c+5],f1_t[c+6]};
	  fc_vec0  = svld1(pg,(float64_t*)&fcm1[0]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[w+8]);
	  tmp0     = svmad_x(pg,cw_vec,fc_vec0,tmp0); tmp1     = svmad_x(pg,cw_vec,fc_vec1,tmp1);
	  fc_vec0  = svld1(pg,(float64_t*)&f1_t[s]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[s+8]);
	  tmp0     = svmad_x(pg,cs_vec,fc_vec0,tmp0); tmp1     = svmad_x(pg,cs_vec,fc_vec1,tmp1);
	  fc_vec0  = svld1(pg,(float64_t*)&f1_t[n]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[n+8]);
	  tmp0     = svmad_x(pg,cn_vec,fc_vec0,tmp0); tmp1     = svmad_x(pg,cn_vec,fc_vec1,tmp1);
	  fc_vec0  = svld1(pg,(float64_t*)&f1_t[b]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[b+8]);
	  tmp0     = svmad_x(pg,cb_vec,fc_vec0,tmp0); tmp1     = svmad_x(pg,cb_vec,fc_vec1,tmp1);
	  fc_vec0  = svld1(pg,(float64_t*)&f1_t[t]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[t+8]);
	  tmp0     = svmad_x(pg,ct_vec,fc_vec0,tmp0); tmp1     = svmad_x(pg,ct_vec,fc_vec1,tmp1);
	  svst1(pg,(float64_t*)&f2_t[c  ],tmp0);      svst1(pg,(float64_t*)&f2_t[c+8],tmp1);
	  
	  c += 16;
	  w += 16;
	  e += 16;
	  n += 16;
	  s += 16;
	  b += 16;
	  t += 16;

	  for (x = 16; x < nx-16; x+=16) {
	    
	    fc_vec0  = svld1(pg,(float64_t*)&f1_t[c]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[c+8]);
	    tmp0     = svmul_x(pg,cc_vec,fc_vec0);      tmp1     = svmul_x(pg,cc_vec,fc_vec1);
	    fc_vec0  = svld1(pg,(float64_t*)&f1_t[e]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[e+8]);
	    tmp0     = svmad_x(pg,ce_vec,fc_vec0,tmp0); tmp1     = svmad_x(pg,ce_vec,fc_vec1,tmp1);
	    fc_vec0  = svld1(pg,(float64_t*)&f1_t[w]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[w+8]);
	    tmp0     = svmad_x(pg,cw_vec,fc_vec0,tmp0); tmp1     = svmad_x(pg,cw_vec,fc_vec1,tmp1);
	    fc_vec0  = svld1(pg,(float64_t*)&f1_t[s]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[s+8]);
	    tmp0     = svmad_x(pg,cs_vec,fc_vec0,tmp0); tmp1     = svmad_x(pg,cs_vec,fc_vec1,tmp1);
	    fc_vec0  = svld1(pg,(float64_t*)&f1_t[n]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[n+8]);
	    tmp0     = svmad_x(pg,cn_vec,fc_vec0,tmp0); tmp1     = svmad_x(pg,cn_vec,fc_vec1,tmp1);
	    fc_vec0  = svld1(pg,(float64_t*)&f1_t[b]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[b+8]);
	    tmp0     = svmad_x(pg,cb_vec,fc_vec0,tmp0); tmp1     = svmad_x(pg,cb_vec,fc_vec1,tmp1);
	    fc_vec0  = svld1(pg,(float64_t*)&f1_t[t]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[t+8]);
	    tmp0     = svmad_x(pg,ct_vec,fc_vec0,tmp0); tmp1     = svmad_x(pg,ct_vec,fc_vec1,tmp1);
	    svst1(pg,(float64_t*)&f2_t[c  ],tmp0);      svst1(pg,(float64_t*)&f2_t[c+8],tmp1);

	    c += 16;
	    w += 16;
	    e += 16;
	    n += 16;
	    s += 16;
	    b += 16;
	    t += 16;
	  }

	  fc_vec0  = svld1(pg,(float64_t*)&f1_t[c]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[c+8]);
	  tmp0     = svmul_x(pg,cc_vec,fc_vec0);      tmp1     = svmul_x(pg,cc_vec,fc_vec1);
	  float64_t fcp1[8] = {f1_t[c+1+8],f1_t[c+2+8],f1_t[c+3+8],f1_t[c+4+8],f1_t[c+5+8],f1_t[c+6+8],f1_t[c+7+8],f1_t[c+7+8]};
	  fc_vec0  = svld1(pg,(float64_t*)&f1_t[e]);  fc_vec1  = svld1(pg,(float64_t*)&fcp1[0]);
	  tmp0     = svmad_x(pg,ce_vec,fc_vec0,tmp0); tmp1     = svmad_x(pg,ce_vec,fc_vec1,tmp1);
	  fc_vec0  = svld1(pg,(float64_t*)&f1_t[w]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[w+8]);
	  tmp0     = svmad_x(pg,cw_vec,fc_vec0,tmp0); tmp1     = svmad_x(pg,cw_vec,fc_vec1,tmp1);
	  fc_vec0  = svld1(pg,(float64_t*)&f1_t[s]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[s+8]);
	  tmp0     = svmad_x(pg,cs_vec,fc_vec0,tmp0); tmp1     = svmad_x(pg,cs_vec,fc_vec1,tmp1);
	  fc_vec0  = svld1(pg,(float64_t*)&f1_t[n]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[n+8]);
	  tmp0     = svmad_x(pg,cn_vec,fc_vec0,tmp0); tmp1     = svmad_x(pg,cn_vec,fc_vec1,tmp1);
	  fc_vec0  = svld1(pg,(float64_t*)&f1_t[b]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[b+8]);
	  tmp0     = svmad_x(pg,cb_vec,fc_vec0,tmp0); tmp1     = svmad_x(pg,cb_vec,fc_vec1,tmp1);
	  fc_vec0  = svld1(pg,(float64_t*)&f1_t[t]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[t+8]);
	  tmp0     = svmad_x(pg,ct_vec,fc_vec0,tmp0); tmp1     = svmad_x(pg,ct_vec,fc_vec1,tmp1);
	  svst1(pg,(float64_t*)&f2_t[c  ],tmp0);      svst1(pg,(float64_t*)&f2_t[c+8],tmp1);

#else
	  svfloat64_t fc_vec0,fc_vec1,fc_vec2,fc_vec3;
	  svfloat64_t tmp0,tmp1,tmp2,tmp3;

	  fc_vec0  = svld1(pg,(float64_t*)&f1_t[c   ]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[c+ 8]);
	  fc_vec2  = svld1(pg,(float64_t*)&f1_t[c+16]);  fc_vec3  = svld1(pg,(float64_t*)&f1_t[c+24]);
	  tmp0     = svmul_x(pg,cc_vec,fc_vec0);         tmp1     = svmul_x(pg,cc_vec,fc_vec1);
	  tmp2     = svmul_x(pg,cc_vec,fc_vec2);         tmp3     = svmul_x(pg,cc_vec,fc_vec3);
	  /* fc_vec0  = svld1(pg,(float64_t*)&f1_t[e   ]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[e+ 8]); */
	  /* fc_vec2  = svld1(pg,(float64_t*)&f1_t[e+16]);  fc_vec3  = svld1(pg,(float64_t*)&f1_t[e+24]); */
	  fc_vec0  = svld1(pg,(float64_t*)&f1_t[c+ 1]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[c+ 9]);
	  fc_vec2  = svld1(pg,(float64_t*)&f1_t[c+17]);  fc_vec3  = svld1(pg,(float64_t*)&f1_t[c+25]);
	  tmp0     = svmad_x(pg,ce_vec,fc_vec0,tmp0);    tmp1     = svmad_x(pg,ce_vec,fc_vec1,tmp1);
	  tmp2     = svmad_x(pg,ce_vec,fc_vec2,tmp2);    tmp3     = svmad_x(pg,ce_vec,fc_vec3,tmp3);
	  float64_t fcm1[8] = {f1_t[c],f1_t[c],f1_t[c+1],f1_t[c+2],f1_t[c+3],f1_t[c+4],f1_t[c+5],f1_t[c+6]};
	  /* fc_vec0  = svld1(pg,(float64_t*)&fcm1[   0]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[w+ 8]); */
	  /* fc_vec2  = svld1(pg,(float64_t*)&f1_t[w+16]);  fc_vec3  = svld1(pg,(float64_t*)&f1_t[w+24]); */
	  fc_vec0  = svld1(pg,(float64_t*)&fcm1[   0]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[c+ 7]);
	  fc_vec2  = svld1(pg,(float64_t*)&f1_t[c+15]);  fc_vec3  = svld1(pg,(float64_t*)&f1_t[c+23]);
	  tmp0     = svmad_x(pg,cw_vec,fc_vec0,tmp0);    tmp1     = svmad_x(pg,cw_vec,fc_vec1,tmp1);
	  tmp2     = svmad_x(pg,cw_vec,fc_vec2,tmp2);    tmp3     = svmad_x(pg,cw_vec,fc_vec3,tmp3);
	  fc_vec0  = svld1(pg,(float64_t*)&f1_t[s   ]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[s+ 8]);
	  fc_vec2  = svld1(pg,(float64_t*)&f1_t[s+16]);  fc_vec3  = svld1(pg,(float64_t*)&f1_t[s+24]);
	  tmp0     = svmad_x(pg,cs_vec,fc_vec0,tmp0);    tmp1     = svmad_x(pg,cs_vec,fc_vec1,tmp1);
	  tmp2     = svmad_x(pg,cs_vec,fc_vec2,tmp2);    tmp3     = svmad_x(pg,cs_vec,fc_vec3,tmp3);
	  fc_vec0  = svld1(pg,(float64_t*)&f1_t[n   ]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[n+ 8]);
	  fc_vec2  = svld1(pg,(float64_t*)&f1_t[n+16]);  fc_vec3  = svld1(pg,(float64_t*)&f1_t[n+24]);
	  tmp0     = svmad_x(pg,cn_vec,fc_vec0,tmp0);    tmp1     = svmad_x(pg,cn_vec,fc_vec1,tmp1);
	  tmp2     = svmad_x(pg,cn_vec,fc_vec2,tmp2);    tmp3     = svmad_x(pg,cn_vec,fc_vec3,tmp3);
	  fc_vec0  = svld1(pg,(float64_t*)&f1_t[b   ]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[b+ 8]);
	  fc_vec2  = svld1(pg,(float64_t*)&f1_t[b+16]);  fc_vec3  = svld1(pg,(float64_t*)&f1_t[b+24]);
	  tmp0     = svmad_x(pg,cb_vec,fc_vec0,tmp0);    tmp1     = svmad_x(pg,cb_vec,fc_vec1,tmp1);
	  tmp2     = svmad_x(pg,cb_vec,fc_vec2,tmp2);    tmp3     = svmad_x(pg,cb_vec,fc_vec3,tmp3);
	  fc_vec0  = svld1(pg,(float64_t*)&f1_t[t   ]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[t+ 8]);
	  fc_vec2  = svld1(pg,(float64_t*)&f1_t[t+16]);  fc_vec3  = svld1(pg,(float64_t*)&f1_t[t+24]);
	  tmp0     = svmad_x(pg,ct_vec,fc_vec0,tmp0);    tmp1     = svmad_x(pg,ct_vec,fc_vec1,tmp1);
	  tmp2     = svmad_x(pg,ct_vec,fc_vec2,tmp2);    tmp3     = svmad_x(pg,ct_vec,fc_vec3,tmp3);
	  svst1(pg,(float64_t*)&f2_t[c   ],tmp0);        svst1(pg,(float64_t*)&f2_t[c+ 8],tmp1);
	  svst1(pg,(float64_t*)&f2_t[c+16],tmp2);        svst1(pg,(float64_t*)&f2_t[c+24],tmp3);
	  
	  c += 32;
	  /* w += 32; */
	  /* e += 32; */
	  n += 32;
	  s += 32;
	  b += 32;
	  t += 32;

	  for (x = 32; x < nx-32; x+=32) {
	    
	    fc_vec0  = svld1(pg,(float64_t*)&f1_t[c   ]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[c+ 8]);
	    fc_vec2  = svld1(pg,(float64_t*)&f1_t[c+16]);  fc_vec3  = svld1(pg,(float64_t*)&f1_t[c+24]);
	    tmp0     = svmul_x(pg,cc_vec,fc_vec0);         tmp1     = svmul_x(pg,cc_vec,fc_vec1);
	    tmp2     = svmul_x(pg,cc_vec,fc_vec2);         tmp3     = svmul_x(pg,cc_vec,fc_vec3);
	    /* fc_vec0  = svld1(pg,(float64_t*)&f1_t[e   ]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[e+ 8]); */
	    /* fc_vec2  = svld1(pg,(float64_t*)&f1_t[e+16]);  fc_vec3  = svld1(pg,(float64_t*)&f1_t[e+24]); */
	    fc_vec0  = svld1(pg,(float64_t*)&f1_t[c+ 1]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[c+ 9]);
	    fc_vec2  = svld1(pg,(float64_t*)&f1_t[c+17]);  fc_vec3  = svld1(pg,(float64_t*)&f1_t[c+25]);
	    tmp0     = svmad_x(pg,ce_vec,fc_vec0,tmp0);    tmp1     = svmad_x(pg,ce_vec,fc_vec1,tmp1);
	    tmp2     = svmad_x(pg,ce_vec,fc_vec2,tmp2);    tmp3     = svmad_x(pg,ce_vec,fc_vec3,tmp3);
	    /* fc_vec0  = svld1(pg,(float64_t*)&f1_t[w   ]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[w+ 8]); */
	    /* fc_vec2  = svld1(pg,(float64_t*)&f1_t[w+16]);  fc_vec3  = svld1(pg,(float64_t*)&f1_t[w+24]); */
	    fc_vec0  = svld1(pg,(float64_t*)&f1_t[c- 1]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[c+ 7]);
	    fc_vec2  = svld1(pg,(float64_t*)&f1_t[c+15]);  fc_vec3  = svld1(pg,(float64_t*)&f1_t[c+23]);
	    tmp0     = svmad_x(pg,cw_vec,fc_vec0,tmp0);    tmp1     = svmad_x(pg,cw_vec,fc_vec1,tmp1);
	    tmp2     = svmad_x(pg,cw_vec,fc_vec2,tmp2);    tmp3     = svmad_x(pg,cw_vec,fc_vec3,tmp3);
	    fc_vec0  = svld1(pg,(float64_t*)&f1_t[s   ]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[s+ 8]);
	    fc_vec2  = svld1(pg,(float64_t*)&f1_t[s+16]);  fc_vec3  = svld1(pg,(float64_t*)&f1_t[s+24]);
	    tmp0     = svmad_x(pg,cs_vec,fc_vec0,tmp0);    tmp1     = svmad_x(pg,cs_vec,fc_vec1,tmp1);
	    tmp2     = svmad_x(pg,cs_vec,fc_vec2,tmp2);    tmp3     = svmad_x(pg,cs_vec,fc_vec3,tmp3);
	    fc_vec0  = svld1(pg,(float64_t*)&f1_t[n   ]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[n+ 8]);
	    fc_vec2  = svld1(pg,(float64_t*)&f1_t[n+16]);  fc_vec3  = svld1(pg,(float64_t*)&f1_t[n+24]);
	    tmp0     = svmad_x(pg,cn_vec,fc_vec0,tmp0);    tmp1     = svmad_x(pg,cn_vec,fc_vec1,tmp1);
	    tmp2     = svmad_x(pg,cn_vec,fc_vec2,tmp2);    tmp3     = svmad_x(pg,cn_vec,fc_vec3,tmp3);
	    fc_vec0  = svld1(pg,(float64_t*)&f1_t[b   ]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[b+ 8]);
	    fc_vec2  = svld1(pg,(float64_t*)&f1_t[b+16]);  fc_vec3  = svld1(pg,(float64_t*)&f1_t[b+24]);
	    tmp0     = svmad_x(pg,cb_vec,fc_vec0,tmp0);    tmp1     = svmad_x(pg,cb_vec,fc_vec1,tmp1);
	    tmp2     = svmad_x(pg,cb_vec,fc_vec2,tmp2);    tmp3     = svmad_x(pg,cb_vec,fc_vec3,tmp3);
	    fc_vec0  = svld1(pg,(float64_t*)&f1_t[t   ]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[t+ 8]);
	    fc_vec2  = svld1(pg,(float64_t*)&f1_t[t+16]);  fc_vec3  = svld1(pg,(float64_t*)&f1_t[t+24]);
	    tmp0     = svmad_x(pg,ct_vec,fc_vec0,tmp0);    tmp1     = svmad_x(pg,ct_vec,fc_vec1,tmp1);
	    tmp2     = svmad_x(pg,ct_vec,fc_vec2,tmp2);    tmp3     = svmad_x(pg,ct_vec,fc_vec3,tmp3);
	    svst1(pg,(float64_t*)&f2_t[c   ],tmp0);        svst1(pg,(float64_t*)&f2_t[c+ 8],tmp1);
	    svst1(pg,(float64_t*)&f2_t[c+16],tmp2);        svst1(pg,(float64_t*)&f2_t[c+24],tmp3);
	    
	    c += 32;
	    /* w += 32; */
	    /* e += 32; */
	    n += 32;
	    s += 32;
	    b += 32;
	    t += 32;

	  }
	  
	  fc_vec0  = svld1(pg,(float64_t*)&f1_t[c   ]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[c+ 8]);
	  fc_vec2  = svld1(pg,(float64_t*)&f1_t[c+16]);  fc_vec3  = svld1(pg,(float64_t*)&f1_t[c+24]);
	  tmp0     = svmul_x(pg,cc_vec,fc_vec0);         tmp1     = svmul_x(pg,cc_vec,fc_vec1);
	  tmp2     = svmul_x(pg,cc_vec,fc_vec2);         tmp3     = svmul_x(pg,cc_vec,fc_vec3);
	  float64_t fcp1[8] = {f1_t[c+1+24],f1_t[c+2+24],f1_t[c+3+24],f1_t[c+4+24],f1_t[c+5+24],f1_t[c+6+24],f1_t[c+7+24],f1_t[c+7+24]};
	  /* fc_vec0  = svld1(pg,(float64_t*)&f1_t[e   ]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[e+ 8]); */
	  /* fc_vec2  = svld1(pg,(float64_t*)&f1_t[e+16]);  fc_vec3  = svld1(pg,(float64_t*)&fcp1[0   ]); */
	  fc_vec0  = svld1(pg,(float64_t*)&f1_t[c+ 1]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[c+ 9]);
	  fc_vec2  = svld1(pg,(float64_t*)&f1_t[c+17]);  fc_vec3  = svld1(pg,(float64_t*)&fcp1[0   ]);
	  tmp0     = svmad_x(pg,ce_vec,fc_vec0,tmp0);    tmp1     = svmad_x(pg,ce_vec,fc_vec1,tmp1);
	  tmp2     = svmad_x(pg,ce_vec,fc_vec2,tmp2);    tmp3     = svmad_x(pg,ce_vec,fc_vec3,tmp3);
	  /* fc_vec0  = svld1(pg,(float64_t*)&f1_t[w   ]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[w+ 8]); */
	  /* fc_vec2  = svld1(pg,(float64_t*)&f1_t[w+16]);  fc_vec3  = svld1(pg,(float64_t*)&f1_t[w+24]); */
	  fc_vec0  = svld1(pg,(float64_t*)&f1_t[c- 1]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[c+ 7]);
	  fc_vec2  = svld1(pg,(float64_t*)&f1_t[c+15]);  fc_vec3  = svld1(pg,(float64_t*)&f1_t[c+23]);
	  tmp0     = svmad_x(pg,cw_vec,fc_vec0,tmp0);    tmp1     = svmad_x(pg,cw_vec,fc_vec1,tmp1);
	  tmp2     = svmad_x(pg,cw_vec,fc_vec2,tmp2);    tmp3     = svmad_x(pg,cw_vec,fc_vec3,tmp3);
	  fc_vec0  = svld1(pg,(float64_t*)&f1_t[s   ]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[s+ 8]);
	  fc_vec2  = svld1(pg,(float64_t*)&f1_t[s+16]);  fc_vec3  = svld1(pg,(float64_t*)&f1_t[s+24]);
	  tmp0     = svmad_x(pg,cs_vec,fc_vec0,tmp0);    tmp1     = svmad_x(pg,cs_vec,fc_vec1,tmp1);
	  tmp2     = svmad_x(pg,cs_vec,fc_vec2,tmp2);    tmp3     = svmad_x(pg,cs_vec,fc_vec3,tmp3);
	  fc_vec0  = svld1(pg,(float64_t*)&f1_t[n   ]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[n+ 8]);
	  fc_vec2  = svld1(pg,(float64_t*)&f1_t[n+16]);  fc_vec3  = svld1(pg,(float64_t*)&f1_t[n+24]);
	  tmp0     = svmad_x(pg,cn_vec,fc_vec0,tmp0);    tmp1     = svmad_x(pg,cn_vec,fc_vec1,tmp1);
	  tmp2     = svmad_x(pg,cn_vec,fc_vec2,tmp2);    tmp3     = svmad_x(pg,cn_vec,fc_vec3,tmp3);
	  fc_vec0  = svld1(pg,(float64_t*)&f1_t[b   ]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[b+ 8]);
	  fc_vec2  = svld1(pg,(float64_t*)&f1_t[b+16]);  fc_vec3  = svld1(pg,(float64_t*)&f1_t[b+24]);
	  tmp0     = svmad_x(pg,cb_vec,fc_vec0,tmp0);    tmp1     = svmad_x(pg,cb_vec,fc_vec1,tmp1);
	  tmp2     = svmad_x(pg,cb_vec,fc_vec2,tmp2);    tmp3     = svmad_x(pg,cb_vec,fc_vec3,tmp3);
	  fc_vec0  = svld1(pg,(float64_t*)&f1_t[t   ]);  fc_vec1  = svld1(pg,(float64_t*)&f1_t[t+ 8]);
	  fc_vec2  = svld1(pg,(float64_t*)&f1_t[t+16]);  fc_vec3  = svld1(pg,(float64_t*)&f1_t[t+24]);
	  tmp0     = svmad_x(pg,ct_vec,fc_vec0,tmp0);    tmp1     = svmad_x(pg,ct_vec,fc_vec1,tmp1);
	  tmp2     = svmad_x(pg,ct_vec,fc_vec2,tmp2);    tmp3     = svmad_x(pg,ct_vec,fc_vec3,tmp3);
	  
	  svst1(pg,(float64_t*)&f2_t[c   ],tmp0);        svst1(pg,(float64_t*)&f2_t[c+ 8],tmp1);
	  svst1(pg,(float64_t*)&f2_t[c+16],tmp2);        svst1(pg,(float64_t*)&f2_t[c+24],tmp3);

#endif

	}
      }
#pragma omp barrier
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
  

