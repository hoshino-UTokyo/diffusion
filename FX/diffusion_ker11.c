#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#ifdef SVE
#include <arm_sve.h>
#endif /* SVE */
#include "diffusion_ker11.h"

#ifndef REAL
#define REAL double
#endif

#ifndef M_PI
#define M_PI (3.1415926535897932384626)
#endif

#ifndef MIN
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#endif

#define UNR 8

void allocate_ker11(REAL **buff_ret, const int nx, const int ny, const int nz) {

  posix_memalign((void**)buff_ret, 64, sizeof(REAL)*nx*ny*nz);

}

void init_ker11(REAL *buff1, const int nx, const int ny, const int nz,
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


void diffusion_ker11(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
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
#if UNR == 2 
	  n = (y == 0)    ? 0 : - nx;
	  s = (y == ny-1) ? 0 :   nx;
	  c =  y * nx + z * nx * ny;
	  svfloat64_t fc_vec0,fc_vec1;
	  svfloat64_t fce_vec0,fce_vec1;
	  svfloat64_t fcw_vec0,fcw_vec1;
	  svfloat64_t fcs_vec0,fcs_vec1;
	  svfloat64_t fcn_vec0,fcn_vec1;
	  svfloat64_t fcb_vec0,fcb_vec1;
	  svfloat64_t fct_vec0,fct_vec1;

	  w = c - 1;
	  e = c + 1;
	  n = c + n;
	  s = c + s;
	  b = c + b0;
	  t = c + t0;
	  /* w = c - 1; */
	  /* e = c + 1; */
	  /* n = c + n; */
	  /* s = c + s; */
	  /* b = c + b; */
	  /* t = c + t; */
	  
	  fc_vec0  = svld1(pg,(float64_t*)&f1_t[c]);
	  fce_vec0 = svld1(pg,(float64_t*)&f1_t[e]);
	  float64_t fcm1_arr[8] = {f1_t[c],f1_t[c],f1_t[c+1],f1_t[c+2],f1_t[c+3],f1_t[c+4],f1_t[c+5],f1_t[c+6]};
	  fcw_vec0 = svld1(pg,(float64_t*)&fcm1_arr[0]);
	  fcs_vec0 = svld1(pg,(float64_t*)&f1_t[s]);
	  fcn_vec0 = svld1(pg,(float64_t*)&f1_t[n]);
	  fcb_vec0 = svld1(pg,(float64_t*)&f1_t[b]);
	  fct_vec0 = svld1(pg,(float64_t*)&f1_t[t]);

	  fc_vec1  = svld1(pg,(float64_t*)&f1_t[c+8]);
	  fce_vec1 = svld1(pg,(float64_t*)&f1_t[e+8]);
	  fcw_vec1 = svld1(pg,(float64_t*)&f1_t[w+8]);
	  fcs_vec1 = svld1(pg,(float64_t*)&f1_t[s+8]);
	  fcn_vec1 = svld1(pg,(float64_t*)&f1_t[n+8]);
	  fcb_vec1 = svld1(pg,(float64_t*)&f1_t[b+8]);
	  fct_vec1 = svld1(pg,(float64_t*)&f1_t[t+8]);
	  
	  svfloat64_t tmp0,tmp1,tmp2,tmp3,tmp4,tmp5;
	  fc_vec0  = svmul_x(pg,cc_vec,fc_vec0);  fc_vec1  = svmul_x(pg,cc_vec,fc_vec1);
	  fcw_vec0 = svmul_x(pg,cw_vec,fcw_vec0); fcw_vec1 = svmul_x(pg,cw_vec,fcw_vec1);
	  fce_vec0 = svmul_x(pg,ce_vec,fce_vec0); fce_vec1 = svmul_x(pg,ce_vec,fce_vec1);
	  fcs_vec0 = svmul_x(pg,cs_vec,fcs_vec0); fcs_vec1 = svmul_x(pg,cs_vec,fcs_vec1);
	  fcn_vec0 = svmul_x(pg,cn_vec,fcn_vec0); fcn_vec1 = svmul_x(pg,cn_vec,fcn_vec1);
	  fcb_vec0 = svmul_x(pg,cb_vec,fcb_vec0); fcb_vec1 = svmul_x(pg,cb_vec,fcb_vec1);
	  fct_vec0 = svmul_x(pg,ct_vec,fct_vec0); fct_vec1 = svmul_x(pg,ct_vec,fct_vec1);

	  tmp0 = svadd_x(pg,fcw_vec0,fce_vec0); tmp1 = svadd_x(pg,fcw_vec1,fce_vec1);
	  tmp2 = svadd_x(pg,fcs_vec0,fcn_vec0); tmp3 = svadd_x(pg,fcs_vec1,fcn_vec1);
	  tmp4 = svadd_x(pg,fct_vec0,fcb_vec0); tmp5 = svadd_x(pg,fct_vec1,fcb_vec1);
	  tmp0 = svadd_x(pg,fc_vec0, tmp0    ); tmp1 = svadd_x(pg,fc_vec1, tmp1    );
	  tmp2 = svadd_x(pg,tmp2   , tmp4    ); tmp3 = svadd_x(pg,tmp3   , tmp5    );
	  tmp0 = svadd_x(pg,tmp0   , tmp2    ); tmp1 = svadd_x(pg,tmp1   , tmp3    );

	  svst1(pg,(float64_t*)&f2_t[c  ],tmp0);
	  svst1(pg,(float64_t*)&f2_t[c+8],tmp1);
	  
	  c += 16;
	  w += 16;
	  e += 16;
	  n += 16;
	  s += 16;
	  b += 16;
	  t += 16;

	  for (x = 16; x < nx-16; x+=16) {
	    
	    fc_vec0  = svld1(pg,(float64_t*)&f1_t[c]);
	    fce_vec0 = svld1(pg,(float64_t*)&f1_t[e]);
	    fcw_vec0 = svld1(pg,(float64_t*)&f1_t[w]);
	    fcs_vec0 = svld1(pg,(float64_t*)&f1_t[s]);
	    fcn_vec0 = svld1(pg,(float64_t*)&f1_t[n]);
	    fcb_vec0 = svld1(pg,(float64_t*)&f1_t[b]);
	    fct_vec0 = svld1(pg,(float64_t*)&f1_t[t]);

	    fc_vec1  = svld1(pg,(float64_t*)&f1_t[c+8]);
	    fce_vec1 = svld1(pg,(float64_t*)&f1_t[e+8]);
	    fcw_vec1 = svld1(pg,(float64_t*)&f1_t[w+8]);
	    fcs_vec1 = svld1(pg,(float64_t*)&f1_t[s+8]);
	    fcn_vec1 = svld1(pg,(float64_t*)&f1_t[n+8]);
	    fcb_vec1 = svld1(pg,(float64_t*)&f1_t[b+8]);
	    fct_vec1 = svld1(pg,(float64_t*)&f1_t[t+8]);
	    
	    /* tmp0 = svmul_x(pg,cc_vec,fc_vec0); tmp1 = svmul_x(pg,cc_vec,fc_vec1); */
	    /* tmp0 = svmad_x(pg,cw_vec,fcw_vec0,tmp0); tmp1 = svmad_x(pg,cw_vec,fcw_vec1,tmp1); */
	    /* tmp0 = svmad_x(pg,ce_vec,fce_vec0,tmp0); tmp1 = svmad_x(pg,ce_vec,fce_vec1,tmp1); */
	    /* tmp0 = svmad_x(pg,cs_vec,fcs_vec0,tmp0); tmp1 = svmad_x(pg,cs_vec,fcs_vec1,tmp1); */
	    /* tmp0 = svmad_x(pg,cn_vec,fcn_vec0,tmp0); tmp1 = svmad_x(pg,cn_vec,fcn_vec1,tmp1); */
	    /* tmp0 = svmad_x(pg,cb_vec,fcb_vec0,tmp0); tmp1 = svmad_x(pg,cb_vec,fcb_vec1,tmp1); */
	    /* tmp0 = svmad_x(pg,ct_vec,fct_vec0,tmp0); tmp1 = svmad_x(pg,ct_vec,fct_vec1,tmp1); */
	    fc_vec0  = svmul_x(pg,cc_vec,fc_vec0);  fc_vec1  = svmul_x(pg,cc_vec,fc_vec1);
	    fcw_vec0 = svmul_x(pg,cw_vec,fcw_vec0); fcw_vec1 = svmul_x(pg,cw_vec,fcw_vec1);
	    fce_vec0 = svmul_x(pg,ce_vec,fce_vec0); fce_vec1 = svmul_x(pg,ce_vec,fce_vec1);
	    fcs_vec0 = svmul_x(pg,cs_vec,fcs_vec0); fcs_vec1 = svmul_x(pg,cs_vec,fcs_vec1);
	    fcn_vec0 = svmul_x(pg,cn_vec,fcn_vec0); fcn_vec1 = svmul_x(pg,cn_vec,fcn_vec1);
	    fcb_vec0 = svmul_x(pg,cb_vec,fcb_vec0); fcb_vec1 = svmul_x(pg,cb_vec,fcb_vec1);
	    fct_vec0 = svmul_x(pg,ct_vec,fct_vec0); fct_vec1 = svmul_x(pg,ct_vec,fct_vec1);
	    
	    tmp0 = svadd_x(pg,fcw_vec0,fce_vec0); tmp1 = svadd_x(pg,fcw_vec1,fce_vec1);
	    tmp2 = svadd_x(pg,fcs_vec0,fcn_vec0); tmp3 = svadd_x(pg,fcs_vec1,fcn_vec1);
	    tmp4 = svadd_x(pg,fct_vec0,fcb_vec0); tmp5 = svadd_x(pg,fct_vec1,fcb_vec1);
	    tmp0 = svadd_x(pg,fc_vec0, tmp0    ); tmp1 = svadd_x(pg,fc_vec1, tmp1    );
	    tmp2 = svadd_x(pg,tmp2   , tmp4    ); tmp3 = svadd_x(pg,tmp3   , tmp5    );
	    tmp0 = svadd_x(pg,tmp0   , tmp2    ); tmp1 = svadd_x(pg,tmp1   , tmp3    );
	    
	    svst1(pg,(float64_t*)&f2_t[c  ],tmp0);
	    svst1(pg,(float64_t*)&f2_t[c+8],tmp1);

	    c += 16;
	    w += 16;
	    e += 16;
	    n += 16;
	    s += 16;
	    b += 16;
	    t += 16;
	  }
	  fc_vec0  = svld1(pg,(float64_t*)&f1_t[c]);
	  fce_vec0 = svld1(pg,(float64_t*)&f1_t[e]);
	  fcw_vec0 = svld1(pg,(float64_t*)&f1_t[w]);
	  fcs_vec0 = svld1(pg,(float64_t*)&f1_t[s]);
	  fcn_vec0 = svld1(pg,(float64_t*)&f1_t[n]);
	  fcb_vec0 = svld1(pg,(float64_t*)&f1_t[b]);
	  fct_vec0 = svld1(pg,(float64_t*)&f1_t[t]);

	  fc_vec1  = svld1(pg,(float64_t*)&f1_t[c+8]);
	  float64_t fcp1_arr[8] = {f1_t[c+1+8],f1_t[c+2+8],f1_t[c+3+8],f1_t[c+4+8],f1_t[c+5+8],f1_t[c+6+8],f1_t[c+7+8],f1_t[c+7+8]};
	  fce_vec1 = svld1(pg,(float64_t*)&fcp1_arr[0]);
	  fcw_vec1 = svld1(pg,(float64_t*)&f1_t[w+8]);
	  fcs_vec1 = svld1(pg,(float64_t*)&f1_t[s+8]);
	  fcn_vec1 = svld1(pg,(float64_t*)&f1_t[n+8]);
	  fcb_vec1 = svld1(pg,(float64_t*)&f1_t[b+8]);
	  fct_vec1 = svld1(pg,(float64_t*)&f1_t[t+8]);
	  /* tmp0 = svmul_x(pg,cc_vec,fc_vec0); tmp1 = svmul_x(pg,cc_vec,fc_vec1); */
	  /* tmp0 = svmad_x(pg,cw_vec,fcw_vec0,tmp0); tmp1 = svmad_x(pg,cw_vec,fcw_vec1,tmp1); */
	  /* tmp0 = svmad_x(pg,ce_vec,fce_vec0,tmp0); tmp1 = svmad_x(pg,ce_vec,fce_vec1,tmp1); */
	  /* tmp0 = svmad_x(pg,cs_vec,fcs_vec0,tmp0); tmp1 = svmad_x(pg,cs_vec,fcs_vec1,tmp1); */
	  /* tmp0 = svmad_x(pg,cn_vec,fcn_vec0,tmp0); tmp1 = svmad_x(pg,cn_vec,fcn_vec1,tmp1); */
	  /* tmp0 = svmad_x(pg,cb_vec,fcb_vec0,tmp0); tmp1 = svmad_x(pg,cb_vec,fcb_vec1,tmp1); */
	  /* tmp0 = svmad_x(pg,ct_vec,fct_vec0,tmp0); tmp1 = svmad_x(pg,ct_vec,fct_vec1,tmp1); */
	  fc_vec0  = svmul_x(pg,cc_vec,fc_vec0);  fc_vec1  = svmul_x(pg,cc_vec,fc_vec1);
	  fcw_vec0 = svmul_x(pg,cw_vec,fcw_vec0); fcw_vec1 = svmul_x(pg,cw_vec,fcw_vec1);
	  fce_vec0 = svmul_x(pg,ce_vec,fce_vec0); fce_vec1 = svmul_x(pg,ce_vec,fce_vec1);
	  fcs_vec0 = svmul_x(pg,cs_vec,fcs_vec0); fcs_vec1 = svmul_x(pg,cs_vec,fcs_vec1);
	  fcn_vec0 = svmul_x(pg,cn_vec,fcn_vec0); fcn_vec1 = svmul_x(pg,cn_vec,fcn_vec1);
	  fcb_vec0 = svmul_x(pg,cb_vec,fcb_vec0); fcb_vec1 = svmul_x(pg,cb_vec,fcb_vec1);
	  fct_vec0 = svmul_x(pg,ct_vec,fct_vec0); fct_vec1 = svmul_x(pg,ct_vec,fct_vec1);

	  tmp0 = svadd_x(pg,fcw_vec0,fce_vec0); tmp1 = svadd_x(pg,fcw_vec1,fce_vec1);
	  tmp2 = svadd_x(pg,fcs_vec0,fcn_vec0); tmp3 = svadd_x(pg,fcs_vec1,fcn_vec1);
	  tmp4 = svadd_x(pg,fct_vec0,fcb_vec0); tmp5 = svadd_x(pg,fct_vec1,fcb_vec1);
	  tmp0 = svadd_x(pg,fc_vec0, tmp0    ); tmp1 = svadd_x(pg,fc_vec1, tmp1    );
	  tmp2 = svadd_x(pg,tmp2   , tmp4    ); tmp3 = svadd_x(pg,tmp3   , tmp5    );
	  tmp0 = svadd_x(pg,tmp0   , tmp2    ); tmp1 = svadd_x(pg,tmp1   , tmp3    );
	  
	  svst1(pg,(float64_t*)&f2_t[c  ],tmp0);
	  svst1(pg,(float64_t*)&f2_t[c+8],tmp1);
#elif UNR == 4
	  n = (y == 0)    ? 0 : - nx;
	  s = (y == ny-1) ? 0 :   nx;
	  b = b0;
	  t = t0;
	  c =  y * nx + z * nx * ny;
	  float64_t fcm1_arr[8] = {f1_t[c],f1_t[c],f1_t[c+1],f1_t[c+2],f1_t[c+3],f1_t[c+4],f1_t[c+5],f1_t[c+6]};
	  svfloat64_t fc_vec0,fc_vec1,fc_vec2,fc_vec3;
	  svfloat64_t fce_vec0,fce_vec1,fce_vec2,fce_vec3;
	  svfloat64_t fcw_vec0,fcw_vec1,fcw_vec2,fcw_vec3;
	  svfloat64_t fcs_vec0,fcs_vec1,fcs_vec2,fcs_vec3;
	  svfloat64_t fcn_vec0,fcn_vec1,fcn_vec2,fcn_vec3;
	  svfloat64_t fcb_vec0,fcb_vec1,fcb_vec2,fcb_vec3;
	  svfloat64_t fct_vec0,fct_vec1,fct_vec2,fct_vec3;
	  
	  fc_vec0  = svld1(svptrue_b64(),(float64_t*)&f1_t[c]);
	  fce_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+1]);
	  fcw_vec0 = svld1(svptrue_b64(),(float64_t*)&fcm1_arr[0]);
	  fcs_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+s]);
	  fcn_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+n]);
	  fcb_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+b]);
	  fct_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+t]);
	  fc_vec1  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*1]);
	  fce_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*1+1]);
	  fcw_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*1-1]);
	  fcs_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*1+s]);
	  fcn_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*1+n]);
	  fcb_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*1+b]);
	  fct_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*1+t]);
	  fc_vec2  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*2]);
	  fce_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*2+1]);
	  fcw_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*2-1]);
	  fcs_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*2+s]);
	  fcn_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*2+n]);
	  fcb_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*2+b]);
	  fct_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*2+t]);
	  fc_vec3  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*3]);
	  fce_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*3+1]);
	  fcw_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*3-1]);
	  fcs_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*3+s]);
	  fcn_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*3+n]);
	  fcb_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*3+b]);
	  fct_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*3+t]);
	  
	  svfloat64_t tmp0,tmp1,tmp2,tmp3;
	  tmp0 = svmul_x(svptrue_b64(),cc_vec,fc_vec0); tmp1 = svmul_x(svptrue_b64(),cc_vec,fc_vec1);
	  tmp2 = svmul_x(svptrue_b64(),cc_vec,fc_vec2); tmp3 = svmul_x(svptrue_b64(),cc_vec,fc_vec3);
	  tmp0 = svmad_x(svptrue_b64(),cw_vec,fcw_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),cw_vec,fcw_vec1,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),cw_vec,fcw_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),cw_vec,fcw_vec3,tmp3);
	  tmp0 = svmad_x(svptrue_b64(),ce_vec,fce_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),ce_vec,fce_vec1,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),ce_vec,fce_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),ce_vec,fce_vec3,tmp3);
	  tmp0 = svmad_x(svptrue_b64(),cs_vec,fcs_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),cs_vec,fcs_vec1,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),cs_vec,fcs_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),cs_vec,fcs_vec3,tmp3);
	  tmp0 = svmad_x(svptrue_b64(),cn_vec,fcn_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),cn_vec,fcn_vec1,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),cn_vec,fcn_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),cn_vec,fcn_vec3,tmp3);
	  tmp0 = svmad_x(svptrue_b64(),cb_vec,fcb_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),cb_vec,fcb_vec1,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),cb_vec,fcb_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),cb_vec,fcb_vec3,tmp3);
	  tmp0 = svmad_x(svptrue_b64(),ct_vec,fct_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),ct_vec,fct_vec1,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),ct_vec,fct_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),ct_vec,fct_vec3,tmp3);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+8*0],tmp0);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+8*1],tmp1);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+8*2],tmp2);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+8*3],tmp3);
	  
	  int xx;
	  for (xx = 32; xx < nx-32; xx+=32) {
	    
	    fc_vec0  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx]);
	    fce_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+1]);
	    fcw_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx-1]);
	    fcs_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+s]);
	    fcn_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+n]);
	    fcb_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+b]);
	    fct_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+t]);
	    fc_vec1  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1]);
	    fce_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+1]);
	    fcw_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1-1]);
	    fcs_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+s]);
	    fcn_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+n]);
	    fcb_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+b]);
	    fct_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+t]);
	    fc_vec2  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2]);
	    fce_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+1]);
	    fcw_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2-1]);
	    fcs_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+s]);
	    fcn_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+n]);
	    fcb_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+b]);
	    fct_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+t]);
	    fc_vec3  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3]);
	    fce_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+1]);
	    fcw_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3-1]);
	    fcs_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+s]);
	    fcn_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+n]);
	    fcb_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+b]);
	    fct_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+t]);
	    
	    tmp0 = svmul_x(svptrue_b64(),cc_vec,fc_vec0); tmp1 = svmul_x(svptrue_b64(),cc_vec,fc_vec1);
	    tmp2 = svmul_x(svptrue_b64(),cc_vec,fc_vec2); tmp3 = svmul_x(svptrue_b64(),cc_vec,fc_vec3);
	    tmp0 = svmad_x(svptrue_b64(),cw_vec,fcw_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),cw_vec,fcw_vec1,tmp1);
	    tmp2 = svmad_x(svptrue_b64(),cw_vec,fcw_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),cw_vec,fcw_vec3,tmp3);
	    tmp0 = svmad_x(svptrue_b64(),ce_vec,fce_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),ce_vec,fce_vec1,tmp1);
	    tmp2 = svmad_x(svptrue_b64(),ce_vec,fce_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),ce_vec,fce_vec3,tmp3);
	    tmp0 = svmad_x(svptrue_b64(),cs_vec,fcs_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),cs_vec,fcs_vec1,tmp1);
	    tmp2 = svmad_x(svptrue_b64(),cs_vec,fcs_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),cs_vec,fcs_vec3,tmp3);
	    tmp0 = svmad_x(svptrue_b64(),cn_vec,fcn_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),cn_vec,fcn_vec1,tmp1);
	    tmp2 = svmad_x(svptrue_b64(),cn_vec,fcn_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),cn_vec,fcn_vec3,tmp3);
	    tmp0 = svmad_x(svptrue_b64(),cb_vec,fcb_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),cb_vec,fcb_vec1,tmp1);
	    tmp2 = svmad_x(svptrue_b64(),cb_vec,fcb_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),cb_vec,fcb_vec3,tmp3);
	    tmp0 = svmad_x(svptrue_b64(),ct_vec,fct_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),ct_vec,fct_vec1,tmp1);
	    tmp2 = svmad_x(svptrue_b64(),ct_vec,fct_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),ct_vec,fct_vec3,tmp3);
	    svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*0],tmp0);
	    svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*1],tmp1);
	    svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*2],tmp2);
	    svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*3],tmp3);
	  }
	  float64_t fcp1_arr[8] = {f1_t[c+xx+8*3+1],f1_t[c+xx+8*3+2],f1_t[c+xx+8*3+3],f1_t[c+xx+8*3+4],f1_t[c+xx+8*3+5],f1_t[c+xx+8*3+6],f1_t[c+xx+8*3+7],f1_t[c+xx+8*3+7]};
	  fc_vec0  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx]);
	  fce_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+1]);
	  fcw_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx-1]);
	  fcs_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+s]);
	  fcn_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+n]);
	  fcb_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+b]);
	  fct_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+t]);
	  fc_vec1  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1]);
	  fce_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+1]);
	  fcw_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1-1]);
	  fcs_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+s]);
	  fcn_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+n]);
	  fcb_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+b]);
	  fct_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+t]);
	  fc_vec2  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2]);
	  fce_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+1]);
	  fcw_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2-1]);
	  fcs_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+s]);
	  fcn_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+n]);
	  fcb_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+b]);
	  fct_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+t]);
	  fc_vec3  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3]);
	  fce_vec3 = svld1(svptrue_b64(),(float64_t*)&fcp1_arr[0]);
	  fcw_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3-1]);
	  fcs_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+s]);
	  fcn_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+n]);
	  fcb_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+b]);
	  fct_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+t]);
	  tmp0 = svmul_x(svptrue_b64(),cc_vec,fc_vec0); tmp1 = svmul_x(svptrue_b64(),cc_vec,fc_vec1);
	  tmp2 = svmul_x(svptrue_b64(),cc_vec,fc_vec2); tmp3 = svmul_x(svptrue_b64(),cc_vec,fc_vec3);
	  tmp0 = svmad_x(svptrue_b64(),cw_vec,fcw_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),cw_vec,fcw_vec1,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),cw_vec,fcw_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),cw_vec,fcw_vec3,tmp3);
	  tmp0 = svmad_x(svptrue_b64(),ce_vec,fce_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),ce_vec,fce_vec1,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),ce_vec,fce_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),ce_vec,fce_vec3,tmp3);
	  tmp0 = svmad_x(svptrue_b64(),cs_vec,fcs_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),cs_vec,fcs_vec1,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),cs_vec,fcs_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),cs_vec,fcs_vec3,tmp3);
	  tmp0 = svmad_x(svptrue_b64(),cn_vec,fcn_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),cn_vec,fcn_vec1,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),cn_vec,fcn_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),cn_vec,fcn_vec3,tmp3);
	  tmp0 = svmad_x(svptrue_b64(),cb_vec,fcb_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),cb_vec,fcb_vec1,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),cb_vec,fcb_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),cb_vec,fcb_vec3,tmp3);
	  tmp0 = svmad_x(svptrue_b64(),ct_vec,fct_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),ct_vec,fct_vec1,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),ct_vec,fct_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),ct_vec,fct_vec3,tmp3);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*0],tmp0);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*1],tmp1);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*2],tmp2);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*3],tmp3);
#elif UNR == 8
	  n = (y == 0)    ? 0 : - nx;
	  s = (y == ny-1) ? 0 :   nx;
	  b = b0;
	  t = t0;
	  c =  y * nx + z * nx * ny;
	  float64_t fcm1_arr[8] = {f1_t[c],f1_t[c],f1_t[c+1],f1_t[c+2],f1_t[c+3],f1_t[c+4],f1_t[c+5],f1_t[c+6]};
	  svfloat64_t fc_vec0,fc_vec1,fc_vec2,fc_vec3,fc_vec4,fc_vec5,fc_vec6,fc_vec7;
	  svfloat64_t fce_vec0,fce_vec1,fce_vec2,fce_vec3,fce_vec4,fce_vec5,fce_vec6,fce_vec7;
	  svfloat64_t fcw_vec0,fcw_vec1,fcw_vec2,fcw_vec3,fcw_vec4,fcw_vec5,fcw_vec6,fcw_vec7;
	  svfloat64_t fcs_vec0,fcs_vec1,fcs_vec2,fcs_vec3,fcs_vec4,fcs_vec5,fcs_vec6,fcs_vec7;
	  svfloat64_t fcn_vec0,fcn_vec1,fcn_vec2,fcn_vec3,fcn_vec4,fcn_vec5,fcn_vec6,fcn_vec7;
	  svfloat64_t fcb_vec0,fcb_vec1,fcb_vec2,fcb_vec3,fcb_vec4,fcb_vec5,fcb_vec6,fcb_vec7;
	  svfloat64_t fct_vec0,fct_vec1,fct_vec2,fct_vec3,fct_vec4,fct_vec5,fct_vec6,fct_vec7;
	  
	  fc_vec0  = svld1(svptrue_b64(),(float64_t*)&f1_t[c]);
	  fce_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+1]);
	  fcw_vec0 = svld1(svptrue_b64(),(float64_t*)&fcm1_arr[0]);
	  fcs_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+s]);
	  fcn_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+n]);
	  fcb_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+b]);
	  fct_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+t]);
	  fc_vec1  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*1]);
	  fce_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*1+1]);
	  fcw_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*1-1]);
	  fcs_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*1+s]);
	  fcn_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*1+n]);
	  fcb_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*1+b]);
	  fct_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*1+t]);
	  fc_vec2  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*2]);
	  fce_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*2+1]);
	  fcw_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*2-1]);
	  fcs_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*2+s]);
	  fcn_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*2+n]);
	  fcb_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*2+b]);
	  fct_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*2+t]);
	  fc_vec3  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*3]);
	  fce_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*3+1]);
	  fcw_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*3-1]);
	  fcs_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*3+s]);
	  fcn_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*3+n]);
	  fcb_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*3+b]);
	  fct_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*3+t]);
	  fc_vec4  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*4]);
	  fce_vec4 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*4+1]);
	  fcw_vec4 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*4-1]);
	  fcs_vec4 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*4+s]);
	  fcn_vec4 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*4+n]);
	  fcb_vec4 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*4+b]);
	  fct_vec4 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*4+t]);
	  fc_vec5  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*5]);
	  fce_vec5 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*5+1]);
	  fcw_vec5 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*5-1]);
	  fcs_vec5 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*5+s]);
	  fcn_vec5 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*5+n]);
	  fcb_vec5 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*5+b]);
	  fct_vec5 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*5+t]);
	  fc_vec6  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*6]);
	  fce_vec6 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*6+1]);
	  fcw_vec6 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*6-1]);
	  fcs_vec6 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*6+s]);
	  fcn_vec6 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*6+n]);
	  fcb_vec6 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*6+b]);
	  fct_vec6 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*6+t]);
	  fc_vec7  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*7]);
	  fce_vec7 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*7+1]);
	  fcw_vec7 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*7-1]);
	  fcs_vec7 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*7+s]);
	  fcn_vec7 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*7+n]);
	  fcb_vec7 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*7+b]);
	  fct_vec7 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*7+t]);
	  
	  svfloat64_t tmp0,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7;
	  tmp0 = svmul_x(svptrue_b64(),cc_vec,fc_vec0); tmp1 = svmul_x(svptrue_b64(),cc_vec,fc_vec1);
	  tmp2 = svmul_x(svptrue_b64(),cc_vec,fc_vec2); tmp3 = svmul_x(svptrue_b64(),cc_vec,fc_vec3);
	  tmp4 = svmul_x(svptrue_b64(),cc_vec,fc_vec4); tmp5 = svmul_x(svptrue_b64(),cc_vec,fc_vec5);
	  tmp6 = svmul_x(svptrue_b64(),cc_vec,fc_vec6); tmp7 = svmul_x(svptrue_b64(),cc_vec,fc_vec7);
	  tmp0 = svmad_x(svptrue_b64(),cw_vec,fcw_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),cw_vec,fcw_vec1,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),cw_vec,fcw_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),cw_vec,fcw_vec3,tmp3);
	  tmp4 = svmad_x(svptrue_b64(),cw_vec,fcw_vec4,tmp4); tmp5 = svmad_x(svptrue_b64(),cw_vec,fcw_vec5,tmp5);
	  tmp6 = svmad_x(svptrue_b64(),cw_vec,fcw_vec6,tmp6); tmp7 = svmad_x(svptrue_b64(),cw_vec,fcw_vec7,tmp7);
	  tmp0 = svmad_x(svptrue_b64(),ce_vec,fce_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),ce_vec,fce_vec1,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),ce_vec,fce_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),ce_vec,fce_vec3,tmp3);
	  tmp4 = svmad_x(svptrue_b64(),ce_vec,fce_vec4,tmp4); tmp5 = svmad_x(svptrue_b64(),ce_vec,fce_vec5,tmp5);
	  tmp6 = svmad_x(svptrue_b64(),ce_vec,fce_vec6,tmp6); tmp7 = svmad_x(svptrue_b64(),ce_vec,fce_vec7,tmp7);
	  tmp0 = svmad_x(svptrue_b64(),cs_vec,fcs_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),cs_vec,fcs_vec1,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),cs_vec,fcs_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),cs_vec,fcs_vec3,tmp3);
	  tmp4 = svmad_x(svptrue_b64(),cs_vec,fcs_vec4,tmp4); tmp5 = svmad_x(svptrue_b64(),cs_vec,fcs_vec5,tmp5);
	  tmp6 = svmad_x(svptrue_b64(),cs_vec,fcs_vec6,tmp6); tmp7 = svmad_x(svptrue_b64(),cs_vec,fcs_vec7,tmp7);
	  tmp0 = svmad_x(svptrue_b64(),cn_vec,fcn_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),cn_vec,fcn_vec1,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),cn_vec,fcn_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),cn_vec,fcn_vec3,tmp3);
	  tmp4 = svmad_x(svptrue_b64(),cn_vec,fcn_vec4,tmp4); tmp5 = svmad_x(svptrue_b64(),cn_vec,fcn_vec5,tmp5);
	  tmp6 = svmad_x(svptrue_b64(),cn_vec,fcn_vec6,tmp6); tmp7 = svmad_x(svptrue_b64(),cn_vec,fcn_vec7,tmp7);
	  tmp0 = svmad_x(svptrue_b64(),cb_vec,fcb_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),cb_vec,fcb_vec1,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),cb_vec,fcb_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),cb_vec,fcb_vec3,tmp3);
	  tmp4 = svmad_x(svptrue_b64(),cb_vec,fcb_vec4,tmp4); tmp5 = svmad_x(svptrue_b64(),cb_vec,fcb_vec5,tmp5);
	  tmp6 = svmad_x(svptrue_b64(),cb_vec,fcb_vec6,tmp6); tmp7 = svmad_x(svptrue_b64(),cb_vec,fcb_vec7,tmp7);
	  tmp0 = svmad_x(svptrue_b64(),ct_vec,fct_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),ct_vec,fct_vec1,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),ct_vec,fct_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),ct_vec,fct_vec3,tmp3);
	  tmp4 = svmad_x(svptrue_b64(),ct_vec,fct_vec4,tmp4); tmp5 = svmad_x(svptrue_b64(),ct_vec,fct_vec5,tmp5);
	  tmp6 = svmad_x(svptrue_b64(),ct_vec,fct_vec6,tmp6); tmp7 = svmad_x(svptrue_b64(),ct_vec,fct_vec7,tmp7);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+8*0],tmp0);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+8*1],tmp1);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+8*2],tmp2);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+8*3],tmp3);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+8*4],tmp4);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+8*5],tmp5);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+8*6],tmp6);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+8*7],tmp7);
	  
	  int xx;
	  for (xx = 64; xx < nx-64; xx+=64) {
	    
	    fc_vec0  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx]);
	    fce_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+1]);
	    fcw_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx-1]);
	    fcs_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+s]);
	    fcn_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+n]);
	    fcb_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+b]);
	    fct_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+t]);
	    fc_vec1  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1]);
	    fce_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+1]);
	    fcw_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1-1]);
	    fcs_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+s]);
	    fcn_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+n]);
	    fcb_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+b]);
	    fct_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+t]);
	    fc_vec2  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2]);
	    fce_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+1]);
	    fcw_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2-1]);
	    fcs_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+s]);
	    fcn_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+n]);
	    fcb_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+b]);
	    fct_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+t]);
	    fc_vec3  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3]);
	    fce_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+1]);
	    fcw_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3-1]);
	    fcs_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+s]);
	    fcn_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+n]);
	    fcb_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+b]);
	    fct_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+t]);
	    fc_vec4  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*4]);
	    fce_vec4 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*4+1]);
	    fcw_vec4 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*4-1]);
	    fcs_vec4 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*4+s]);
	    fcn_vec4 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*4+n]);
	    fcb_vec4 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*4+b]);
	    fct_vec4 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*4+t]);
	    fc_vec5  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*5]);
	    fce_vec5 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*5+1]);
	    fcw_vec5 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*5-1]);
	    fcs_vec5 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*5+s]);
	    fcn_vec5 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*5+n]);
	    fcb_vec5 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*5+b]);
	    fct_vec5 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*5+t]);
	    fc_vec6  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*6]);
	    fce_vec6 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*6+1]);
	    fcw_vec6 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*6-1]);
	    fcs_vec6 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*6+s]);
	    fcn_vec6 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*6+n]);
	    fcb_vec6 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*6+b]);
	    fct_vec6 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*6+t]);
	    fc_vec7  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*7]);
	    fce_vec7 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*7+1]);
	    fcw_vec7 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*7-1]);
	    fcs_vec7 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*7+s]);
	    fcn_vec7 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*7+n]);
	    fcb_vec7 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*7+b]);
	    fct_vec7 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*7+t]);
	    
	    tmp0 = svmul_x(svptrue_b64(),cc_vec,fc_vec0); tmp1 = svmul_x(svptrue_b64(),cc_vec,fc_vec1);
	    tmp2 = svmul_x(svptrue_b64(),cc_vec,fc_vec2); tmp3 = svmul_x(svptrue_b64(),cc_vec,fc_vec3);
	    tmp4 = svmul_x(svptrue_b64(),cc_vec,fc_vec4); tmp5 = svmul_x(svptrue_b64(),cc_vec,fc_vec5);
	    tmp6 = svmul_x(svptrue_b64(),cc_vec,fc_vec6); tmp7 = svmul_x(svptrue_b64(),cc_vec,fc_vec7);
	    tmp0 = svmad_x(svptrue_b64(),cw_vec,fcw_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),cw_vec,fcw_vec1,tmp1);
	    tmp2 = svmad_x(svptrue_b64(),cw_vec,fcw_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),cw_vec,fcw_vec3,tmp3);
	    tmp4 = svmad_x(svptrue_b64(),cw_vec,fcw_vec4,tmp4); tmp5 = svmad_x(svptrue_b64(),cw_vec,fcw_vec5,tmp5);
	    tmp6 = svmad_x(svptrue_b64(),cw_vec,fcw_vec6,tmp6); tmp7 = svmad_x(svptrue_b64(),cw_vec,fcw_vec7,tmp7);
	    tmp0 = svmad_x(svptrue_b64(),ce_vec,fce_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),ce_vec,fce_vec1,tmp1);
	    tmp2 = svmad_x(svptrue_b64(),ce_vec,fce_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),ce_vec,fce_vec3,tmp3);
	    tmp4 = svmad_x(svptrue_b64(),ce_vec,fce_vec4,tmp4); tmp5 = svmad_x(svptrue_b64(),ce_vec,fce_vec5,tmp5);
	    tmp6 = svmad_x(svptrue_b64(),ce_vec,fce_vec6,tmp6); tmp7 = svmad_x(svptrue_b64(),ce_vec,fce_vec7,tmp7);
	    tmp0 = svmad_x(svptrue_b64(),cs_vec,fcs_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),cs_vec,fcs_vec1,tmp1);
	    tmp2 = svmad_x(svptrue_b64(),cs_vec,fcs_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),cs_vec,fcs_vec3,tmp3);
	    tmp4 = svmad_x(svptrue_b64(),cs_vec,fcs_vec4,tmp4); tmp5 = svmad_x(svptrue_b64(),cs_vec,fcs_vec5,tmp5);
	    tmp6 = svmad_x(svptrue_b64(),cs_vec,fcs_vec6,tmp6); tmp7 = svmad_x(svptrue_b64(),cs_vec,fcs_vec7,tmp7);
	    tmp0 = svmad_x(svptrue_b64(),cn_vec,fcn_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),cn_vec,fcn_vec1,tmp1);
	    tmp2 = svmad_x(svptrue_b64(),cn_vec,fcn_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),cn_vec,fcn_vec3,tmp3);
	    tmp4 = svmad_x(svptrue_b64(),cn_vec,fcn_vec4,tmp4); tmp5 = svmad_x(svptrue_b64(),cn_vec,fcn_vec5,tmp5);
	    tmp6 = svmad_x(svptrue_b64(),cn_vec,fcn_vec6,tmp6); tmp7 = svmad_x(svptrue_b64(),cn_vec,fcn_vec7,tmp7);
	    tmp0 = svmad_x(svptrue_b64(),cb_vec,fcb_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),cb_vec,fcb_vec1,tmp1);
	    tmp2 = svmad_x(svptrue_b64(),cb_vec,fcb_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),cb_vec,fcb_vec3,tmp3);
	    tmp4 = svmad_x(svptrue_b64(),cb_vec,fcb_vec4,tmp4); tmp5 = svmad_x(svptrue_b64(),cb_vec,fcb_vec5,tmp5);
	    tmp6 = svmad_x(svptrue_b64(),cb_vec,fcb_vec6,tmp6); tmp7 = svmad_x(svptrue_b64(),cb_vec,fcb_vec7,tmp7);
	    tmp0 = svmad_x(svptrue_b64(),ct_vec,fct_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),ct_vec,fct_vec1,tmp1);
	    tmp2 = svmad_x(svptrue_b64(),ct_vec,fct_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),ct_vec,fct_vec3,tmp3);
	    tmp4 = svmad_x(svptrue_b64(),ct_vec,fct_vec4,tmp4); tmp5 = svmad_x(svptrue_b64(),ct_vec,fct_vec5,tmp5);
	    tmp6 = svmad_x(svptrue_b64(),ct_vec,fct_vec6,tmp6); tmp7 = svmad_x(svptrue_b64(),ct_vec,fct_vec7,tmp7);
	    svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*0],tmp0);
	    svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*1],tmp1);
	    svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*2],tmp2);
	    svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*3],tmp3);
	    svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*4],tmp4);
	    svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*5],tmp5);
	    svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*6],tmp6);
	    svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*7],tmp7);
	  }
	  float64_t fcp1_arr[8] = {f1_t[c+xx+8*7+1],f1_t[c+xx+8*7+2],f1_t[c+xx+8*7+3],f1_t[c+xx+8*7+4],f1_t[c+xx+8*7+5],f1_t[c+xx+8*7+6],f1_t[c+xx+8*7+7],f1_t[c+xx+8*7+7]};
	  fc_vec0  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx]);
	  fce_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+1]);
	  fcw_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx-1]);
	  fcs_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+s]);
	  fcn_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+n]);
	  fcb_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+b]);
	  fct_vec0 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+t]);
	  fc_vec1  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1]);
	  fce_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+1]);
	  fcw_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1-1]);
	  fcs_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+s]);
	  fcn_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+n]);
	  fcb_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+b]);
	  fct_vec1 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+t]);
	  fc_vec2  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2]);
	  fce_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+1]);
	  fcw_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2-1]);
	  fcs_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+s]);
	  fcn_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+n]);
	  fcb_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+b]);
	  fct_vec2 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+t]);
	  fc_vec3  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3]);
	  fce_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+1]);
	  fcw_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3-1]);
	  fcs_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+s]);
	  fcn_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+n]);
	  fcb_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+b]);
	  fct_vec3 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+t]);
	  fc_vec4  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*4]);
	  fce_vec4 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*4+1]);
	  fcw_vec4 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*4-1]);
	  fcs_vec4 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*4+s]);
	  fcn_vec4 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*4+n]);
	  fcb_vec4 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*4+b]);
	  fct_vec4 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*4+t]);
	  fc_vec5  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*5]);
	  fce_vec5 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*5+1]);
	  fcw_vec5 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*5-1]);
	  fcs_vec5 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*5+s]);
	  fcn_vec5 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*5+n]);
	  fcb_vec5 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*5+b]);
	  fct_vec5 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*5+t]);
	  fc_vec6  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*6]);
	  fce_vec6 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*6+1]);
	  fcw_vec6 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*6-1]);
	  fcs_vec6 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*6+s]);
	  fcn_vec6 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*6+n]);
	  fcb_vec6 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*6+b]);
	  fct_vec6 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*6+t]);
	  fc_vec7  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*7]);
	  fce_vec7 = svld1(svptrue_b64(),(float64_t*)&fcp1_arr[0]);
	  fcw_vec7 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*7-1]);
	  fcs_vec7 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*7+s]);
	  fcn_vec7 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*7+n]);
	  fcb_vec7 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*7+b]);
	  fct_vec7 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*7+t]);
	  tmp0 = svmul_x(svptrue_b64(),cc_vec,fc_vec0); tmp1 = svmul_x(svptrue_b64(),cc_vec,fc_vec1);
	  tmp2 = svmul_x(svptrue_b64(),cc_vec,fc_vec2); tmp3 = svmul_x(svptrue_b64(),cc_vec,fc_vec3);
	  tmp4 = svmul_x(svptrue_b64(),cc_vec,fc_vec4); tmp5 = svmul_x(svptrue_b64(),cc_vec,fc_vec5);
	  tmp6 = svmul_x(svptrue_b64(),cc_vec,fc_vec6); tmp7 = svmul_x(svptrue_b64(),cc_vec,fc_vec7);
	  tmp0 = svmad_x(svptrue_b64(),cw_vec,fcw_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),cw_vec,fcw_vec1,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),cw_vec,fcw_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),cw_vec,fcw_vec3,tmp3);
	  tmp4 = svmad_x(svptrue_b64(),cw_vec,fcw_vec4,tmp4); tmp5 = svmad_x(svptrue_b64(),cw_vec,fcw_vec5,tmp5);
	  tmp6 = svmad_x(svptrue_b64(),cw_vec,fcw_vec6,tmp6); tmp7 = svmad_x(svptrue_b64(),cw_vec,fcw_vec7,tmp7);
	  tmp0 = svmad_x(svptrue_b64(),ce_vec,fce_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),ce_vec,fce_vec1,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),ce_vec,fce_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),ce_vec,fce_vec3,tmp3);
	  tmp4 = svmad_x(svptrue_b64(),ce_vec,fce_vec4,tmp4); tmp5 = svmad_x(svptrue_b64(),ce_vec,fce_vec5,tmp5);
	  tmp6 = svmad_x(svptrue_b64(),ce_vec,fce_vec6,tmp6); tmp7 = svmad_x(svptrue_b64(),ce_vec,fce_vec7,tmp7);
	  tmp0 = svmad_x(svptrue_b64(),cs_vec,fcs_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),cs_vec,fcs_vec1,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),cs_vec,fcs_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),cs_vec,fcs_vec3,tmp3);
	  tmp4 = svmad_x(svptrue_b64(),cs_vec,fcs_vec4,tmp4); tmp5 = svmad_x(svptrue_b64(),cs_vec,fcs_vec5,tmp5);
	  tmp6 = svmad_x(svptrue_b64(),cs_vec,fcs_vec6,tmp6); tmp7 = svmad_x(svptrue_b64(),cs_vec,fcs_vec7,tmp7);
	  tmp0 = svmad_x(svptrue_b64(),cn_vec,fcn_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),cn_vec,fcn_vec1,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),cn_vec,fcn_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),cn_vec,fcn_vec3,tmp3);
	  tmp4 = svmad_x(svptrue_b64(),cn_vec,fcn_vec4,tmp4); tmp5 = svmad_x(svptrue_b64(),cn_vec,fcn_vec5,tmp5);
	  tmp6 = svmad_x(svptrue_b64(),cn_vec,fcn_vec6,tmp6); tmp7 = svmad_x(svptrue_b64(),cn_vec,fcn_vec7,tmp7);
	  tmp0 = svmad_x(svptrue_b64(),cb_vec,fcb_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),cb_vec,fcb_vec1,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),cb_vec,fcb_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),cb_vec,fcb_vec3,tmp3);
	  tmp4 = svmad_x(svptrue_b64(),cb_vec,fcb_vec4,tmp4); tmp5 = svmad_x(svptrue_b64(),cb_vec,fcb_vec5,tmp5);
	  tmp6 = svmad_x(svptrue_b64(),cb_vec,fcb_vec6,tmp6); tmp7 = svmad_x(svptrue_b64(),cb_vec,fcb_vec7,tmp7);
	  tmp0 = svmad_x(svptrue_b64(),ct_vec,fct_vec0,tmp0); tmp1 = svmad_x(svptrue_b64(),ct_vec,fct_vec1,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),ct_vec,fct_vec2,tmp2); tmp3 = svmad_x(svptrue_b64(),ct_vec,fct_vec3,tmp3);
	  tmp4 = svmad_x(svptrue_b64(),ct_vec,fct_vec4,tmp4); tmp5 = svmad_x(svptrue_b64(),ct_vec,fct_vec5,tmp5);
	  tmp6 = svmad_x(svptrue_b64(),ct_vec,fct_vec6,tmp6); tmp7 = svmad_x(svptrue_b64(),ct_vec,fct_vec7,tmp7);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*0],tmp0);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*1],tmp1);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*2],tmp2);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*3],tmp3);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*4],tmp4);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*5],tmp5);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*6],tmp6);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*7],tmp7);
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
  

