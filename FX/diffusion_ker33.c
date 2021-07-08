#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#ifdef SVE
#include <arm_sve.h>
#endif /* SVE */
#include "diffusion_ker33.h"

#ifndef REAL
#define REAL double
#endif

#ifndef M_PI
#define M_PI (3.1415926535897932384626)
#endif

#ifndef MIN
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#endif

void allocate_ker33(REAL **buff_ret, const int nx, const int ny, const int nz) {

  posix_memalign((void**)buff_ret, 64, sizeof(REAL)*nx*ny*nz);

}

void init_ker33(REAL *buff1, const int nx, const int ny, const int nz,
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


void diffusion_ker33(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
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
	/* b0 = (z == 0)    ? 0 : - nx * ny; */
	/* t0 = (z == nz-1) ? 0 :   nx * ny; */
	b = (z == 0)    ? 0 : - nx * ny;
	t = (z == nz-1) ? 0 :   nx * ny;
	for (y = ystr; y < yend; y+=2) {
	  n = (y == 0)    ? 0 : - nx;
	  s = (y == ny-2) ? 0 :   nx;
	  c =  y * nx + z * nx * ny;

	  /* w = c - 1; */
	  /* e = c + 1; */
	  /* n = c + n; */
	  /* s = c + s; */
	  /* b = c + b0; */
	  /* t = c + t0; */

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
	  /* svfloat64_t fc_vec00,fc_vec01,fc_vec02,fc_vec03; */
	  /* svfloat64_t fc_vec10,fc_vec11,fc_vec12,fc_vec13; */
	  /* svfloat64_t tmp00,tmp01,tmp02,tmp03; */
	  /* svfloat64_t tmp10,tmp11,tmp12,tmp13; */
	  /* // n */
	  /* fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+n      ]); fc_vec01  = svld1(pg,(float64_t*)&f1_t[c+n+ 8   ]); */
	  /* fc_vec02  = svld1(pg,(float64_t*)&f1_t[c+n+16   ]); fc_vec03  = svld1(pg,(float64_t*)&f1_t[c+n+24   ]); */
	  /* fc_vec10  = svld1(pg,(float64_t*)&f1_t[c        ]); fc_vec11  = svld1(pg,(float64_t*)&f1_t[c  + 8   ]); */
	  /* fc_vec12  = svld1(pg,(float64_t*)&f1_t[c  +16   ]); fc_vec13  = svld1(pg,(float64_t*)&f1_t[c  +24   ]); */
	  /* tmp00     = svmul_x(pg,cn_vec,fc_vec00);            tmp01     = svmul_x(pg,cn_vec,fc_vec01); */
	  /* tmp02     = svmul_x(pg,cn_vec,fc_vec02);            tmp03     = svmul_x(pg,cn_vec,fc_vec03); */
	  /* tmp10     = svmul_x(pg,cn_vec,fc_vec10);            tmp11     = svmul_x(pg,cn_vec,fc_vec11); */
	  /* tmp12     = svmul_x(pg,cn_vec,fc_vec12);            tmp13     = svmul_x(pg,cn_vec,fc_vec13); */
	  /* // c */
	  /* fc_vec00  = fc_vec10;                               fc_vec01  = fc_vec11; */
	  /* fc_vec02  = fc_vec12;                               fc_vec03  = fc_vec13; */
	  /* fc_vec10  = svld1(pg,(float64_t*)&f1_t[c     +nx]); fc_vec11  = svld1(pg,(float64_t*)&f1_t[c  + 8+nx]); */
	  /* fc_vec12  = svld1(pg,(float64_t*)&f1_t[c  +16+nx]); fc_vec13  = svld1(pg,(float64_t*)&f1_t[c  +24+nx]); */
	  /* tmp00     = svmad_x(pg,cc_vec,fc_vec00,tmp00);      tmp01     = svmad_x(pg,cc_vec,fc_vec01,tmp01); */
	  /* tmp02     = svmad_x(pg,cc_vec,fc_vec02,tmp02);      tmp03     = svmad_x(pg,cc_vec,fc_vec03,tmp03); */
	  /* tmp10     = svmad_x(pg,cc_vec,fc_vec10,tmp10);      tmp11     = svmad_x(pg,cc_vec,fc_vec11,tmp11); */
	  /* tmp12     = svmad_x(pg,cc_vec,fc_vec12,tmp12);      tmp13     = svmad_x(pg,cc_vec,fc_vec13,tmp13); */
	  /* // s */
	  /* fc_vec00  = fc_vec10;                               fc_vec01  = fc_vec11; */
	  /* fc_vec02  = fc_vec12;                               fc_vec03  = fc_vec13; */
	  /* fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+s   +nx]); fc_vec11  = svld1(pg,(float64_t*)&f1_t[c+s+ 8+nx]); */
	  /* fc_vec12  = svld1(pg,(float64_t*)&f1_t[c+s+16+nx]); fc_vec13  = svld1(pg,(float64_t*)&f1_t[c+s+24+nx]); */
	  /* tmp00     = svmad_x(pg,cs_vec,fc_vec00,tmp00);      tmp01     = svmad_x(pg,cs_vec,fc_vec01,tmp01); */
	  /* tmp02     = svmad_x(pg,cs_vec,fc_vec02,tmp02);      tmp03     = svmad_x(pg,cs_vec,fc_vec03,tmp03); */
	  /* tmp10     = svmad_x(pg,cs_vec,fc_vec10,tmp10);      tmp11     = svmad_x(pg,cs_vec,fc_vec11,tmp11); */
	  /* tmp12     = svmad_x(pg,cs_vec,fc_vec12,tmp12);      tmp13     = svmad_x(pg,cs_vec,fc_vec13,tmp13); */
	  /* // w */
	  /* float64_t fcm1[8] = {f1_t[c],f1_t[c],f1_t[c+1],f1_t[c+2],f1_t[c+3],f1_t[c+4],f1_t[c+5],f1_t[c+6]}; */
	  /* float64_t fcm2[8] = {f1_t[c+nx],f1_t[c+nx],f1_t[c+1+nx],f1_t[c+2+nx],f1_t[c+3+nx],f1_t[c+4+nx],f1_t[c+5+nx],f1_t[c+6+nx]}; */
	  /* fc_vec00  = svld1(pg,(float64_t*)&fcm1[0        ]); fc_vec01  = svld1(pg,(float64_t*)&f1_t[c-1+ 8   ]); */
	  /* fc_vec02  = svld1(pg,(float64_t*)&f1_t[c-1+16   ]); fc_vec03  = svld1(pg,(float64_t*)&f1_t[c-1+24   ]); */
	  /* fc_vec10  = svld1(pg,(float64_t*)&fcm2[0        ]); fc_vec01  = svld1(pg,(float64_t*)&f1_t[c-1+ 8+nx]); */
	  /* fc_vec12  = svld1(pg,(float64_t*)&f1_t[c-1+16+nx]); fc_vec03  = svld1(pg,(float64_t*)&f1_t[c-1+24+nx]); */
	  /* tmp00     = svmad_x(pg,cw_vec,fc_vec00,tmp00);      tmp01     = svmad_x(pg,cw_vec,fc_vec01,tmp01); */
	  /* tmp02     = svmad_x(pg,cw_vec,fc_vec02,tmp02);      tmp03     = svmad_x(pg,cw_vec,fc_vec03,tmp03); */
	  /* tmp10     = svmad_x(pg,cw_vec,fc_vec10,tmp10);      tmp11     = svmad_x(pg,cw_vec,fc_vec11,tmp11); */
	  /* tmp12     = svmad_x(pg,cw_vec,fc_vec12,tmp12);      tmp13     = svmad_x(pg,cw_vec,fc_vec13,tmp13); */
	  /* // e */
	  /* fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+1      ]); fc_vec01  = svld1(pg,(float64_t*)&f1_t[c+1+ 8   ]); */
	  /* fc_vec02  = svld1(pg,(float64_t*)&f1_t[c+1+16   ]); fc_vec03  = svld1(pg,(float64_t*)&f1_t[c+1+24   ]); */
	  /* fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+1   +nx]); fc_vec11  = svld1(pg,(float64_t*)&f1_t[c+1+ 8+nx]); */
	  /* fc_vec12  = svld1(pg,(float64_t*)&f1_t[c+1+16+nx]); fc_vec13  = svld1(pg,(float64_t*)&f1_t[c+1+24+nx]); */
	  /* tmp00     = svmad_x(pg,ce_vec,fc_vec00,tmp00);      tmp01     = svmad_x(pg,ce_vec,fc_vec01,tmp01); */
	  /* tmp02     = svmad_x(pg,ce_vec,fc_vec02,tmp02);      tmp03     = svmad_x(pg,ce_vec,fc_vec03,tmp03); */
	  /* tmp10     = svmad_x(pg,ce_vec,fc_vec10,tmp10);      tmp11     = svmad_x(pg,ce_vec,fc_vec11,tmp11); */
	  /* tmp12     = svmad_x(pg,ce_vec,fc_vec12,tmp12);      tmp13     = svmad_x(pg,ce_vec,fc_vec13,tmp13); */
	  /* // b */
	  /* fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+b      ]); fc_vec01  = svld1(pg,(float64_t*)&f1_t[c+b+ 8   ]); */
	  /* fc_vec02  = svld1(pg,(float64_t*)&f1_t[c+b+16   ]); fc_vec03  = svld1(pg,(float64_t*)&f1_t[c+b+24   ]); */
	  /* fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+b   +nx]); fc_vec11  = svld1(pg,(float64_t*)&f1_t[c+b+ 8+nx]); */
	  /* fc_vec12  = svld1(pg,(float64_t*)&f1_t[c+b+16+nx]); fc_vec13  = svld1(pg,(float64_t*)&f1_t[c+b+24+nx]); */
	  /* tmp00     = svmad_x(pg,cb_vec,fc_vec00,tmp00);      tmp01     = svmad_x(pg,cb_vec,fc_vec01,tmp01); */
	  /* tmp02     = svmad_x(pg,cb_vec,fc_vec02,tmp02);      tmp03     = svmad_x(pg,cb_vec,fc_vec03,tmp03); */
	  /* tmp10     = svmad_x(pg,cb_vec,fc_vec10,tmp10);      tmp11     = svmad_x(pg,cb_vec,fc_vec11,tmp11); */
	  /* tmp12     = svmad_x(pg,cb_vec,fc_vec12,tmp12);      tmp13     = svmad_x(pg,cb_vec,fc_vec13,tmp13); */
	  /* // t */
	  /* fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+t      ]); fc_vec01  = svld1(pg,(float64_t*)&f1_t[c+t+ 8   ]); */
	  /* fc_vec02  = svld1(pg,(float64_t*)&f1_t[c+t+16   ]); fc_vec03  = svld1(pg,(float64_t*)&f1_t[c+t+24   ]); */
	  /* fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+t   +nx]); fc_vec11  = svld1(pg,(float64_t*)&f1_t[c+t+ 8+nx]); */
	  /* fc_vec12  = svld1(pg,(float64_t*)&f1_t[c+t+16+nx]); fc_vec13  = svld1(pg,(float64_t*)&f1_t[c+t+24+nx]); */
	  /* tmp00     = svmad_x(pg,ct_vec,fc_vec00,tmp00);      tmp01     = svmad_x(pg,ct_vec,fc_vec01,tmp01); */
	  /* tmp02     = svmad_x(pg,ct_vec,fc_vec02,tmp02);      tmp03     = svmad_x(pg,ct_vec,fc_vec03,tmp03); */
	  /* tmp10     = svmad_x(pg,ct_vec,fc_vec10,tmp10);      tmp11     = svmad_x(pg,ct_vec,fc_vec11,tmp11); */
	  /* tmp12     = svmad_x(pg,ct_vec,fc_vec12,tmp12);      tmp13     = svmad_x(pg,ct_vec,fc_vec13,tmp13); */
	  
	  /* svst1(pg,(float64_t*)&f2_t[c      ],tmp00);         svst1(pg,(float64_t*)&f2_t[c+ 8   ],tmp01); */
	  /* svst1(pg,(float64_t*)&f2_t[c+16   ],tmp02);         svst1(pg,(float64_t*)&f2_t[c+24   ],tmp03); */
	  /* svst1(pg,(float64_t*)&f2_t[c   +nx],tmp10);         svst1(pg,(float64_t*)&f2_t[c+ 8+nx],tmp11); */
	  /* svst1(pg,(float64_t*)&f2_t[c+16+nx],tmp12);         svst1(pg,(float64_t*)&f2_t[c+24+nx],tmp13); */

	  /* c += 32; */
	  /* /\* w += 32; *\/ */
	  /* /\* e += 32; *\/ */
	  /* /\* n += 32; *\/ */
	  /* /\* s += 32; *\/ */
	  /* /\* b += 32; *\/ */
	  /* /\* t += 32; *\/ */

	  /* for (x = 32; x < nx-32; x+=32) { */
	    
	  /*   // n */
	  /*   fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+n      ]); fc_vec01  = svld1(pg,(float64_t*)&f1_t[c+n+ 8   ]); */
	  /*   fc_vec02  = svld1(pg,(float64_t*)&f1_t[c+n+16   ]); fc_vec03  = svld1(pg,(float64_t*)&f1_t[c+n+24   ]); */
	  /*   fc_vec10  = svld1(pg,(float64_t*)&f1_t[c        ]); fc_vec11  = svld1(pg,(float64_t*)&f1_t[c  + 8   ]); */
	  /*   fc_vec12  = svld1(pg,(float64_t*)&f1_t[c  +16   ]); fc_vec13  = svld1(pg,(float64_t*)&f1_t[c  +24   ]); */
	  /*   tmp00     = svmul_x(pg,cn_vec,fc_vec00);            tmp01     = svmul_x(pg,cn_vec,fc_vec01); */
	  /*   tmp02     = svmul_x(pg,cn_vec,fc_vec02);            tmp03     = svmul_x(pg,cn_vec,fc_vec03); */
	  /*   tmp10     = svmul_x(pg,cn_vec,fc_vec10);            tmp11     = svmul_x(pg,cn_vec,fc_vec11); */
	  /*   tmp12     = svmul_x(pg,cn_vec,fc_vec12);            tmp13     = svmul_x(pg,cn_vec,fc_vec13); */
	  /*   // c */
	  /*   fc_vec00  = fc_vec10;                               fc_vec01  = fc_vec11; */
	  /*   fc_vec02  = fc_vec12;                               fc_vec03  = fc_vec13; */
	  /*   fc_vec10  = svld1(pg,(float64_t*)&f1_t[c     +nx]); fc_vec11  = svld1(pg,(float64_t*)&f1_t[c  + 8+nx]); */
	  /*   fc_vec12  = svld1(pg,(float64_t*)&f1_t[c  +16+nx]); fc_vec13  = svld1(pg,(float64_t*)&f1_t[c  +24+nx]); */
	  /*   tmp00     = svmad_x(pg,cc_vec,fc_vec00,tmp00);      tmp01     = svmad_x(pg,cc_vec,fc_vec01,tmp01); */
	  /*   tmp02     = svmad_x(pg,cc_vec,fc_vec02,tmp02);      tmp03     = svmad_x(pg,cc_vec,fc_vec03,tmp03); */
	  /*   tmp10     = svmad_x(pg,cc_vec,fc_vec10,tmp10);      tmp11     = svmad_x(pg,cc_vec,fc_vec11,tmp11); */
	  /*   tmp12     = svmad_x(pg,cc_vec,fc_vec12,tmp12);      tmp13     = svmad_x(pg,cc_vec,fc_vec13,tmp13); */
	  /*   // s */
	  /*   fc_vec00  = fc_vec10;                               fc_vec01  = fc_vec11; */
	  /*   fc_vec02  = fc_vec12;                               fc_vec03  = fc_vec13; */
	  /*   fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+s   +nx]); fc_vec11  = svld1(pg,(float64_t*)&f1_t[c+s+ 8+nx]); */
	  /*   fc_vec12  = svld1(pg,(float64_t*)&f1_t[c+s+16+nx]); fc_vec13  = svld1(pg,(float64_t*)&f1_t[c+s+24+nx]); */
	  /*   tmp00     = svmad_x(pg,cs_vec,fc_vec00,tmp00);      tmp01     = svmad_x(pg,cs_vec,fc_vec01,tmp01); */
	  /*   tmp02     = svmad_x(pg,cs_vec,fc_vec02,tmp02);      tmp03     = svmad_x(pg,cs_vec,fc_vec03,tmp03); */
	  /*   tmp10     = svmad_x(pg,cs_vec,fc_vec10,tmp10);      tmp11     = svmad_x(pg,cs_vec,fc_vec11,tmp11); */
	  /*   tmp12     = svmad_x(pg,cs_vec,fc_vec12,tmp12);      tmp13     = svmad_x(pg,cs_vec,fc_vec13,tmp13); */
	  /*   // w */
	  /*   fc_vec00  = svld1(pg,(float64_t*)&f1_t[c-1      ]); fc_vec01  = svld1(pg,(float64_t*)&f1_t[c-1+ 8   ]); */
	  /*   fc_vec02  = svld1(pg,(float64_t*)&f1_t[c-1+16   ]); fc_vec03  = svld1(pg,(float64_t*)&f1_t[c-1+24   ]); */
	  /*   fc_vec10  = svld1(pg,(float64_t*)&f1_t[c-1   +nx]); fc_vec01  = svld1(pg,(float64_t*)&f1_t[c-1+ 8+nx]); */
	  /*   fc_vec12  = svld1(pg,(float64_t*)&f1_t[c-1+16+nx]); fc_vec03  = svld1(pg,(float64_t*)&f1_t[c-1+24+nx]); */
	  /*   tmp00     = svmad_x(pg,cw_vec,fc_vec00,tmp00);      tmp01     = svmad_x(pg,cw_vec,fc_vec01,tmp01); */
	  /*   tmp02     = svmad_x(pg,cw_vec,fc_vec02,tmp02);      tmp03     = svmad_x(pg,cw_vec,fc_vec03,tmp03); */
	  /*   tmp10     = svmad_x(pg,cw_vec,fc_vec10,tmp10);      tmp11     = svmad_x(pg,cw_vec,fc_vec11,tmp11); */
	  /*   tmp12     = svmad_x(pg,cw_vec,fc_vec12,tmp12);      tmp13     = svmad_x(pg,cw_vec,fc_vec13,tmp13); */
	  /*   // e */
	  /*   fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+1      ]); fc_vec01  = svld1(pg,(float64_t*)&f1_t[c+1+ 8   ]); */
	  /*   fc_vec02  = svld1(pg,(float64_t*)&f1_t[c+1+16   ]); fc_vec03  = svld1(pg,(float64_t*)&f1_t[c+1+24   ]); */
	  /*   fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+1   +nx]); fc_vec11  = svld1(pg,(float64_t*)&f1_t[c+1+ 8+nx]); */
	  /*   fc_vec12  = svld1(pg,(float64_t*)&f1_t[c+1+16+nx]); fc_vec13  = svld1(pg,(float64_t*)&f1_t[c+1+24+nx]); */
	  /*   tmp00     = svmad_x(pg,ce_vec,fc_vec00,tmp00);      tmp01     = svmad_x(pg,ce_vec,fc_vec01,tmp01); */
	  /*   tmp02     = svmad_x(pg,ce_vec,fc_vec02,tmp02);      tmp03     = svmad_x(pg,ce_vec,fc_vec03,tmp03); */
	  /*   tmp10     = svmad_x(pg,ce_vec,fc_vec10,tmp10);      tmp11     = svmad_x(pg,ce_vec,fc_vec11,tmp11); */
	  /*   tmp12     = svmad_x(pg,ce_vec,fc_vec12,tmp12);      tmp13     = svmad_x(pg,ce_vec,fc_vec13,tmp13); */
	  /*   // b */
	  /*   fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+b      ]); fc_vec01  = svld1(pg,(float64_t*)&f1_t[c+b+ 8   ]); */
	  /*   fc_vec02  = svld1(pg,(float64_t*)&f1_t[c+b+16   ]); fc_vec03  = svld1(pg,(float64_t*)&f1_t[c+b+24   ]); */
	  /*   fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+b   +nx]); fc_vec11  = svld1(pg,(float64_t*)&f1_t[c+b+ 8+nx]); */
	  /*   fc_vec12  = svld1(pg,(float64_t*)&f1_t[c+b+16+nx]); fc_vec13  = svld1(pg,(float64_t*)&f1_t[c+b+24+nx]); */
	  /*   tmp00     = svmad_x(pg,cb_vec,fc_vec00,tmp00);      tmp01     = svmad_x(pg,cb_vec,fc_vec01,tmp01); */
	  /*   tmp02     = svmad_x(pg,cb_vec,fc_vec02,tmp02);      tmp03     = svmad_x(pg,cb_vec,fc_vec03,tmp03); */
	  /*   tmp10     = svmad_x(pg,cb_vec,fc_vec10,tmp10);      tmp11     = svmad_x(pg,cb_vec,fc_vec11,tmp11); */
	  /*   tmp12     = svmad_x(pg,cb_vec,fc_vec12,tmp12);      tmp13     = svmad_x(pg,cb_vec,fc_vec13,tmp13); */
	  /*   // t */
	  /*   fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+t      ]); fc_vec01  = svld1(pg,(float64_t*)&f1_t[c+t+ 8   ]); */
	  /*   fc_vec02  = svld1(pg,(float64_t*)&f1_t[c+t+16   ]); fc_vec03  = svld1(pg,(float64_t*)&f1_t[c+t+24   ]); */
	  /*   fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+t   +nx]); fc_vec11  = svld1(pg,(float64_t*)&f1_t[c+t+ 8+nx]); */
	  /*   fc_vec12  = svld1(pg,(float64_t*)&f1_t[c+t+16+nx]); fc_vec13  = svld1(pg,(float64_t*)&f1_t[c+t+24+nx]); */
	  /*   tmp00     = svmad_x(pg,ct_vec,fc_vec00,tmp00);      tmp01     = svmad_x(pg,ct_vec,fc_vec01,tmp01); */
	  /*   tmp02     = svmad_x(pg,ct_vec,fc_vec02,tmp02);      tmp03     = svmad_x(pg,ct_vec,fc_vec03,tmp03); */
	  /*   tmp10     = svmad_x(pg,ct_vec,fc_vec10,tmp10);      tmp11     = svmad_x(pg,ct_vec,fc_vec11,tmp11); */
	  /*   tmp12     = svmad_x(pg,ct_vec,fc_vec12,tmp12);      tmp13     = svmad_x(pg,ct_vec,fc_vec13,tmp13); */

	  /*   svst1(pg,(float64_t*)&f2_t[c      ],tmp00);         svst1(pg,(float64_t*)&f2_t[c+ 8   ],tmp01); */
	  /*   svst1(pg,(float64_t*)&f2_t[c+16   ],tmp02);         svst1(pg,(float64_t*)&f2_t[c+24   ],tmp03); */
	  /*   svst1(pg,(float64_t*)&f2_t[c   +nx],tmp10);         svst1(pg,(float64_t*)&f2_t[c+ 8+nx],tmp11); */
	  /*   svst1(pg,(float64_t*)&f2_t[c+16+nx],tmp12);         svst1(pg,(float64_t*)&f2_t[c+24+nx],tmp13); */
	    
	  /*   c += 32; */
	  /*   /\* w += 32; *\/ */
	  /*   /\* e += 32; *\/ */
	  /*   /\* n += 32; *\/ */
	  /*   /\* s += 32; *\/ */
	  /*   /\* b += 32; *\/ */
	  /*   /\* t += 32; *\/ */

	  /* } */
	  
	  /* // n */
	  /* fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+n      ]); fc_vec01  = svld1(pg,(float64_t*)&f1_t[c+n+ 8   ]); */
	  /* fc_vec02  = svld1(pg,(float64_t*)&f1_t[c+n+16   ]); fc_vec03  = svld1(pg,(float64_t*)&f1_t[c+n+24   ]); */
	  /* fc_vec10  = svld1(pg,(float64_t*)&f1_t[c        ]); fc_vec11  = svld1(pg,(float64_t*)&f1_t[c  + 8   ]); */
	  /* fc_vec12  = svld1(pg,(float64_t*)&f1_t[c  +16   ]); fc_vec13  = svld1(pg,(float64_t*)&f1_t[c  +24   ]); */
	  /* tmp00     = svmul_x(pg,cn_vec,fc_vec00);            tmp01     = svmul_x(pg,cn_vec,fc_vec01); */
	  /* tmp02     = svmul_x(pg,cn_vec,fc_vec02);            tmp03     = svmul_x(pg,cn_vec,fc_vec03); */
	  /* tmp10     = svmul_x(pg,cn_vec,fc_vec10);            tmp11     = svmul_x(pg,cn_vec,fc_vec11); */
	  /* tmp12     = svmul_x(pg,cn_vec,fc_vec12);            tmp13     = svmul_x(pg,cn_vec,fc_vec13); */
	  /* // c */
	  /* fc_vec00  = fc_vec10;                               fc_vec01  = fc_vec11; */
	  /* fc_vec02  = fc_vec12;                               fc_vec03  = fc_vec13; */
	  /* fc_vec10  = svld1(pg,(float64_t*)&f1_t[c     +nx]); fc_vec11  = svld1(pg,(float64_t*)&f1_t[c  + 8+nx]); */
	  /* fc_vec12  = svld1(pg,(float64_t*)&f1_t[c  +16+nx]); fc_vec13  = svld1(pg,(float64_t*)&f1_t[c  +24+nx]); */
	  /* tmp00     = svmad_x(pg,cc_vec,fc_vec00,tmp00);      tmp01     = svmad_x(pg,cc_vec,fc_vec01,tmp01); */
	  /* tmp02     = svmad_x(pg,cc_vec,fc_vec02,tmp02);      tmp03     = svmad_x(pg,cc_vec,fc_vec03,tmp03); */
	  /* tmp10     = svmad_x(pg,cc_vec,fc_vec10,tmp10);      tmp11     = svmad_x(pg,cc_vec,fc_vec11,tmp11); */
	  /* tmp12     = svmad_x(pg,cc_vec,fc_vec12,tmp12);      tmp13     = svmad_x(pg,cc_vec,fc_vec13,tmp13); */
	  /* // s */
	  /* fc_vec00  = fc_vec10;                               fc_vec01  = fc_vec11; */
	  /* fc_vec02  = fc_vec12;                               fc_vec03  = fc_vec13; */
	  /* fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+s   +nx]); fc_vec11  = svld1(pg,(float64_t*)&f1_t[c+s+ 8+nx]); */
	  /* fc_vec12  = svld1(pg,(float64_t*)&f1_t[c+s+16+nx]); fc_vec13  = svld1(pg,(float64_t*)&f1_t[c+s+24+nx]); */
	  /* tmp00     = svmad_x(pg,cs_vec,fc_vec00,tmp00);      tmp01     = svmad_x(pg,cs_vec,fc_vec01,tmp01); */
	  /* tmp02     = svmad_x(pg,cs_vec,fc_vec02,tmp02);      tmp03     = svmad_x(pg,cs_vec,fc_vec03,tmp03); */
	  /* tmp10     = svmad_x(pg,cs_vec,fc_vec10,tmp10);      tmp11     = svmad_x(pg,cs_vec,fc_vec11,tmp11); */
	  /* tmp12     = svmad_x(pg,cs_vec,fc_vec12,tmp12);      tmp13     = svmad_x(pg,cs_vec,fc_vec13,tmp13); */
	  /* // w */
	  /* fc_vec00  = svld1(pg,(float64_t*)&f1_t[c-1      ]); fc_vec01  = svld1(pg,(float64_t*)&f1_t[c-1+ 8   ]); */
	  /* fc_vec02  = svld1(pg,(float64_t*)&f1_t[c-1+16   ]); fc_vec03  = svld1(pg,(float64_t*)&f1_t[c-1+24   ]); */
	  /* fc_vec10  = svld1(pg,(float64_t*)&f1_t[c-1   +nx]); fc_vec01  = svld1(pg,(float64_t*)&f1_t[c-1+ 8+nx]); */
	  /* fc_vec12  = svld1(pg,(float64_t*)&f1_t[c-1+16+nx]); fc_vec03  = svld1(pg,(float64_t*)&f1_t[c-1+24+nx]); */
	  /* tmp00     = svmad_x(pg,cw_vec,fc_vec00,tmp00);      tmp01     = svmad_x(pg,cw_vec,fc_vec01,tmp01); */
	  /* tmp02     = svmad_x(pg,cw_vec,fc_vec02,tmp02);      tmp03     = svmad_x(pg,cw_vec,fc_vec03,tmp03); */
	  /* tmp10     = svmad_x(pg,cw_vec,fc_vec10,tmp10);      tmp11     = svmad_x(pg,cw_vec,fc_vec11,tmp11); */
	  /* tmp12     = svmad_x(pg,cw_vec,fc_vec12,tmp12);      tmp13     = svmad_x(pg,cw_vec,fc_vec13,tmp13); */
	  /* // e */
	  /* float64_t fcp1[8] = {f1_t[c+1+24],f1_t[c+2+24],f1_t[c+3+24],f1_t[c+4+24],f1_t[c+5+24],f1_t[c+6+24],f1_t[c+7+24],f1_t[c+7+24]}; */
	  /* float64_t fcp2[8] = {f1_t[c+1+24+nx],f1_t[c+2+24+nx],f1_t[c+3+24+nx],f1_t[c+4+24+nx],f1_t[c+5+24+nx],f1_t[c+6+24+nx],f1_t[c+7+24+nx],f1_t[c+7+24+nx]}; */
	  /* fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+1      ]); fc_vec01  = svld1(pg,(float64_t*)&f1_t[c+1+ 8   ]); */
	  /* fc_vec02  = svld1(pg,(float64_t*)&f1_t[c+1+16   ]); fc_vec03  = svld1(pg,(float64_t*)&fcp1[0        ]); */
	  /* fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+1   +nx]); fc_vec11  = svld1(pg,(float64_t*)&f1_t[c+1+ 8+nx]); */
	  /* fc_vec12  = svld1(pg,(float64_t*)&f1_t[c+1+16+nx]); fc_vec13  = svld1(pg,(float64_t*)&fcp2[0        ]); */
	  /* tmp00     = svmad_x(pg,ce_vec,fc_vec00,tmp00);      tmp01     = svmad_x(pg,ce_vec,fc_vec01,tmp01); */
	  /* tmp02     = svmad_x(pg,ce_vec,fc_vec02,tmp02);      tmp03     = svmad_x(pg,ce_vec,fc_vec03,tmp03); */
	  /* tmp10     = svmad_x(pg,ce_vec,fc_vec10,tmp10);      tmp11     = svmad_x(pg,ce_vec,fc_vec11,tmp11); */
	  /* tmp12     = svmad_x(pg,ce_vec,fc_vec12,tmp12);      tmp13     = svmad_x(pg,ce_vec,fc_vec13,tmp13); */
	  /* // b */
	  /* fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+b      ]); fc_vec01  = svld1(pg,(float64_t*)&f1_t[c+b+ 8   ]); */
	  /* fc_vec02  = svld1(pg,(float64_t*)&f1_t[c+b+16   ]); fc_vec03  = svld1(pg,(float64_t*)&f1_t[c+b+24   ]); */
	  /* fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+b   +nx]); fc_vec11  = svld1(pg,(float64_t*)&f1_t[c+b+ 8+nx]); */
	  /* fc_vec12  = svld1(pg,(float64_t*)&f1_t[c+b+16+nx]); fc_vec13  = svld1(pg,(float64_t*)&f1_t[c+b+24+nx]); */
	  /* tmp00     = svmad_x(pg,cb_vec,fc_vec00,tmp00);      tmp01     = svmad_x(pg,cb_vec,fc_vec01,tmp01); */
	  /* tmp02     = svmad_x(pg,cb_vec,fc_vec02,tmp02);      tmp03     = svmad_x(pg,cb_vec,fc_vec03,tmp03); */
	  /* tmp10     = svmad_x(pg,cb_vec,fc_vec10,tmp10);      tmp11     = svmad_x(pg,cb_vec,fc_vec11,tmp11); */
	  /* tmp12     = svmad_x(pg,cb_vec,fc_vec12,tmp12);      tmp13     = svmad_x(pg,cb_vec,fc_vec13,tmp13); */
	  /* // t */
	  /* fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+t      ]); fc_vec01  = svld1(pg,(float64_t*)&f1_t[c+t+ 8   ]); */
	  /* fc_vec02  = svld1(pg,(float64_t*)&f1_t[c+t+16   ]); fc_vec03  = svld1(pg,(float64_t*)&f1_t[c+t+24   ]); */
	  /* fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+t   +nx]); fc_vec11  = svld1(pg,(float64_t*)&f1_t[c+t+ 8+nx]); */
	  /* fc_vec12  = svld1(pg,(float64_t*)&f1_t[c+t+16+nx]); fc_vec13  = svld1(pg,(float64_t*)&f1_t[c+t+24+nx]); */
	  /* tmp00     = svmad_x(pg,ct_vec,fc_vec00,tmp00);      tmp01     = svmad_x(pg,ct_vec,fc_vec01,tmp01); */
	  /* tmp02     = svmad_x(pg,ct_vec,fc_vec02,tmp02);      tmp03     = svmad_x(pg,ct_vec,fc_vec03,tmp03); */
	  /* tmp10     = svmad_x(pg,ct_vec,fc_vec10,tmp10);      tmp11     = svmad_x(pg,ct_vec,fc_vec11,tmp11); */
	  /* tmp12     = svmad_x(pg,ct_vec,fc_vec12,tmp12);      tmp13     = svmad_x(pg,ct_vec,fc_vec13,tmp13); */

	  /* svst1(pg,(float64_t*)&f2_t[c      ],tmp00);         svst1(pg,(float64_t*)&f2_t[c+ 8   ],tmp01); */
	  /* svst1(pg,(float64_t*)&f2_t[c+16   ],tmp02);         svst1(pg,(float64_t*)&f2_t[c+24   ],tmp03); */
	  /* svst1(pg,(float64_t*)&f2_t[c   +nx],tmp10);         svst1(pg,(float64_t*)&f2_t[c+ 8+nx],tmp11); */
	  /* svst1(pg,(float64_t*)&f2_t[c+16+nx],tmp12);         svst1(pg,(float64_t*)&f2_t[c+24+nx],tmp13); */
	  
	  /*  */


	  svfloat64_t fc_vec00;
	  svfloat64_t fc_vec10;
	  svfloat64_t tmp00;
	  svfloat64_t tmp10;
#if 0
	  // n
	  fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+n      ]);
	  fc_vec10  = svld1(pg,(float64_t*)&f1_t[c        ]);
	  tmp00     = svmul_x(pg,cn_vec,fc_vec00);
	  tmp10     = svmul_x(pg,cn_vec,fc_vec10);
	  // c
	  fc_vec00  = fc_vec10;
	  fc_vec10  = svld1(pg,(float64_t*)&f1_t[c     +nx]);
	  tmp00     = svmad_x(pg,cc_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,cc_vec,fc_vec10,tmp10);
	  // s
	  fc_vec00  = fc_vec10;
	  fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+s   +nx]);
	  tmp00     = svmad_x(pg,cs_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,cs_vec,fc_vec10,tmp10);
	  // w
	  float64_t fcm1[8] = {f1_t[c],f1_t[c],f1_t[c+1],f1_t[c+2],f1_t[c+3],f1_t[c+4],f1_t[c+5],f1_t[c+6]};
	  float64_t fcm2[8] = {f1_t[c+nx],f1_t[c+nx],f1_t[c+1+nx],f1_t[c+2+nx],f1_t[c+3+nx],f1_t[c+4+nx],f1_t[c+5+nx],f1_t[c+6+nx]};
	  fc_vec00  = svld1(pg,(float64_t*)&fcm1[0        ]);
	  fc_vec10  = svld1(pg,(float64_t*)&fcm2[0        ]);
	  tmp00     = svmad_x(pg,cw_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,cw_vec,fc_vec10,tmp10);
	  // e
	  fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+1      ]);
	  fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+1   +nx]);
	  tmp00     = svmad_x(pg,ce_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,ce_vec,fc_vec10,tmp10);
	  // b
	  fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+b      ]);
	  fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+b   +nx]);
	  tmp00     = svmad_x(pg,cb_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,cb_vec,fc_vec10,tmp10);
	  // t
	  fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+t      ]);
	  fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+t   +nx]);
	  tmp00     = svmad_x(pg,ct_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,ct_vec,fc_vec10,tmp10);
	  
	  svst1(pg,(float64_t*)&f2_t[c      ],tmp00);
	  svst1(pg,(float64_t*)&f2_t[c   +nx],tmp10);
#else
	  // s
	  fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+s   +nx]);
	  fc_vec00  = svld1(pg,(float64_t*)&f1_t[c     +nx]);
	  tmp00     = svmul_x(pg,cs_vec,fc_vec00);
	  tmp10     = svmul_x(pg,cs_vec,fc_vec10);
	  // c
	  fc_vec10  = fc_vec00;
	  fc_vec00  = svld1(pg,(float64_t*)&f1_t[c        ]);
	  tmp00     = svmad_x(pg,cc_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,cc_vec,fc_vec10,tmp10);
	  // n
	  fc_vec10  = fc_vec00;
	  fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+n      ]);
	  tmp00     = svmad_x(pg,cn_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,cn_vec,fc_vec10,tmp10);
	  // w
	  float64_t fcm2[8] = {f1_t[c+nx],f1_t[c+nx],f1_t[c+1+nx],f1_t[c+2+nx],f1_t[c+3+nx],f1_t[c+4+nx],f1_t[c+5+nx],f1_t[c+6+nx]};
	  float64_t fcm1[8] = {f1_t[c],f1_t[c],f1_t[c+1],f1_t[c+2],f1_t[c+3],f1_t[c+4],f1_t[c+5],f1_t[c+6]};
	  fc_vec10  = svld1(pg,(float64_t*)&fcm2[0        ]);
	  fc_vec00  = svld1(pg,(float64_t*)&fcm1[0        ]);
	  tmp00     = svmad_x(pg,cw_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,cw_vec,fc_vec10,tmp10);
	  // e
	  fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+1   +nx]);
	  fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+1      ]);
	  tmp00     = svmad_x(pg,ce_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,ce_vec,fc_vec10,tmp10);
	  // b
	  fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+b   +nx]);
	  fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+b      ]);
	  tmp00     = svmad_x(pg,cb_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,cb_vec,fc_vec10,tmp10);
	  // t
	  fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+t   +nx]);
	  fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+t      ]);
	  tmp00     = svmad_x(pg,ct_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,ct_vec,fc_vec10,tmp10);
	  
	  svst1(pg,(float64_t*)&f2_t[c   +nx],tmp10);
	  svst1(pg,(float64_t*)&f2_t[c      ],tmp00);
#endif	  

	  c += 8;
	  /* w += 32; */
	  /* e += 32; */
	  /* n += 32; */
	  /* s += 32; */
	  /* b += 32; */
	  /* t += 32; */

	  for (x = 8; x < nx-8; x+=8) {
	    
#if 0
	    // n
	    fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+n      ]);
	    fc_vec10  = svld1(pg,(float64_t*)&f1_t[c        ]);
	    tmp00     = svmul_x(pg,cn_vec,fc_vec00);
	    tmp10     = svmul_x(pg,cn_vec,fc_vec10);
	    // c
	    fc_vec00  = fc_vec10;
	    fc_vec10  = svld1(pg,(float64_t*)&f1_t[c     +nx]);
	    tmp00     = svmad_x(pg,cc_vec,fc_vec00,tmp00);
	    tmp10     = svmad_x(pg,cc_vec,fc_vec10,tmp10);
	    // s
	    fc_vec00  = fc_vec10;
	    fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+s   +nx]);
	    tmp00     = svmad_x(pg,cs_vec,fc_vec00,tmp00);
	    tmp10     = svmad_x(pg,cs_vec,fc_vec10,tmp10);
	    // w
	    fc_vec00  = svld1(pg,(float64_t*)&f1_t[c-1      ]);
	    fc_vec10  = svld1(pg,(float64_t*)&f1_t[c-1   +nx]);
	    tmp00     = svmad_x(pg,cw_vec,fc_vec00,tmp00);
	    tmp10     = svmad_x(pg,cw_vec,fc_vec10,tmp10);
	    // e
	    fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+1      ]);
	    fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+1   +nx]);
	    tmp00     = svmad_x(pg,ce_vec,fc_vec00,tmp00);
	    tmp10     = svmad_x(pg,ce_vec,fc_vec10,tmp10);
	    // b
	    fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+b      ]);
	    fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+b   +nx]);
	    tmp00     = svmad_x(pg,cb_vec,fc_vec00,tmp00);
	    tmp10     = svmad_x(pg,cb_vec,fc_vec10,tmp10);
	    // t
	    fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+t      ]);
	    fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+t   +nx]);
	    tmp00     = svmad_x(pg,ct_vec,fc_vec00,tmp00);
	    tmp10     = svmad_x(pg,ct_vec,fc_vec10,tmp10);
	    
	    svst1(pg,(float64_t*)&f2_t[c      ],tmp00);
	    svst1(pg,(float64_t*)&f2_t[c   +nx],tmp10);
#else
	    // s
	    fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+s   +nx]);
	    fc_vec00  = svld1(pg,(float64_t*)&f1_t[c     +nx]);
	    tmp00     = svmul_x(pg,cs_vec,fc_vec00);
	    tmp10     = svmul_x(pg,cs_vec,fc_vec10);
	    // c
	    fc_vec10  = fc_vec00;
	    fc_vec00  = svld1(pg,(float64_t*)&f1_t[c        ]);
	    tmp00     = svmad_x(pg,cc_vec,fc_vec00,tmp00);
	    tmp10     = svmad_x(pg,cc_vec,fc_vec10,tmp10);
	    // n
	    fc_vec10  = fc_vec00;
	    fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+n      ]);
	    tmp00     = svmad_x(pg,cn_vec,fc_vec00,tmp00);
	    tmp10     = svmad_x(pg,cn_vec,fc_vec10,tmp10);
	    // w
	    fc_vec10  = svld1(pg,(float64_t*)&f1_t[c-1   +nx]);
	    fc_vec00  = svld1(pg,(float64_t*)&f1_t[c-1      ]);
	    tmp00     = svmad_x(pg,cw_vec,fc_vec00,tmp00);
	    tmp10     = svmad_x(pg,cw_vec,fc_vec10,tmp10);
	    // e
	    fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+1   +nx]);
	    fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+1      ]);
	    tmp00     = svmad_x(pg,ce_vec,fc_vec00,tmp00);
	    tmp10     = svmad_x(pg,ce_vec,fc_vec10,tmp10);
	    // b
	    fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+b   +nx]);
	    fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+b      ]);
	    tmp00     = svmad_x(pg,cb_vec,fc_vec00,tmp00);
	    tmp10     = svmad_x(pg,cb_vec,fc_vec10,tmp10);
	    // t
	    fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+t   +nx]);
	    fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+t      ]);
	    tmp00     = svmad_x(pg,ct_vec,fc_vec00,tmp00);
	    tmp10     = svmad_x(pg,ct_vec,fc_vec10,tmp10);
	    
	    svst1(pg,(float64_t*)&f2_t[c   +nx],tmp10);
	    svst1(pg,(float64_t*)&f2_t[c      ],tmp00);
#endif	  
	    
	    c += 8;
	    /* w += 32; */
	    /* e += 32; */
	    /* n += 32; */
	    /* s += 32; */
	    /* b += 32; */
	    /* t += 32; */

	  }
	  
#if 0
	  // n
	  fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+n      ]);
	  fc_vec10  = svld1(pg,(float64_t*)&f1_t[c        ]);
	  tmp00     = svmul_x(pg,cn_vec,fc_vec00);
	  tmp10     = svmul_x(pg,cn_vec,fc_vec10);
	  // c
	  fc_vec00  = fc_vec10;
	  fc_vec10  = svld1(pg,(float64_t*)&f1_t[c     +nx]);
	  tmp00     = svmad_x(pg,cc_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,cc_vec,fc_vec10,tmp10);
	  // s
	  fc_vec00  = fc_vec10;
	  fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+s   +nx]);
	  tmp00     = svmad_x(pg,cs_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,cs_vec,fc_vec10,tmp10);
	  // w
	  fc_vec00  = svld1(pg,(float64_t*)&f1_t[c-1      ]);
	  fc_vec10  = svld1(pg,(float64_t*)&f1_t[c-1   +nx]);
	  tmp00     = svmad_x(pg,cw_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,cw_vec,fc_vec10,tmp10);
	  // e
	  float64_t fcp1[8] = {f1_t[c+1],f1_t[c+2],f1_t[c+3],f1_t[c+4],f1_t[c+5],f1_t[c+6],f1_t[c+7],f1_t[c+7]};
	  float64_t fcp2[8] = {f1_t[c+1+nx],f1_t[c+2+nx],f1_t[c+3+nx],f1_t[c+4+nx],f1_t[c+5+nx],f1_t[c+6+nx],f1_t[c+7+nx],f1_t[c+7+nx]};
	  fc_vec00  = svld1(pg,(float64_t*)&fcp1[0        ]);
	  fc_vec10  = svld1(pg,(float64_t*)&fcp2[0        ]);
	  tmp00     = svmad_x(pg,ce_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,ce_vec,fc_vec10,tmp10);
	  // b
	  fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+b      ]);
	  fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+b   +nx]);
	  tmp00     = svmad_x(pg,cb_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,cb_vec,fc_vec10,tmp10);
	  // t
	  fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+t      ]);
	  fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+t   +nx]);
	  tmp00     = svmad_x(pg,ct_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,ct_vec,fc_vec10,tmp10);
	  
	  svst1(pg,(float64_t*)&f2_t[c      ],tmp00);
	  svst1(pg,(float64_t*)&f2_t[c   +nx],tmp10);
#else
	  // s
	  fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+s   +nx]);
	  fc_vec00  = svld1(pg,(float64_t*)&f1_t[c     +nx]);
	  tmp00     = svmul_x(pg,cs_vec,fc_vec00);
	  tmp10     = svmul_x(pg,cs_vec,fc_vec10);
	  // c
	  fc_vec10  = fc_vec00;
	  fc_vec00  = svld1(pg,(float64_t*)&f1_t[c        ]);
	  tmp00     = svmad_x(pg,cc_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,cc_vec,fc_vec10,tmp10);
	  // n
	  fc_vec10  = fc_vec00;
	  fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+n      ]);
	  tmp00     = svmad_x(pg,cn_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,cn_vec,fc_vec10,tmp10);
	  // w
	  fc_vec10  = svld1(pg,(float64_t*)&f1_t[c-1   +nx]);
	  fc_vec00  = svld1(pg,(float64_t*)&f1_t[c-1      ]);
	  tmp00     = svmad_x(pg,cw_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,cw_vec,fc_vec10,tmp10);
	  // e
	  float64_t fcp2[8] = {f1_t[c+1+nx],f1_t[c+2+nx],f1_t[c+3+nx],f1_t[c+4+nx],f1_t[c+5+nx],f1_t[c+6+nx],f1_t[c+7+nx],f1_t[c+7+nx]};
	  float64_t fcp1[8] = {f1_t[c+1],f1_t[c+2],f1_t[c+3],f1_t[c+4],f1_t[c+5],f1_t[c+6],f1_t[c+7],f1_t[c+7]};
	  fc_vec10  = svld1(pg,(float64_t*)&fcp2[0        ]);
	  fc_vec00  = svld1(pg,(float64_t*)&fcp1[0        ]);
	  tmp00     = svmad_x(pg,ce_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,ce_vec,fc_vec10,tmp10);
	  // b
	  fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+b   +nx]);
	  fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+b      ]);
	  tmp00     = svmad_x(pg,cb_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,cb_vec,fc_vec10,tmp10);
	  // t
	  fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+t   +nx]);
	  fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+t      ]);
	  tmp00     = svmad_x(pg,ct_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,ct_vec,fc_vec10,tmp10);
	  
	  svst1(pg,(float64_t*)&f2_t[c   +nx],tmp10);
	  svst1(pg,(float64_t*)&f2_t[c      ],tmp00);
#endif	  
	  /* // n */
	  /* fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+n      ]); */
	  /* fc_vec10  = svld1(pg,(float64_t*)&f1_t[c        ]); */
	  /* tmp00     = svmul_x(pg,cn_vec,fc_vec00); */
	  /* tmp10     = svmul_x(pg,cn_vec,fc_vec10); */
	  /* // c */
	  /* fc_vec00  = fc_vec10; */
	  /* //fc_vec00  = svld1(pg,(float64_t*)&f1_t[c        ]); */
	  /* fc_vec10  = svld1(pg,(float64_t*)&f1_t[c     +nx]); */
	  /* tmp00     = svmad_x(pg,cc_vec,fc_vec00,tmp00); */
	  /* tmp10     = svmad_x(pg,cc_vec,fc_vec10,tmp10); */
	  /* // s */
	  /* fc_vec00  = fc_vec10; */
	  /* //	  fc_vec00  = svld1(pg,(float64_t*)&f1_t[c     +nx]); */
	  /* //fc_vec00  = svld1(pg,(float64_t*)&f1_t[c+s      ]); */
	  /* fc_vec10  = svld1(pg,(float64_t*)&f1_t[c+s   +nx]); */
	  /* tmp00     = svmad_x(pg,cs_vec,fc_vec00,tmp00); */
	  /* tmp10     = svmad_x(pg,cs_vec,fc_vec10,tmp10); */
	  /*  */
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
  

