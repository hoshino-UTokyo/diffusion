#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#ifdef SVE
#include <arm_sve.h>
#endif /* SVE */
#include "diffusion_ker39.h"

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

void allocate_ker39(REAL **buff_ret, const int nx, const int ny, const int nz) {

  posix_memalign((void**)buff_ret, 64, sizeof(REAL)*nx*ny*nz);

}

void init_ker39(REAL *buff1, const int nx, const int ny, const int nz,
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


void diffusion_ker39(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
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

    svfloat64_t z0, z1, z2, z3, z4, z5, z6, z7, z8, z9;
    svfloat64_t z10,z11,z12,z13,z14,z15,z16,z17,z18,z19;
    svfloat64_t z20,z21,z22,z23,z24,z25,z26,z27,z28,z29;
    svfloat64_t z30,z31;
    z0 = svdup_f64(cc); // cc_vec
    z1 = svdup_f64(cw); // cw_vec
    z2 = svdup_f64(ce); // ce_vec
    z3 = svdup_f64(cn); // cn_vec
    z4 = svdup_f64(cs); // cs_vec
    z5 = svdup_f64(cb); // cb_vec
    z6 = svdup_f64(ct); // ct_vec
    const svbool_t p0 = svptrue_b64();
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
	  svfloat64_t fz20,fz21;
	  svfloat64_t fz10,fz11;
	  svfloat64_t fz30,fz31;
	  svfloat64_t fz40,fz41;
	  svfloat64_t fz50,fz51;
	  svfloat64_t fz60,fz61;

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
	  
	  fc_vec0  = svld1(p0,(float64_t*)&f1_t[c]);
	  fz20 = svld1(p0,(float64_t*)&f1_t[e]);
	  float64_t fcm1_arr[8] = {f1_t[c],f1_t[c],f1_t[c+1],f1_t[c+2],f1_t[c+3],f1_t[c+4],f1_t[c+5],f1_t[c+6]};
	  fz10 = svld1(p0,(float64_t*)&fcm1_arr[0]);
	  fz30 = svld1(p0,(float64_t*)&f1_t[s]);
	  fz40 = svld1(p0,(float64_t*)&f1_t[n]);
	  fz50 = svld1(p0,(float64_t*)&f1_t[b]);
	  fz60 = svld1(p0,(float64_t*)&f1_t[t]);

	  fc_vec1  = svld1(p0,(float64_t*)&f1_t[c+8]);
	  fz21 = svld1(p0,(float64_t*)&f1_t[e+8]);
	  fz11 = svld1(p0,(float64_t*)&f1_t[w+8]);
	  fz31 = svld1(p0,(float64_t*)&f1_t[s+8]);
	  fz41 = svld1(p0,(float64_t*)&f1_t[n+8]);
	  fz51 = svld1(p0,(float64_t*)&f1_t[b+8]);
	  fz61 = svld1(p0,(float64_t*)&f1_t[t+8]);
	  
	  svfloat64_t tmp0,tmp1,tmp2,tmp3,tmp4,tmp5;
	  fc_vec0  = svmul_x(p0,z0,fc_vec0);  fc_vec1  = svmul_x(p0,z0,fc_vec1);
	  fz10 = svmul_x(p0,z1,fz10); fz11 = svmul_x(p0,z1,fz11);
	  fz20 = svmul_x(p0,z2,fz20); fz21 = svmul_x(p0,z2,fz21);
	  fz30 = svmul_x(p0,z3,fz30); fz31 = svmul_x(p0,z3,fz31);
	  fz40 = svmul_x(p0,z4,fz40); fz41 = svmul_x(p0,z4,fz41);
	  fz50 = svmul_x(p0,z5,fz50); fz51 = svmul_x(p0,z5,fz51);
	  fz60 = svmul_x(p0,z6,fz60); fz61 = svmul_x(p0,z6,fz61);

	  tmp0 = svadd_x(p0,fz10,fz20); tmp1 = svadd_x(p0,fz11,fz21);
	  tmp2 = svadd_x(p0,fz30,fz40); tmp3 = svadd_x(p0,fz31,fz41);
	  tmp4 = svadd_x(p0,fz60,fz50); tmp5 = svadd_x(p0,fz61,fz51);
	  tmp0 = svadd_x(p0,fc_vec0, tmp0    ); tmp1 = svadd_x(p0,fc_vec1, tmp1    );
	  tmp2 = svadd_x(p0,tmp2   , tmp4    ); tmp3 = svadd_x(p0,tmp3   , tmp5    );
	  tmp0 = svadd_x(p0,tmp0   , tmp2    ); tmp1 = svadd_x(p0,tmp1   , tmp3    );

	  svst1(p0,(float64_t*)&f2_t[c  ],tmp0);
	  svst1(p0,(float64_t*)&f2_t[c+8],tmp1);
	  
	  c += 16;
	  w += 16;
	  e += 16;
	  n += 16;
	  s += 16;
	  b += 16;
	  t += 16;

	  for (x = 16; x < nx-16; x+=16) {
	    
	    fc_vec0  = svld1(p0,(float64_t*)&f1_t[c]);
	    fz20 = svld1(p0,(float64_t*)&f1_t[e]);
	    fz10 = svld1(p0,(float64_t*)&f1_t[w]);
	    fz30 = svld1(p0,(float64_t*)&f1_t[s]);
	    fz40 = svld1(p0,(float64_t*)&f1_t[n]);
	    fz50 = svld1(p0,(float64_t*)&f1_t[b]);
	    fz60 = svld1(p0,(float64_t*)&f1_t[t]);

	    fc_vec1  = svld1(p0,(float64_t*)&f1_t[c+8]);
	    fz21 = svld1(p0,(float64_t*)&f1_t[e+8]);
	    fz11 = svld1(p0,(float64_t*)&f1_t[w+8]);
	    fz31 = svld1(p0,(float64_t*)&f1_t[s+8]);
	    fz41 = svld1(p0,(float64_t*)&f1_t[n+8]);
	    fz51 = svld1(p0,(float64_t*)&f1_t[b+8]);
	    fz61 = svld1(p0,(float64_t*)&f1_t[t+8]);
	    
	    /* tmp0 = svmul_x(p0,z0,fc_vec0); tmp1 = svmul_x(p0,z0,fc_vec1); */
	    /* tmp0 = svmad_x(p0,z1,fz10,tmp0); tmp1 = svmad_x(p0,z1,fz11,tmp1); */
	    /* tmp0 = svmad_x(p0,z2,fz20,tmp0); tmp1 = svmad_x(p0,z2,fz21,tmp1); */
	    /* tmp0 = svmad_x(p0,z3,fz30,tmp0); tmp1 = svmad_x(p0,z3,fz31,tmp1); */
	    /* tmp0 = svmad_x(p0,z4,fz40,tmp0); tmp1 = svmad_x(p0,z4,fz41,tmp1); */
	    /* tmp0 = svmad_x(p0,z5,fz50,tmp0); tmp1 = svmad_x(p0,z5,fz51,tmp1); */
	    /* tmp0 = svmad_x(p0,z6,fz60,tmp0); tmp1 = svmad_x(p0,z6,fz61,tmp1); */
	    fc_vec0  = svmul_x(p0,z0,fc_vec0);  fc_vec1  = svmul_x(p0,z0,fc_vec1);
	    fz10 = svmul_x(p0,z1,fz10); fz11 = svmul_x(p0,z1,fz11);
	    fz20 = svmul_x(p0,z2,fz20); fz21 = svmul_x(p0,z2,fz21);
	    fz30 = svmul_x(p0,z3,fz30); fz31 = svmul_x(p0,z3,fz31);
	    fz40 = svmul_x(p0,z4,fz40); fz41 = svmul_x(p0,z4,fz41);
	    fz50 = svmul_x(p0,z5,fz50); fz51 = svmul_x(p0,z5,fz51);
	    fz60 = svmul_x(p0,z6,fz60); fz61 = svmul_x(p0,z6,fz61);
	    
	    tmp0 = svadd_x(p0,fz10,fz20); tmp1 = svadd_x(p0,fz11,fz21);
	    tmp2 = svadd_x(p0,fz30,fz40); tmp3 = svadd_x(p0,fz31,fz41);
	    tmp4 = svadd_x(p0,fz60,fz50); tmp5 = svadd_x(p0,fz61,fz51);
	    tmp0 = svadd_x(p0,fc_vec0, tmp0    ); tmp1 = svadd_x(p0,fc_vec1, tmp1    );
	    tmp2 = svadd_x(p0,tmp2   , tmp4    ); tmp3 = svadd_x(p0,tmp3   , tmp5    );
	    tmp0 = svadd_x(p0,tmp0   , tmp2    ); tmp1 = svadd_x(p0,tmp1   , tmp3    );
	    
	    svst1(p0,(float64_t*)&f2_t[c  ],tmp0);
	    svst1(p0,(float64_t*)&f2_t[c+8],tmp1);

	    c += 16;
	    w += 16;
	    e += 16;
	    n += 16;
	    s += 16;
	    b += 16;
	    t += 16;
	  }
	  fc_vec0  = svld1(p0,(float64_t*)&f1_t[c]);
	  fz20 = svld1(p0,(float64_t*)&f1_t[e]);
	  fz10 = svld1(p0,(float64_t*)&f1_t[w]);
	  fz30 = svld1(p0,(float64_t*)&f1_t[s]);
	  fz40 = svld1(p0,(float64_t*)&f1_t[n]);
	  fz50 = svld1(p0,(float64_t*)&f1_t[b]);
	  fz60 = svld1(p0,(float64_t*)&f1_t[t]);

	  fc_vec1  = svld1(p0,(float64_t*)&f1_t[c+8]);
	  float64_t fcp1_arr[8] = {f1_t[c+1+8],f1_t[c+2+8],f1_t[c+3+8],f1_t[c+4+8],f1_t[c+5+8],f1_t[c+6+8],f1_t[c+7+8],f1_t[c+7+8]};
	  fz21 = svld1(p0,(float64_t*)&fcp1_arr[0]);
	  fz11 = svld1(p0,(float64_t*)&f1_t[w+8]);
	  fz31 = svld1(p0,(float64_t*)&f1_t[s+8]);
	  fz41 = svld1(p0,(float64_t*)&f1_t[n+8]);
	  fz51 = svld1(p0,(float64_t*)&f1_t[b+8]);
	  fz61 = svld1(p0,(float64_t*)&f1_t[t+8]);
	  /* tmp0 = svmul_x(p0,z0,fc_vec0); tmp1 = svmul_x(p0,z0,fc_vec1); */
	  /* tmp0 = svmad_x(p0,z1,fz10,tmp0); tmp1 = svmad_x(p0,z1,fz11,tmp1); */
	  /* tmp0 = svmad_x(p0,z2,fz20,tmp0); tmp1 = svmad_x(p0,z2,fz21,tmp1); */
	  /* tmp0 = svmad_x(p0,z3,fz30,tmp0); tmp1 = svmad_x(p0,z3,fz31,tmp1); */
	  /* tmp0 = svmad_x(p0,z4,fz40,tmp0); tmp1 = svmad_x(p0,z4,fz41,tmp1); */
	  /* tmp0 = svmad_x(p0,z5,fz50,tmp0); tmp1 = svmad_x(p0,z5,fz51,tmp1); */
	  /* tmp0 = svmad_x(p0,z6,fz60,tmp0); tmp1 = svmad_x(p0,z6,fz61,tmp1); */
	  fc_vec0  = svmul_x(p0,z0,fc_vec0);  fc_vec1  = svmul_x(p0,z0,fc_vec1);
	  fz10 = svmul_x(p0,z1,fz10); fz11 = svmul_x(p0,z1,fz11);
	  fz20 = svmul_x(p0,z2,fz20); fz21 = svmul_x(p0,z2,fz21);
	  fz30 = svmul_x(p0,z3,fz30); fz31 = svmul_x(p0,z3,fz31);
	  fz40 = svmul_x(p0,z4,fz40); fz41 = svmul_x(p0,z4,fz41);
	  fz50 = svmul_x(p0,z5,fz50); fz51 = svmul_x(p0,z5,fz51);
	  fz60 = svmul_x(p0,z6,fz60); fz61 = svmul_x(p0,z6,fz61);

	  tmp0 = svadd_x(p0,fz10,fz20); tmp1 = svadd_x(p0,fz11,fz21);
	  tmp2 = svadd_x(p0,fz30,fz40); tmp3 = svadd_x(p0,fz31,fz41);
	  tmp4 = svadd_x(p0,fz60,fz50); tmp5 = svadd_x(p0,fz61,fz51);
	  tmp0 = svadd_x(p0,fc_vec0, tmp0    ); tmp1 = svadd_x(p0,fc_vec1, tmp1    );
	  tmp2 = svadd_x(p0,tmp2   , tmp4    ); tmp3 = svadd_x(p0,tmp3   , tmp5    );
	  tmp0 = svadd_x(p0,tmp0   , tmp2    ); tmp1 = svadd_x(p0,tmp1   , tmp3    );
	  
	  svst1(p0,(float64_t*)&f2_t[c  ],tmp0);
	  svst1(p0,(float64_t*)&f2_t[c+8],tmp1);
#elif UNR == 4
	  n = (y == 0)    ? 0 : - nx;
	  s = (y == ny-1) ? 0 :   nx;
	  b = b0;
	  t = t0;
	  c =  y * nx + z * nx * ny;
	  float64_t fcm1_arr[8] = {f1_t[c],f1_t[c],f1_t[c+1],f1_t[c+2],f1_t[c+3],f1_t[c+4],f1_t[c+5],f1_t[c+6]};
	  svfloat64_t fc_vec0,fc_vec1,fc_vec2,fc_vec3;
	  svfloat64_t fz20,fz21,fz22,fz23;
	  svfloat64_t fz10,fz11,fz12,fz13;
	  svfloat64_t fz30,fz31,fz32,fz33;
	  svfloat64_t fz40,fz41,fz42,fz43;
	  svfloat64_t fz50,fz51,fz52,fz53;
	  svfloat64_t fz60,fz61,fz62,fz63;
	  
	  fc_vec0  = svld1(svptrue_b64(),(float64_t*)&f1_t[c]);
	  fz20 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+1]);
	  fz10 = svld1(svptrue_b64(),(float64_t*)&fcm1_arr[0]);
	  fz30 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+s]);
	  fz40 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+n]);
	  fz50 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+b]);
	  fz60 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+t]);
	  fc_vec1  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*1]);
	  fz21 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*1+1]);
	  fz11 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*1-1]);
	  fz31 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*1+s]);
	  fz41 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*1+n]);
	  fz51 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*1+b]);
	  fz61 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*1+t]);
	  fc_vec2  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*2]);
	  fz22 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*2+1]);
	  fz12 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*2-1]);
	  fz32 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*2+s]);
	  fz42 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*2+n]);
	  fz52 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*2+b]);
	  fz62 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*2+t]);
	  fc_vec3  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*3]);
	  fz23 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*3+1]);
	  fz13 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*3-1]);
	  fz33 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*3+s]);
	  fz43 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*3+n]);
	  fz53 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*3+b]);
	  fz63 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*3+t]);
	  
	  svfloat64_t tmp0,tmp1,tmp2,tmp3;
	  tmp0 = svmul_x(svptrue_b64(),z0,fc_vec0); tmp1 = svmul_x(svptrue_b64(),z0,fc_vec1);
	  tmp2 = svmul_x(svptrue_b64(),z0,fc_vec2); tmp3 = svmul_x(svptrue_b64(),z0,fc_vec3);
	  tmp0 = svmad_x(svptrue_b64(),z1,fz10,tmp0); tmp1 = svmad_x(svptrue_b64(),z1,fz11,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),z1,fz12,tmp2); tmp3 = svmad_x(svptrue_b64(),z1,fz13,tmp3);
	  tmp0 = svmad_x(svptrue_b64(),z2,fz20,tmp0); tmp1 = svmad_x(svptrue_b64(),z2,fz21,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),z2,fz22,tmp2); tmp3 = svmad_x(svptrue_b64(),z2,fz23,tmp3);
	  tmp0 = svmad_x(svptrue_b64(),z3,fz30,tmp0); tmp1 = svmad_x(svptrue_b64(),z3,fz31,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),z3,fz32,tmp2); tmp3 = svmad_x(svptrue_b64(),z3,fz33,tmp3);
	  tmp0 = svmad_x(svptrue_b64(),z4,fz40,tmp0); tmp1 = svmad_x(svptrue_b64(),z4,fz41,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),z4,fz42,tmp2); tmp3 = svmad_x(svptrue_b64(),z4,fz43,tmp3);
	  tmp0 = svmad_x(svptrue_b64(),z5,fz50,tmp0); tmp1 = svmad_x(svptrue_b64(),z5,fz51,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),z5,fz52,tmp2); tmp3 = svmad_x(svptrue_b64(),z5,fz53,tmp3);
	  tmp0 = svmad_x(svptrue_b64(),z6,fz60,tmp0); tmp1 = svmad_x(svptrue_b64(),z6,fz61,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),z6,fz62,tmp2); tmp3 = svmad_x(svptrue_b64(),z6,fz63,tmp3);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+8*0],tmp0);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+8*1],tmp1);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+8*2],tmp2);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+8*3],tmp3);
	  
	  int xx;
	  for (xx = 32; xx < nx-32; xx+=32) {
	    
	    fc_vec0  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx]);
	    fz20 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+1]);
	    fz10 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx-1]);
	    fz30 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+s]);
	    fz40 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+n]);
	    fz50 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+b]);
	    fz60 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+t]);
	    fc_vec1  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1]);
	    fz21 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+1]);
	    fz11 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1-1]);
	    fz31 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+s]);
	    fz41 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+n]);
	    fz51 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+b]);
	    fz61 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+t]);
	    fc_vec2  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2]);
	    fz22 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+1]);
	    fz12 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2-1]);
	    fz32 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+s]);
	    fz42 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+n]);
	    fz52 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+b]);
	    fz62 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+t]);
	    fc_vec3  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3]);
	    fz23 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+1]);
	    fz13 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3-1]);
	    fz33 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+s]);
	    fz43 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+n]);
	    fz53 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+b]);
	    fz63 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+t]);
	    
	    tmp0 = svmul_x(svptrue_b64(),z0,fc_vec0); tmp1 = svmul_x(svptrue_b64(),z0,fc_vec1);
	    tmp2 = svmul_x(svptrue_b64(),z0,fc_vec2); tmp3 = svmul_x(svptrue_b64(),z0,fc_vec3);
	    tmp0 = svmad_x(svptrue_b64(),z1,fz10,tmp0); tmp1 = svmad_x(svptrue_b64(),z1,fz11,tmp1);
	    tmp2 = svmad_x(svptrue_b64(),z1,fz12,tmp2); tmp3 = svmad_x(svptrue_b64(),z1,fz13,tmp3);
	    tmp0 = svmad_x(svptrue_b64(),z2,fz20,tmp0); tmp1 = svmad_x(svptrue_b64(),z2,fz21,tmp1);
	    tmp2 = svmad_x(svptrue_b64(),z2,fz22,tmp2); tmp3 = svmad_x(svptrue_b64(),z2,fz23,tmp3);
	    tmp0 = svmad_x(svptrue_b64(),z3,fz30,tmp0); tmp1 = svmad_x(svptrue_b64(),z3,fz31,tmp1);
	    tmp2 = svmad_x(svptrue_b64(),z3,fz32,tmp2); tmp3 = svmad_x(svptrue_b64(),z3,fz33,tmp3);
	    tmp0 = svmad_x(svptrue_b64(),z4,fz40,tmp0); tmp1 = svmad_x(svptrue_b64(),z4,fz41,tmp1);
	    tmp2 = svmad_x(svptrue_b64(),z4,fz42,tmp2); tmp3 = svmad_x(svptrue_b64(),z4,fz43,tmp3);
	    tmp0 = svmad_x(svptrue_b64(),z5,fz50,tmp0); tmp1 = svmad_x(svptrue_b64(),z5,fz51,tmp1);
	    tmp2 = svmad_x(svptrue_b64(),z5,fz52,tmp2); tmp3 = svmad_x(svptrue_b64(),z5,fz53,tmp3);
	    tmp0 = svmad_x(svptrue_b64(),z6,fz60,tmp0); tmp1 = svmad_x(svptrue_b64(),z6,fz61,tmp1);
	    tmp2 = svmad_x(svptrue_b64(),z6,fz62,tmp2); tmp3 = svmad_x(svptrue_b64(),z6,fz63,tmp3);
	    svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*0],tmp0);
	    svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*1],tmp1);
	    svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*2],tmp2);
	    svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*3],tmp3);
	  }
	  float64_t fcp1_arr[8] = {f1_t[c+xx+8*3+1],f1_t[c+xx+8*3+2],f1_t[c+xx+8*3+3],f1_t[c+xx+8*3+4],f1_t[c+xx+8*3+5],f1_t[c+xx+8*3+6],f1_t[c+xx+8*3+7],f1_t[c+xx+8*3+7]};
	  fc_vec0  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx]);
	  fz20 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+1]);
	  fz10 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx-1]);
	  fz30 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+s]);
	  fz40 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+n]);
	  fz50 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+b]);
	  fz60 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+t]);
	  fc_vec1  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1]);
	  fz21 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+1]);
	  fz11 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1-1]);
	  fz31 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+s]);
	  fz41 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+n]);
	  fz51 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+b]);
	  fz61 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+t]);
	  fc_vec2  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2]);
	  fz22 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+1]);
	  fz12 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2-1]);
	  fz32 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+s]);
	  fz42 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+n]);
	  fz52 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+b]);
	  fz62 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+t]);
	  fc_vec3  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3]);
	  fz23 = svld1(svptrue_b64(),(float64_t*)&fcp1_arr[0]);
	  fz13 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3-1]);
	  fz33 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+s]);
	  fz43 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+n]);
	  fz53 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+b]);
	  fz63 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+t]);
	  tmp0 = svmul_x(svptrue_b64(),z0,fc_vec0); tmp1 = svmul_x(svptrue_b64(),z0,fc_vec1);
	  tmp2 = svmul_x(svptrue_b64(),z0,fc_vec2); tmp3 = svmul_x(svptrue_b64(),z0,fc_vec3);
	  tmp0 = svmad_x(svptrue_b64(),z1,fz10,tmp0); tmp1 = svmad_x(svptrue_b64(),z1,fz11,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),z1,fz12,tmp2); tmp3 = svmad_x(svptrue_b64(),z1,fz13,tmp3);
	  tmp0 = svmad_x(svptrue_b64(),z2,fz20,tmp0); tmp1 = svmad_x(svptrue_b64(),z2,fz21,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),z2,fz22,tmp2); tmp3 = svmad_x(svptrue_b64(),z2,fz23,tmp3);
	  tmp0 = svmad_x(svptrue_b64(),z3,fz30,tmp0); tmp1 = svmad_x(svptrue_b64(),z3,fz31,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),z3,fz32,tmp2); tmp3 = svmad_x(svptrue_b64(),z3,fz33,tmp3);
	  tmp0 = svmad_x(svptrue_b64(),z4,fz40,tmp0); tmp1 = svmad_x(svptrue_b64(),z4,fz41,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),z4,fz42,tmp2); tmp3 = svmad_x(svptrue_b64(),z4,fz43,tmp3);
	  tmp0 = svmad_x(svptrue_b64(),z5,fz50,tmp0); tmp1 = svmad_x(svptrue_b64(),z5,fz51,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),z5,fz52,tmp2); tmp3 = svmad_x(svptrue_b64(),z5,fz53,tmp3);
	  tmp0 = svmad_x(svptrue_b64(),z6,fz60,tmp0); tmp1 = svmad_x(svptrue_b64(),z6,fz61,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),z6,fz62,tmp2); tmp3 = svmad_x(svptrue_b64(),z6,fz63,tmp3);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*0],tmp0);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*1],tmp1);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*2],tmp2);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+xx+8*3],tmp3);
#elif UNR == 8

	  int32_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9;
	  int32_t x10,x11,x12,x13,x14,x15,x16,x17,x18,x19;
	  int32_t x20,x21,x22,x23,x24,x25,x26,x27,x28,x29;

	  n = (y == 0)    ? 0 : - nx;
	  s = (y == ny-1) ? 0 :   nx;
	  b = b0;
	  t = t0;
	  c =  y * nx + z * nx * ny;
	  float64_t fcm1_arr[8] = {f1_t[c],f1_t[c],f1_t[c+1],f1_t[c+2],f1_t[c+3],f1_t[c+4],f1_t[c+5],f1_t[c+6]};

	  // c

	  x0 = c + 0;
	  x1 = c + 8;
	  x2 = c + 16;
	  x3 = c + 24;
	  x4 = c + 32;
	  x5 = c + 40;
	  x6 = c + 48;
	  x7 = c + 56;
	  
	  z7 = svld1(p0,(float64_t*)&f1_t[x0]); 
	  z8 = svld1(p0,(float64_t*)&f1_t[x1]); 
	  z9 = svld1(p0,(float64_t*)&f1_t[x2]); 
	  z10= svld1(p0,(float64_t*)&f1_t[x3]); 
	  z11= svld1(p0,(float64_t*)&f1_t[x4]); 
	  z12= svld1(p0,(float64_t*)&f1_t[x5]); 
	  z13= svld1(p0,(float64_t*)&f1_t[x6]); 
	  z14= svld1(p0,(float64_t*)&f1_t[x7]); 

	  z15= svmul_x(p0,z0,z7); 
	  z16= svmul_x(p0,z0,z8); 
	  z17= svmul_x(p0,z0,z9); 
	  z18= svmul_x(p0,z0,z10); 
	  z19= svmul_x(p0,z0,z11); 
	  z20= svmul_x(p0,z0,z12); 
	  z21= svmul_x(p0,z0,z13); 
	  z22= svmul_x(p0,z0,z14);

	  // w

	  /* x10 = x0 - 1; */
	  x11 = x1 - 1;
	  x12 = x2 - 1;
	  x13 = x3 - 1;
	  x14 = x4 - 1;
	  x15 = x5 - 1;
	  x16 = x6 - 1;
	  x17 = x7 - 1;
	  
	  z7 = svld1(p0,(float64_t*)&fcm1_arr[0]); 
	  z8 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z9 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10= svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11= svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12= svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13= svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14= svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z1,z7, z15); 
	  z16= svmad_x(p0,z1,z8, z16); 
	  z17= svmad_x(p0,z1,z9, z17); 
	  z18= svmad_x(p0,z1,z10,z18); 
	  z19= svmad_x(p0,z1,z11,z19); 
	  z20= svmad_x(p0,z1,z12,z20); 
	  z21= svmad_x(p0,z1,z13,z21); 
	  z22= svmad_x(p0,z1,z14,z22);

	  // e

	  x10 = x0 + 1;
	  x11 = x1 + 1;
	  x12 = x2 + 1;
	  x13 = x3 + 1;
	  x14 = x4 + 1;
	  x15 = x5 + 1;
	  x16 = x6 + 1;
	  x17 = x7 + 1;
	  
	  z7 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z8 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z9 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10= svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11= svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12= svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13= svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14= svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z2,z7, z15); 
	  z16= svmad_x(p0,z2,z8, z16); 
	  z17= svmad_x(p0,z2,z9, z17); 
	  z18= svmad_x(p0,z2,z10,z18); 
	  z19= svmad_x(p0,z2,z11,z19); 
	  z20= svmad_x(p0,z2,z12,z20); 
	  z21= svmad_x(p0,z2,z13,z21); 
	  z22= svmad_x(p0,z2,z14,z22); 

	  // n

	  x10 = x0 + n;
	  x11 = x1 + n;
	  x12 = x2 + n;
	  x13 = x3 + n;
	  x14 = x4 + n;
	  x15 = x5 + n;
	  x16 = x6 + n;
	  x17 = x7 + n;
	  
	  z7 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z8 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z9 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10= svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11= svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12= svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13= svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14= svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z3,z7, z15); 
	  z16= svmad_x(p0,z3,z8, z16); 
	  z17= svmad_x(p0,z3,z9, z17); 
	  z18= svmad_x(p0,z3,z10,z18); 
	  z19= svmad_x(p0,z3,z11,z19); 
	  z20= svmad_x(p0,z3,z12,z20); 
	  z21= svmad_x(p0,z3,z13,z21); 
	  z22= svmad_x(p0,z3,z14,z22); 
	  
	  // s

	  x10 = x0 + s;
	  x11 = x1 + s;
	  x12 = x2 + s;
	  x13 = x3 + s;
	  x14 = x4 + s;
	  x15 = x5 + s;
	  x16 = x6 + s;
	  x17 = x7 + s;
	  
	  z7 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z8 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z9 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10= svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11= svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12= svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13= svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14= svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z4,z7, z15); 
	  z16= svmad_x(p0,z4,z8, z16); 
	  z17= svmad_x(p0,z4,z9, z17); 
	  z18= svmad_x(p0,z4,z10,z18); 
	  z19= svmad_x(p0,z4,z11,z19); 
	  z20= svmad_x(p0,z4,z12,z20); 
	  z21= svmad_x(p0,z4,z13,z21); 
	  z22= svmad_x(p0,z4,z14,z22); 

	  // b

	  x10 = x0 + b;
	  x11 = x1 + b;
	  x12 = x2 + b;
	  x13 = x3 + b;
	  x14 = x4 + b;
	  x15 = x5 + b;
	  x16 = x6 + b;
	  x17 = x7 + b;
	  
	  z7 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z8 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z9 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10= svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11= svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12= svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13= svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14= svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z5,z7, z15); 
	  z16= svmad_x(p0,z5,z8, z16); 
	  z17= svmad_x(p0,z5,z9, z17); 
	  z18= svmad_x(p0,z5,z10,z18); 
	  z19= svmad_x(p0,z5,z11,z19); 
	  z20= svmad_x(p0,z5,z12,z20); 
	  z21= svmad_x(p0,z5,z13,z21); 
	  z22= svmad_x(p0,z5,z14,z22); 

	  // t

	  x10 = x0 + t;
	  x11 = x1 + t;
	  x12 = x2 + t;
	  x13 = x3 + t;
	  x14 = x4 + t;
	  x15 = x5 + t;
	  x16 = x6 + t;
	  x17 = x7 + t;
	  
	  z7 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z8 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z9 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10= svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11= svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12= svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13= svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14= svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z6,z7, z15); 
	  z16= svmad_x(p0,z6,z8, z16); 
	  z17= svmad_x(p0,z6,z9, z17); 
	  z18= svmad_x(p0,z6,z10,z18); 
	  z19= svmad_x(p0,z6,z11,z19); 
	  z20= svmad_x(p0,z6,z12,z20); 
	  z21= svmad_x(p0,z6,z13,z21); 
	  z22= svmad_x(p0,z6,z14,z22); 
	  
	  svst1(p0,(float64_t*)&f2_t[x0],z15);
	  svst1(p0,(float64_t*)&f2_t[x1],z16);
	  svst1(p0,(float64_t*)&f2_t[x2],z17);
	  svst1(p0,(float64_t*)&f2_t[x3],z18);
	  svst1(p0,(float64_t*)&f2_t[x4],z19);
	  svst1(p0,(float64_t*)&f2_t[x5],z20);
	  svst1(p0,(float64_t*)&f2_t[x6],z21);
	  svst1(p0,(float64_t*)&f2_t[x7],z22);

	  for (x20 = 64; x20 < nx-64; x20+=64) {
	    
	    // c
	    
	    x0 = x0 + 64;
	    x1 = x1 + 64;
	    x2 = x2 + 64;
	    x3 = x3 + 64;
	    x4 = x4 + 64;
	    x5 = x5 + 64;
	    x6 = x6 + 64;
	    x7 = x7 + 64;
	  
	    z7 = svld1(p0,(float64_t*)&f1_t[x0]); 
	    z8 = svld1(p0,(float64_t*)&f1_t[x1]); 
	    z9 = svld1(p0,(float64_t*)&f1_t[x2]); 
	    z10= svld1(p0,(float64_t*)&f1_t[x3]); 
	    z11= svld1(p0,(float64_t*)&f1_t[x4]); 
	    z12= svld1(p0,(float64_t*)&f1_t[x5]); 
	    z13= svld1(p0,(float64_t*)&f1_t[x6]); 
	    z14= svld1(p0,(float64_t*)&f1_t[x7]); 
	    
	    z15= svmul_x(p0,z0,z7); 
	    z16= svmul_x(p0,z0,z8); 
	    z17= svmul_x(p0,z0,z9); 
	    z18= svmul_x(p0,z0,z10); 
	    z19= svmul_x(p0,z0,z11); 
	    z20= svmul_x(p0,z0,z12); 
	    z21= svmul_x(p0,z0,z13); 
	    z22= svmul_x(p0,z0,z14);
	    
	    // w
	    
	    x10 = x0 - 1;
	    x11 = x1 - 1;
	    x12 = x2 - 1;
	    x13 = x3 - 1;
	    x14 = x4 - 1;
	    x15 = x5 - 1;
	    x16 = x6 - 1;
	    x17 = x7 - 1;
	    
	    z7 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z8 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z9 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10= svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11= svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12= svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13= svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14= svld1(p0,(float64_t*)&f1_t[x17]); 
	    
	    z15= svmad_x(p0,z1,z7, z15); 
	    z16= svmad_x(p0,z1,z8, z16); 
	    z17= svmad_x(p0,z1,z9, z17); 
	    z18= svmad_x(p0,z1,z10,z18); 
	    z19= svmad_x(p0,z1,z11,z19); 
	    z20= svmad_x(p0,z1,z12,z20); 
	    z21= svmad_x(p0,z1,z13,z21); 
	    z22= svmad_x(p0,z1,z14,z22);
	    
	    // e
	    
	    x10 = x0 + 1;
	    x11 = x1 + 1;
	    x12 = x2 + 1;
	    x13 = x3 + 1;
	    x14 = x4 + 1;
	    x15 = x5 + 1;
	    x16 = x6 + 1;
	    x17 = x7 + 1;
	    
	    z7 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z8 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z9 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10= svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11= svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12= svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13= svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14= svld1(p0,(float64_t*)&f1_t[x17]); 
	    
	    z15= svmad_x(p0,z2,z7, z15); 
	    z16= svmad_x(p0,z2,z8, z16); 
	    z17= svmad_x(p0,z2,z9, z17); 
	    z18= svmad_x(p0,z2,z10,z18); 
	    z19= svmad_x(p0,z2,z11,z19); 
	    z20= svmad_x(p0,z2,z12,z20); 
	    z21= svmad_x(p0,z2,z13,z21); 
	    z22= svmad_x(p0,z2,z14,z22); 
	    
	    // n
	    
	    x10 = x0 + n;
	    x11 = x1 + n;
	    x12 = x2 + n;
	    x13 = x3 + n;
	    x14 = x4 + n;
	    x15 = x5 + n;
	    x16 = x6 + n;
	    x17 = x7 + n;
	    
	    z7 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z8 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z9 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10= svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11= svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12= svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13= svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14= svld1(p0,(float64_t*)&f1_t[x17]); 
	    
	    z15= svmad_x(p0,z3,z7, z15); 
	    z16= svmad_x(p0,z3,z8, z16); 
	    z17= svmad_x(p0,z3,z9, z17); 
	    z18= svmad_x(p0,z3,z10,z18); 
	    z19= svmad_x(p0,z3,z11,z19); 
	    z20= svmad_x(p0,z3,z12,z20); 
	    z21= svmad_x(p0,z3,z13,z21); 
	    z22= svmad_x(p0,z3,z14,z22);
	    
	    // s
	    
	    x10 = x0 + s;
	    x11 = x1 + s;
	    x12 = x2 + s;
	    x13 = x3 + s;
	    x14 = x4 + s;
	    x15 = x5 + s;
	    x16 = x6 + s;
	    x17 = x7 + s;
	    
	    z7 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z8 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z9 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10= svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11= svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12= svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13= svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14= svld1(p0,(float64_t*)&f1_t[x17]); 
	    
	    z15= svmad_x(p0,z4,z7, z15); 
	    z16= svmad_x(p0,z4,z8, z16); 
	    z17= svmad_x(p0,z4,z9, z17); 
	    z18= svmad_x(p0,z4,z10,z18); 
	    z19= svmad_x(p0,z4,z11,z19); 
	    z20= svmad_x(p0,z4,z12,z20); 
	    z21= svmad_x(p0,z4,z13,z21); 
	    z22= svmad_x(p0,z4,z14,z22); 
	    
	    // b
	    
	    x10 = x0 + b;
	    x11 = x1 + b;
	    x12 = x2 + b;
	    x13 = x3 + b;
	    x14 = x4 + b;
	    x15 = x5 + b;
	    x16 = x6 + b;
	    x17 = x7 + b;
	    
	    z7 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z8 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z9 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10= svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11= svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12= svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13= svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14= svld1(p0,(float64_t*)&f1_t[x17]); 
	    
	    z15= svmad_x(p0,z5,z7, z15); 
	    z16= svmad_x(p0,z5,z8, z16); 
	    z17= svmad_x(p0,z5,z9, z17); 
	    z18= svmad_x(p0,z5,z10,z18); 
	    z19= svmad_x(p0,z5,z11,z19); 
	    z20= svmad_x(p0,z5,z12,z20); 
	    z21= svmad_x(p0,z5,z13,z21); 
	    z22= svmad_x(p0,z5,z14,z22); 
	    
	    // t
	    
	    x10 = x0 + t;
	    x11 = x1 + t;
	    x12 = x2 + t;
	    x13 = x3 + t;
	    x14 = x4 + t;
	    x15 = x5 + t;
	    x16 = x6 + t;
	    x17 = x7 + t;
	    
	    z7 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z8 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z9 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10= svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11= svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12= svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13= svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14= svld1(p0,(float64_t*)&f1_t[x17]); 
	    
	    z15= svmad_x(p0,z6,z7, z15); 
	    z16= svmad_x(p0,z6,z8, z16); 
	    z17= svmad_x(p0,z6,z9, z17); 
	    z18= svmad_x(p0,z6,z10,z18); 
	    z19= svmad_x(p0,z6,z11,z19); 
	    z20= svmad_x(p0,z6,z12,z20); 
	    z21= svmad_x(p0,z6,z13,z21); 
	    z22= svmad_x(p0,z6,z14,z22); 
	    
	    svst1(p0,(float64_t*)&f2_t[x0],z15);
	    svst1(p0,(float64_t*)&f2_t[x1],z16);
	    svst1(p0,(float64_t*)&f2_t[x2],z17);
	    svst1(p0,(float64_t*)&f2_t[x3],z18);
	    svst1(p0,(float64_t*)&f2_t[x4],z19);
	    svst1(p0,(float64_t*)&f2_t[x5],z20);
	    svst1(p0,(float64_t*)&f2_t[x6],z21);
	    svst1(p0,(float64_t*)&f2_t[x7],z22);
	  }
	  

	  // c
	  
	  x0 = x0 + 64;
	  x1 = x1 + 64;
	  x2 = x2 + 64;
	  x3 = x3 + 64;
	  x4 = x4 + 64;
	  x5 = x5 + 64;
	  x6 = x6 + 64;
	  x7 = x7 + 64;
	  
	  z7 = svld1(p0,(float64_t*)&f1_t[x0]); 
	  z8 = svld1(p0,(float64_t*)&f1_t[x1]); 
	  z9 = svld1(p0,(float64_t*)&f1_t[x2]); 
	  z10= svld1(p0,(float64_t*)&f1_t[x3]); 
	  z11= svld1(p0,(float64_t*)&f1_t[x4]); 
	  z12= svld1(p0,(float64_t*)&f1_t[x5]); 
	  z13= svld1(p0,(float64_t*)&f1_t[x6]); 
	  z14= svld1(p0,(float64_t*)&f1_t[x7]); 
	  
	  z15= svmul_x(p0,z0,z7); 
	  z16= svmul_x(p0,z0,z8); 
	  z17= svmul_x(p0,z0,z9); 
	  z18= svmul_x(p0,z0,z10); 
	  z19= svmul_x(p0,z0,z11); 
	  z20= svmul_x(p0,z0,z12); 
	  z21= svmul_x(p0,z0,z13); 
	  z22= svmul_x(p0,z0,z14);
	  
	  // w
	  
	  x10 = x0 - 1;
	  x11 = x1 - 1;
	  x12 = x2 - 1;
	  x13 = x3 - 1;
	  x14 = x4 - 1;
	  x15 = x5 - 1;
	  x16 = x6 - 1;
	  x17 = x7 - 1;
	  
	  z7 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z8 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z9 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10= svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11= svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12= svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13= svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14= svld1(p0,(float64_t*)&f1_t[x17]); 
	  
	  z15= svmad_x(p0,z1,z7, z15); 
	  z16= svmad_x(p0,z1,z8, z16); 
	  z17= svmad_x(p0,z1,z9, z17); 
	  z18= svmad_x(p0,z1,z10,z18); 
	  z19= svmad_x(p0,z1,z11,z19); 
	  z20= svmad_x(p0,z1,z12,z20); 
	  z21= svmad_x(p0,z1,z13,z21); 
	  z22= svmad_x(p0,z1,z14,z22);
	  
	  // e
	  
	  x10 = x0 + 1;
	  x11 = x1 + 1;
	  x12 = x2 + 1;
	  x13 = x3 + 1;
	  x14 = x4 + 1;
	  x15 = x5 + 1;
	  x16 = x6 + 1;
	  /* x17 = x7 + 1; */

	  float64_t fcp1_arr[8] = {f1_t[x7+1],f1_t[x7+2],f1_t[x7+3],f1_t[x7+4],f1_t[x7+5],f1_t[x7+6],f1_t[x7+7],f1_t[x7+7]};
	  
	  z7 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z8 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z9 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10= svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11= svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12= svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13= svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14= svld1(p0,(float64_t*)&fcp1_arr[0]); 
	  
	  z15= svmad_x(p0,z2,z7, z15); 
	  z16= svmad_x(p0,z2,z8, z16); 
	  z17= svmad_x(p0,z2,z9, z17); 
	  z18= svmad_x(p0,z2,z10,z18); 
	  z19= svmad_x(p0,z2,z11,z19); 
	  z20= svmad_x(p0,z2,z12,z20); 
	  z21= svmad_x(p0,z2,z13,z21); 
	  z22= svmad_x(p0,z2,z14,z22); 
	  
	  // n
	  
	  x10 = x0 + n;
	  x11 = x1 + n;
	  x12 = x2 + n;
	  x13 = x3 + n;
	  x14 = x4 + n;
	  x15 = x5 + n;
	  x16 = x6 + n;
	  x17 = x7 + n;
	  
	  z7 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z8 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z9 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10= svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11= svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12= svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13= svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14= svld1(p0,(float64_t*)&f1_t[x17]); 
	  
	  z15= svmad_x(p0,z3,z7, z15); 
	  z16= svmad_x(p0,z3,z8, z16); 
	  z17= svmad_x(p0,z3,z9, z17); 
	  z18= svmad_x(p0,z3,z10,z18); 
	  z19= svmad_x(p0,z3,z11,z19); 
	  z20= svmad_x(p0,z3,z12,z20); 
	  z21= svmad_x(p0,z3,z13,z21); 
	  z22= svmad_x(p0,z3,z14,z22); 
	  
	  // s
	  
	  x10 = x0 + s;
	  x11 = x1 + s;
	  x12 = x2 + s;
	  x13 = x3 + s;
	  x14 = x4 + s;
	  x15 = x5 + s;
	  x16 = x6 + s;
	  x17 = x7 + s;
	  
	  z7 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z8 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z9 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10= svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11= svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12= svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13= svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14= svld1(p0,(float64_t*)&f1_t[x17]); 
	  
	  z15= svmad_x(p0,z4,z7, z15); 
	  z16= svmad_x(p0,z4,z8, z16); 
	  z17= svmad_x(p0,z4,z9, z17); 
	  z18= svmad_x(p0,z4,z10,z18); 
	  z19= svmad_x(p0,z4,z11,z19); 
	  z20= svmad_x(p0,z4,z12,z20); 
	  z21= svmad_x(p0,z4,z13,z21); 
	  z22= svmad_x(p0,z4,z14,z22); 
	  
	  // b
	  
	  x10 = x0 + b;
	  x11 = x1 + b;
	  x12 = x2 + b;
	  x13 = x3 + b;
	  x14 = x4 + b;
	  x15 = x5 + b;
	  x16 = x6 + b;
	  x17 = x7 + b;
	  
	  z7 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z8 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z9 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10= svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11= svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12= svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13= svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14= svld1(p0,(float64_t*)&f1_t[x17]); 
	  
	  z15= svmad_x(p0,z5,z7, z15); 
	  z16= svmad_x(p0,z5,z8, z16); 
	  z17= svmad_x(p0,z5,z9, z17); 
	  z18= svmad_x(p0,z5,z10,z18); 
	  z19= svmad_x(p0,z5,z11,z19); 
	  z20= svmad_x(p0,z5,z12,z20); 
	  z21= svmad_x(p0,z5,z13,z21); 
	  z22= svmad_x(p0,z5,z14,z22); 
	  
	  // t
	  
	  x10 = x0 + t;
	  x11 = x1 + t;
	  x12 = x2 + t;
	  x13 = x3 + t;
	  x14 = x4 + t;
	  x15 = x5 + t;
	  x16 = x6 + t;
	  x17 = x7 + t;
	  
	  z7 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z8 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z9 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10= svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11= svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12= svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13= svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14= svld1(p0,(float64_t*)&f1_t[x17]); 
	  
	  z15= svmad_x(p0,z6,z7, z15); 
	  z16= svmad_x(p0,z6,z8, z16); 
	  z17= svmad_x(p0,z6,z9, z17); 
	  z18= svmad_x(p0,z6,z10,z18); 
	  z19= svmad_x(p0,z6,z11,z19); 
	  z20= svmad_x(p0,z6,z12,z20); 
	  z21= svmad_x(p0,z6,z13,z21); 
	  z22= svmad_x(p0,z6,z14,z22); 
	  
	  svst1(p0,(float64_t*)&f2_t[x0],z15);
	  svst1(p0,(float64_t*)&f2_t[x1],z16);
	  svst1(p0,(float64_t*)&f2_t[x2],z17);
	  svst1(p0,(float64_t*)&f2_t[x3],z18);
	  svst1(p0,(float64_t*)&f2_t[x4],z19);
	  svst1(p0,(float64_t*)&f2_t[x5],z20);
	  svst1(p0,(float64_t*)&f2_t[x6],z21);
	  svst1(p0,(float64_t*)&f2_t[x7],z22);
	  
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
  

