#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#ifdef SVE
#include <arm_sve.h>
#endif /* SVE */
#include "diffusion_ker40.h"

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

void allocate_ker40(REAL **buff_ret, const int nx, const int ny, const int nz) {

  posix_memalign((void**)buff_ret, 64, sizeof(REAL)*nx*ny*nz);

}

void init_ker40(REAL *buff1, const int nx, const int ny, const int nz,
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


void diffusion_ker40(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
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

    svfloat64_t z00,z01,z02,z03,z04,z05,z06,z07,z08,z09;
    svfloat64_t z10,z11,z12,z13,z14,z15,z16,z17,z18,z19;
    svfloat64_t z20,z21,z22,z23,z24,z25,z26,z27,z28,z29;
    //    svfloat64_t z30,z31;
    svint32_t z30,z31;

    int32_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9;
    int32_t x10,x11,x12,x13,x14,x15,x16,x17,x18,x19;
    int32_t x20,x21,x22,x23,x24,x25,x26,x27,x28,x29;
    int32_t x30,x31;
    
    z00 = svdup_f64(cc); // cc_vec
    z01 = svdup_f64(cw); // cw_vec
    z02 = svdup_f64(ce); // ce_vec
    z03 = svdup_f64(cn); // cn_vec
    z04 = svdup_f64(cs); // cs_vec
    z05 = svdup_f64(cb); // cb_vec
    z06 = svdup_f64(ct); // ct_vec
    const svbool_t p0 = svptrue_b64();
    const svbool_t p1 = svptrue_b32();
    const int32_t ptmp1[16] = {1};
    const int32_t ptmp2[16] = {0,1};
    const int32_t ptmp3[16] = {0,0,1};
    const int32_t ptmp4[16] = {0,0,0,1};
    const int32_t ptmp5[16] = {0,0,0,0,1};
    const int32_t ptmp6[16] = {0,0,0,0,0,1};
    const int32_t ptmp7[16] = {0,0,0,0,0,0,1};
    const int32_t ptmp8[16] = {0,0,0,0,0,0,0,1};
    x30 = 1;
    z30 = svld1(p1,(int32_t*)&ptmp1[0]);
    const svbool_t p1000000000000000 = svcmpeq(p1,z30,x30);
    z30 = svld1(p1,(int32_t*)&ptmp2[0]);
    const svbool_t p0100000000000000 = svcmpeq(p1,z30,x30);
    z30 = svld1(p1,(int32_t*)&ptmp3[0]);
    const svbool_t p0010000000000000 = svcmpeq(p1,z30,x30);
    z30 = svld1(p1,(int32_t*)&ptmp4[0]);
    const svbool_t p0001000000000000 = svcmpeq(p1,z30,x30);
    z30 = svld1(p1,(int32_t*)&ptmp5[0]);
    const svbool_t p0000100000000000 = svcmpeq(p1,z30,x30);
    z30 = svld1(p1,(int32_t*)&ptmp6[0]);
    const svbool_t p0000010000000000 = svcmpeq(p1,z30,x30);
    z30 = svld1(p1,(int32_t*)&ptmp7[0]);
    const svbool_t p0000001000000000 = svcmpeq(p1,z30,x30);
    z30 = svld1(p1,(int32_t*)&ptmp8[0]);
    const svbool_t p0000000100000000 = svcmpeq(p1,z30,x30);

    /* x0 = 0; */
    /* x1 = 8; */
    /* x2 = 16; */
    /* x3 = 24; */
    /* x4 = 32; */
    /* x5 = 40; */
    /* x6 = 48; */
    /* x7 = 56; */
    /* int32_t index_tmp[16] = {x0,x1,x2,x3,x4,x5,x6,x7}; */
    /* z30 = svld1(p1,(int32_t*)&index_tmp[0]); */
    
    /* x30 = -1; */
    /* z31 = svadd_x(p1,z30,x30); */
    /* x10 = svaddv(p1000000000000000,z31); */
    /* x11 = svaddv(p0100000000000000,z31); */
    /* x12 = svaddv(p0010000000000000,z31); */
    /* x13 = svaddv(p0001000000000000,z31); */
    /* x14 = svaddv(p0000100000000000,z31); */
    /* x15 = svaddv(p0000010000000000,z31); */
    /* x16 = svaddv(p0000001000000000,z31); */
    /* x17 = svaddv(p0000000100000000,z31); */

    /* printf("test index: %d,%d,%d,%d,%d,%d,%d,%d\n",x10,x11,x12,x13,x14,x15,x16,x17); */

    
    do {
      for (z = zstr; z < zend; z++) {
	b0 = (z == 0)    ? 0 : - nx * ny;
	t0 = (z == nz-1) ? 0 :   nx * ny;
	for (y = ystr; y < yend; y++) {
#if UNR == 2 
	  n = (y == 0)    ? 0 : - nx;
	  s = (y == ny-1) ? 0 :   nx;
	  c =  y * nx + z * nx * ny;

	  w = c - 1;
	  e = c + 1;
	  n = c + n;
	  s = c + s;
	  b = c + b0;
	  t = c + t0;
	  
	  z07 = svld1(p0,(float64_t*)&f1_t[c]); z08 = svld1(p0,(float64_t*)&f1_t[c+8]);
	  z09 = svmul_x(p0,z00,z07);            z10 = svmul_x(p0,z00,z08);
	  z07 = svld1(p0,(float64_t*)&f1_t[e]); z08 = svld1(p0,(float64_t*)&f1_t[e+8]);
	  z09 = svmad_x(p0,z01,z07,z09);        z10 = svmad_x(p0,z01,z08,z10);
	  float64_t tmp1[8] = {f1_t[c],f1_t[c],f1_t[c+1],f1_t[c+2],f1_t[c+3],f1_t[c+4],f1_t[c+5],f1_t[c+6]};
	  z07 = svld1(p0,(float64_t*)&tmp1[0]); z08 = svld1(p0,(float64_t*)&f1_t[w+8]);
	  z09 = svmad_x(p0,z02,z07,z09);        z10 = svmad_x(p0,z02,z08,z10);
	  z07 = svld1(p0,(float64_t*)&f1_t[s]); z08 = svld1(p0,(float64_t*)&f1_t[s+8]);
	  z09 = svmad_x(p0,z03,z07,z09);        z10 = svmad_x(p0,z03,z08,z10);
	  z07 = svld1(p0,(float64_t*)&f1_t[n]); z08 = svld1(p0,(float64_t*)&f1_t[n+8]);
	  z09 = svmad_x(p0,z04,z07,z09);        z10 = svmad_x(p0,z04,z08,z10);
	  z07 = svld1(p0,(float64_t*)&f1_t[b]); z08 = svld1(p0,(float64_t*)&f1_t[b+8]);
	  z09 = svmad_x(p0,z05,z07,z09);        z10 = svmad_x(p0,z05,z08,z10);
	  z07 = svld1(p0,(float64_t*)&f1_t[t]); z08 = svld1(p0,(float64_t*)&f1_t[t+8]);
	  z09 = svmad_x(p0,z06,z07,z09);        z10 = svmad_x(p0,z06,z08,z10);

	  svst1(p0,(float64_t*)&f2_t[c  ],z09);
	  svst1(p0,(float64_t*)&f2_t[c+8],z10);
	  
	  c += 16;
	  w += 16;
	  e += 16;
	  n += 16;
	  s += 16;
	  b += 16;
	  t += 16;

	  for (x = 16; x < nx-16; x+=16) {
	    
	    z07 = svld1(p0,(float64_t*)&f1_t[c]); z08 = svld1(p0,(float64_t*)&f1_t[c+8]);
	    z09 = svmul_x(p0,z00,z07);            z10 = svmul_x(p0,z00,z08);
	    z07 = svld1(p0,(float64_t*)&f1_t[e]); z08 = svld1(p0,(float64_t*)&f1_t[e+8]);
	    z09 = svmad_x(p0,z01,z07,z09);        z10 = svmad_x(p0,z01,z08,z10);
	    z07 = svld1(p0,(float64_t*)&f1_t[w]); z08 = svld1(p0,(float64_t*)&f1_t[w+8]);
	    z09 = svmad_x(p0,z02,z07,z09);        z10 = svmad_x(p0,z02,z08,z10);
	    z07 = svld1(p0,(float64_t*)&f1_t[s]); z08 = svld1(p0,(float64_t*)&f1_t[s+8]);
	    z09 = svmad_x(p0,z03,z07,z09);        z10 = svmad_x(p0,z03,z08,z10);
	    z07 = svld1(p0,(float64_t*)&f1_t[n]); z08 = svld1(p0,(float64_t*)&f1_t[n+8]);
	    z09 = svmad_x(p0,z04,z07,z09);        z10 = svmad_x(p0,z04,z08,z10);
	    z07 = svld1(p0,(float64_t*)&f1_t[b]); z08 = svld1(p0,(float64_t*)&f1_t[b+8]);
	    z09 = svmad_x(p0,z05,z07,z09);        z10 = svmad_x(p0,z05,z08,z10);
	    z07 = svld1(p0,(float64_t*)&f1_t[t]); z08 = svld1(p0,(float64_t*)&f1_t[t+8]);
	    z09 = svmad_x(p0,z06,z07,z09);        z10 = svmad_x(p0,z06,z08,z10);
	    
	    svst1(p0,(float64_t*)&f2_t[c  ],z09);
	    svst1(p0,(float64_t*)&f2_t[c+8],z10);
	  
	    c += 16;
	    w += 16;
	    e += 16;
	    n += 16;
	    s += 16;
	    b += 16;
	    t += 16;
	  }
	  z07 = svld1(p0,(float64_t*)&f1_t[c]); z08 = svld1(p0,(float64_t*)&f1_t[c+8]);
	  z09 = svmul_x(p0,z00,z07);            z10 = svmul_x(p0,z00,z08);
	  float64_t tmp2[8] = {f1_t[c+1+8],f1_t[c+2+8],f1_t[c+3+8],f1_t[c+4+8],f1_t[c+5+8],f1_t[c+6+8],f1_t[c+7+8],f1_t[c+7+8]};
	  z07 = svld1(p0,(float64_t*)&f1_t[e]); z08 = svld1(p0,(float64_t*)&tmp2[0]);
	  z09 = svmad_x(p0,z01,z07,z09);        z10 = svmad_x(p0,z01,z08,z10);
	  z07 = svld1(p0,(float64_t*)&f1_t[w]); z08 = svld1(p0,(float64_t*)&f1_t[w+8]);
	  z09 = svmad_x(p0,z02,z07,z09);        z10 = svmad_x(p0,z02,z08,z10);
	  z07 = svld1(p0,(float64_t*)&f1_t[s]); z08 = svld1(p0,(float64_t*)&f1_t[s+8]);
	  z09 = svmad_x(p0,z03,z07,z09);        z10 = svmad_x(p0,z03,z08,z10);
	  z07 = svld1(p0,(float64_t*)&f1_t[n]); z08 = svld1(p0,(float64_t*)&f1_t[n+8]);
	  z09 = svmad_x(p0,z04,z07,z09);        z10 = svmad_x(p0,z04,z08,z10);
	  z07 = svld1(p0,(float64_t*)&f1_t[b]); z08 = svld1(p0,(float64_t*)&f1_t[b+8]);
	  z09 = svmad_x(p0,z05,z07,z09);        z10 = svmad_x(p0,z05,z08,z10);
	  z07 = svld1(p0,(float64_t*)&f1_t[t]); z08 = svld1(p0,(float64_t*)&f1_t[t+8]);
	  z09 = svmad_x(p0,z06,z07,z09);        z10 = svmad_x(p0,z06,z08,z10);
	  
	  svst1(p0,(float64_t*)&f2_t[c  ],z09);
	  svst1(p0,(float64_t*)&f2_t[c+8],z10);
	  
#elif UNR == 4
	  n = (y == 0)    ? 0 : - nx;
	  s = (y == ny-1) ? 0 :   nx;
	  b = b0;
	  t = t0;
	  c =  y * nx + z * nx * ny;
	  float64_t fcm1_arr[8] = {f1_t[c],f1_t[c],f1_t[c+1],f1_t[c+2],f1_t[c+3],f1_t[c+4],f1_t[c+5],f1_t[c+6]};
	  svfloat64_t fc_vec0,fc_vec1,fc_vec2,fc_vec3;
	  svfloat64_t fz20,fz21,fz22,fz23;
	  svfloat64_t fz010,fz011,fz012,fz013;
	  svfloat64_t fz30,fz31,fz32,fz33;
	  svfloat64_t fz40,fz41,fz42,fz43;
	  svfloat64_t fz50,fz51,fz52,fz53;
	  svfloat64_t fz60,fz61,fz62,fz63;
	  
	  fc_vec0  = svld1(svptrue_b64(),(float64_t*)&f1_t[c]);
	  fz20 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+1]);
	  fz010 = svld1(svptrue_b64(),(float64_t*)&fcm1_arr[0]);
	  fz30 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+s]);
	  fz40 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+n]);
	  fz50 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+b]);
	  fz60 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+t]);
	  fc_vec1  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*1]);
	  fz21 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*1+1]);
	  fz011 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*1-1]);
	  fz31 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*1+s]);
	  fz41 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*1+n]);
	  fz51 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*1+b]);
	  fz61 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*1+t]);
	  fc_vec2  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*2]);
	  fz22 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*2+1]);
	  fz012 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*2-1]);
	  fz32 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*2+s]);
	  fz42 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*2+n]);
	  fz52 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*2+b]);
	  fz62 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*2+t]);
	  fc_vec3  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*3]);
	  fz23 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*3+1]);
	  fz013 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*3-1]);
	  fz33 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*3+s]);
	  fz43 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*3+n]);
	  fz53 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*3+b]);
	  fz63 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+8*3+t]);
	  
	  svfloat64_t tmp0,tmp1,tmp2,tmp3;
	  tmp0 = svmul_x(svptrue_b64(),z00,fc_vec0); tmp1 = svmul_x(svptrue_b64(),z00,fc_vec1);
	  tmp2 = svmul_x(svptrue_b64(),z00,fc_vec2); tmp3 = svmul_x(svptrue_b64(),z00,fc_vec3);
	  tmp0 = svmad_x(svptrue_b64(),z01,fz010,tmp0); tmp1 = svmad_x(svptrue_b64(),z01,fz011,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),z01,fz012,tmp2); tmp3 = svmad_x(svptrue_b64(),z01,fz013,tmp3);
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
	    fz010 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx-1]);
	    fz30 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+s]);
	    fz40 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+n]);
	    fz50 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+b]);
	    fz60 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+t]);
	    fc_vec1  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1]);
	    fz21 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+1]);
	    fz011 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1-1]);
	    fz31 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+s]);
	    fz41 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+n]);
	    fz51 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+b]);
	    fz61 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+t]);
	    fc_vec2  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2]);
	    fz22 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+1]);
	    fz012 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2-1]);
	    fz32 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+s]);
	    fz42 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+n]);
	    fz52 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+b]);
	    fz62 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+t]);
	    fc_vec3  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3]);
	    fz23 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+1]);
	    fz013 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3-1]);
	    fz33 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+s]);
	    fz43 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+n]);
	    fz53 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+b]);
	    fz63 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+t]);
	    
	    tmp0 = svmul_x(svptrue_b64(),z00,fc_vec0); tmp1 = svmul_x(svptrue_b64(),z00,fc_vec1);
	    tmp2 = svmul_x(svptrue_b64(),z00,fc_vec2); tmp3 = svmul_x(svptrue_b64(),z00,fc_vec3);
	    tmp0 = svmad_x(svptrue_b64(),z01,fz010,tmp0); tmp1 = svmad_x(svptrue_b64(),z01,fz011,tmp1);
	    tmp2 = svmad_x(svptrue_b64(),z01,fz012,tmp2); tmp3 = svmad_x(svptrue_b64(),z01,fz013,tmp3);
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
	  fz010 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx-1]);
	  fz30 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+s]);
	  fz40 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+n]);
	  fz50 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+b]);
	  fz60 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+t]);
	  fc_vec1  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1]);
	  fz21 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+1]);
	  fz011 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1-1]);
	  fz31 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+s]);
	  fz41 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+n]);
	  fz51 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+b]);
	  fz61 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*1+t]);
	  fc_vec2  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2]);
	  fz22 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+1]);
	  fz012 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2-1]);
	  fz32 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+s]);
	  fz42 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+n]);
	  fz52 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+b]);
	  fz62 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*2+t]);
	  fc_vec3  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3]);
	  fz23 = svld1(svptrue_b64(),(float64_t*)&fcp1_arr[0]);
	  fz013 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3-1]);
	  fz33 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+s]);
	  fz43 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+n]);
	  fz53 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+b]);
	  fz63 = svld1(svptrue_b64(),(float64_t*)&f1_t[c+xx+8*3+t]);
	  tmp0 = svmul_x(svptrue_b64(),z00,fc_vec0); tmp1 = svmul_x(svptrue_b64(),z00,fc_vec1);
	  tmp2 = svmul_x(svptrue_b64(),z00,fc_vec2); tmp3 = svmul_x(svptrue_b64(),z00,fc_vec3);
	  tmp0 = svmad_x(svptrue_b64(),z01,fz010,tmp0); tmp1 = svmad_x(svptrue_b64(),z01,fz011,tmp1);
	  tmp2 = svmad_x(svptrue_b64(),z01,fz012,tmp2); tmp3 = svmad_x(svptrue_b64(),z01,fz013,tmp3);
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


	  x29 = 64;

	  n = (y == 0)    ? 0 : - nx;
	  s = (y == ny-1) ? 0 :   nx;
	  b = b0;
	  t = t0;
	  c =  y * nx + z * nx * ny;
	  float64_t fcm1_arr[8] = {f1_t[c],f1_t[c],f1_t[c+1],f1_t[c+2],f1_t[c+3],f1_t[c+4],f1_t[c+5],f1_t[c+6]};

	  // c

#if 1

	  x0 = c + 0;
	  x1 = c + 8;
	  x2 = c + 16;
	  x3 = c + 24;
	  x4 = c + 32;
	  x5 = c + 40;
	  x6 = c + 48;
	  x7 = c + 56;
	  
	  z07 = svld1(p0,(float64_t*)&f1_t[x0]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x1]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x2]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x3]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x4]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x5]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x6]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x7]); 

	  z15= svmul_x(p0,z00,z07); 
	  z16= svmul_x(p0,z00,z08); 
	  z17= svmul_x(p0,z00,z09); 
	  z18= svmul_x(p0,z00,z10); 
	  z19= svmul_x(p0,z00,z11); 
	  z20= svmul_x(p0,z00,z12); 
	  z21= svmul_x(p0,z00,z13); 
	  z22= svmul_x(p0,z00,z14);

#if 0
	  // w

	  /* x10 = x0 - 1; */
	  x11 = x1 - 1;
	  x12 = x2 - 1;
	  x13 = x3 - 1;
	  x14 = x4 - 1;
	  x15 = x5 - 1;
	  x16 = x6 - 1;
	  x17 = x7 - 1;
	  
	  z07 = svld1(p0,(float64_t*)&fcm1_arr[0]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z01,z07,z15); 
	  z16= svmad_x(p0,z01,z08,z16); 
	  z17= svmad_x(p0,z01,z09,z17); 
	  z18= svmad_x(p0,z01,z10,z18); 
	  z19= svmad_x(p0,z01,z11,z19); 
	  z20= svmad_x(p0,z01,z12,z20); 
	  z21= svmad_x(p0,z01,z13,z21); 
	  z22= svmad_x(p0,z01,z14,z22);

	  // e

	  x10 = x0 + 1;
	  x11 = x1 + 1;
	  x12 = x2 + 1;
	  x13 = x3 + 1;
	  x14 = x4 + 1;
	  x15 = x5 + 1;
	  x16 = x6 + 1;
	  x17 = x7 + 1;
	  
	  z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z02,z07,z15); 
	  z16= svmad_x(p0,z02,z08,z16); 
	  z17= svmad_x(p0,z02,z09,z17); 
	  z18= svmad_x(p0,z02,z10,z18); 
	  z19= svmad_x(p0,z02,z11,z19); 
	  z20= svmad_x(p0,z02,z12,z20); 
	  z21= svmad_x(p0,z02,z13,z21); 
	  z22= svmad_x(p0,z02,z14,z22);

	  // n

	  x10 = x0 + n;
	  x11 = x1 + n;
	  x12 = x2 + n;
	  x13 = x3 + n;
	  x14 = x4 + n;
	  x15 = x5 + n;
	  x16 = x6 + n;
	  x17 = x7 + n;
	  
	  z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z03,z07,z15); 
	  z16= svmad_x(p0,z03,z08,z16); 
	  z17= svmad_x(p0,z03,z09,z17); 
	  z18= svmad_x(p0,z03,z10,z18); 
	  z19= svmad_x(p0,z03,z11,z19); 
	  z20= svmad_x(p0,z03,z12,z20); 
	  z21= svmad_x(p0,z03,z13,z21); 
	  z22= svmad_x(p0,z03,z14,z22);
	  
	  // s

	  x10 = x0 + s;
	  x11 = x1 + s;
	  x12 = x2 + s;
	  x13 = x3 + s;
	  x14 = x4 + s;
	  x15 = x5 + s;
	  x16 = x6 + s;
	  x17 = x7 + s;
	  
	  z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z04,z07,z15); 
	  z16= svmad_x(p0,z04,z08,z16); 
	  z17= svmad_x(p0,z04,z09,z17); 
	  z18= svmad_x(p0,z04,z10,z18); 
	  z19= svmad_x(p0,z04,z11,z19); 
	  z20= svmad_x(p0,z04,z12,z20); 
	  z21= svmad_x(p0,z04,z13,z21); 
	  z22= svmad_x(p0,z04,z14,z22);

	  // b

	  x10 = x0 + b;
	  x11 = x1 + b;
	  x12 = x2 + b;
	  x13 = x3 + b;
	  x14 = x4 + b;
	  x15 = x5 + b;
	  x16 = x6 + b;
	  x17 = x7 + b;
	  
	  z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z05,z07,z15); 
	  z16= svmad_x(p0,z05,z08,z16); 
	  z17= svmad_x(p0,z05,z09,z17); 
	  z18= svmad_x(p0,z05,z10,z18); 
	  z19= svmad_x(p0,z05,z11,z19); 
	  z20= svmad_x(p0,z05,z12,z20); 
	  z21= svmad_x(p0,z05,z13,z21); 
	  z22= svmad_x(p0,z05,z14,z22);

	  // t

	  x10 = x0 + t;
	  x11 = x1 + t;
	  x12 = x2 + t;
	  x13 = x3 + t;
	  x14 = x4 + t;
	  x15 = x5 + t;
	  x16 = x6 + t;
	  x17 = x7 + t;
	  
	  z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z06,z07,z15); 
	  z16= svmad_x(p0,z06,z08,z16); 
	  z17= svmad_x(p0,z06,z09,z17); 
	  z18= svmad_x(p0,z06,z10,z18); 
	  z19= svmad_x(p0,z06,z11,z19); 
	  z20= svmad_x(p0,z06,z12,z20); 
	  z21= svmad_x(p0,z06,z13,z21); 
	  z22= svmad_x(p0,z06,z14,z22);
	  
	  svst1(p0,(float64_t*)&f2_t[x0],z15);
	  svst1(p0,(float64_t*)&f2_t[x1],z16);
	  svst1(p0,(float64_t*)&f2_t[x2],z17);
	  svst1(p0,(float64_t*)&f2_t[x3],z18);
	  svst1(p0,(float64_t*)&f2_t[x4],z19);
	  svst1(p0,(float64_t*)&f2_t[x5],z20);
	  svst1(p0,(float64_t*)&f2_t[x6],z21);
	  svst1(p0,(float64_t*)&f2_t[x7],z22);
#else
	  x20 = s;
	  // w
	  x31 = (x20 >> 20) - 1;
	  x10 = x0 + x31;
	  x11 = x1 + x31;
	  x12 = x2 + x31;
	  x13 = x3 + x31;
	  x14 = x4 + x31;
	  x15 = x5 + x31;
	  x16 = x6 + x31;
	  x17 = x7 + x31;
	  
	  z07 = svld1(p0,(float64_t*)&fcm1_arr[0]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z01,z07,z15); 
	  z16= svmad_x(p0,z01,z08,z16); 
	  z17= svmad_x(p0,z01,z09,z17); 
	  z18= svmad_x(p0,z01,z10,z18); 
	  z19= svmad_x(p0,z01,z11,z19); 
	  z20= svmad_x(p0,z01,z12,z20); 
	  z21= svmad_x(p0,z01,z13,z21); 
	  z22= svmad_x(p0,z01,z14,z22);

	  // e

	  x31 = (x20 >> 21) + 1;
	  x10 = x0 + x31;
	  x11 = x1 + x31;
	  x12 = x2 + x31;
	  x13 = x3 + x31;
	  x14 = x4 + x31;
	  x15 = x5 + x31;
	  x16 = x6 + x31;
	  x17 = x7 + x31;
	    
	  z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z02,z07,z15); 
	  z16= svmad_x(p0,z02,z08,z16); 
	  z17= svmad_x(p0,z02,z09,z17); 
	  z18= svmad_x(p0,z02,z10,z18); 
	  z19= svmad_x(p0,z02,z11,z19); 
	  z20= svmad_x(p0,z02,z12,z20); 
	  z21= svmad_x(p0,z02,z13,z21); 
	  z22= svmad_x(p0,z02,z14,z22);

	  // n

	  x31 = (x20 >> 22) + n;
	  x10 = x0 + x31;
	  x11 = x1 + x31;
	  x12 = x2 + x31;
	  x13 = x3 + x31;
	  x14 = x4 + x31;
	  x15 = x5 + x31;
	  x16 = x6 + x31;
	  x17 = x7 + x31;
	  
	  z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z03,z07,z15); 
	  z16= svmad_x(p0,z03,z08,z16); 
	  z17= svmad_x(p0,z03,z09,z17); 
	  z18= svmad_x(p0,z03,z10,z18); 
	  z19= svmad_x(p0,z03,z11,z19); 
	  z20= svmad_x(p0,z03,z12,z20); 
	  z21= svmad_x(p0,z03,z13,z21); 
	  z22= svmad_x(p0,z03,z14,z22);
	  
	  // s

	  x31 = (x20 >> 23) + s;
	  x10 = x0 + x31;
	  x11 = x1 + x31;
	  x12 = x2 + x31;
	  x13 = x3 + x31;
	  x14 = x4 + x31;
	  x15 = x5 + x31;
	  x16 = x6 + x31;
	  x17 = x7 + x31;
	    
	  z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z04,z07,z15); 
	  z16= svmad_x(p0,z04,z08,z16); 
	  z17= svmad_x(p0,z04,z09,z17); 
	  z18= svmad_x(p0,z04,z10,z18); 
	  z19= svmad_x(p0,z04,z11,z19); 
	  z20= svmad_x(p0,z04,z12,z20); 
	  z21= svmad_x(p0,z04,z13,z21); 
	  z22= svmad_x(p0,z04,z14,z22);

	  // b

	  x31 = (x20 >> 24) + b;
	  x10 = x0 + x31;
	  x11 = x1 + x31;
	  x12 = x2 + x31;
	  x13 = x3 + x31;
	  x14 = x4 + x31;
	  x15 = x5 + x31;
	  x16 = x6 + x31;
	  x17 = x7 + x31;
	    
	  z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z05,z07,z15); 
	  z16= svmad_x(p0,z05,z08,z16); 
	  z17= svmad_x(p0,z05,z09,z17); 
	  z18= svmad_x(p0,z05,z10,z18); 
	  z19= svmad_x(p0,z05,z11,z19); 
	  z20= svmad_x(p0,z05,z12,z20); 
	  z21= svmad_x(p0,z05,z13,z21); 
	  z22= svmad_x(p0,z05,z14,z22);

	  // t

	  x31 = (x20 >> 25) + t;
	  x10 = x0 + x31;
	  x11 = x1 + x31;
	  x12 = x2 + x31;
	  x13 = x3 + x31;
	  x14 = x4 + x31;
	  x15 = x5 + x31;
	  x16 = x6 + x31;
	  x17 = x7 + x31;
	  
	  z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z06,z07,z15); 
	  z16= svmad_x(p0,z06,z08,z16); 
	  z17= svmad_x(p0,z06,z09,z17); 
	  z18= svmad_x(p0,z06,z10,z18); 
	  z19= svmad_x(p0,z06,z11,z19); 
	  z20= svmad_x(p0,z06,z12,z20); 
	  z21= svmad_x(p0,z06,z13,z21); 
	  z22= svmad_x(p0,z06,z14,z22);
	  
	  svst1(p0,(float64_t*)&f2_t[x0],z15);
	  svst1(p0,(float64_t*)&f2_t[x1],z16);
	  svst1(p0,(float64_t*)&f2_t[x2],z17);
	  svst1(p0,(float64_t*)&f2_t[x3],z18);
	  svst1(p0,(float64_t*)&f2_t[x4],z19);
	  svst1(p0,(float64_t*)&f2_t[x5],z20);
	  svst1(p0,(float64_t*)&f2_t[x6],z21);
	  svst1(p0,(float64_t*)&f2_t[x7],z22);
#endif
	  

	  for (x20 = 64; x20 < nx-64; x20+=64) {
	    
#if 0
	    x31 = (x20 >> 20) + 1;
	    // c

	    x0 = x0 + 64;
	    x1 = x1 + 64;
	    x2 = x2 + 64;
	    x3 = x3 + 64;
	    x4 = x4 + 64;
	    x5 = x5 + 64;
	    x6 = x6 + 64;
	    x7 = x7 + 64;
	  
	    z07 = svld1(p0,(float64_t*)&f1_t[x0]); 
	    z08 = svld1(p0,(float64_t*)&f1_t[x1]); 
	    z09 = svld1(p0,(float64_t*)&f1_t[x2]); 
	    z10 = svld1(p0,(float64_t*)&f1_t[x3]); 
	    z11 = svld1(p0,(float64_t*)&f1_t[x4]); 
	    z12 = svld1(p0,(float64_t*)&f1_t[x5]); 
	    z13 = svld1(p0,(float64_t*)&f1_t[x6]); 
	    z14 = svld1(p0,(float64_t*)&f1_t[x7]); 

	    z15= svmul_x(p0,z00,z07); 
	    z16= svmul_x(p0,z00,z08); 
	    z17= svmul_x(p0,z00,z09); 
	    z18= svmul_x(p0,z00,z10); 
	    z19= svmul_x(p0,z00,z11); 
	    z20= svmul_x(p0,z00,z12); 
	    z21= svmul_x(p0,z00,z13); 
	    z22= svmul_x(p0,z00,z14);

	    // w

	    /* x10 = x0 - 1; */
	    /* x11 = x1 - 1; */
	    /* x12 = x2 - 1; */
	    /* x13 = x3 - 1; */
	    /* x14 = x4 - 1; */
	    /* x15 = x5 - 1; */
	    /* x16 = x6 - 1; */
	    /* x17 = x7 - 1; */

	    x10 = x0 * x31 - 1;
	    x11 = x1 * x31 - 1;
	    x12 = x2 * x31 - 1;
	    x13 = x3 * x31 - 1;
	    x14 = x4 * x31 - 1;
	    x15 = x5 * x31 - 1;
	    x16 = x6 * x31 - 1;
	    x17 = x7 * x31 - 1;
	  
	    z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	    z15= svmad_x(p0,z01,z07,z15); 
	    z16= svmad_x(p0,z01,z08,z16); 
	    z17= svmad_x(p0,z01,z09,z17); 
	    z18= svmad_x(p0,z01,z10,z18); 
	    z19= svmad_x(p0,z01,z11,z19); 
	    z20= svmad_x(p0,z01,z12,z20); 
	    z21= svmad_x(p0,z01,z13,z21); 
	    z22= svmad_x(p0,z01,z14,z22);

	    // e

	    /* x10 = x0 + 1; */
	    /* x11 = x1 + 1; */
	    /* x12 = x2 + 1; */
	    /* x13 = x3 + 1; */
	    /* x14 = x4 + 1; */
	    /* x15 = x5 + 1; */
	    /* x16 = x6 + 1; */
	    /* x17 = x7 + 1; */

	    x10 = x0 * x31 + 1;
	    x11 = x1 * x31 + 1;
	    x12 = x2 * x31 + 1;
	    x13 = x3 * x31 + 1;
	    x14 = x4 * x31 + 1;
	    x15 = x5 * x31 + 1;
	    x16 = x6 * x31 + 1;
	    x17 = x7 * x31 + 1;
	    
	    z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	    z15= svmad_x(p0,z02,z07,z15); 
	    z16= svmad_x(p0,z02,z08,z16); 
	    z17= svmad_x(p0,z02,z09,z17); 
	    z18= svmad_x(p0,z02,z10,z18); 
	    z19= svmad_x(p0,z02,z11,z19); 
	    z20= svmad_x(p0,z02,z12,z20); 
	    z21= svmad_x(p0,z02,z13,z21); 
	    z22= svmad_x(p0,z02,z14,z22);

	    // n

	    /* x31 = x20 >> 22; */
	    /* x10 = x0 + n + x31; */
	    /* x11 = x1 + n + x31; */
	    /* x12 = x2 + n + x31; */
	    /* x13 = x3 + n + x31; */
	    /* x14 = x4 + n + x31; */
	    /* x15 = x5 + n + x31; */
	    /* x16 = x6 + n + x31; */
	    /* x17 = x7 + n + x31; */
	    /* x10 = x0 + n; */
	    /* x11 = x1 + n; */
	    /* x12 = x2 + n; */
	    /* x13 = x3 + n; */
	    /* x14 = x4 + n; */
	    /* x15 = x5 + n; */
	    /* x16 = x6 + n; */
	    /* x17 = x7 + n; */

	    x10 = x0 * x31 + n;
	    x11 = x1 * x31 + n;
	    x12 = x2 * x31 + n;
	    x13 = x3 * x31 + n;
	    x14 = x4 * x31 + n;
	    x15 = x5 * x31 + n;
	    x16 = x6 * x31 + n;
	    x17 = x7 * x31 + n;
	  
	    z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	    z15= svmad_x(p0,z03,z07,z15); 
	    z16= svmad_x(p0,z03,z08,z16); 
	    z17= svmad_x(p0,z03,z09,z17); 
	    z18= svmad_x(p0,z03,z10,z18); 
	    z19= svmad_x(p0,z03,z11,z19); 
	    z20= svmad_x(p0,z03,z12,z20); 
	    z21= svmad_x(p0,z03,z13,z21); 
	    z22= svmad_x(p0,z03,z14,z22);
	  
	    // s

	    /* x31 = x20 >> 23; */
	    /* x10 = x0 + s + x31; */
	    /* x11 = x1 + s + x31; */
	    /* x12 = x2 + s + x31; */
	    /* x13 = x3 + s + x31; */
	    /* x14 = x4 + s + x31; */
	    /* x15 = x5 + s + x31; */
	    /* x16 = x6 + s + x31; */
	    /* x17 = x7 + s + x31; */

	    x10 = x0 * x31 + s;
	    x11 = x1 * x31 + s;
	    x12 = x2 * x31 + s;
	    x13 = x3 * x31 + s;
	    x14 = x4 * x31 + s;
	    x15 = x5 * x31 + s;
	    x16 = x6 * x31 + s;
	    x17 = x7 * x31 + s;
	    
	    z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	    z15= svmad_x(p0,z04,z07,z15); 
	    z16= svmad_x(p0,z04,z08,z16); 
	    z17= svmad_x(p0,z04,z09,z17); 
	    z18= svmad_x(p0,z04,z10,z18); 
	    z19= svmad_x(p0,z04,z11,z19); 
	    z20= svmad_x(p0,z04,z12,z20); 
	    z21= svmad_x(p0,z04,z13,z21); 
	    z22= svmad_x(p0,z04,z14,z22);

	    // b

	    /* x31 = x20 >> 24; */
	    /* x10 = x0 + b + x31; */
	    /* x11 = x1 + b + x31; */
	    /* x12 = x2 + b + x31; */
	    /* x13 = x3 + b + x31; */
	    /* x14 = x4 + b + x31; */
	    /* x15 = x5 + b + x31; */
	    /* x16 = x6 + b + x31; */
	    /* x17 = x7 + b + x31; */

	    x10 = x0 * x31 + b;
	    x11 = x1 * x31 + b;
	    x12 = x2 * x31 + b;
	    x13 = x3 * x31 + b;
	    x14 = x4 * x31 + b;
	    x15 = x5 * x31 + b;
	    x16 = x6 * x31 + b;
	    x17 = x7 * x31 + b;
	    
	    z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	    z15= svmad_x(p0,z05,z07,z15); 
	    z16= svmad_x(p0,z05,z08,z16); 
	    z17= svmad_x(p0,z05,z09,z17); 
	    z18= svmad_x(p0,z05,z10,z18); 
	    z19= svmad_x(p0,z05,z11,z19); 
	    z20= svmad_x(p0,z05,z12,z20); 
	    z21= svmad_x(p0,z05,z13,z21); 
	    z22= svmad_x(p0,z05,z14,z22);

	    // t

	    /* x31 = x20 >> 25; */
	    /* x10 = x0 + t + x31; */
	    /* x11 = x1 + t + x31; */
	    /* x12 = x2 + t + x31; */
	    /* x13 = x3 + t + x31; */
	    /* x14 = x4 + t + x31; */
	    /* x15 = x5 + t + x31; */
	    /* x16 = x6 + t + x31; */
	    /* x17 = x7 + t + x31; */
	    x10 = x0 * x31 + t;
	    x11 = x1 * x31 + t;
	    x12 = x2 * x31 + t;
	    x13 = x3 * x31 + t;
	    x14 = x4 * x31 + t;
	    x15 = x5 * x31 + t;
	    x16 = x6 * x31 + t;
	    x17 = x7 * x31 + t;
	  
	    z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	    z15= svmad_x(p0,z06,z07,z15); 
	    z16= svmad_x(p0,z06,z08,z16); 
	    z17= svmad_x(p0,z06,z09,z17); 
	    z18= svmad_x(p0,z06,z10,z18); 
	    z19= svmad_x(p0,z06,z11,z19); 
	    z20= svmad_x(p0,z06,z12,z20); 
	    z21= svmad_x(p0,z06,z13,z21); 
	    z22= svmad_x(p0,z06,z14,z22);
	  
	    svst1(p0,(float64_t*)&f2_t[x0],z15);
	    svst1(p0,(float64_t*)&f2_t[x1],z16);
	    svst1(p0,(float64_t*)&f2_t[x2],z17);
	    svst1(p0,(float64_t*)&f2_t[x3],z18);
	    svst1(p0,(float64_t*)&f2_t[x4],z19);
	    svst1(p0,(float64_t*)&f2_t[x5],z20);
	    svst1(p0,(float64_t*)&f2_t[x6],z21);
	    svst1(p0,(float64_t*)&f2_t[x7],z22);
#else
	    // c

	    x0 = x0 + 64;
	    x1 = x1 + 64;
	    x2 = x2 + 64;
	    x3 = x3 + 64;
	    x4 = x4 + 64;
	    x5 = x5 + 64;
	    x6 = x6 + 64;
	    x7 = x7 + 64;
	  
	    z07 = svld1(p0,(float64_t*)&f1_t[x0]); 
	    z08 = svld1(p0,(float64_t*)&f1_t[x1]); 
	    z09 = svld1(p0,(float64_t*)&f1_t[x2]); 
	    z10 = svld1(p0,(float64_t*)&f1_t[x3]); 
	    z11 = svld1(p0,(float64_t*)&f1_t[x4]); 
	    z12 = svld1(p0,(float64_t*)&f1_t[x5]); 
	    z13 = svld1(p0,(float64_t*)&f1_t[x6]); 
	    z14 = svld1(p0,(float64_t*)&f1_t[x7]); 

	    z15= svmul_x(p0,z00,z07); 
	    z16= svmul_x(p0,z00,z08); 
	    z17= svmul_x(p0,z00,z09); 
	    z18= svmul_x(p0,z00,z10); 
	    z19= svmul_x(p0,z00,z11); 
	    z20= svmul_x(p0,z00,z12); 
	    z21= svmul_x(p0,z00,z13); 
	    z22= svmul_x(p0,z00,z14);

	    // w

	    x31 = (x20 >> 20) - 1;
	    x10 = x0 + x31;
	    x11 = x1 + x31;
	    x12 = x2 + x31;
	    x13 = x3 + x31;
	    x14 = x4 + x31;
	    x15 = x5 + x31;
	    x16 = x6 + x31;
	    x17 = x7 + x31;
	  
	    z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	    z15= svmad_x(p0,z01,z07,z15); 
	    z16= svmad_x(p0,z01,z08,z16); 
	    z17= svmad_x(p0,z01,z09,z17); 
	    z18= svmad_x(p0,z01,z10,z18); 
	    z19= svmad_x(p0,z01,z11,z19); 
	    z20= svmad_x(p0,z01,z12,z20); 
	    z21= svmad_x(p0,z01,z13,z21); 
	    z22= svmad_x(p0,z01,z14,z22);

	    // e

	    x31 = (x20 >> 21) + 1;
	    x10 = x0 + x31;
	    x11 = x1 + x31;
	    x12 = x2 + x31;
	    x13 = x3 + x31;
	    x14 = x4 + x31;
	    x15 = x5 + x31;
	    x16 = x6 + x31;
	    x17 = x7 + x31;
	    
	    z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	    z15= svmad_x(p0,z02,z07,z15); 
	    z16= svmad_x(p0,z02,z08,z16); 
	    z17= svmad_x(p0,z02,z09,z17); 
	    z18= svmad_x(p0,z02,z10,z18); 
	    z19= svmad_x(p0,z02,z11,z19); 
	    z20= svmad_x(p0,z02,z12,z20); 
	    z21= svmad_x(p0,z02,z13,z21); 
	    z22= svmad_x(p0,z02,z14,z22);

	    // n

	    x31 = (x20 >> 22) + n;
	    x10 = x0 + x31;
	    x11 = x1 + x31;
	    x12 = x2 + x31;
	    x13 = x3 + x31;
	    x14 = x4 + x31;
	    x15 = x5 + x31;
	    x16 = x6 + x31;
	    x17 = x7 + x31;
	  
	    z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	    z15= svmad_x(p0,z03,z07,z15); 
	    z16= svmad_x(p0,z03,z08,z16); 
	    z17= svmad_x(p0,z03,z09,z17); 
	    z18= svmad_x(p0,z03,z10,z18); 
	    z19= svmad_x(p0,z03,z11,z19); 
	    z20= svmad_x(p0,z03,z12,z20); 
	    z21= svmad_x(p0,z03,z13,z21); 
	    z22= svmad_x(p0,z03,z14,z22);
	  
	    // s

	    x31 = (x20 >> 23) + s;
	    x10 = x0 + x31;
	    x11 = x1 + x31;
	    x12 = x2 + x31;
	    x13 = x3 + x31;
	    x14 = x4 + x31;
	    x15 = x5 + x31;
	    x16 = x6 + x31;
	    x17 = x7 + x31;
	    
	    z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	    z15= svmad_x(p0,z04,z07,z15); 
	    z16= svmad_x(p0,z04,z08,z16); 
	    z17= svmad_x(p0,z04,z09,z17); 
	    z18= svmad_x(p0,z04,z10,z18); 
	    z19= svmad_x(p0,z04,z11,z19); 
	    z20= svmad_x(p0,z04,z12,z20); 
	    z21= svmad_x(p0,z04,z13,z21); 
	    z22= svmad_x(p0,z04,z14,z22);

	    // b

	    x31 = (x20 >> 24) + b;
	    x10 = x0 + x31;
	    x11 = x1 + x31;
	    x12 = x2 + x31;
	    x13 = x3 + x31;
	    x14 = x4 + x31;
	    x15 = x5 + x31;
	    x16 = x6 + x31;
	    x17 = x7 + x31;
	    
	    z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	    z15= svmad_x(p0,z05,z07,z15); 
	    z16= svmad_x(p0,z05,z08,z16); 
	    z17= svmad_x(p0,z05,z09,z17); 
	    z18= svmad_x(p0,z05,z10,z18); 
	    z19= svmad_x(p0,z05,z11,z19); 
	    z20= svmad_x(p0,z05,z12,z20); 
	    z21= svmad_x(p0,z05,z13,z21); 
	    z22= svmad_x(p0,z05,z14,z22);

	    // t

	    x31 = (x20 >> 25) + t;
	    x10 = x0 + x31;
	    x11 = x1 + x31;
	    x12 = x2 + x31;
	    x13 = x3 + x31;
	    x14 = x4 + x31;
	    x15 = x5 + x31;
	    x16 = x6 + x31;
	    x17 = x7 + x31;
	  
	    z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	    z15= svmad_x(p0,z06,z07,z15); 
	    z16= svmad_x(p0,z06,z08,z16); 
	    z17= svmad_x(p0,z06,z09,z17); 
	    z18= svmad_x(p0,z06,z10,z18); 
	    z19= svmad_x(p0,z06,z11,z19); 
	    z20= svmad_x(p0,z06,z12,z20); 
	    z21= svmad_x(p0,z06,z13,z21); 
	    z22= svmad_x(p0,z06,z14,z22);
	  
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
	  

	  // c

	  x0 = x0 + 64;
	  x1 = x1 + 64;
	  x2 = x2 + 64;
	  x3 = x3 + 64;
	  x4 = x4 + 64;
	  x5 = x5 + 64;
	  x6 = x6 + 64;
	  x7 = x7 + 64;
	  
	  z07 = svld1(p0,(float64_t*)&f1_t[x0]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x1]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x2]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x3]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x4]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x5]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x6]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x7]); 

	  z15= svmul_x(p0,z00,z07); 
	  z16= svmul_x(p0,z00,z08); 
	  z17= svmul_x(p0,z00,z09); 
	  z18= svmul_x(p0,z00,z10); 
	  z19= svmul_x(p0,z00,z11); 
	  z20= svmul_x(p0,z00,z12); 
	  z21= svmul_x(p0,z00,z13); 
	  z22= svmul_x(p0,z00,z14);

#if 0
	  // w

	  x10 = x0 - 1;
	  x11 = x1 - 1;
	  x12 = x2 - 1;
	  x13 = x3 - 1;
	  x14 = x4 - 1;
	  x15 = x5 - 1;
	  x16 = x6 - 1;
	  x17 = x7 - 1;
	  
	  z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z01,z07,z15); 
	  z16= svmad_x(p0,z01,z08,z16); 
	  z17= svmad_x(p0,z01,z09,z17); 
	  z18= svmad_x(p0,z01,z10,z18); 
	  z19= svmad_x(p0,z01,z11,z19); 
	  z20= svmad_x(p0,z01,z12,z20); 
	  z21= svmad_x(p0,z01,z13,z21); 
	  z22= svmad_x(p0,z01,z14,z22);

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
	  z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&fcp1_arr[0]); 

	  z15= svmad_x(p0,z02,z07,z15); 
	  z16= svmad_x(p0,z02,z08,z16); 
	  z17= svmad_x(p0,z02,z09,z17); 
	  z18= svmad_x(p0,z02,z10,z18); 
	  z19= svmad_x(p0,z02,z11,z19); 
	  z20= svmad_x(p0,z02,z12,z20); 
	  z21= svmad_x(p0,z02,z13,z21); 
	  z22= svmad_x(p0,z02,z14,z22);

	  // n

	  x10 = x0 + n;
	  x11 = x1 + n;
	  x12 = x2 + n;
	  x13 = x3 + n;
	  x14 = x4 + n;
	  x15 = x5 + n;
	  x16 = x6 + n;
	  x17 = x7 + n;
	  
	  z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z03,z07,z15); 
	  z16= svmad_x(p0,z03,z08,z16); 
	  z17= svmad_x(p0,z03,z09,z17); 
	  z18= svmad_x(p0,z03,z10,z18); 
	  z19= svmad_x(p0,z03,z11,z19); 
	  z20= svmad_x(p0,z03,z12,z20); 
	  z21= svmad_x(p0,z03,z13,z21); 
	  z22= svmad_x(p0,z03,z14,z22);
	  
	  // s

	  x10 = x0 + s;
	  x11 = x1 + s;
	  x12 = x2 + s;
	  x13 = x3 + s;
	  x14 = x4 + s;
	  x15 = x5 + s;
	  x16 = x6 + s;
	  x17 = x7 + s;
	  
	  z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z04,z07,z15); 
	  z16= svmad_x(p0,z04,z08,z16); 
	  z17= svmad_x(p0,z04,z09,z17); 
	  z18= svmad_x(p0,z04,z10,z18); 
	  z19= svmad_x(p0,z04,z11,z19); 
	  z20= svmad_x(p0,z04,z12,z20); 
	  z21= svmad_x(p0,z04,z13,z21); 
	  z22= svmad_x(p0,z04,z14,z22);

	  // b

	  x10 = x0 + b;
	  x11 = x1 + b;
	  x12 = x2 + b;
	  x13 = x3 + b;
	  x14 = x4 + b;
	  x15 = x5 + b;
	  x16 = x6 + b;
	  x17 = x7 + b;
	  
	  z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z05,z07,z15); 
	  z16= svmad_x(p0,z05,z08,z16); 
	  z17= svmad_x(p0,z05,z09,z17); 
	  z18= svmad_x(p0,z05,z10,z18); 
	  z19= svmad_x(p0,z05,z11,z19); 
	  z20= svmad_x(p0,z05,z12,z20); 
	  z21= svmad_x(p0,z05,z13,z21); 
	  z22= svmad_x(p0,z05,z14,z22);

	  // t

	  x10 = x0 + t;
	  x11 = x1 + t;
	  x12 = x2 + t;
	  x13 = x3 + t;
	  x14 = x4 + t;
	  x15 = x5 + t;
	  x16 = x6 + t;
	  x17 = x7 + t;
	  
	  z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z06,z07,z15); 
	  z16= svmad_x(p0,z06,z08,z16); 
	  z17= svmad_x(p0,z06,z09,z17); 
	  z18= svmad_x(p0,z06,z10,z18); 
	  z19= svmad_x(p0,z06,z11,z19); 
	  z20= svmad_x(p0,z06,z12,z20); 
	  z21= svmad_x(p0,z06,z13,z21); 
	  z22= svmad_x(p0,z06,z14,z22);
	  
	  svst1(p0,(float64_t*)&f2_t[x0],z15);
	  svst1(p0,(float64_t*)&f2_t[x1],z16);
	  svst1(p0,(float64_t*)&f2_t[x2],z17);
	  svst1(p0,(float64_t*)&f2_t[x3],z18);
	  svst1(p0,(float64_t*)&f2_t[x4],z19);
	  svst1(p0,(float64_t*)&f2_t[x5],z20);
	  svst1(p0,(float64_t*)&f2_t[x6],z21);
	  svst1(p0,(float64_t*)&f2_t[x7],z22);
#else
	  // w

	  x31 = (x20 >> 20) - 1;
	  x10 = x0 + x31;
	  x11 = x1 + x31;
	  x12 = x2 + x31;
	  x13 = x3 + x31;
	  x14 = x4 + x31;
	  x15 = x5 + x31;
	  x16 = x6 + x31;
	  x17 = x7 + x31;
	  
	  z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z01,z07,z15); 
	  z16= svmad_x(p0,z01,z08,z16); 
	  z17= svmad_x(p0,z01,z09,z17); 
	  z18= svmad_x(p0,z01,z10,z18); 
	  z19= svmad_x(p0,z01,z11,z19); 
	  z20= svmad_x(p0,z01,z12,z20); 
	  z21= svmad_x(p0,z01,z13,z21); 
	  z22= svmad_x(p0,z01,z14,z22);

	  // e

	  x31 = (x20 >> 21) + 1;
	  x10 = x0 + x31;
	  x11 = x1 + x31;
	  x12 = x2 + x31;
	  x13 = x3 + x31;
	  x14 = x4 + x31;
	  x15 = x5 + x31;
	  x16 = x6 + x31;
	  x17 = x7 + x31;
	    
	  float64_t fcp1_arr[8] = {f1_t[x7+1],f1_t[x7+2],f1_t[x7+3],f1_t[x7+4],f1_t[x7+5],f1_t[x7+6],f1_t[x7+7],f1_t[x7+7]};
	  z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&fcp1_arr[0]);

	  z15= svmad_x(p0,z02,z07,z15); 
	  z16= svmad_x(p0,z02,z08,z16); 
	  z17= svmad_x(p0,z02,z09,z17); 
	  z18= svmad_x(p0,z02,z10,z18); 
	  z19= svmad_x(p0,z02,z11,z19); 
	  z20= svmad_x(p0,z02,z12,z20); 
	  z21= svmad_x(p0,z02,z13,z21); 
	  z22= svmad_x(p0,z02,z14,z22);

	  // n

	  x31 = (x20 >> 22) + n;
	  x10 = x0 + x31;
	  x11 = x1 + x31;
	  x12 = x2 + x31;
	  x13 = x3 + x31;
	  x14 = x4 + x31;
	  x15 = x5 + x31;
	  x16 = x6 + x31;
	  x17 = x7 + x31;
	  
	  z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z03,z07,z15); 
	  z16= svmad_x(p0,z03,z08,z16); 
	  z17= svmad_x(p0,z03,z09,z17); 
	  z18= svmad_x(p0,z03,z10,z18); 
	  z19= svmad_x(p0,z03,z11,z19); 
	  z20= svmad_x(p0,z03,z12,z20); 
	  z21= svmad_x(p0,z03,z13,z21); 
	  z22= svmad_x(p0,z03,z14,z22);
	  
	  // s

	  x31 = (x20 >> 23) + s;
	  x10 = x0 + x31;
	  x11 = x1 + x31;
	  x12 = x2 + x31;
	  x13 = x3 + x31;
	  x14 = x4 + x31;
	  x15 = x5 + x31;
	  x16 = x6 + x31;
	  x17 = x7 + x31;
	    
	  z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z04,z07,z15); 
	  z16= svmad_x(p0,z04,z08,z16); 
	  z17= svmad_x(p0,z04,z09,z17); 
	  z18= svmad_x(p0,z04,z10,z18); 
	  z19= svmad_x(p0,z04,z11,z19); 
	  z20= svmad_x(p0,z04,z12,z20); 
	  z21= svmad_x(p0,z04,z13,z21); 
	  z22= svmad_x(p0,z04,z14,z22);

	  // b

	  x31 = (x20 >> 24) + b;
	  x10 = x0 + x31;
	  x11 = x1 + x31;
	  x12 = x2 + x31;
	  x13 = x3 + x31;
	  x14 = x4 + x31;
	  x15 = x5 + x31;
	  x16 = x6 + x31;
	  x17 = x7 + x31;
	    
	  z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z05,z07,z15); 
	  z16= svmad_x(p0,z05,z08,z16); 
	  z17= svmad_x(p0,z05,z09,z17); 
	  z18= svmad_x(p0,z05,z10,z18); 
	  z19= svmad_x(p0,z05,z11,z19); 
	  z20= svmad_x(p0,z05,z12,z20); 
	  z21= svmad_x(p0,z05,z13,z21); 
	  z22= svmad_x(p0,z05,z14,z22);

	  // t

	  x31 = (x20 >> 25) + t;
	  x10 = x0 + x31;
	  x11 = x1 + x31;
	  x12 = x2 + x31;
	  x13 = x3 + x31;
	  x14 = x4 + x31;
	  x15 = x5 + x31;
	  x16 = x6 + x31;
	  x17 = x7 + x31;
	  
	  z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z06,z07,z15); 
	  z16= svmad_x(p0,z06,z08,z16); 
	  z17= svmad_x(p0,z06,z09,z17); 
	  z18= svmad_x(p0,z06,z10,z18); 
	  z19= svmad_x(p0,z06,z11,z19); 
	  z20= svmad_x(p0,z06,z12,z20); 
	  z21= svmad_x(p0,z06,z13,z21); 
	  z22= svmad_x(p0,z06,z14,z22);
	  
	  svst1(p0,(float64_t*)&f2_t[x0],z15);
	  svst1(p0,(float64_t*)&f2_t[x1],z16);
	  svst1(p0,(float64_t*)&f2_t[x2],z17);
	  svst1(p0,(float64_t*)&f2_t[x3],z18);
	  svst1(p0,(float64_t*)&f2_t[x4],z19);
	  svst1(p0,(float64_t*)&f2_t[x5],z20);
	  svst1(p0,(float64_t*)&f2_t[x6],z21);
	  svst1(p0,(float64_t*)&f2_t[x7],z22);

#endif
	  
#else

	  x0 = c + 0;
	  x1 = c + 8;
	  x2 = c + 16;
	  x3 = c + 24;
	  x4 = c + 32;
	  x5 = c + 40;
	  x6 = c + 48;
	  x7 = c + 56;
	  int32_t index_tmp[16] = {x0,x1,x2,x3,x4,x5,x6,x7};
	  z30 = svld1(p1,(int32_t*)&index_tmp[0]);
	  
	  z07 = svld1(p0,(float64_t*)&f1_t[x0]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x1]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x2]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x3]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x4]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x5]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x6]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x7]); 

	  z15= svmul_x(p0,z00,z07); 
	  z16= svmul_x(p0,z00,z08); 
	  z17= svmul_x(p0,z00,z09); 
	  z18= svmul_x(p0,z00,z10); 
	  z19= svmul_x(p0,z00,z11); 
	  z20= svmul_x(p0,z00,z12); 
	  z21= svmul_x(p0,z00,z13); 
	  z22= svmul_x(p0,z00,z14);

	  // w
	  x30 = -1;
	  z31 = svadd_x(p1,z30,x30);
	  svst1(p1,(int32_t*)&index_tmp[0],z31);
	  x10 = index_tmp[0];
	  x11 = index_tmp[1];
	  x12 = index_tmp[2];
	  x13 = index_tmp[3];
	  x14 = index_tmp[4];
	  x15 = index_tmp[5];
	  x16 = index_tmp[6];
	  x17 = index_tmp[7];
	  
	  z07 = svld1(p0,(float64_t*)&fcm1_arr[0]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z01,z07,z15); 
	  z16= svmad_x(p0,z01,z08,z16); 
	  z17= svmad_x(p0,z01,z09,z17); 
	  z18= svmad_x(p0,z01,z10,z18); 
	  z19= svmad_x(p0,z01,z11,z19); 
	  z20= svmad_x(p0,z01,z12,z20); 
	  z21= svmad_x(p0,z01,z13,z21); 
	  z22= svmad_x(p0,z01,z14,z22);

	  // e

	  x30 = 1;
	  z31 = svadd_x(p1,z30,x30);
	  svst1(p1,(int32_t*)&index_tmp[0],z31);
	  x10 = index_tmp[0];
	  x11 = index_tmp[1];
	  x12 = index_tmp[2];
	  x13 = index_tmp[3];
	  x14 = index_tmp[4];
	  x15 = index_tmp[5];
	  x16 = index_tmp[6];
	  x17 = index_tmp[7];

	  z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z02,z07,z15); 
	  z16= svmad_x(p0,z02,z08,z16); 
	  z17= svmad_x(p0,z02,z09,z17); 
	  z18= svmad_x(p0,z02,z10,z18); 
	  z19= svmad_x(p0,z02,z11,z19); 
	  z20= svmad_x(p0,z02,z12,z20); 
	  z21= svmad_x(p0,z02,z13,z21); 
	  z22= svmad_x(p0,z02,z14,z22);

	  // n
	  
	  x30 = n;
	  z31 = svadd_x(p1,z30,x30);
	  svst1(p1,(int32_t*)&index_tmp[0],z31);
	  x10 = index_tmp[0];
	  x11 = index_tmp[1];
	  x12 = index_tmp[2];
	  x13 = index_tmp[3];
	  x14 = index_tmp[4];
	  x15 = index_tmp[5];
	  x16 = index_tmp[6];
	  x17 = index_tmp[7];

	  z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z03,z07,z15); 
	  z16= svmad_x(p0,z03,z08,z16); 
	  z17= svmad_x(p0,z03,z09,z17); 
	  z18= svmad_x(p0,z03,z10,z18); 
	  z19= svmad_x(p0,z03,z11,z19); 
	  z20= svmad_x(p0,z03,z12,z20); 
	  z21= svmad_x(p0,z03,z13,z21); 
	  z22= svmad_x(p0,z03,z14,z22);
	  
	  // s

	  x30 = s;
	  z31 = svadd_x(p1,z30,x30);
	  svst1(p1,(int32_t*)&index_tmp[0],z31);
	  x10 = index_tmp[0];
	  x11 = index_tmp[1];
	  x12 = index_tmp[2];
	  x13 = index_tmp[3];
	  x14 = index_tmp[4];
	  x15 = index_tmp[5];
	  x16 = index_tmp[6];
	  x17 = index_tmp[7];

	  z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z04,z07,z15); 
	  z16= svmad_x(p0,z04,z08,z16); 
	  z17= svmad_x(p0,z04,z09,z17); 
	  z18= svmad_x(p0,z04,z10,z18); 
	  z19= svmad_x(p0,z04,z11,z19); 
	  z20= svmad_x(p0,z04,z12,z20); 
	  z21= svmad_x(p0,z04,z13,z21); 
	  z22= svmad_x(p0,z04,z14,z22);

	  // b

	  x30 = b;
	  z31 = svadd_x(p1,z30,x30);
	  svst1(p1,(int32_t*)&index_tmp[0],z31);
	  x10 = index_tmp[0];
	  x11 = index_tmp[1];
	  x12 = index_tmp[2];
	  x13 = index_tmp[3];
	  x14 = index_tmp[4];
	  x15 = index_tmp[5];
	  x16 = index_tmp[6];
	  x17 = index_tmp[7];

	  z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z05,z07,z15); 
	  z16= svmad_x(p0,z05,z08,z16); 
	  z17= svmad_x(p0,z05,z09,z17); 
	  z18= svmad_x(p0,z05,z10,z18); 
	  z19= svmad_x(p0,z05,z11,z19); 
	  z20= svmad_x(p0,z05,z12,z20); 
	  z21= svmad_x(p0,z05,z13,z21); 
	  z22= svmad_x(p0,z05,z14,z22);

	  // t

	  x30 = t;
	  z31 = svadd_x(p1,z30,x30);
	  svst1(p1,(int32_t*)&index_tmp[0],z31);
	  x10 = index_tmp[0];
	  x11 = index_tmp[1];
	  x12 = index_tmp[2];
	  x13 = index_tmp[3];
	  x14 = index_tmp[4];
	  x15 = index_tmp[5];
	  x16 = index_tmp[6];
	  x17 = index_tmp[7];

	  z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	  z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	  z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	  z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	  z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	  z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	  z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	  z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	  z15= svmad_x(p0,z06,z07,z15); 
	  z16= svmad_x(p0,z06,z08,z16); 
	  z17= svmad_x(p0,z06,z09,z17); 
	  z18= svmad_x(p0,z06,z10,z18); 
	  z19= svmad_x(p0,z06,z11,z19); 
	  z20= svmad_x(p0,z06,z12,z20); 
	  z21= svmad_x(p0,z06,z13,z21); 
	  z22= svmad_x(p0,z06,z14,z22);
	  
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


	    x30 = 64;
	    z31 = svadd_x(p1,z30,x30);
	    z30 = z31;
	    svst1(p1,(int32_t*)&index_tmp[0],z31);
	    x10 = index_tmp[0];
	    x11 = index_tmp[1];
	    x12 = index_tmp[2];
	    x13 = index_tmp[3];
	    x14 = index_tmp[4];
	    x15 = index_tmp[5];
	    x16 = index_tmp[6];
	    x17 = index_tmp[7];
	  
	    z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	    z15= svmul_x(p0,z00,z07); 
	    z16= svmul_x(p0,z00,z08); 
	    z17= svmul_x(p0,z00,z09); 
	    z18= svmul_x(p0,z00,z10); 
	    z19= svmul_x(p0,z00,z11); 
	    z20= svmul_x(p0,z00,z12); 
	    z21= svmul_x(p0,z00,z13); 
	    z22= svmul_x(p0,z00,z14);

	    // w
	    x30 = -1;
	    z31 = svadd_x(p1,z30,x30);
	    svst1(p1,(int32_t*)&index_tmp[0],z31);
	    x10 = index_tmp[0];
	    x11 = index_tmp[1];
	    x12 = index_tmp[2];
	    x13 = index_tmp[3];
	    x14 = index_tmp[4];
	    x15 = index_tmp[5];
	    x16 = index_tmp[6];
	    x17 = index_tmp[7];
	  
	    z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	    z15= svmad_x(p0,z01,z07,z15); 
	    z16= svmad_x(p0,z01,z08,z16); 
	    z17= svmad_x(p0,z01,z09,z17); 
	    z18= svmad_x(p0,z01,z10,z18); 
	    z19= svmad_x(p0,z01,z11,z19); 
	    z20= svmad_x(p0,z01,z12,z20); 
	    z21= svmad_x(p0,z01,z13,z21); 
	    z22= svmad_x(p0,z01,z14,z22);

	    // e

	    x30 = 1;
	    z31 = svadd_x(p1,z30,x30);
	    svst1(p1,(int32_t*)&index_tmp[0],z31);
	    x10 = index_tmp[0];
	    x11 = index_tmp[1];
	    x12 = index_tmp[2];
	    x13 = index_tmp[3];
	    x14 = index_tmp[4];
	    x15 = index_tmp[5];
	    x16 = index_tmp[6];
	    x17 = index_tmp[7];

	    z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	    z15= svmad_x(p0,z02,z07,z15); 
	    z16= svmad_x(p0,z02,z08,z16); 
	    z17= svmad_x(p0,z02,z09,z17); 
	    z18= svmad_x(p0,z02,z10,z18); 
	    z19= svmad_x(p0,z02,z11,z19); 
	    z20= svmad_x(p0,z02,z12,z20); 
	    z21= svmad_x(p0,z02,z13,z21); 
	    z22= svmad_x(p0,z02,z14,z22);

	    // n
	  
	    x30 = n;
	    z31 = svadd_x(p1,z30,x30);
	    svst1(p1,(int32_t*)&index_tmp[0],z31);
	    x10 = index_tmp[0];
	    x11 = index_tmp[1];
	    x12 = index_tmp[2];
	    x13 = index_tmp[3];
	    x14 = index_tmp[4];
	    x15 = index_tmp[5];
	    x16 = index_tmp[6];
	    x17 = index_tmp[7];

	    z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	    z15= svmad_x(p0,z03,z07,z15); 
	    z16= svmad_x(p0,z03,z08,z16); 
	    z17= svmad_x(p0,z03,z09,z17); 
	    z18= svmad_x(p0,z03,z10,z18); 
	    z19= svmad_x(p0,z03,z11,z19); 
	    z20= svmad_x(p0,z03,z12,z20); 
	    z21= svmad_x(p0,z03,z13,z21); 
	    z22= svmad_x(p0,z03,z14,z22);
	  
	    // s

	    x30 = s;
	    z31 = svadd_x(p1,z30,x30);
	    svst1(p1,(int32_t*)&index_tmp[0],z31);
	    x10 = index_tmp[0];
	    x11 = index_tmp[1];
	    x12 = index_tmp[2];
	    x13 = index_tmp[3];
	    x14 = index_tmp[4];
	    x15 = index_tmp[5];
	    x16 = index_tmp[6];
	    x17 = index_tmp[7];

	    z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	    z15= svmad_x(p0,z04,z07,z15); 
	    z16= svmad_x(p0,z04,z08,z16); 
	    z17= svmad_x(p0,z04,z09,z17); 
	    z18= svmad_x(p0,z04,z10,z18); 
	    z19= svmad_x(p0,z04,z11,z19); 
	    z20= svmad_x(p0,z04,z12,z20); 
	    z21= svmad_x(p0,z04,z13,z21); 
	    z22= svmad_x(p0,z04,z14,z22);

	    // b

	    x30 = b;
	    z31 = svadd_x(p1,z30,x30);
	    svst1(p1,(int32_t*)&index_tmp[0],z31);
	    x10 = index_tmp[0];
	    x11 = index_tmp[1];
	    x12 = index_tmp[2];
	    x13 = index_tmp[3];
	    x14 = index_tmp[4];
	    x15 = index_tmp[5];
	    x16 = index_tmp[6];
	    x17 = index_tmp[7];

	    z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	    z15= svmad_x(p0,z05,z07,z15); 
	    z16= svmad_x(p0,z05,z08,z16); 
	    z17= svmad_x(p0,z05,z09,z17); 
	    z18= svmad_x(p0,z05,z10,z18); 
	    z19= svmad_x(p0,z05,z11,z19); 
	    z20= svmad_x(p0,z05,z12,z20); 
	    z21= svmad_x(p0,z05,z13,z21); 
	    z22= svmad_x(p0,z05,z14,z22);

	    // t

	    x30 = t;
	    z31 = svadd_x(p1,z30,x30);
	    svst1(p1,(int32_t*)&index_tmp[0],z31);
	    x10 = index_tmp[0];
	    x11 = index_tmp[1];
	    x12 = index_tmp[2];
	    x13 = index_tmp[3];
	    x14 = index_tmp[4];
	    x15 = index_tmp[5];
	    x16 = index_tmp[6];
	    x17 = index_tmp[7];

	    z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	    z15= svmad_x(p0,z06,z07,z15); 
	    z16= svmad_x(p0,z06,z08,z16); 
	    z17= svmad_x(p0,z06,z09,z17); 
	    z18= svmad_x(p0,z06,z10,z18); 
	    z19= svmad_x(p0,z06,z11,z19); 
	    z20= svmad_x(p0,z06,z12,z20); 
	    z21= svmad_x(p0,z06,z13,z21); 
	    z22= svmad_x(p0,z06,z14,z22);
	  
	    svst1(p0,(float64_t*)&f2_t[x0],z15);
	    svst1(p0,(float64_t*)&f2_t[x1],z16);
	    svst1(p0,(float64_t*)&f2_t[x2],z17);
	    svst1(p0,(float64_t*)&f2_t[x3],z18);
	    svst1(p0,(float64_t*)&f2_t[x4],z19);
	    svst1(p0,(float64_t*)&f2_t[x5],z20);
	    svst1(p0,(float64_t*)&f2_t[x6],z21);
	    svst1(p0,(float64_t*)&f2_t[x7],z22);

	    
	  }
	  
	    x0 = x0 + 64;
	    x1 = x1 + 64;
	    x2 = x2 + 64;
	    x3 = x3 + 64;
	    x4 = x4 + 64;
	    x5 = x5 + 64;
	    x6 = x6 + 64;
	    x7 = x7 + 64;

	  // c

	    x30 = 64;
	    z31 = svadd_x(p1,z30,x30);
	    z30 = z31;
	    svst1(p1,(int32_t*)&index_tmp[0],z31);
	    x10 = index_tmp[0];
	    x11 = index_tmp[1];
	    x12 = index_tmp[2];
	    x13 = index_tmp[3];
	    x14 = index_tmp[4];
	    x15 = index_tmp[5];
	    x16 = index_tmp[6];
	    x17 = index_tmp[7];
	  
	    z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	    z15= svmul_x(p0,z00,z07); 
	    z16= svmul_x(p0,z00,z08); 
	    z17= svmul_x(p0,z00,z09); 
	    z18= svmul_x(p0,z00,z10); 
	    z19= svmul_x(p0,z00,z11); 
	    z20= svmul_x(p0,z00,z12); 
	    z21= svmul_x(p0,z00,z13); 
	    z22= svmul_x(p0,z00,z14);

	    // w
	    x30 = -1;
	    z31 = svadd_x(p1,z30,x30);
	    svst1(p1,(int32_t*)&index_tmp[0],z31);
	    x10 = index_tmp[0];
	    x11 = index_tmp[1];
	    x12 = index_tmp[2];
	    x13 = index_tmp[3];
	    x14 = index_tmp[4];
	    x15 = index_tmp[5];
	    x16 = index_tmp[6];
	    x17 = index_tmp[7];
	  
	    z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	    z15= svmad_x(p0,z01,z07,z15); 
	    z16= svmad_x(p0,z01,z08,z16); 
	    z17= svmad_x(p0,z01,z09,z17); 
	    z18= svmad_x(p0,z01,z10,z18); 
	    z19= svmad_x(p0,z01,z11,z19); 
	    z20= svmad_x(p0,z01,z12,z20); 
	    z21= svmad_x(p0,z01,z13,z21); 
	    z22= svmad_x(p0,z01,z14,z22);

	    // e

	    x30 = 1;
	    z31 = svadd_x(p1,z30,x30);
	    svst1(p1,(int32_t*)&index_tmp[0],z31);
	    x10 = index_tmp[0];
	    x11 = index_tmp[1];
	    x12 = index_tmp[2];
	    x13 = index_tmp[3];
	    x14 = index_tmp[4];
	    x15 = index_tmp[5];
	    x16 = index_tmp[6];
	    x17 = index_tmp[7];

	    //	  float64_t fcp1_arr[8] = {f1_t[x7+1],f1_t[x7+2],f1_t[x7+3],f1_t[x7+4],f1_t[x7+5],f1_t[x7+6],f1_t[x7+7],f1_t[x7+7]};
	  float64_t fcp1_arr[8] = {f1_t[x17],f1_t[x17+1],f1_t[x17+2],f1_t[x17+3],f1_t[x17+4],f1_t[x17+5],f1_t[x17+6],f1_t[x17+6]};
	    z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14 = svld1(p0,(float64_t*)&fcp1_arr[0]);

	    z15= svmad_x(p0,z02,z07,z15); 
	    z16= svmad_x(p0,z02,z08,z16); 
	    z17= svmad_x(p0,z02,z09,z17); 
	    z18= svmad_x(p0,z02,z10,z18); 
	    z19= svmad_x(p0,z02,z11,z19); 
	    z20= svmad_x(p0,z02,z12,z20); 
	    z21= svmad_x(p0,z02,z13,z21); 
	    z22= svmad_x(p0,z02,z14,z22);

	    // n
	  
	    x30 = n;
	    z31 = svadd_x(p1,z30,x30);
	    svst1(p1,(int32_t*)&index_tmp[0],z31);
	    x10 = index_tmp[0];
	    x11 = index_tmp[1];
	    x12 = index_tmp[2];
	    x13 = index_tmp[3];
	    x14 = index_tmp[4];
	    x15 = index_tmp[5];
	    x16 = index_tmp[6];
	    x17 = index_tmp[7];

	    z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	    z15= svmad_x(p0,z03,z07,z15); 
	    z16= svmad_x(p0,z03,z08,z16); 
	    z17= svmad_x(p0,z03,z09,z17); 
	    z18= svmad_x(p0,z03,z10,z18); 
	    z19= svmad_x(p0,z03,z11,z19); 
	    z20= svmad_x(p0,z03,z12,z20); 
	    z21= svmad_x(p0,z03,z13,z21); 
	    z22= svmad_x(p0,z03,z14,z22);
	  
	    // s

	    x30 = s;
	    z31 = svadd_x(p1,z30,x30);
	    svst1(p1,(int32_t*)&index_tmp[0],z31);
	    x10 = index_tmp[0];
	    x11 = index_tmp[1];
	    x12 = index_tmp[2];
	    x13 = index_tmp[3];
	    x14 = index_tmp[4];
	    x15 = index_tmp[5];
	    x16 = index_tmp[6];
	    x17 = index_tmp[7];

	    z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	    z15= svmad_x(p0,z04,z07,z15); 
	    z16= svmad_x(p0,z04,z08,z16); 
	    z17= svmad_x(p0,z04,z09,z17); 
	    z18= svmad_x(p0,z04,z10,z18); 
	    z19= svmad_x(p0,z04,z11,z19); 
	    z20= svmad_x(p0,z04,z12,z20); 
	    z21= svmad_x(p0,z04,z13,z21); 
	    z22= svmad_x(p0,z04,z14,z22);

	    // b

	    x30 = b;
	    z31 = svadd_x(p1,z30,x30);
	    svst1(p1,(int32_t*)&index_tmp[0],z31);
	    x10 = index_tmp[0];
	    x11 = index_tmp[1];
	    x12 = index_tmp[2];
	    x13 = index_tmp[3];
	    x14 = index_tmp[4];
	    x15 = index_tmp[5];
	    x16 = index_tmp[6];
	    x17 = index_tmp[7];

	    z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	    z15= svmad_x(p0,z05,z07,z15); 
	    z16= svmad_x(p0,z05,z08,z16); 
	    z17= svmad_x(p0,z05,z09,z17); 
	    z18= svmad_x(p0,z05,z10,z18); 
	    z19= svmad_x(p0,z05,z11,z19); 
	    z20= svmad_x(p0,z05,z12,z20); 
	    z21= svmad_x(p0,z05,z13,z21); 
	    z22= svmad_x(p0,z05,z14,z22);

	    // t

	    x30 = t;
	    z31 = svadd_x(p1,z30,x30);
	    svst1(p1,(int32_t*)&index_tmp[0],z31);
	    x10 = index_tmp[0];
	    x11 = index_tmp[1];
	    x12 = index_tmp[2];
	    x13 = index_tmp[3];
	    x14 = index_tmp[4];
	    x15 = index_tmp[5];
	    x16 = index_tmp[6];
	    x17 = index_tmp[7];

	    z07 = svld1(p0,(float64_t*)&f1_t[x10]); 
	    z08 = svld1(p0,(float64_t*)&f1_t[x11]); 
	    z09 = svld1(p0,(float64_t*)&f1_t[x12]); 
	    z10 = svld1(p0,(float64_t*)&f1_t[x13]); 
	    z11 = svld1(p0,(float64_t*)&f1_t[x14]); 
	    z12 = svld1(p0,(float64_t*)&f1_t[x15]); 
	    z13 = svld1(p0,(float64_t*)&f1_t[x16]); 
	    z14 = svld1(p0,(float64_t*)&f1_t[x17]); 

	    z15= svmad_x(p0,z06,z07,z15); 
	    z16= svmad_x(p0,z06,z08,z16); 
	    z17= svmad_x(p0,z06,z09,z17); 
	    z18= svmad_x(p0,z06,z10,z18); 
	    z19= svmad_x(p0,z06,z11,z19); 
	    z20= svmad_x(p0,z06,z12,z20); 
	    z21= svmad_x(p0,z06,z13,z21); 
	    z22= svmad_x(p0,z06,z14,z22);
	  
	    svst1(p0,(float64_t*)&f2_t[x0],z15);
	    svst1(p0,(float64_t*)&f2_t[x1],z16);
	    svst1(p0,(float64_t*)&f2_t[x2],z17);
	    svst1(p0,(float64_t*)&f2_t[x3],z18);
	    svst1(p0,(float64_t*)&f2_t[x4],z19);
	    svst1(p0,(float64_t*)&f2_t[x5],z20);
	    svst1(p0,(float64_t*)&f2_t[x6],z21);
	    svst1(p0,(float64_t*)&f2_t[x7],z22);

	  
	  /* for (x20 = 64; x20 < nx-64; x20+=64) { */
	    
	  /*   // c */

	  /*   x30 = 64; */
	  /*   z31 = svadd_x(p1,z30,x30); */
	  /*   z30 = z31; */
	  /*   x10 = svandv(p1000000000000000,z31); */
	  /*   x11 = svandv(p0100000000000000,z31); */
	  /*   x12 = svandv(p0010000000000000,z31); */
	  /*   x13 = svandv(p0001000000000000,z31); */
	  /*   x14 = svandv(p0000100000000000,z31); */
	  /*   x15 = svandv(p0000010000000000,z31); */
	  /*   x16 = svandv(p0000001000000000,z31); */
	  /*   x17 = svandv(p0000000100000000,z31); */
	    	  
	  /*   z07 = svld1(p0,(float64_t*)&f1_t[x10]);  */
	  /*   z08 = svld1(p0,(float64_t*)&f1_t[x11]);  */
	  /*   z09 = svld1(p0,(float64_t*)&f1_t[x12]);  */
	  /*   z10 = svld1(p0,(float64_t*)&f1_t[x13]);  */
	  /*   z11 = svld1(p0,(float64_t*)&f1_t[x14]);  */
	  /*   z12 = svld1(p0,(float64_t*)&f1_t[x15]);  */
	  /*   z13 = svld1(p0,(float64_t*)&f1_t[x16]);  */
	  /*   z14 = svld1(p0,(float64_t*)&f1_t[x17]);  */

	  /*   z15= svmul_x(p0,z00,z07);  */
	  /*   z16= svmul_x(p0,z00,z08);  */
	  /*   z17= svmul_x(p0,z00,z09);  */
	  /*   z18= svmul_x(p0,z00,z10);  */
	  /*   z19= svmul_x(p0,z00,z11);  */
	  /*   z20= svmul_x(p0,z00,z12);  */
	  /*   z21= svmul_x(p0,z00,z13);  */
	  /*   z22= svmul_x(p0,z00,z14); */

	  /*   // w */
	  /*   x30 = -1; */
	  /*   z31 = svadd_x(p1,z30,x30); */
	  /*   x10 = svandv(p1000000000000000,z31); */
	  /*   x11 = svandv(p0100000000000000,z31); */
	  /*   x12 = svandv(p0010000000000000,z31); */
	  /*   x13 = svandv(p0001000000000000,z31); */
	  /*   x14 = svandv(p0000100000000000,z31); */
	  /*   x15 = svandv(p0000010000000000,z31); */
	  /*   x16 = svandv(p0000001000000000,z31); */
	  /*   x17 = svandv(p0000000100000000,z31); */
	  
	  /*   z07 = svld1(p0,(float64_t*)&f1_t[x10]);  */
	  /*   z08 = svld1(p0,(float64_t*)&f1_t[x11]);  */
	  /*   z09 = svld1(p0,(float64_t*)&f1_t[x12]);  */
	  /*   z10 = svld1(p0,(float64_t*)&f1_t[x13]);  */
	  /*   z11 = svld1(p0,(float64_t*)&f1_t[x14]);  */
	  /*   z12 = svld1(p0,(float64_t*)&f1_t[x15]);  */
	  /*   z13 = svld1(p0,(float64_t*)&f1_t[x16]);  */
	  /*   z14 = svld1(p0,(float64_t*)&f1_t[x17]);  */

	  /*   z15= svmad_x(p0,z01,z07,z15);  */
	  /*   z16= svmad_x(p0,z01,z08,z16);  */
	  /*   z17= svmad_x(p0,z01,z09,z17);  */
	  /*   z18= svmad_x(p0,z01,z10,z18);  */
	  /*   z19= svmad_x(p0,z01,z11,z19);  */
	  /*   z20= svmad_x(p0,z01,z12,z20);  */
	  /*   z21= svmad_x(p0,z01,z13,z21);  */
	  /*   z22= svmad_x(p0,z01,z14,z22); */

	  /*   // e */

	  /*   x30 = 1; */
	  /*   z31 = svadd_x(p1,z30,x30); */
	  /*   x10 = svandv(p1000000000000000,z31); */
	  /*   x11 = svandv(p0100000000000000,z31); */
	  /*   x12 = svandv(p0010000000000000,z31); */
	  /*   x13 = svandv(p0001000000000000,z31); */
	  /*   x14 = svandv(p0000100000000000,z31); */
	  /*   x15 = svandv(p0000010000000000,z31); */
	  /*   x16 = svandv(p0000001000000000,z31); */
	  /*   x17 = svandv(p0000000100000000,z31); */

	  /*   z07 = svld1(p0,(float64_t*)&f1_t[x10]);  */
	  /*   z08 = svld1(p0,(float64_t*)&f1_t[x11]);  */
	  /*   z09 = svld1(p0,(float64_t*)&f1_t[x12]);  */
	  /*   z10 = svld1(p0,(float64_t*)&f1_t[x13]);  */
	  /*   z11 = svld1(p0,(float64_t*)&f1_t[x14]);  */
	  /*   z12 = svld1(p0,(float64_t*)&f1_t[x15]);  */
	  /*   z13 = svld1(p0,(float64_t*)&f1_t[x16]);  */
	  /*   z14 = svld1(p0,(float64_t*)&f1_t[x17]);  */

	  /*   z15= svmad_x(p0,z02,z07,z15);  */
	  /*   z16= svmad_x(p0,z02,z08,z16);  */
	  /*   z17= svmad_x(p0,z02,z09,z17);  */
	  /*   z18= svmad_x(p0,z02,z10,z18);  */
	  /*   z19= svmad_x(p0,z02,z11,z19);  */
	  /*   z20= svmad_x(p0,z02,z12,z20);  */
	  /*   z21= svmad_x(p0,z02,z13,z21);  */
	  /*   z22= svmad_x(p0,z02,z14,z22); */

	  /*   // n */
	  
	  /*   x30 = n; */
	  /*   z31 = svadd_x(p1,z30,x30); */
	  /*   x10 = svandv(p1000000000000000,z31); */
	  /*   x11 = svandv(p0100000000000000,z31); */
	  /*   x12 = svandv(p0010000000000000,z31); */
	  /*   x13 = svandv(p0001000000000000,z31); */
	  /*   x14 = svandv(p0000100000000000,z31); */
	  /*   x15 = svandv(p0000010000000000,z31); */
	  /*   x16 = svandv(p0000001000000000,z31); */
	  /*   x17 = svandv(p0000000100000000,z31); */

	  /*   z07 = svld1(p0,(float64_t*)&f1_t[x10]);  */
	  /*   z08 = svld1(p0,(float64_t*)&f1_t[x11]);  */
	  /*   z09 = svld1(p0,(float64_t*)&f1_t[x12]);  */
	  /*   z10 = svld1(p0,(float64_t*)&f1_t[x13]);  */
	  /*   z11 = svld1(p0,(float64_t*)&f1_t[x14]);  */
	  /*   z12 = svld1(p0,(float64_t*)&f1_t[x15]);  */
	  /*   z13 = svld1(p0,(float64_t*)&f1_t[x16]);  */
	  /*   z14 = svld1(p0,(float64_t*)&f1_t[x17]);  */

	  /*   z15= svmad_x(p0,z03,z07,z15);  */
	  /*   z16= svmad_x(p0,z03,z08,z16);  */
	  /*   z17= svmad_x(p0,z03,z09,z17);  */
	  /*   z18= svmad_x(p0,z03,z10,z18);  */
	  /*   z19= svmad_x(p0,z03,z11,z19);  */
	  /*   z20= svmad_x(p0,z03,z12,z20);  */
	  /*   z21= svmad_x(p0,z03,z13,z21);  */
	  /*   z22= svmad_x(p0,z03,z14,z22); */
	  
	  /*   // s */

	  /*   x30 = s; */
	  /*   z31 = svadd_x(p1,z30,x30); */
	  /*   x10 = svandv(p1000000000000000,z31); */
	  /*   x11 = svandv(p0100000000000000,z31); */
	  /*   x12 = svandv(p0010000000000000,z31); */
	  /*   x13 = svandv(p0001000000000000,z31); */
	  /*   x14 = svandv(p0000100000000000,z31); */
	  /*   x15 = svandv(p0000010000000000,z31); */
	  /*   x16 = svandv(p0000001000000000,z31); */
	  /*   x17 = svandv(p0000000100000000,z31); */

	  /*   z07 = svld1(p0,(float64_t*)&f1_t[x10]);  */
	  /*   z08 = svld1(p0,(float64_t*)&f1_t[x11]);  */
	  /*   z09 = svld1(p0,(float64_t*)&f1_t[x12]);  */
	  /*   z10 = svld1(p0,(float64_t*)&f1_t[x13]);  */
	  /*   z11 = svld1(p0,(float64_t*)&f1_t[x14]);  */
	  /*   z12 = svld1(p0,(float64_t*)&f1_t[x15]);  */
	  /*   z13 = svld1(p0,(float64_t*)&f1_t[x16]);  */
	  /*   z14 = svld1(p0,(float64_t*)&f1_t[x17]);  */

	  /*   z15= svmad_x(p0,z04,z07,z15);  */
	  /*   z16= svmad_x(p0,z04,z08,z16);  */
	  /*   z17= svmad_x(p0,z04,z09,z17);  */
	  /*   z18= svmad_x(p0,z04,z10,z18);  */
	  /*   z19= svmad_x(p0,z04,z11,z19);  */
	  /*   z20= svmad_x(p0,z04,z12,z20);  */
	  /*   z21= svmad_x(p0,z04,z13,z21);  */
	  /*   z22= svmad_x(p0,z04,z14,z22); */

	  /*   // b */

	  /*   x30 = b; */
	  /*   z31 = svadd_x(p1,z30,x30); */
	  /*   x10 = svandv(p1000000000000000,z31); */
	  /*   x11 = svandv(p0100000000000000,z31); */
	  /*   x12 = svandv(p0010000000000000,z31); */
	  /*   x13 = svandv(p0001000000000000,z31); */
	  /*   x14 = svandv(p0000100000000000,z31); */
	  /*   x15 = svandv(p0000010000000000,z31); */
	  /*   x16 = svandv(p0000001000000000,z31); */
	  /*   x17 = svandv(p0000000100000000,z31); */

	  /*   z07 = svld1(p0,(float64_t*)&f1_t[x10]);  */
	  /*   z08 = svld1(p0,(float64_t*)&f1_t[x11]);  */
	  /*   z09 = svld1(p0,(float64_t*)&f1_t[x12]);  */
	  /*   z10 = svld1(p0,(float64_t*)&f1_t[x13]);  */
	  /*   z11 = svld1(p0,(float64_t*)&f1_t[x14]);  */
	  /*   z12 = svld1(p0,(float64_t*)&f1_t[x15]);  */
	  /*   z13 = svld1(p0,(float64_t*)&f1_t[x16]);  */
	  /*   z14 = svld1(p0,(float64_t*)&f1_t[x17]);  */

	  /*   z15= svmad_x(p0,z05,z07,z15);  */
	  /*   z16= svmad_x(p0,z05,z08,z16);  */
	  /*   z17= svmad_x(p0,z05,z09,z17);  */
	  /*   z18= svmad_x(p0,z05,z10,z18);  */
	  /*   z19= svmad_x(p0,z05,z11,z19);  */
	  /*   z20= svmad_x(p0,z05,z12,z20);  */
	  /*   z21= svmad_x(p0,z05,z13,z21);  */
	  /*   z22= svmad_x(p0,z05,z14,z22); */

	  /*   // t */

	  /*   x30 = t; */
	  /*   z31 = svadd_x(p1,z30,x30); */
	  /*   x10 = svandv(p1000000000000000,z31); */
	  /*   x11 = svandv(p0100000000000000,z31); */
	  /*   x12 = svandv(p0010000000000000,z31); */
	  /*   x13 = svandv(p0001000000000000,z31); */
	  /*   x14 = svandv(p0000100000000000,z31); */
	  /*   x15 = svandv(p0000010000000000,z31); */
	  /*   x16 = svandv(p0000001000000000,z31); */
	  /*   x17 = svandv(p0000000100000000,z31); */

	  /*   z07 = svld1(p0,(float64_t*)&f1_t[x10]);  */
	  /*   z08 = svld1(p0,(float64_t*)&f1_t[x11]);  */
	  /*   z09 = svld1(p0,(float64_t*)&f1_t[x12]);  */
	  /*   z10 = svld1(p0,(float64_t*)&f1_t[x13]);  */
	  /*   z11 = svld1(p0,(float64_t*)&f1_t[x14]);  */
	  /*   z12 = svld1(p0,(float64_t*)&f1_t[x15]);  */
	  /*   z13 = svld1(p0,(float64_t*)&f1_t[x16]);  */
	  /*   z14 = svld1(p0,(float64_t*)&f1_t[x17]);  */

	  /*   z15= svmad_x(p0,z06,z07,z15);  */
	  /*   z16= svmad_x(p0,z06,z08,z16);  */
	  /*   z17= svmad_x(p0,z06,z09,z17);  */
	  /*   z18= svmad_x(p0,z06,z10,z18);  */
	  /*   z19= svmad_x(p0,z06,z11,z19);  */
	  /*   z20= svmad_x(p0,z06,z12,z20);  */
	  /*   z21= svmad_x(p0,z06,z13,z21);  */
	  /*   z22= svmad_x(p0,z06,z14,z22); */
	  
	  /*   svst1(p0,(float64_t*)&f2_t[x0],z15); */
	  /*   svst1(p0,(float64_t*)&f2_t[x1],z16); */
	  /*   svst1(p0,(float64_t*)&f2_t[x2],z17); */
	  /*   svst1(p0,(float64_t*)&f2_t[x3],z18); */
	  /*   svst1(p0,(float64_t*)&f2_t[x4],z19); */
	  /*   svst1(p0,(float64_t*)&f2_t[x5],z20); */
	  /*   svst1(p0,(float64_t*)&f2_t[x6],z21); */
	  /*   svst1(p0,(float64_t*)&f2_t[x7],z22); */


	  /* } */

	  /* // c */

	  /* x30 = 64; */
	  /* z31 = svadd_x(p1,z30,x30); */
	  /* z30 = z31; */
	  /* x10 = svandv(p1000000000000000,z31); */
	  /* x11 = svandv(p0100000000000000,z31); */
	  /* x12 = svandv(p0010000000000000,z31); */
	  /* x13 = svandv(p0001000000000000,z31); */
	  /* x14 = svandv(p0000100000000000,z31); */
	  /* x15 = svandv(p0000010000000000,z31); */
	  /* x16 = svandv(p0000001000000000,z31); */
	  /* x17 = svandv(p0000000100000000,z31); */
	    	  
	  /* z07 = svld1(p0,(float64_t*)&f1_t[x10]);  */
	  /* z08 = svld1(p0,(float64_t*)&f1_t[x11]);  */
	  /* z09 = svld1(p0,(float64_t*)&f1_t[x12]);  */
	  /* z10 = svld1(p0,(float64_t*)&f1_t[x13]);  */
	  /* z11 = svld1(p0,(float64_t*)&f1_t[x14]);  */
	  /* z12 = svld1(p0,(float64_t*)&f1_t[x15]);  */
	  /* z13 = svld1(p0,(float64_t*)&f1_t[x16]);  */
	  /* z14 = svld1(p0,(float64_t*)&f1_t[x17]);  */

	  /* z15= svmul_x(p0,z00,z07);  */
	  /* z16= svmul_x(p0,z00,z08);  */
	  /* z17= svmul_x(p0,z00,z09);  */
	  /* z18= svmul_x(p0,z00,z10);  */
	  /* z19= svmul_x(p0,z00,z11);  */
	  /* z20= svmul_x(p0,z00,z12);  */
	  /* z21= svmul_x(p0,z00,z13);  */
	  /* z22= svmul_x(p0,z00,z14); */

	  /* // w */
	  /* x30 = -1; */
	  /* z31 = svadd_x(p1,z30,x30); */
	  /* x10 = svandv(p1000000000000000,z31); */
	  /* x11 = svandv(p0100000000000000,z31); */
	  /* x12 = svandv(p0010000000000000,z31); */
	  /* x13 = svandv(p0001000000000000,z31); */
	  /* x14 = svandv(p0000100000000000,z31); */
	  /* x15 = svandv(p0000010000000000,z31); */
	  /* x16 = svandv(p0000001000000000,z31); */
	  /* x17 = svandv(p0000000100000000,z31); */
	  
	  /* z07 = svld1(p0,(float64_t*)&f1_t[x10]);  */
	  /* z08 = svld1(p0,(float64_t*)&f1_t[x11]);  */
	  /* z09 = svld1(p0,(float64_t*)&f1_t[x12]);  */
	  /* z10 = svld1(p0,(float64_t*)&f1_t[x13]);  */
	  /* z11 = svld1(p0,(float64_t*)&f1_t[x14]);  */
	  /* z12 = svld1(p0,(float64_t*)&f1_t[x15]);  */
	  /* z13 = svld1(p0,(float64_t*)&f1_t[x16]);  */
	  /* z14 = svld1(p0,(float64_t*)&f1_t[x17]);  */

	  /* z15= svmad_x(p0,z01,z07,z15);  */
	  /* z16= svmad_x(p0,z01,z08,z16);  */
	  /* z17= svmad_x(p0,z01,z09,z17);  */
	  /* z18= svmad_x(p0,z01,z10,z18);  */
	  /* z19= svmad_x(p0,z01,z11,z19);  */
	  /* z20= svmad_x(p0,z01,z12,z20);  */
	  /* z21= svmad_x(p0,z01,z13,z21);  */
	  /* z22= svmad_x(p0,z01,z14,z22); */

	  /* // e */

	  /* x30 = 1; */
	  /* z31 = svadd_x(p1,z30,x30); */
	  /* x10 = svandv(p1000000000000000,z31); */
	  /* x11 = svandv(p0100000000000000,z31); */
	  /* x12 = svandv(p0010000000000000,z31); */
	  /* x13 = svandv(p0001000000000000,z31); */
	  /* x14 = svandv(p0000100000000000,z31); */
	  /* x15 = svandv(p0000010000000000,z31); */
	  /* x16 = svandv(p0000001000000000,z31); */
	  /* x17 = svandv(p0000000100000000,z31); */

	  /* float64_t fcp1_arr[8] = {f1_t[x17],f1_t[x17+1],f1_t[x17+2],f1_t[x17+3],f1_t[x17+4],f1_t[x17+5],f1_t[x17+6],f1_t[x17+6]}; */
	  /* z07 = svld1(p0,(float64_t*)&f1_t[x10]);  */
	  /* z08 = svld1(p0,(float64_t*)&f1_t[x11]);  */
	  /* z09 = svld1(p0,(float64_t*)&f1_t[x12]);  */
	  /* z10 = svld1(p0,(float64_t*)&f1_t[x13]);  */
	  /* z11 = svld1(p0,(float64_t*)&f1_t[x14]);  */
	  /* z12 = svld1(p0,(float64_t*)&f1_t[x15]);  */
	  /* z13 = svld1(p0,(float64_t*)&f1_t[x16]);  */
	  /* z14 = svld1(p0,(float64_t*)&fcp1_arr[0]); */

	  /* z15= svmad_x(p0,z02,z07,z15);  */
	  /* z16= svmad_x(p0,z02,z08,z16);  */
	  /* z17= svmad_x(p0,z02,z09,z17);  */
	  /* z18= svmad_x(p0,z02,z10,z18);  */
	  /* z19= svmad_x(p0,z02,z11,z19);  */
	  /* z20= svmad_x(p0,z02,z12,z20);  */
	  /* z21= svmad_x(p0,z02,z13,z21);  */
	  /* z22= svmad_x(p0,z02,z14,z22); */

	  /* // n */
	  
	  /* x30 = n; */
	  /* z31 = svadd_x(p1,z30,x30); */
	  /* x10 = svandv(p1000000000000000,z31); */
	  /* x11 = svandv(p0100000000000000,z31); */
	  /* x12 = svandv(p0010000000000000,z31); */
	  /* x13 = svandv(p0001000000000000,z31); */
	  /* x14 = svandv(p0000100000000000,z31); */
	  /* x15 = svandv(p0000010000000000,z31); */
	  /* x16 = svandv(p0000001000000000,z31); */
	  /* x17 = svandv(p0000000100000000,z31); */

	  /* z07 = svld1(p0,(float64_t*)&f1_t[x10]);  */
	  /* z08 = svld1(p0,(float64_t*)&f1_t[x11]);  */
	  /* z09 = svld1(p0,(float64_t*)&f1_t[x12]);  */
	  /* z10 = svld1(p0,(float64_t*)&f1_t[x13]);  */
	  /* z11 = svld1(p0,(float64_t*)&f1_t[x14]);  */
	  /* z12 = svld1(p0,(float64_t*)&f1_t[x15]);  */
	  /* z13 = svld1(p0,(float64_t*)&f1_t[x16]);  */
	  /* z14 = svld1(p0,(float64_t*)&f1_t[x17]);  */

	  /* z15= svmad_x(p0,z03,z07,z15);  */
	  /* z16= svmad_x(p0,z03,z08,z16);  */
	  /* z17= svmad_x(p0,z03,z09,z17);  */
	  /* z18= svmad_x(p0,z03,z10,z18);  */
	  /* z19= svmad_x(p0,z03,z11,z19);  */
	  /* z20= svmad_x(p0,z03,z12,z20);  */
	  /* z21= svmad_x(p0,z03,z13,z21);  */
	  /* z22= svmad_x(p0,z03,z14,z22); */
	  
	  /* // s */

	  /* x30 = s; */
	  /* z31 = svadd_x(p1,z30,x30); */
	  /* x10 = svandv(p1000000000000000,z31); */
	  /* x11 = svandv(p0100000000000000,z31); */
	  /* x12 = svandv(p0010000000000000,z31); */
	  /* x13 = svandv(p0001000000000000,z31); */
	  /* x14 = svandv(p0000100000000000,z31); */
	  /* x15 = svandv(p0000010000000000,z31); */
	  /* x16 = svandv(p0000001000000000,z31); */
	  /* x17 = svandv(p0000000100000000,z31); */

	  /* z07 = svld1(p0,(float64_t*)&f1_t[x10]);  */
	  /* z08 = svld1(p0,(float64_t*)&f1_t[x11]);  */
	  /* z09 = svld1(p0,(float64_t*)&f1_t[x12]);  */
	  /* z10 = svld1(p0,(float64_t*)&f1_t[x13]);  */
	  /* z11 = svld1(p0,(float64_t*)&f1_t[x14]);  */
	  /* z12 = svld1(p0,(float64_t*)&f1_t[x15]);  */
	  /* z13 = svld1(p0,(float64_t*)&f1_t[x16]);  */
	  /* z14 = svld1(p0,(float64_t*)&f1_t[x17]);  */

	  /* z15= svmad_x(p0,z04,z07,z15);  */
	  /* z16= svmad_x(p0,z04,z08,z16);  */
	  /* z17= svmad_x(p0,z04,z09,z17);  */
	  /* z18= svmad_x(p0,z04,z10,z18);  */
	  /* z19= svmad_x(p0,z04,z11,z19);  */
	  /* z20= svmad_x(p0,z04,z12,z20);  */
	  /* z21= svmad_x(p0,z04,z13,z21);  */
	  /* z22= svmad_x(p0,z04,z14,z22); */

	  /* // b */

	  /* x30 = b; */
	  /* z31 = svadd_x(p1,z30,x30); */
	  /* x10 = svandv(p1000000000000000,z31); */
	  /* x11 = svandv(p0100000000000000,z31); */
	  /* x12 = svandv(p0010000000000000,z31); */
	  /* x13 = svandv(p0001000000000000,z31); */
	  /* x14 = svandv(p0000100000000000,z31); */
	  /* x15 = svandv(p0000010000000000,z31); */
	  /* x16 = svandv(p0000001000000000,z31); */
	  /* x17 = svandv(p0000000100000000,z31); */

	  /* z07 = svld1(p0,(float64_t*)&f1_t[x10]);  */
	  /* z08 = svld1(p0,(float64_t*)&f1_t[x11]);  */
	  /* z09 = svld1(p0,(float64_t*)&f1_t[x12]);  */
	  /* z10 = svld1(p0,(float64_t*)&f1_t[x13]);  */
	  /* z11 = svld1(p0,(float64_t*)&f1_t[x14]);  */
	  /* z12 = svld1(p0,(float64_t*)&f1_t[x15]);  */
	  /* z13 = svld1(p0,(float64_t*)&f1_t[x16]);  */
	  /* z14 = svld1(p0,(float64_t*)&f1_t[x17]);  */

	  /* z15= svmad_x(p0,z05,z07,z15);  */
	  /* z16= svmad_x(p0,z05,z08,z16);  */
	  /* z17= svmad_x(p0,z05,z09,z17);  */
	  /* z18= svmad_x(p0,z05,z10,z18);  */
	  /* z19= svmad_x(p0,z05,z11,z19);  */
	  /* z20= svmad_x(p0,z05,z12,z20);  */
	  /* z21= svmad_x(p0,z05,z13,z21);  */
	  /* z22= svmad_x(p0,z05,z14,z22); */

	  /* // t */

	  /* x30 = t; */
	  /* z31 = svadd_x(p1,z30,x30); */
	  /* x10 = svandv(p1000000000000000,z31); */
	  /* x11 = svandv(p0100000000000000,z31); */
	  /* x12 = svandv(p0010000000000000,z31); */
	  /* x13 = svandv(p0001000000000000,z31); */
	  /* x14 = svandv(p0000100000000000,z31); */
	  /* x15 = svandv(p0000010000000000,z31); */
	  /* x16 = svandv(p0000001000000000,z31); */
	  /* x17 = svandv(p0000000100000000,z31); */

	  /* z07 = svld1(p0,(float64_t*)&f1_t[x10]);  */
	  /* z08 = svld1(p0,(float64_t*)&f1_t[x11]);  */
	  /* z09 = svld1(p0,(float64_t*)&f1_t[x12]);  */
	  /* z10 = svld1(p0,(float64_t*)&f1_t[x13]);  */
	  /* z11 = svld1(p0,(float64_t*)&f1_t[x14]);  */
	  /* z12 = svld1(p0,(float64_t*)&f1_t[x15]);  */
	  /* z13 = svld1(p0,(float64_t*)&f1_t[x16]);  */
	  /* z14 = svld1(p0,(float64_t*)&f1_t[x17]);  */

	  /* z15= svmad_x(p0,z06,z07,z15);  */
	  /* z16= svmad_x(p0,z06,z08,z16);  */
	  /* z17= svmad_x(p0,z06,z09,z17);  */
	  /* z18= svmad_x(p0,z06,z10,z18);  */
	  /* z19= svmad_x(p0,z06,z11,z19);  */
	  /* z20= svmad_x(p0,z06,z12,z20);  */
	  /* z21= svmad_x(p0,z06,z13,z21);  */
	  /* z22= svmad_x(p0,z06,z14,z22);  */
	  
	  /* svst1(p0,(float64_t*)&f2_t[x0],z15); */
	  /* svst1(p0,(float64_t*)&f2_t[x1],z16); */
	  /* svst1(p0,(float64_t*)&f2_t[x2],z17); */
	  /* svst1(p0,(float64_t*)&f2_t[x3],z18); */
	  /* svst1(p0,(float64_t*)&f2_t[x4],z19); */
	  /* svst1(p0,(float64_t*)&f2_t[x5],z20); */
	  /* svst1(p0,(float64_t*)&f2_t[x6],z21); */
	  /* svst1(p0,(float64_t*)&f2_t[x7],z22); */

	  
#endif
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
  

