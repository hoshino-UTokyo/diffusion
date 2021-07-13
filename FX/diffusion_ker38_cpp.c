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


void allocate_ker38(double **buff_ret, const int nx, const int ny, const int nz) {

  posix_memalign((void**)buff_ret, 64, sizeof(double)*nx*ny*nz);

}

void init_ker38(double *buff1, const int nx, const int ny, const int nz,
  const double kx, const double ky, const double kz,
  const double dx, const double dy, const double dz,
  const double kappa, const double time) {

  double ax, ay, az;
  int jz, jy, jx;
  ax = exp(-kappa*time*(kx*kx));
  ay = exp(-kappa*time*(ky*ky));
  az = exp(-kappa*time*(kz*kz));
#pragma omp parallel private(jx,jy,jz)
  {







    int xx,yy;
    int tid = omp_get_thread_num();
    int nth = omp_get_num_threads();
    int ty = tid;
    int ychunk = 8*nth;
    int yystr = ty*8;






    for (yy = yystr; yy < ny; yy+= ychunk) {
      for (jz = 0; jz < nz; jz++) {
 for (jy = yy; jy < ((yy+8) < (ny) ? (yy+8) : (ny)); jy++) {
   for (jx = 0; jx < nx; jx++) {
     int j = jz*nx*ny + jy*nx + jx;
     double x = dx*((double)(jx + 0.5));
     double y = dy*((double)(jy + 0.5));
     double z = dz*((double)(jz + 0.5));
     double f0 = (double)0.125
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


void diffusion_ker38(double *restrict f1, double *restrict f2, int nx, int ny, int nz,
       double ce, double cw, double cn, double cs, double ct,
       double cb, double cc, double dt,
       double **f1_ret, double **f2_ret, double *time_ret, int *count_ret) {

#pragma omp parallel
  {
    double time = 0.0;
    int count = 0;
    double *restrict f1_t = f1;
    double *restrict f2_t = f2;
    int c, w, e, n, s, b, t,b0, t0, n0, s0;
    int z, y, x, xx, yy, zz, h;
    int izm,izc,izp,iy,ix,tmp,halo;
    int step;
    int xstr,xend,ystr,yend;
    int id2,id1;

    int tid = omp_get_thread_num();
    int nth = omp_get_num_threads();
    int ty = tid;
    int ychunk = 8*nth;
    int yystr = ty*8;

    double *temporal;
    temporal = (double*)malloc(sizeof(double)*(3 -1)*(2*1 +1)*(8 +2*(1 +3 -2))*nx);
    int tbx = nx;
    int tby = (8 +2*(1 +3 -2));
    int tbz = (2*1 +1);

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

# 1 "ker38.inc" 1

{
  izm = 0;
  izc = 1;
  izp = 2;

  for (h = 0; h < (1 +3 -2); h++){
    z = h;
    step = 0;

    halo = (1 +3 -2)-step;
    b0 = (z == 0) ? 0 : - nx * ny;
    t0 = (z == nz-1) ? 0 : nx * ny;
    n0 = -nx;
    s0 = nx;


    ystr = 0;
    iy = (1 +3 -2);







    yend = yy+8 +halo;


    id2 = 0+iy*nx+izp*nx*tby+step*nx*tby*tbz;

    for (y = ystr; y < yend; y++) {

      n0 = (y == 0) ? 0 : - nx;




      x = 0;
      {
 c = x + y*nx + z*nx*ny;
 float64_t fcm1_arr[8] = {f1_t[c+x],f1_t[c+x],f1_t[c+x+1],f1_t[c+x+2],f1_t[c+x+3],f1_t[c+x+4],f1_t[c+x+5],f1_t[c+x+6]};
 svfloat64_t fc_vec = svld1(pg,(float64_t*)&f1_t[c+x]);
 svfloat64_t fce_vec = svld1(pg,(float64_t*)&f1_t[c+x+1]);
 svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
 svfloat64_t fcs_vec = svld1(pg,(float64_t*)&f1_t[c+x+s0]);
 svfloat64_t fcn_vec = svld1(pg,(float64_t*)&f1_t[c+x+n0]);
 svfloat64_t fcb_vec = svld1(pg,(float64_t*)&f1_t[c+x+b0]);
 svfloat64_t fct_vec = svld1(pg,(float64_t*)&f1_t[c+x+t0]);
 svfloat64_t tmp0,tmp1,tmp2;
 fc_vec = svmul_x(pg,cc_vec,fc_vec);
 fce_vec = svmul_x(pg,ce_vec,fce_vec);
 fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
 fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
 fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
 fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
 fct_vec = svmul_x(pg,ct_vec,fct_vec);
 tmp0 = svadd_x(pg,fce_vec,fcw_vec);
 tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
 tmp2 = svadd_x(pg,fct_vec,fcb_vec);
 tmp0 = svadd_x(pg,fc_vec, tmp0);
 tmp1 = svadd_x(pg,tmp1, tmp2);
 tmp0 = svadd_x(pg,tmp0, tmp1);
 svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
      }
      for (x = 8; x < nx-8; x+=8) {
       svfloat64_t fc_vec = svld1(pg,(float64_t*)&f1_t[c+x]);
       svfloat64_t fce_vec = svld1(pg,(float64_t*)&f1_t[c+x+1]);
       svfloat64_t fcw_vec = svld1(pg,(float64_t*)&f1_t[c+x-1]);
       svfloat64_t fcs_vec = svld1(pg,(float64_t*)&f1_t[c+x+s0]);
       svfloat64_t fcn_vec = svld1(pg,(float64_t*)&f1_t[c+x+n0]);
       svfloat64_t fcb_vec = svld1(pg,(float64_t*)&f1_t[c+x+b0]);
       svfloat64_t fct_vec = svld1(pg,(float64_t*)&f1_t[c+x+t0]);
       svfloat64_t tmp0,tmp1,tmp2;
       fc_vec = svmul_x(pg,cc_vec,fc_vec);
       fce_vec = svmul_x(pg,ce_vec,fce_vec);
       fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
       fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
       fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
       fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
       fct_vec = svmul_x(pg,ct_vec,fct_vec);
       tmp0 = svadd_x(pg,fce_vec,fcw_vec);
       tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
       tmp2 = svadd_x(pg,fct_vec,fcb_vec);
       tmp0 = svadd_x(pg,fc_vec, tmp0);
       tmp1 = svadd_x(pg,tmp1, tmp2);
       tmp0 = svadd_x(pg,tmp0, tmp1);
       svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
      }
      int i = 0;

      int remainder = 8;
      svbool_t pgt = svwhilelt_b64(i,remainder);
      {
 float64_t fcp1_arr[8];
 for(int i = 0; i < remainder; i++){
   if(i == remainder-1){
     fcp1_arr[i] = f1_t[c+x+i];
   }else{
     fcp1_arr[i] = f1_t[c+x+1+i];
   }
 }
 svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
       svfloat64_t fc_vec = svld1(pgt,(float64_t*)&f1_t[c+x]);
       svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&f1_t[c+x-1]);
       svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&f1_t[c+x+s0]);
       svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&f1_t[c+x+n0]);
       svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&f1_t[c+x+b0]);
       svfloat64_t fct_vec = svld1(pgt,(float64_t*)&f1_t[c+x+t0]);
 svfloat64_t tmp0,tmp1,tmp2;
 fc_vec = svmul_x(pgt,cc_vec,fc_vec);
 fce_vec = svmul_x(pgt,ce_vec,fce_vec);
 fcw_vec = svmul_x(pgt,cw_vec,fcw_vec);
 fcn_vec = svmul_x(pgt,cn_vec,fcn_vec);
 fcs_vec = svmul_x(pgt,cs_vec,fcs_vec);
 fcb_vec = svmul_x(pgt,cb_vec,fcb_vec);
 fct_vec = svmul_x(pgt,ct_vec,fct_vec);
 tmp0 = svadd_x(pgt,fce_vec,fcw_vec);
 tmp1 = svadd_x(pgt,fcn_vec,fcs_vec);
 tmp2 = svadd_x(pgt,fct_vec,fcb_vec);
 tmp0 = svadd_x(pgt,fc_vec, tmp0);
 tmp1 = svadd_x(pgt,tmp1, tmp2);
 tmp0 = svadd_x(pgt,tmp0, tmp1);
 svst1(pgt,(float64_t*)&temporal[id2+x],tmp0);
      }
      id2 += nx;
    }

    for(z = h-1; z >= 0; z--){
      b = (z == 0) ? 0 : (izm-izc)*nx*tby;
      t = (z == nz-1) ? 0 : (izp-izc)*nx*tby;
      n = -nx;
      s = nx;

      halo = (1 +3 -2)-(step+1);

      ystr = 0;
      iy = (1 +3 -2);







      yend = yy+8 +halo;


      id2 = 0+iy*nx+izp*nx*tby+(step+1)*nx*tby*tbz;
      id1 = 0+iy*nx+izc*nx*tby+step*nx*tby*tbz;

      for (y = ystr; y < yend; y++) {

 n = (y == 0) ? 0 : -nx;




 x = 0;
 {
   float64_t fcm1_arr[8] = {temporal[id1+x],temporal[id1+x],temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6]};
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
 }
 for (x = 8; x < nx-8; x+=8) {
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
 }
 x = nx-8;
 int i = 0;

 int remainder = 8;
 svbool_t pgt = svwhilelt_b64(i,remainder);
 {

   float64_t fcp1_arr[8];
   for(int i = 0; i < remainder; i++){
     if(i == remainder-1){
       fcp1_arr[i] = temporal[id1+x+i];
     }else{
       fcp1_arr[i] = temporal[id1+x+1+i];
     }
   }
   svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
   svfloat64_t fc_vec = svld1(pgt,(float64_t*)&temporal[id1+x]);
   svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pgt,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pgt,cc_vec,fc_vec);
   fce_vec = svmul_x(pgt,ce_vec,fce_vec);
   fcw_vec = svmul_x(pgt,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pgt,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pgt,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pgt,cb_vec,fcb_vec);
   fct_vec = svmul_x(pgt,ct_vec,fct_vec);
   tmp0 = svadd_x(pgt,fce_vec,fcw_vec);
   tmp1 = svadd_x(pgt,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pgt,fct_vec,fcb_vec);
   tmp0 = svadd_x(pgt,fc_vec, tmp0);
   tmp1 = svadd_x(pgt,tmp1, tmp2);
   tmp0 = svadd_x(pgt,tmp0, tmp1);
   svst1(pgt,(float64_t*)&temporal[id2+x],tmp0);
 }
 id2 += nx;
 id1 += nx;
      }
      step++;
    }
    tmp = izm;
    izm = izc;
    izc = izp;
    izp = tmp;

  }

  for (zz = 0; zz < nz-(1 +3 -2); zz++) {
    z = zz+(1 +3 -2);
    b0 = (z == 0) ? 0 : - nx * ny;
    t0 = (z == nz-1) ? 0 : nx * ny;
    n0 = -nx;
    s0 = nx;

    step = 0;
    halo = (1 +3 -2)-step;


    ystr = 0;
    iy = (1 +3 -2);







    yend = yy+8 +halo;

    id2 = 0+iy*nx+izp*nx*tby+step*nx*tby*tbz;

    for (y = ystr; y < yend; y++) {

      n0 = (y == 0) ? 0 : - nx;




      x = 0;
      {
 c = x + y*nx + z*nx*ny;
 float64_t fcm1_arr[8] = {f1_t[c+x],f1_t[c+x],f1_t[c+x+1],f1_t[c+x+2],f1_t[c+x+3],f1_t[c+x+4],f1_t[c+x+5],f1_t[c+x+6]};
 svfloat64_t fc_vec = svld1(pg,(float64_t*)&f1_t[c+x]);
 svfloat64_t fce_vec = svld1(pg,(float64_t*)&f1_t[c+x+1]);
 svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
 svfloat64_t fcs_vec = svld1(pg,(float64_t*)&f1_t[c+x+s0]);
 svfloat64_t fcn_vec = svld1(pg,(float64_t*)&f1_t[c+x+n0]);
 svfloat64_t fcb_vec = svld1(pg,(float64_t*)&f1_t[c+x+b0]);
 svfloat64_t fct_vec = svld1(pg,(float64_t*)&f1_t[c+x+t0]);
 svfloat64_t tmp0,tmp1,tmp2;
 fc_vec = svmul_x(pg,cc_vec,fc_vec);
 fce_vec = svmul_x(pg,ce_vec,fce_vec);
 fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
 fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
 fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
 fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
 fct_vec = svmul_x(pg,ct_vec,fct_vec);
 tmp0 = svadd_x(pg,fce_vec,fcw_vec);
 tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
 tmp2 = svadd_x(pg,fct_vec,fcb_vec);
 tmp0 = svadd_x(pg,fc_vec, tmp0);
 tmp1 = svadd_x(pg,tmp1, tmp2);
 tmp0 = svadd_x(pg,tmp0, tmp1);
 svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
      }
      for (x = 8; x < nx-8; x+=8) {
       svfloat64_t fc_vec = svld1(pg,(float64_t*)&f1_t[c+x]);
       svfloat64_t fce_vec = svld1(pg,(float64_t*)&f1_t[c+x+1]);
       svfloat64_t fcw_vec = svld1(pg,(float64_t*)&f1_t[c+x-1]);
       svfloat64_t fcs_vec = svld1(pg,(float64_t*)&f1_t[c+x+s0]);
       svfloat64_t fcn_vec = svld1(pg,(float64_t*)&f1_t[c+x+n0]);
       svfloat64_t fcb_vec = svld1(pg,(float64_t*)&f1_t[c+x+b0]);
       svfloat64_t fct_vec = svld1(pg,(float64_t*)&f1_t[c+x+t0]);
       svfloat64_t tmp0,tmp1,tmp2;
       fc_vec = svmul_x(pg,cc_vec,fc_vec);
       fce_vec = svmul_x(pg,ce_vec,fce_vec);
       fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
       fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
       fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
       fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
       fct_vec = svmul_x(pg,ct_vec,fct_vec);
       tmp0 = svadd_x(pg,fce_vec,fcw_vec);
       tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
       tmp2 = svadd_x(pg,fct_vec,fcb_vec);
       tmp0 = svadd_x(pg,fc_vec, tmp0);
       tmp1 = svadd_x(pg,tmp1, tmp2);
       tmp0 = svadd_x(pg,tmp0, tmp1);
       svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
      }
      x = nx-8;
      int i = 0;
      int remainder = 8;

      svbool_t pgt = svwhilelt_b64(i,remainder);
      {
 float64_t fcp1_arr[8];
 for(int i = 0; i < remainder; i++){
   if(i == remainder-1){
     fcp1_arr[i] = f1_t[c+x+i];
   }else{
     fcp1_arr[i] = f1_t[c+x+1+i];
   }
 }
 svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
       svfloat64_t fc_vec = svld1(pgt,(float64_t*)&f1_t[c+x]);
       svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&f1_t[c+x-1]);
       svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&f1_t[c+x+s0]);
       svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&f1_t[c+x+n0]);
       svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&f1_t[c+x+b0]);
       svfloat64_t fct_vec = svld1(pgt,(float64_t*)&f1_t[c+x+t0]);
 svfloat64_t tmp0,tmp1,tmp2;
 fc_vec = svmul_x(pgt,cc_vec,fc_vec);
 fce_vec = svmul_x(pgt,ce_vec,fce_vec);
 fcw_vec = svmul_x(pgt,cw_vec,fcw_vec);
 fcn_vec = svmul_x(pgt,cn_vec,fcn_vec);
 fcs_vec = svmul_x(pgt,cs_vec,fcs_vec);
 fcb_vec = svmul_x(pgt,cb_vec,fcb_vec);
 fct_vec = svmul_x(pgt,ct_vec,fct_vec);
 tmp0 = svadd_x(pgt,fce_vec,fcw_vec);
 tmp1 = svadd_x(pgt,fcn_vec,fcs_vec);
 tmp2 = svadd_x(pgt,fct_vec,fcb_vec);
 tmp0 = svadd_x(pgt,fc_vec, tmp0);
 tmp1 = svadd_x(pgt,tmp1, tmp2);
 tmp0 = svadd_x(pgt,tmp0, tmp1);
 svst1(pgt,(float64_t*)&temporal[id2+x],tmp0);
      }
      id2 += nx;
    }

    for(z = zz+(1 +3 -2)-1; z >= zz+1; z--){
      b = (z == 0) ? 0 : (izm-izc)*nx*tby;
      t = (z == nz-1) ? 0 : (izp-izc)*nx*tby;
      n = -nx;
      s = nx;

      halo = (1 +3 -2)-(step+1);

      ystr = 0;
      iy = (1 +3 -2);







      yend = yy+8 +halo;


      id2 = 0+iy*nx+izp*nx*tby+(step+1)*nx*tby*tbz;
      id1 = 0+iy*nx+izc*nx*tby+step*nx*tby*tbz;

      for (y = ystr; y < yend; y++) {

 n = (y == 0) ? 0 : -nx;




 x = 0;
 {
   float64_t fcm1_arr[8] = {temporal[id1+x],temporal[id1+x],temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6]};
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&temporal[id2+x],tmp0);


 }

 for (x = 8; x < nx-8; x+=8) {
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&temporal[id2+x],tmp0);


 }
 x = nx-8;
 int remainder = 8;
 int i = 0;

 svbool_t pgt = svwhilelt_b64(i,remainder);
 {
   float64_t fcp1_arr[8];
   for(int i = 0; i < remainder; i++){
     if(i == remainder-1){
       fcp1_arr[i] = temporal[id1+x+i];
     }else{
       fcp1_arr[i] = temporal[id1+x+1+i];
     }
   }
   svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
   svfloat64_t fc_vec = svld1(pgt,(float64_t*)&temporal[id1+x]);
   svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pgt,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pgt,cc_vec,fc_vec);
   fce_vec = svmul_x(pgt,ce_vec,fce_vec);
   fcw_vec = svmul_x(pgt,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pgt,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pgt,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pgt,cb_vec,fcb_vec);
   fct_vec = svmul_x(pgt,ct_vec,fct_vec);
   tmp0 = svadd_x(pgt,fce_vec,fcw_vec);
   tmp1 = svadd_x(pgt,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pgt,fct_vec,fcb_vec);
   tmp0 = svadd_x(pgt,fc_vec, tmp0);
   tmp1 = svadd_x(pgt,tmp1, tmp2);
   tmp0 = svadd_x(pgt,tmp0, tmp1);
   svst1(pgt,(float64_t*)&temporal[id2+x],tmp0);
 }


 id1 += nx;
 id2 += nx;
      }
      step++;

    }

    {
      z = zz;
      b = (z == 0) ? 0 : (izm-izc)*nx*tby;
      t = (z == nz-1) ? 0 : (izp-izc)*nx*tby;
      n = -nx;
      s = nx;

      halo = (1 +3 -2)-(step+1);

      ystr = 0;
      iy = (1 +3 -2);







      yend = yy+8 +halo;

      id1 = 0+iy*nx+izc*nx*tby+step*nx*tby*tbz;
      for (y = ystr; y < yend; y++) {


 n = (y == 0) ? 0 : -nx;




 x = 0;
 c = y*nx + z*nx*ny;
 {
   float64_t fcm1_arr[8] = {temporal[id1+x],temporal[id1+x],temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6]};
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&f2_t[c+x],tmp0);

 }

 for (x = 8; x < nx-8; x+=8) {
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&f2_t[c+x],tmp0);
 }
 x = nx-8;
 int i = 0;
 int remainder = 8;

 svbool_t pgt = svwhilelt_b64(i,remainder);
 {
   float64_t fcp1_arr[8];
   for(int i = 0; i < remainder; i++){
     if(i == remainder-1){
       fcp1_arr[i] = temporal[id1+x+i];
     }else{
       fcp1_arr[i] = temporal[id1+x+1+i];
     }
   }
   svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
   svfloat64_t fc_vec = svld1(pgt,(float64_t*)&temporal[id1+x]);
   svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pgt,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pgt,cc_vec,fc_vec);
   fce_vec = svmul_x(pgt,ce_vec,fce_vec);
   fcw_vec = svmul_x(pgt,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pgt,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pgt,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pgt,cb_vec,fcb_vec);
   fct_vec = svmul_x(pgt,ct_vec,fct_vec);
   tmp0 = svadd_x(pgt,fce_vec,fcw_vec);
   tmp1 = svadd_x(pgt,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pgt,fct_vec,fcb_vec);
   tmp0 = svadd_x(pgt,fc_vec, tmp0);
   tmp1 = svadd_x(pgt,tmp1, tmp2);
   tmp0 = svadd_x(pgt,tmp0, tmp1);
   svst1(pgt,(float64_t*)&f2_t[c+x],tmp0);

 }

 id1 += nx;
      }
    }
    tmp = izm;
    izm = izc;
    izc = izp;
    izp = tmp;

  }

  for (zz = nz-(1 +3 -2), h = (1 +3 -2)-1; zz < nz; zz++, h--) {

    step = (1 +3 -2)-1-h;

    for(z = zz+h ; z >= zz+1; z--){
      b = (z == 0) ? 0 : (izm-izc)*nx*tby;
      t = (z == nz-1) ? 0 : (izp-izc)*nx*tby;
      n = -nx;
      s = nx;

      halo = (1 +3 -2)-(step+1);

      ystr = 0;
      iy = (1 +3 -2);







      yend = yy+8 +halo;


      id2 = 0+iy*nx+izp*nx*tby+(step+1)*nx*tby*tbz;
      id1 = 0+iy*nx+izc*nx*tby+step*nx*tby*tbz;

      for (y = ystr; y < yend; y++) {

 n = (y == 0) ? 0 : -nx;




 x = 0;
 {
   float64_t fcm1_arr[8] = {temporal[id1+x],temporal[id1+x],temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6]};
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
 }

 for (x = 8; x < nx-8; x+=8) {
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
 }
 x = nx-8;
 int i = 0;
 int remainder = 8;

 svbool_t pgt = svwhilelt_b64(i,remainder);
 {
   float64_t fcp1_arr[8];
   for(int i = 0; i < remainder; i++){
     if(i == remainder-1){
       fcp1_arr[i] = temporal[id1+x+i];
     }else{
       fcp1_arr[i] = temporal[id1+x+1+i];
     }
   }
   svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
   svfloat64_t fc_vec = svld1(pgt,(float64_t*)&temporal[id1+x]);
   svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pgt,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pgt,cc_vec,fc_vec);
   fce_vec = svmul_x(pgt,ce_vec,fce_vec);
   fcw_vec = svmul_x(pgt,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pgt,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pgt,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pgt,cb_vec,fcb_vec);
   fct_vec = svmul_x(pgt,ct_vec,fct_vec);
   tmp0 = svadd_x(pgt,fce_vec,fcw_vec);
   tmp1 = svadd_x(pgt,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pgt,fct_vec,fcb_vec);
   tmp0 = svadd_x(pgt,fc_vec, tmp0);
   tmp1 = svadd_x(pgt,tmp1, tmp2);
   tmp0 = svadd_x(pgt,tmp0, tmp1);
   svst1(pgt,(float64_t*)&temporal[id2+x],tmp0);
 }
 id2 += nx;
 id1 += nx;
      }

      step++;

    }

    {
      z = zz;
      b = (z == 0) ? 0 : (izm-izc)*nx*tby;
      t = (z == nz-1) ? 0 : (izp-izc)*nx*tby;
      n = -nx;
      s = nx;
      halo = (1 +3 -2)-(step+1);


      ystr = 0;
      iy = (1 +3 -2);







      yend = yy+8 +halo;


      id1 = 0+iy*nx+izc*nx*tby+step*nx*tby*tbz;
      for (y = ystr; y < yend; y++) {

 n = (y == 0) ? 0 : -nx;




 x = 0;
 c = x + y*nx + z*nx*ny;
 {
   float64_t fcm1_arr[8] = {temporal[id1+x],temporal[id1+x],temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6]};
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&f2_t[c+x],tmp0);

 }

 for (x = 8; x < nx-8; x+=8) {
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&f2_t[c+x],tmp0);
 }
 x = nx-8;
 int i = 0;
 int remainder = 8;

 svbool_t pgt = svwhilelt_b64(i,remainder);
 {

   float64_t fcp1_arr[8];
   for(int i = 0; i < remainder; i++){
     if(i == remainder-1){
       fcp1_arr[i] = temporal[id1+x+i];
     }else{
       fcp1_arr[i] = temporal[id1+x+1+i];
     }
   }
   svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
   svfloat64_t fc_vec = svld1(pgt,(float64_t*)&temporal[id1+x]);
   svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pgt,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pgt,cc_vec,fc_vec);
   fce_vec = svmul_x(pgt,ce_vec,fce_vec);
   fcw_vec = svmul_x(pgt,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pgt,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pgt,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pgt,cb_vec,fcb_vec);
   fct_vec = svmul_x(pgt,ct_vec,fct_vec);
   tmp0 = svadd_x(pgt,fce_vec,fcw_vec);
   tmp1 = svadd_x(pgt,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pgt,fct_vec,fcb_vec);
   tmp0 = svadd_x(pgt,fc_vec, tmp0);
   tmp1 = svadd_x(pgt,tmp1, tmp2);
   tmp0 = svadd_x(pgt,tmp0, tmp1);
   svst1(pgt,(float64_t*)&f2_t[c+x],tmp0);

 }
 id1 += nx;
      }
    }

    tmp = izm;
    izm = izc;
    izc = izp;
    izp = tmp;
  }
}
# 136 "diffusion_ker38.c" 2

 }else if(yy >= ny-8){

# 1 "ker38.inc" 1

{
  izm = 0;
  izc = 1;
  izp = 2;

  for (h = 0; h < (1 +3 -2); h++){
    z = h;
    step = 0;

    halo = (1 +3 -2)-step;
    b0 = (z == 0) ? 0 : - nx * ny;
    t0 = (z == nz-1) ? 0 : nx * ny;
    n0 = -nx;
    s0 = nx;





    ystr = yy-halo;
    iy = step;


    yend = ny;




    id2 = 0+iy*nx+izp*nx*tby+step*nx*tby*tbz;

    for (y = ystr; y < yend; y++) {




      s0 = (y == ny-1) ? 0 : nx;

      x = 0;
      {
 c = x + y*nx + z*nx*ny;
 float64_t fcm1_arr[8] = {f1_t[c+x],f1_t[c+x],f1_t[c+x+1],f1_t[c+x+2],f1_t[c+x+3],f1_t[c+x+4],f1_t[c+x+5],f1_t[c+x+6]};
 svfloat64_t fc_vec = svld1(pg,(float64_t*)&f1_t[c+x]);
 svfloat64_t fce_vec = svld1(pg,(float64_t*)&f1_t[c+x+1]);
 svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
 svfloat64_t fcs_vec = svld1(pg,(float64_t*)&f1_t[c+x+s0]);
 svfloat64_t fcn_vec = svld1(pg,(float64_t*)&f1_t[c+x+n0]);
 svfloat64_t fcb_vec = svld1(pg,(float64_t*)&f1_t[c+x+b0]);
 svfloat64_t fct_vec = svld1(pg,(float64_t*)&f1_t[c+x+t0]);
 svfloat64_t tmp0,tmp1,tmp2;
 fc_vec = svmul_x(pg,cc_vec,fc_vec);
 fce_vec = svmul_x(pg,ce_vec,fce_vec);
 fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
 fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
 fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
 fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
 fct_vec = svmul_x(pg,ct_vec,fct_vec);
 tmp0 = svadd_x(pg,fce_vec,fcw_vec);
 tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
 tmp2 = svadd_x(pg,fct_vec,fcb_vec);
 tmp0 = svadd_x(pg,fc_vec, tmp0);
 tmp1 = svadd_x(pg,tmp1, tmp2);
 tmp0 = svadd_x(pg,tmp0, tmp1);
 svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
      }
      for (x = 8; x < nx-8; x+=8) {
       svfloat64_t fc_vec = svld1(pg,(float64_t*)&f1_t[c+x]);
       svfloat64_t fce_vec = svld1(pg,(float64_t*)&f1_t[c+x+1]);
       svfloat64_t fcw_vec = svld1(pg,(float64_t*)&f1_t[c+x-1]);
       svfloat64_t fcs_vec = svld1(pg,(float64_t*)&f1_t[c+x+s0]);
       svfloat64_t fcn_vec = svld1(pg,(float64_t*)&f1_t[c+x+n0]);
       svfloat64_t fcb_vec = svld1(pg,(float64_t*)&f1_t[c+x+b0]);
       svfloat64_t fct_vec = svld1(pg,(float64_t*)&f1_t[c+x+t0]);
       svfloat64_t tmp0,tmp1,tmp2;
       fc_vec = svmul_x(pg,cc_vec,fc_vec);
       fce_vec = svmul_x(pg,ce_vec,fce_vec);
       fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
       fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
       fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
       fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
       fct_vec = svmul_x(pg,ct_vec,fct_vec);
       tmp0 = svadd_x(pg,fce_vec,fcw_vec);
       tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
       tmp2 = svadd_x(pg,fct_vec,fcb_vec);
       tmp0 = svadd_x(pg,fc_vec, tmp0);
       tmp1 = svadd_x(pg,tmp1, tmp2);
       tmp0 = svadd_x(pg,tmp0, tmp1);
       svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
      }
      int i = 0;

      int remainder = 8;
      svbool_t pgt = svwhilelt_b64(i,remainder);
      {
 float64_t fcp1_arr[8];
 for(int i = 0; i < remainder; i++){
   if(i == remainder-1){
     fcp1_arr[i] = f1_t[c+x+i];
   }else{
     fcp1_arr[i] = f1_t[c+x+1+i];
   }
 }
 svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
       svfloat64_t fc_vec = svld1(pgt,(float64_t*)&f1_t[c+x]);
       svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&f1_t[c+x-1]);
       svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&f1_t[c+x+s0]);
       svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&f1_t[c+x+n0]);
       svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&f1_t[c+x+b0]);
       svfloat64_t fct_vec = svld1(pgt,(float64_t*)&f1_t[c+x+t0]);
 svfloat64_t tmp0,tmp1,tmp2;
 fc_vec = svmul_x(pgt,cc_vec,fc_vec);
 fce_vec = svmul_x(pgt,ce_vec,fce_vec);
 fcw_vec = svmul_x(pgt,cw_vec,fcw_vec);
 fcn_vec = svmul_x(pgt,cn_vec,fcn_vec);
 fcs_vec = svmul_x(pgt,cs_vec,fcs_vec);
 fcb_vec = svmul_x(pgt,cb_vec,fcb_vec);
 fct_vec = svmul_x(pgt,ct_vec,fct_vec);
 tmp0 = svadd_x(pgt,fce_vec,fcw_vec);
 tmp1 = svadd_x(pgt,fcn_vec,fcs_vec);
 tmp2 = svadd_x(pgt,fct_vec,fcb_vec);
 tmp0 = svadd_x(pgt,fc_vec, tmp0);
 tmp1 = svadd_x(pgt,tmp1, tmp2);
 tmp0 = svadd_x(pgt,tmp0, tmp1);
 svst1(pgt,(float64_t*)&temporal[id2+x],tmp0);
      }
      id2 += nx;
    }

    for(z = h-1; z >= 0; z--){
      b = (z == 0) ? 0 : (izm-izc)*nx*tby;
      t = (z == nz-1) ? 0 : (izp-izc)*nx*tby;
      n = -nx;
      s = nx;

      halo = (1 +3 -2)-(step+1);




      ystr = yy-halo;
      iy = step+1;


      yend = ny;




      id2 = 0+iy*nx+izp*nx*tby+(step+1)*nx*tby*tbz;
      id1 = 0+iy*nx+izc*nx*tby+step*nx*tby*tbz;

      for (y = ystr; y < yend; y++) {




 s = (y == ny-1) ? 0 : nx;

 x = 0;
 {
   float64_t fcm1_arr[8] = {temporal[id1+x],temporal[id1+x],temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6]};
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
 }
 for (x = 8; x < nx-8; x+=8) {
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
 }
 x = nx-8;
 int i = 0;

 int remainder = 8;
 svbool_t pgt = svwhilelt_b64(i,remainder);
 {

   float64_t fcp1_arr[8];
   for(int i = 0; i < remainder; i++){
     if(i == remainder-1){
       fcp1_arr[i] = temporal[id1+x+i];
     }else{
       fcp1_arr[i] = temporal[id1+x+1+i];
     }
   }
   svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
   svfloat64_t fc_vec = svld1(pgt,(float64_t*)&temporal[id1+x]);
   svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pgt,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pgt,cc_vec,fc_vec);
   fce_vec = svmul_x(pgt,ce_vec,fce_vec);
   fcw_vec = svmul_x(pgt,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pgt,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pgt,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pgt,cb_vec,fcb_vec);
   fct_vec = svmul_x(pgt,ct_vec,fct_vec);
   tmp0 = svadd_x(pgt,fce_vec,fcw_vec);
   tmp1 = svadd_x(pgt,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pgt,fct_vec,fcb_vec);
   tmp0 = svadd_x(pgt,fc_vec, tmp0);
   tmp1 = svadd_x(pgt,tmp1, tmp2);
   tmp0 = svadd_x(pgt,tmp0, tmp1);
   svst1(pgt,(float64_t*)&temporal[id2+x],tmp0);
 }
 id2 += nx;
 id1 += nx;
      }
      step++;
    }
    tmp = izm;
    izm = izc;
    izc = izp;
    izp = tmp;

  }

  for (zz = 0; zz < nz-(1 +3 -2); zz++) {
    z = zz+(1 +3 -2);
    b0 = (z == 0) ? 0 : - nx * ny;
    t0 = (z == nz-1) ? 0 : nx * ny;
    n0 = -nx;
    s0 = nx;

    step = 0;
    halo = (1 +3 -2)-step;





    ystr = yy-halo;
    iy = step;


    yend = ny;



    id2 = 0+iy*nx+izp*nx*tby+step*nx*tby*tbz;

    for (y = ystr; y < yend; y++) {




      s0 = (y == ny-1) ? 0 : nx;

      x = 0;
      {
 c = x + y*nx + z*nx*ny;
 float64_t fcm1_arr[8] = {f1_t[c+x],f1_t[c+x],f1_t[c+x+1],f1_t[c+x+2],f1_t[c+x+3],f1_t[c+x+4],f1_t[c+x+5],f1_t[c+x+6]};
 svfloat64_t fc_vec = svld1(pg,(float64_t*)&f1_t[c+x]);
 svfloat64_t fce_vec = svld1(pg,(float64_t*)&f1_t[c+x+1]);
 svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
 svfloat64_t fcs_vec = svld1(pg,(float64_t*)&f1_t[c+x+s0]);
 svfloat64_t fcn_vec = svld1(pg,(float64_t*)&f1_t[c+x+n0]);
 svfloat64_t fcb_vec = svld1(pg,(float64_t*)&f1_t[c+x+b0]);
 svfloat64_t fct_vec = svld1(pg,(float64_t*)&f1_t[c+x+t0]);
 svfloat64_t tmp0,tmp1,tmp2;
 fc_vec = svmul_x(pg,cc_vec,fc_vec);
 fce_vec = svmul_x(pg,ce_vec,fce_vec);
 fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
 fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
 fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
 fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
 fct_vec = svmul_x(pg,ct_vec,fct_vec);
 tmp0 = svadd_x(pg,fce_vec,fcw_vec);
 tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
 tmp2 = svadd_x(pg,fct_vec,fcb_vec);
 tmp0 = svadd_x(pg,fc_vec, tmp0);
 tmp1 = svadd_x(pg,tmp1, tmp2);
 tmp0 = svadd_x(pg,tmp0, tmp1);
 svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
      }
      for (x = 8; x < nx-8; x+=8) {
       svfloat64_t fc_vec = svld1(pg,(float64_t*)&f1_t[c+x]);
       svfloat64_t fce_vec = svld1(pg,(float64_t*)&f1_t[c+x+1]);
       svfloat64_t fcw_vec = svld1(pg,(float64_t*)&f1_t[c+x-1]);
       svfloat64_t fcs_vec = svld1(pg,(float64_t*)&f1_t[c+x+s0]);
       svfloat64_t fcn_vec = svld1(pg,(float64_t*)&f1_t[c+x+n0]);
       svfloat64_t fcb_vec = svld1(pg,(float64_t*)&f1_t[c+x+b0]);
       svfloat64_t fct_vec = svld1(pg,(float64_t*)&f1_t[c+x+t0]);
       svfloat64_t tmp0,tmp1,tmp2;
       fc_vec = svmul_x(pg,cc_vec,fc_vec);
       fce_vec = svmul_x(pg,ce_vec,fce_vec);
       fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
       fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
       fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
       fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
       fct_vec = svmul_x(pg,ct_vec,fct_vec);
       tmp0 = svadd_x(pg,fce_vec,fcw_vec);
       tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
       tmp2 = svadd_x(pg,fct_vec,fcb_vec);
       tmp0 = svadd_x(pg,fc_vec, tmp0);
       tmp1 = svadd_x(pg,tmp1, tmp2);
       tmp0 = svadd_x(pg,tmp0, tmp1);
       svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
      }
      x = nx-8;
      int i = 0;
      int remainder = 8;

      svbool_t pgt = svwhilelt_b64(i,remainder);
      {
 float64_t fcp1_arr[8];
 for(int i = 0; i < remainder; i++){
   if(i == remainder-1){
     fcp1_arr[i] = f1_t[c+x+i];
   }else{
     fcp1_arr[i] = f1_t[c+x+1+i];
   }
 }
 svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
       svfloat64_t fc_vec = svld1(pgt,(float64_t*)&f1_t[c+x]);
       svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&f1_t[c+x-1]);
       svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&f1_t[c+x+s0]);
       svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&f1_t[c+x+n0]);
       svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&f1_t[c+x+b0]);
       svfloat64_t fct_vec = svld1(pgt,(float64_t*)&f1_t[c+x+t0]);
 svfloat64_t tmp0,tmp1,tmp2;
 fc_vec = svmul_x(pgt,cc_vec,fc_vec);
 fce_vec = svmul_x(pgt,ce_vec,fce_vec);
 fcw_vec = svmul_x(pgt,cw_vec,fcw_vec);
 fcn_vec = svmul_x(pgt,cn_vec,fcn_vec);
 fcs_vec = svmul_x(pgt,cs_vec,fcs_vec);
 fcb_vec = svmul_x(pgt,cb_vec,fcb_vec);
 fct_vec = svmul_x(pgt,ct_vec,fct_vec);
 tmp0 = svadd_x(pgt,fce_vec,fcw_vec);
 tmp1 = svadd_x(pgt,fcn_vec,fcs_vec);
 tmp2 = svadd_x(pgt,fct_vec,fcb_vec);
 tmp0 = svadd_x(pgt,fc_vec, tmp0);
 tmp1 = svadd_x(pgt,tmp1, tmp2);
 tmp0 = svadd_x(pgt,tmp0, tmp1);
 svst1(pgt,(float64_t*)&temporal[id2+x],tmp0);
      }
      id2 += nx;
    }

    for(z = zz+(1 +3 -2)-1; z >= zz+1; z--){
      b = (z == 0) ? 0 : (izm-izc)*nx*tby;
      t = (z == nz-1) ? 0 : (izp-izc)*nx*tby;
      n = -nx;
      s = nx;

      halo = (1 +3 -2)-(step+1);




      ystr = yy-halo;
      iy = step+1;


      yend = ny;




      id2 = 0+iy*nx+izp*nx*tby+(step+1)*nx*tby*tbz;
      id1 = 0+iy*nx+izc*nx*tby+step*nx*tby*tbz;

      for (y = ystr; y < yend; y++) {




 s = (y == ny-1) ? 0 : nx;

 x = 0;
 {
   float64_t fcm1_arr[8] = {temporal[id1+x],temporal[id1+x],temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6]};
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&temporal[id2+x],tmp0);


 }

 for (x = 8; x < nx-8; x+=8) {
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&temporal[id2+x],tmp0);


 }
 x = nx-8;
 int remainder = 8;
 int i = 0;

 svbool_t pgt = svwhilelt_b64(i,remainder);
 {
   float64_t fcp1_arr[8];
   for(int i = 0; i < remainder; i++){
     if(i == remainder-1){
       fcp1_arr[i] = temporal[id1+x+i];
     }else{
       fcp1_arr[i] = temporal[id1+x+1+i];
     }
   }
   svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
   svfloat64_t fc_vec = svld1(pgt,(float64_t*)&temporal[id1+x]);
   svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pgt,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pgt,cc_vec,fc_vec);
   fce_vec = svmul_x(pgt,ce_vec,fce_vec);
   fcw_vec = svmul_x(pgt,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pgt,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pgt,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pgt,cb_vec,fcb_vec);
   fct_vec = svmul_x(pgt,ct_vec,fct_vec);
   tmp0 = svadd_x(pgt,fce_vec,fcw_vec);
   tmp1 = svadd_x(pgt,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pgt,fct_vec,fcb_vec);
   tmp0 = svadd_x(pgt,fc_vec, tmp0);
   tmp1 = svadd_x(pgt,tmp1, tmp2);
   tmp0 = svadd_x(pgt,tmp0, tmp1);
   svst1(pgt,(float64_t*)&temporal[id2+x],tmp0);
 }


 id1 += nx;
 id2 += nx;
      }
      step++;

    }

    {
      z = zz;
      b = (z == 0) ? 0 : (izm-izc)*nx*tby;
      t = (z == nz-1) ? 0 : (izp-izc)*nx*tby;
      n = -nx;
      s = nx;

      halo = (1 +3 -2)-(step+1);




      ystr = yy-halo;
      iy = step+1;


      yend = ny;



      id1 = 0+iy*nx+izc*nx*tby+step*nx*tby*tbz;
      for (y = ystr; y < yend; y++) {





 s = (y == ny-1) ? 0 : nx;

 x = 0;
 c = y*nx + z*nx*ny;
 {
   float64_t fcm1_arr[8] = {temporal[id1+x],temporal[id1+x],temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6]};
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&f2_t[c+x],tmp0);

 }

 for (x = 8; x < nx-8; x+=8) {
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&f2_t[c+x],tmp0);
 }
 x = nx-8;
 int i = 0;
 int remainder = 8;

 svbool_t pgt = svwhilelt_b64(i,remainder);
 {
   float64_t fcp1_arr[8];
   for(int i = 0; i < remainder; i++){
     if(i == remainder-1){
       fcp1_arr[i] = temporal[id1+x+i];
     }else{
       fcp1_arr[i] = temporal[id1+x+1+i];
     }
   }
   svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
   svfloat64_t fc_vec = svld1(pgt,(float64_t*)&temporal[id1+x]);
   svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pgt,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pgt,cc_vec,fc_vec);
   fce_vec = svmul_x(pgt,ce_vec,fce_vec);
   fcw_vec = svmul_x(pgt,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pgt,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pgt,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pgt,cb_vec,fcb_vec);
   fct_vec = svmul_x(pgt,ct_vec,fct_vec);
   tmp0 = svadd_x(pgt,fce_vec,fcw_vec);
   tmp1 = svadd_x(pgt,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pgt,fct_vec,fcb_vec);
   tmp0 = svadd_x(pgt,fc_vec, tmp0);
   tmp1 = svadd_x(pgt,tmp1, tmp2);
   tmp0 = svadd_x(pgt,tmp0, tmp1);
   svst1(pgt,(float64_t*)&f2_t[c+x],tmp0);

 }

 id1 += nx;
      }
    }
    tmp = izm;
    izm = izc;
    izc = izp;
    izp = tmp;

  }

  for (zz = nz-(1 +3 -2), h = (1 +3 -2)-1; zz < nz; zz++, h--) {

    step = (1 +3 -2)-1-h;

    for(z = zz+h ; z >= zz+1; z--){
      b = (z == 0) ? 0 : (izm-izc)*nx*tby;
      t = (z == nz-1) ? 0 : (izp-izc)*nx*tby;
      n = -nx;
      s = nx;

      halo = (1 +3 -2)-(step+1);




      ystr = yy-halo;
      iy = step+1;


      yend = ny;




      id2 = 0+iy*nx+izp*nx*tby+(step+1)*nx*tby*tbz;
      id1 = 0+iy*nx+izc*nx*tby+step*nx*tby*tbz;

      for (y = ystr; y < yend; y++) {




 s = (y == ny-1) ? 0 : nx;

 x = 0;
 {
   float64_t fcm1_arr[8] = {temporal[id1+x],temporal[id1+x],temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6]};
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
 }

 for (x = 8; x < nx-8; x+=8) {
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
 }
 x = nx-8;
 int i = 0;
 int remainder = 8;

 svbool_t pgt = svwhilelt_b64(i,remainder);
 {
   float64_t fcp1_arr[8];
   for(int i = 0; i < remainder; i++){
     if(i == remainder-1){
       fcp1_arr[i] = temporal[id1+x+i];
     }else{
       fcp1_arr[i] = temporal[id1+x+1+i];
     }
   }
   svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
   svfloat64_t fc_vec = svld1(pgt,(float64_t*)&temporal[id1+x]);
   svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pgt,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pgt,cc_vec,fc_vec);
   fce_vec = svmul_x(pgt,ce_vec,fce_vec);
   fcw_vec = svmul_x(pgt,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pgt,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pgt,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pgt,cb_vec,fcb_vec);
   fct_vec = svmul_x(pgt,ct_vec,fct_vec);
   tmp0 = svadd_x(pgt,fce_vec,fcw_vec);
   tmp1 = svadd_x(pgt,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pgt,fct_vec,fcb_vec);
   tmp0 = svadd_x(pgt,fc_vec, tmp0);
   tmp1 = svadd_x(pgt,tmp1, tmp2);
   tmp0 = svadd_x(pgt,tmp0, tmp1);
   svst1(pgt,(float64_t*)&temporal[id2+x],tmp0);
 }
 id2 += nx;
 id1 += nx;
      }

      step++;

    }

    {
      z = zz;
      b = (z == 0) ? 0 : (izm-izc)*nx*tby;
      t = (z == nz-1) ? 0 : (izp-izc)*nx*tby;
      n = -nx;
      s = nx;
      halo = (1 +3 -2)-(step+1);





      ystr = yy-halo;
      iy = step+1;


      yend = ny;




      id1 = 0+iy*nx+izc*nx*tby+step*nx*tby*tbz;
      for (y = ystr; y < yend; y++) {




 s = (y == ny-1) ? 0 : nx;

 x = 0;
 c = x + y*nx + z*nx*ny;
 {
   float64_t fcm1_arr[8] = {temporal[id1+x],temporal[id1+x],temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6]};
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&f2_t[c+x],tmp0);

 }

 for (x = 8; x < nx-8; x+=8) {
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&f2_t[c+x],tmp0);
 }
 x = nx-8;
 int i = 0;
 int remainder = 8;

 svbool_t pgt = svwhilelt_b64(i,remainder);
 {

   float64_t fcp1_arr[8];
   for(int i = 0; i < remainder; i++){
     if(i == remainder-1){
       fcp1_arr[i] = temporal[id1+x+i];
     }else{
       fcp1_arr[i] = temporal[id1+x+1+i];
     }
   }
   svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
   svfloat64_t fc_vec = svld1(pgt,(float64_t*)&temporal[id1+x]);
   svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pgt,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pgt,cc_vec,fc_vec);
   fce_vec = svmul_x(pgt,ce_vec,fce_vec);
   fcw_vec = svmul_x(pgt,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pgt,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pgt,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pgt,cb_vec,fcb_vec);
   fct_vec = svmul_x(pgt,ct_vec,fct_vec);
   tmp0 = svadd_x(pgt,fce_vec,fcw_vec);
   tmp1 = svadd_x(pgt,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pgt,fct_vec,fcb_vec);
   tmp0 = svadd_x(pgt,fc_vec, tmp0);
   tmp1 = svadd_x(pgt,tmp1, tmp2);
   tmp0 = svadd_x(pgt,tmp0, tmp1);
   svst1(pgt,(float64_t*)&f2_t[c+x],tmp0);

 }
 id1 += nx;
      }
    }

    tmp = izm;
    izm = izc;
    izc = izp;
    izp = tmp;
  }
}
# 140 "diffusion_ker38.c" 2

 }else{

# 1 "ker38.inc" 1

{
  izm = 0;
  izc = 1;
  izp = 2;

  for (h = 0; h < (1 +3 -2); h++){
    z = h;
    step = 0;

    halo = (1 +3 -2)-step;
    b0 = (z == 0) ? 0 : - nx * ny;
    t0 = (z == nz-1) ? 0 : nx * ny;
    n0 = -nx;
    s0 = nx;





    ystr = yy-halo;
    iy = step;




    yend = yy+8 +halo;


    id2 = 0+iy*nx+izp*nx*tby+step*nx*tby*tbz;

    for (y = ystr; y < yend; y++) {






      x = 0;
      {
 c = x + y*nx + z*nx*ny;
 float64_t fcm1_arr[8] = {f1_t[c+x],f1_t[c+x],f1_t[c+x+1],f1_t[c+x+2],f1_t[c+x+3],f1_t[c+x+4],f1_t[c+x+5],f1_t[c+x+6]};
 svfloat64_t fc_vec = svld1(pg,(float64_t*)&f1_t[c+x]);
 svfloat64_t fce_vec = svld1(pg,(float64_t*)&f1_t[c+x+1]);
 svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
 svfloat64_t fcs_vec = svld1(pg,(float64_t*)&f1_t[c+x+s0]);
 svfloat64_t fcn_vec = svld1(pg,(float64_t*)&f1_t[c+x+n0]);
 svfloat64_t fcb_vec = svld1(pg,(float64_t*)&f1_t[c+x+b0]);
 svfloat64_t fct_vec = svld1(pg,(float64_t*)&f1_t[c+x+t0]);
 svfloat64_t tmp0,tmp1,tmp2;
 fc_vec = svmul_x(pg,cc_vec,fc_vec);
 fce_vec = svmul_x(pg,ce_vec,fce_vec);
 fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
 fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
 fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
 fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
 fct_vec = svmul_x(pg,ct_vec,fct_vec);
 tmp0 = svadd_x(pg,fce_vec,fcw_vec);
 tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
 tmp2 = svadd_x(pg,fct_vec,fcb_vec);
 tmp0 = svadd_x(pg,fc_vec, tmp0);
 tmp1 = svadd_x(pg,tmp1, tmp2);
 tmp0 = svadd_x(pg,tmp0, tmp1);
 svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
      }
      for (x = 8; x < nx-8; x+=8) {
       svfloat64_t fc_vec = svld1(pg,(float64_t*)&f1_t[c+x]);
       svfloat64_t fce_vec = svld1(pg,(float64_t*)&f1_t[c+x+1]);
       svfloat64_t fcw_vec = svld1(pg,(float64_t*)&f1_t[c+x-1]);
       svfloat64_t fcs_vec = svld1(pg,(float64_t*)&f1_t[c+x+s0]);
       svfloat64_t fcn_vec = svld1(pg,(float64_t*)&f1_t[c+x+n0]);
       svfloat64_t fcb_vec = svld1(pg,(float64_t*)&f1_t[c+x+b0]);
       svfloat64_t fct_vec = svld1(pg,(float64_t*)&f1_t[c+x+t0]);
       svfloat64_t tmp0,tmp1,tmp2;
       fc_vec = svmul_x(pg,cc_vec,fc_vec);
       fce_vec = svmul_x(pg,ce_vec,fce_vec);
       fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
       fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
       fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
       fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
       fct_vec = svmul_x(pg,ct_vec,fct_vec);
       tmp0 = svadd_x(pg,fce_vec,fcw_vec);
       tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
       tmp2 = svadd_x(pg,fct_vec,fcb_vec);
       tmp0 = svadd_x(pg,fc_vec, tmp0);
       tmp1 = svadd_x(pg,tmp1, tmp2);
       tmp0 = svadd_x(pg,tmp0, tmp1);
       svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
      }
      int i = 0;

      int remainder = 8;
      svbool_t pgt = svwhilelt_b64(i,remainder);
      {
 float64_t fcp1_arr[8];
 for(int i = 0; i < remainder; i++){
   if(i == remainder-1){
     fcp1_arr[i] = f1_t[c+x+i];
   }else{
     fcp1_arr[i] = f1_t[c+x+1+i];
   }
 }
 svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
       svfloat64_t fc_vec = svld1(pgt,(float64_t*)&f1_t[c+x]);
       svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&f1_t[c+x-1]);
       svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&f1_t[c+x+s0]);
       svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&f1_t[c+x+n0]);
       svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&f1_t[c+x+b0]);
       svfloat64_t fct_vec = svld1(pgt,(float64_t*)&f1_t[c+x+t0]);
 svfloat64_t tmp0,tmp1,tmp2;
 fc_vec = svmul_x(pgt,cc_vec,fc_vec);
 fce_vec = svmul_x(pgt,ce_vec,fce_vec);
 fcw_vec = svmul_x(pgt,cw_vec,fcw_vec);
 fcn_vec = svmul_x(pgt,cn_vec,fcn_vec);
 fcs_vec = svmul_x(pgt,cs_vec,fcs_vec);
 fcb_vec = svmul_x(pgt,cb_vec,fcb_vec);
 fct_vec = svmul_x(pgt,ct_vec,fct_vec);
 tmp0 = svadd_x(pgt,fce_vec,fcw_vec);
 tmp1 = svadd_x(pgt,fcn_vec,fcs_vec);
 tmp2 = svadd_x(pgt,fct_vec,fcb_vec);
 tmp0 = svadd_x(pgt,fc_vec, tmp0);
 tmp1 = svadd_x(pgt,tmp1, tmp2);
 tmp0 = svadd_x(pgt,tmp0, tmp1);
 svst1(pgt,(float64_t*)&temporal[id2+x],tmp0);
      }
      id2 += nx;
    }

    for(z = h-1; z >= 0; z--){
      b = (z == 0) ? 0 : (izm-izc)*nx*tby;
      t = (z == nz-1) ? 0 : (izp-izc)*nx*tby;
      n = -nx;
      s = nx;

      halo = (1 +3 -2)-(step+1);




      ystr = yy-halo;
      iy = step+1;




      yend = yy+8 +halo;


      id2 = 0+iy*nx+izp*nx*tby+(step+1)*nx*tby*tbz;
      id1 = 0+iy*nx+izc*nx*tby+step*nx*tby*tbz;

      for (y = ystr; y < yend; y++) {






 x = 0;
 {
   float64_t fcm1_arr[8] = {temporal[id1+x],temporal[id1+x],temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6]};
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
 }
 for (x = 8; x < nx-8; x+=8) {
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
 }
 x = nx-8;
 int i = 0;

 int remainder = 8;
 svbool_t pgt = svwhilelt_b64(i,remainder);
 {

   float64_t fcp1_arr[8];
   for(int i = 0; i < remainder; i++){
     if(i == remainder-1){
       fcp1_arr[i] = temporal[id1+x+i];
     }else{
       fcp1_arr[i] = temporal[id1+x+1+i];
     }
   }
   svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
   svfloat64_t fc_vec = svld1(pgt,(float64_t*)&temporal[id1+x]);
   svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pgt,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pgt,cc_vec,fc_vec);
   fce_vec = svmul_x(pgt,ce_vec,fce_vec);
   fcw_vec = svmul_x(pgt,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pgt,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pgt,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pgt,cb_vec,fcb_vec);
   fct_vec = svmul_x(pgt,ct_vec,fct_vec);
   tmp0 = svadd_x(pgt,fce_vec,fcw_vec);
   tmp1 = svadd_x(pgt,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pgt,fct_vec,fcb_vec);
   tmp0 = svadd_x(pgt,fc_vec, tmp0);
   tmp1 = svadd_x(pgt,tmp1, tmp2);
   tmp0 = svadd_x(pgt,tmp0, tmp1);
   svst1(pgt,(float64_t*)&temporal[id2+x],tmp0);
 }
 id2 += nx;
 id1 += nx;
      }
      step++;
    }
    tmp = izm;
    izm = izc;
    izc = izp;
    izp = tmp;

  }

  for (zz = 0; zz < nz-(1 +3 -2); zz++) {
    z = zz+(1 +3 -2);
    b0 = (z == 0) ? 0 : - nx * ny;
    t0 = (z == nz-1) ? 0 : nx * ny;
    n0 = -nx;
    s0 = nx;

    step = 0;
    halo = (1 +3 -2)-step;





    ystr = yy-halo;
    iy = step;




    yend = yy+8 +halo;

    id2 = 0+iy*nx+izp*nx*tby+step*nx*tby*tbz;

    for (y = ystr; y < yend; y++) {






      x = 0;
      {
 c = x + y*nx + z*nx*ny;
 float64_t fcm1_arr[8] = {f1_t[c+x],f1_t[c+x],f1_t[c+x+1],f1_t[c+x+2],f1_t[c+x+3],f1_t[c+x+4],f1_t[c+x+5],f1_t[c+x+6]};
 svfloat64_t fc_vec = svld1(pg,(float64_t*)&f1_t[c+x]);
 svfloat64_t fce_vec = svld1(pg,(float64_t*)&f1_t[c+x+1]);
 svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
 svfloat64_t fcs_vec = svld1(pg,(float64_t*)&f1_t[c+x+s0]);
 svfloat64_t fcn_vec = svld1(pg,(float64_t*)&f1_t[c+x+n0]);
 svfloat64_t fcb_vec = svld1(pg,(float64_t*)&f1_t[c+x+b0]);
 svfloat64_t fct_vec = svld1(pg,(float64_t*)&f1_t[c+x+t0]);
 svfloat64_t tmp0,tmp1,tmp2;
 fc_vec = svmul_x(pg,cc_vec,fc_vec);
 fce_vec = svmul_x(pg,ce_vec,fce_vec);
 fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
 fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
 fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
 fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
 fct_vec = svmul_x(pg,ct_vec,fct_vec);
 tmp0 = svadd_x(pg,fce_vec,fcw_vec);
 tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
 tmp2 = svadd_x(pg,fct_vec,fcb_vec);
 tmp0 = svadd_x(pg,fc_vec, tmp0);
 tmp1 = svadd_x(pg,tmp1, tmp2);
 tmp0 = svadd_x(pg,tmp0, tmp1);
 svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
      }
      for (x = 8; x < nx-8; x+=8) {
       svfloat64_t fc_vec = svld1(pg,(float64_t*)&f1_t[c+x]);
       svfloat64_t fce_vec = svld1(pg,(float64_t*)&f1_t[c+x+1]);
       svfloat64_t fcw_vec = svld1(pg,(float64_t*)&f1_t[c+x-1]);
       svfloat64_t fcs_vec = svld1(pg,(float64_t*)&f1_t[c+x+s0]);
       svfloat64_t fcn_vec = svld1(pg,(float64_t*)&f1_t[c+x+n0]);
       svfloat64_t fcb_vec = svld1(pg,(float64_t*)&f1_t[c+x+b0]);
       svfloat64_t fct_vec = svld1(pg,(float64_t*)&f1_t[c+x+t0]);
       svfloat64_t tmp0,tmp1,tmp2;
       fc_vec = svmul_x(pg,cc_vec,fc_vec);
       fce_vec = svmul_x(pg,ce_vec,fce_vec);
       fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
       fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
       fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
       fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
       fct_vec = svmul_x(pg,ct_vec,fct_vec);
       tmp0 = svadd_x(pg,fce_vec,fcw_vec);
       tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
       tmp2 = svadd_x(pg,fct_vec,fcb_vec);
       tmp0 = svadd_x(pg,fc_vec, tmp0);
       tmp1 = svadd_x(pg,tmp1, tmp2);
       tmp0 = svadd_x(pg,tmp0, tmp1);
       svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
      }
      x = nx-8;
      int i = 0;
      int remainder = 8;

      svbool_t pgt = svwhilelt_b64(i,remainder);
      {
 float64_t fcp1_arr[8];
 for(int i = 0; i < remainder; i++){
   if(i == remainder-1){
     fcp1_arr[i] = f1_t[c+x+i];
   }else{
     fcp1_arr[i] = f1_t[c+x+1+i];
   }
 }
 svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
       svfloat64_t fc_vec = svld1(pgt,(float64_t*)&f1_t[c+x]);
       svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&f1_t[c+x-1]);
       svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&f1_t[c+x+s0]);
       svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&f1_t[c+x+n0]);
       svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&f1_t[c+x+b0]);
       svfloat64_t fct_vec = svld1(pgt,(float64_t*)&f1_t[c+x+t0]);
 svfloat64_t tmp0,tmp1,tmp2;
 fc_vec = svmul_x(pgt,cc_vec,fc_vec);
 fce_vec = svmul_x(pgt,ce_vec,fce_vec);
 fcw_vec = svmul_x(pgt,cw_vec,fcw_vec);
 fcn_vec = svmul_x(pgt,cn_vec,fcn_vec);
 fcs_vec = svmul_x(pgt,cs_vec,fcs_vec);
 fcb_vec = svmul_x(pgt,cb_vec,fcb_vec);
 fct_vec = svmul_x(pgt,ct_vec,fct_vec);
 tmp0 = svadd_x(pgt,fce_vec,fcw_vec);
 tmp1 = svadd_x(pgt,fcn_vec,fcs_vec);
 tmp2 = svadd_x(pgt,fct_vec,fcb_vec);
 tmp0 = svadd_x(pgt,fc_vec, tmp0);
 tmp1 = svadd_x(pgt,tmp1, tmp2);
 tmp0 = svadd_x(pgt,tmp0, tmp1);
 svst1(pgt,(float64_t*)&temporal[id2+x],tmp0);
      }
      id2 += nx;
    }

    for(z = zz+(1 +3 -2)-1; z >= zz+1; z--){
      b = (z == 0) ? 0 : (izm-izc)*nx*tby;
      t = (z == nz-1) ? 0 : (izp-izc)*nx*tby;
      n = -nx;
      s = nx;

      halo = (1 +3 -2)-(step+1);




      ystr = yy-halo;
      iy = step+1;




      yend = yy+8 +halo;


      id2 = 0+iy*nx+izp*nx*tby+(step+1)*nx*tby*tbz;
      id1 = 0+iy*nx+izc*nx*tby+step*nx*tby*tbz;

      for (y = ystr; y < yend; y++) {






 x = 0;
 {
   float64_t fcm1_arr[8] = {temporal[id1+x],temporal[id1+x],temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6]};
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&temporal[id2+x],tmp0);


 }

 for (x = 8; x < nx-8; x+=8) {
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&temporal[id2+x],tmp0);


 }
 x = nx-8;
 int remainder = 8;
 int i = 0;

 svbool_t pgt = svwhilelt_b64(i,remainder);
 {
   float64_t fcp1_arr[8];
   for(int i = 0; i < remainder; i++){
     if(i == remainder-1){
       fcp1_arr[i] = temporal[id1+x+i];
     }else{
       fcp1_arr[i] = temporal[id1+x+1+i];
     }
   }
   svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
   svfloat64_t fc_vec = svld1(pgt,(float64_t*)&temporal[id1+x]);
   svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pgt,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pgt,cc_vec,fc_vec);
   fce_vec = svmul_x(pgt,ce_vec,fce_vec);
   fcw_vec = svmul_x(pgt,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pgt,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pgt,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pgt,cb_vec,fcb_vec);
   fct_vec = svmul_x(pgt,ct_vec,fct_vec);
   tmp0 = svadd_x(pgt,fce_vec,fcw_vec);
   tmp1 = svadd_x(pgt,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pgt,fct_vec,fcb_vec);
   tmp0 = svadd_x(pgt,fc_vec, tmp0);
   tmp1 = svadd_x(pgt,tmp1, tmp2);
   tmp0 = svadd_x(pgt,tmp0, tmp1);
   svst1(pgt,(float64_t*)&temporal[id2+x],tmp0);
 }


 id1 += nx;
 id2 += nx;
      }
      step++;

    }

    {
      z = zz;
      b = (z == 0) ? 0 : (izm-izc)*nx*tby;
      t = (z == nz-1) ? 0 : (izp-izc)*nx*tby;
      n = -nx;
      s = nx;

      halo = (1 +3 -2)-(step+1);




      ystr = yy-halo;
      iy = step+1;




      yend = yy+8 +halo;

      id1 = 0+iy*nx+izc*nx*tby+step*nx*tby*tbz;
      for (y = ystr; y < yend; y++) {







 x = 0;
 c = y*nx + z*nx*ny;
 {
   float64_t fcm1_arr[8] = {temporal[id1+x],temporal[id1+x],temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6]};
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&f2_t[c+x],tmp0);

 }

 for (x = 8; x < nx-8; x+=8) {
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&f2_t[c+x],tmp0);
 }
 x = nx-8;
 int i = 0;
 int remainder = 8;

 svbool_t pgt = svwhilelt_b64(i,remainder);
 {
   float64_t fcp1_arr[8];
   for(int i = 0; i < remainder; i++){
     if(i == remainder-1){
       fcp1_arr[i] = temporal[id1+x+i];
     }else{
       fcp1_arr[i] = temporal[id1+x+1+i];
     }
   }
   svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
   svfloat64_t fc_vec = svld1(pgt,(float64_t*)&temporal[id1+x]);
   svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pgt,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pgt,cc_vec,fc_vec);
   fce_vec = svmul_x(pgt,ce_vec,fce_vec);
   fcw_vec = svmul_x(pgt,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pgt,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pgt,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pgt,cb_vec,fcb_vec);
   fct_vec = svmul_x(pgt,ct_vec,fct_vec);
   tmp0 = svadd_x(pgt,fce_vec,fcw_vec);
   tmp1 = svadd_x(pgt,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pgt,fct_vec,fcb_vec);
   tmp0 = svadd_x(pgt,fc_vec, tmp0);
   tmp1 = svadd_x(pgt,tmp1, tmp2);
   tmp0 = svadd_x(pgt,tmp0, tmp1);
   svst1(pgt,(float64_t*)&f2_t[c+x],tmp0);

 }

 id1 += nx;
      }
    }
    tmp = izm;
    izm = izc;
    izc = izp;
    izp = tmp;

  }

  for (zz = nz-(1 +3 -2), h = (1 +3 -2)-1; zz < nz; zz++, h--) {

    step = (1 +3 -2)-1-h;

    for(z = zz+h ; z >= zz+1; z--){
      b = (z == 0) ? 0 : (izm-izc)*nx*tby;
      t = (z == nz-1) ? 0 : (izp-izc)*nx*tby;
      n = -nx;
      s = nx;

      halo = (1 +3 -2)-(step+1);




      ystr = yy-halo;
      iy = step+1;




      yend = yy+8 +halo;


      id2 = 0+iy*nx+izp*nx*tby+(step+1)*nx*tby*tbz;
      id1 = 0+iy*nx+izc*nx*tby+step*nx*tby*tbz;

      for (y = ystr; y < yend; y++) {






 x = 0;
 {
   float64_t fcm1_arr[8] = {temporal[id1+x],temporal[id1+x],temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6]};
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
 }

 for (x = 8; x < nx-8; x+=8) {
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
 }
 x = nx-8;
 int i = 0;
 int remainder = 8;

 svbool_t pgt = svwhilelt_b64(i,remainder);
 {
   float64_t fcp1_arr[8];
   for(int i = 0; i < remainder; i++){
     if(i == remainder-1){
       fcp1_arr[i] = temporal[id1+x+i];
     }else{
       fcp1_arr[i] = temporal[id1+x+1+i];
     }
   }
   svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
   svfloat64_t fc_vec = svld1(pgt,(float64_t*)&temporal[id1+x]);
   svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pgt,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pgt,cc_vec,fc_vec);
   fce_vec = svmul_x(pgt,ce_vec,fce_vec);
   fcw_vec = svmul_x(pgt,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pgt,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pgt,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pgt,cb_vec,fcb_vec);
   fct_vec = svmul_x(pgt,ct_vec,fct_vec);
   tmp0 = svadd_x(pgt,fce_vec,fcw_vec);
   tmp1 = svadd_x(pgt,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pgt,fct_vec,fcb_vec);
   tmp0 = svadd_x(pgt,fc_vec, tmp0);
   tmp1 = svadd_x(pgt,tmp1, tmp2);
   tmp0 = svadd_x(pgt,tmp0, tmp1);
   svst1(pgt,(float64_t*)&temporal[id2+x],tmp0);
 }
 id2 += nx;
 id1 += nx;
      }

      step++;

    }

    {
      z = zz;
      b = (z == 0) ? 0 : (izm-izc)*nx*tby;
      t = (z == nz-1) ? 0 : (izp-izc)*nx*tby;
      n = -nx;
      s = nx;
      halo = (1 +3 -2)-(step+1);





      ystr = yy-halo;
      iy = step+1;




      yend = yy+8 +halo;


      id1 = 0+iy*nx+izc*nx*tby+step*nx*tby*tbz;
      for (y = ystr; y < yend; y++) {






 x = 0;
 c = x + y*nx + z*nx*ny;
 {
   float64_t fcm1_arr[8] = {temporal[id1+x],temporal[id1+x],temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6]};
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&f2_t[c+x],tmp0);

 }

 for (x = 8; x < nx-8; x+=8) {
   svfloat64_t fc_vec = svld1(pg,(float64_t*)&temporal[id1+x]);
   svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
   svfloat64_t fcw_vec = svld1(pg,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pg,cc_vec,fc_vec);
   fce_vec = svmul_x(pg,ce_vec,fce_vec);
   fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
   fct_vec = svmul_x(pg,ct_vec,fct_vec);
   tmp0 = svadd_x(pg,fce_vec,fcw_vec);
   tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pg,fct_vec,fcb_vec);
   tmp0 = svadd_x(pg,fc_vec, tmp0);
   tmp1 = svadd_x(pg,tmp1, tmp2);
   tmp0 = svadd_x(pg,tmp0, tmp1);
   svst1(pg,(float64_t*)&f2_t[c+x],tmp0);
 }
 x = nx-8;
 int i = 0;
 int remainder = 8;

 svbool_t pgt = svwhilelt_b64(i,remainder);
 {

   float64_t fcp1_arr[8];
   for(int i = 0; i < remainder; i++){
     if(i == remainder-1){
       fcp1_arr[i] = temporal[id1+x+i];
     }else{
       fcp1_arr[i] = temporal[id1+x+1+i];
     }
   }
   svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
   svfloat64_t fc_vec = svld1(pgt,(float64_t*)&temporal[id1+x]);
   svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&temporal[id1+x-1]);
   svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&temporal[id1+x+s]);
   svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&temporal[id1+x+n]);
   svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&temporal[id1+x+b]);
   svfloat64_t fct_vec = svld1(pgt,(float64_t*)&temporal[id1+x+t]);
   svfloat64_t tmp0,tmp1,tmp2;
   fc_vec = svmul_x(pgt,cc_vec,fc_vec);
   fce_vec = svmul_x(pgt,ce_vec,fce_vec);
   fcw_vec = svmul_x(pgt,cw_vec,fcw_vec);
   fcn_vec = svmul_x(pgt,cn_vec,fcn_vec);
   fcs_vec = svmul_x(pgt,cs_vec,fcs_vec);
   fcb_vec = svmul_x(pgt,cb_vec,fcb_vec);
   fct_vec = svmul_x(pgt,ct_vec,fct_vec);
   tmp0 = svadd_x(pgt,fce_vec,fcw_vec);
   tmp1 = svadd_x(pgt,fcn_vec,fcs_vec);
   tmp2 = svadd_x(pgt,fct_vec,fcb_vec);
   tmp0 = svadd_x(pgt,fc_vec, tmp0);
   tmp1 = svadd_x(pgt,tmp1, tmp2);
   tmp0 = svadd_x(pgt,tmp0, tmp1);
   svst1(pgt,(float64_t*)&f2_t[c+x],tmp0);

 }
 id1 += nx;
      }
    }

    tmp = izm;
    izm = izc;
    izc = izp;
    izp = tmp;
  }
}
# 144 "diffusion_ker38.c" 2

 }
      }

#pragma omp barrier
      double *tmp = f1_t;
      f1_t = f2_t;
      f2_t = tmp;
      time += 3*dt;

      count+=3;

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
