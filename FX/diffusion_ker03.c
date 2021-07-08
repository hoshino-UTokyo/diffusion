#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifdef SVE
#include <arm_sve.h>
#endif /* SVE */
#include "diffusion_ker03.h"

#ifndef REAL
#define REAL double
#endif

#ifndef M_PI
#define M_PI (3.1415926535897932384626)
#endif

void allocate_ker03(REAL **buff_ret, const int nx, const int ny, const int nz) {

  posix_memalign((void**)buff_ret, 64, sizeof(REAL)*nx*ny*nz);

}

void init_ker03(REAL *buff1, const int nx, const int ny, const int nz,
		const REAL kx, const REAL ky, const REAL kz,
		const REAL dx, const REAL dy, const REAL dz,
		const REAL kappa, const REAL time) {

  REAL ax, ay, az;
  int jz, jy, jx;
  ax = exp(-kappa*time*(kx*kx));
  ay = exp(-kappa*time*(ky*ky));
  az = exp(-kappa*time*(kz*kz));
#pragma omp parallel for private(jx,jy,jz)
  for (jz = 0; jz < nz; jz++) {
    for (jy = 0; jy < ny; jy++) {
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

void diffusion_ker03(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		     REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		     REAL cb, REAL cc, REAL dt,
		     REAL **f1_ret, REAL **f2_ret, REAL *time_ret, int *count_ret) {

  REAL time = 0.0;
  int count = 0;
  REAL *restrict f1_t = f1;
  REAL *restrict f2_t = f2;
  int c, n, s, b, t;
  int z, y, x;
  
  do {
#pragma omp parallel private(x,y,z,c,n,s,b,t)
    {
      const svfloat64_t cc_vec = svdup_f64(cc);
      const svfloat64_t cw_vec = svdup_f64(cw);
      const svfloat64_t ce_vec = svdup_f64(ce);
      const svfloat64_t cs_vec = svdup_f64(cs);
      const svfloat64_t cn_vec = svdup_f64(cn);
      const svfloat64_t cb_vec = svdup_f64(cb);
      const svfloat64_t ct_vec = svdup_f64(ct);
#pragma omp for
      for (z = 0; z < nz; z++) {
	for (y = 0; y < ny; y++) {
	  b = (z == 0)    ? 0 : - nx * ny;
	  t = (z == nz-1) ? 0 :   nx * ny;
	  n = (y == 0)    ? 0 : - nx;
	  s = (y == ny-1) ? 0 :   nx;
	  c =  y * nx + z * nx * ny;
	  float64_t fcm1_arr[8] = {f1_t[c],f1_t[c],f1_t[c+1],f1_t[c+2],f1_t[c+3],f1_t[c+4],f1_t[c+5],f1_t[c+6]};
	  svfloat64_t fc_vec  = svld1(svptrue_b64(),(float64_t*)&f1_t[c]);
	  svfloat64_t fce_vec = svld1(svptrue_b64(),(float64_t*)&f1_t[c+1]);
	  svfloat64_t fcw_vec = svld1(svptrue_b64(),(float64_t*)&fcm1_arr[0]);
	  svfloat64_t fcs_vec = svld1(svptrue_b64(),(float64_t*)&f1_t[c+s]);
	  svfloat64_t fcn_vec = svld1(svptrue_b64(),(float64_t*)&f1_t[c+n]);
	  svfloat64_t fcb_vec = svld1(svptrue_b64(),(float64_t*)&f1_t[c+b]);
	  svfloat64_t fct_vec = svld1(svptrue_b64(),(float64_t*)&f1_t[c+t]);
	  svfloat64_t tmp = svmul_x(svptrue_b64(),cc_vec,fc_vec);
	  tmp = svmad_x(svptrue_b64(),cw_vec,fcw_vec,tmp);
	  tmp = svmad_x(svptrue_b64(),ce_vec,fce_vec,tmp);
	  tmp = svmad_x(svptrue_b64(),cs_vec,fcs_vec,tmp);
	  tmp = svmad_x(svptrue_b64(),cn_vec,fcn_vec,tmp);
	  tmp = svmad_x(svptrue_b64(),cb_vec,fcb_vec,tmp);
	  tmp = svmad_x(svptrue_b64(),ct_vec,fct_vec,tmp);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c],tmp);

	  for (x = 8; x < nx-8; x+=8) {
	    fc_vec  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+x]);
	    fce_vec = svld1(svptrue_b64(),(float64_t*)&f1_t[c+x+1]);
	    fcw_vec = svld1(svptrue_b64(),(float64_t*)&f1_t[c+x-1]);
	    fcs_vec = svld1(svptrue_b64(),(float64_t*)&f1_t[c+x+s]);
	    fcn_vec = svld1(svptrue_b64(),(float64_t*)&f1_t[c+x+n]);
	    fcb_vec = svld1(svptrue_b64(),(float64_t*)&f1_t[c+x+b]);
	    fct_vec = svld1(svptrue_b64(),(float64_t*)&f1_t[c+x+t]);
	    tmp = svmul_x(svptrue_b64(),cc_vec,fc_vec);
	    tmp = svmad_x(svptrue_b64(),cw_vec,fcw_vec,tmp);
	    tmp = svmad_x(svptrue_b64(),ce_vec,fce_vec,tmp);
	    tmp = svmad_x(svptrue_b64(),cs_vec,fcs_vec,tmp);
	    tmp = svmad_x(svptrue_b64(),cn_vec,fcn_vec,tmp);
	    tmp = svmad_x(svptrue_b64(),cb_vec,fcb_vec,tmp);
	    tmp = svmad_x(svptrue_b64(),ct_vec,fct_vec,tmp);
	    svst1(svptrue_b64(),(float64_t*)&f2_t[c+x],tmp);
	  }
	  float64_t fcp1_arr[8] = {f1_t[c+x+1],f1_t[c+x+2],f1_t[c+x+3],f1_t[c+x+4],f1_t[c+x+5],f1_t[c+x+6],f1_t[c+x+7],f1_t[c+x+7]};
	  fc_vec  = svld1(svptrue_b64(),(float64_t*)&f1_t[c+x]);
	  fce_vec = svld1(svptrue_b64(),(float64_t*)&fcp1_arr[0]);
	  fcw_vec = svld1(svptrue_b64(),(float64_t*)&f1_t[c+x-1]);
	  fcs_vec = svld1(svptrue_b64(),(float64_t*)&f1_t[c+x+s]);
	  fcn_vec = svld1(svptrue_b64(),(float64_t*)&f1_t[c+x+n]);
	  fcb_vec = svld1(svptrue_b64(),(float64_t*)&f1_t[c+x+b]);
	  fct_vec = svld1(svptrue_b64(),(float64_t*)&f1_t[c+x+t]);
	  tmp = svmul_x(svptrue_b64(),cc_vec,fc_vec);
	  tmp = svmad_x(svptrue_b64(),cw_vec,fcw_vec,tmp);
	  tmp = svmad_x(svptrue_b64(),ce_vec,fce_vec,tmp);
	  tmp = svmad_x(svptrue_b64(),cs_vec,fcs_vec,tmp);
	  tmp = svmad_x(svptrue_b64(),cn_vec,fcn_vec,tmp);
	  tmp = svmad_x(svptrue_b64(),cb_vec,fcb_vec,tmp);
	  tmp = svmad_x(svptrue_b64(),ct_vec,fct_vec,tmp);
	  svst1(svptrue_b64(),(float64_t*)&f2_t[c+x],tmp);

	}
      }
    }
    REAL *tmp = f1_t;
    f1_t = f2_t;
    f2_t = tmp;
    time += dt;
    count++;
    
  } while (time + 0.5*dt < 0.1);
  *f1_ret = f1_t; *f2_ret = f2_t;
  *time_ret = time;      
  *count_ret = count;        
  
  return;
}
