#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifdef SVE
#include <arm_sve.h>
#endif /* SVE */
#include "immintrin.h"
#include "diffusion_ker13.h"

#ifndef REAL
#define REAL double
#endif

#ifndef M_PI
#define M_PI (3.1415926535897932384626)
#endif

void allocate_ker13(REAL **buff_ret, const int nx, const int ny, const int nz) {

  posix_memalign((void**)buff_ret, 64, sizeof(REAL)*nx*ny*nz);

}

void init_ker13(REAL *buff1, const int nx, const int ny, const int nz,
		const REAL kx, const REAL ky, const REAL kz,
		const REAL dx, const REAL dy, const REAL dz,
		const REAL kappa, const REAL time) {

  REAL ax, ay, az;
  int jz, jy, jx;
  ax = exp(-kappa*time*(kx*kx));
  ay = exp(-kappa*time*(ky*ky));
  az = exp(-kappa*time*(kz*kz));
#pragma omp parallel private(jx,jy,jz)
  for (jz = 0; jz < nz; jz++) {
#pragma omp for nowait
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

void diffusion_ker13(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
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
      const __m512d cc_vec = _mm512_set1_pd(cc);
      const __m512d cw_vec = _mm512_set1_pd(cw);
      const __m512d ce_vec = _mm512_set1_pd(ce);
      const __m512d cs_vec = _mm512_set1_pd(cs);
      const __m512d cn_vec = _mm512_set1_pd(cn);
      const __m512d cb_vec = _mm512_set1_pd(cb);
      const __m512d ct_vec = _mm512_set1_pd(ct);
      for (z = 0; z < nz; z++) {
#pragma omp for nowait
	for (y = 0; y < ny; y++) {
	  b = (z == 0)    ? 0 : - nx * ny;
	  t = (z == nz-1) ? 0 :   nx * ny;
	  n = (y == 0)    ? 0 : - nx;
	  s = (y == ny-1) ? 0 :   nx;
	  c =  y * nx + z * nx * ny;
	  REAL fcm1_arr[8] = {f1_t[c],f1_t[c],f1_t[c+1],f1_t[c+2],f1_t[c+3],f1_t[c+4],f1_t[c+5],f1_t[c+6]};
	  __m512d fc_vec  = _mm512_load_pd(f1_t+c);
	  __m512d fce_vec = _mm512_load_pd(f1_t+c+1);
	  __m512d fcw_vec = _mm512_load_pd(fcm1_arr);
	  __m512d fcs_vec = _mm512_load_pd(f1_t+c+s);
	  __m512d fcn_vec = _mm512_load_pd(f1_t+c+n);
	  __m512d fcb_vec = _mm512_load_pd(f1_t+c+b);
	  __m512d fct_vec = _mm512_load_pd(f1_t+c+t);
	  __m512d tmp0,tmp1,tmp2;
          fc_vec  = _mm512_mul_pd(cc_vec,fc_vec);
          fce_vec = _mm512_mul_pd(ce_vec,fce_vec);
          fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec);
          fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec);
          fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec);
          fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec);
          fct_vec = _mm512_mul_pd(ct_vec,fct_vec);
          tmp0 = _mm512_add_pd(fce_vec,fcw_vec);
          tmp1 = _mm512_add_pd(fcn_vec,fcs_vec);
          tmp2 = _mm512_add_pd(fct_vec,fcb_vec);
          tmp0 = _mm512_add_pd(fc_vec, tmp0);
          tmp1 = _mm512_add_pd(tmp1,   tmp2);
          tmp0 = _mm512_add_pd(tmp0,   tmp1);
	  _mm512_store_pd(f2_t+c,tmp0);

	  for (x = 8; x < nx-8; x+=8) {
	    fc_vec  = _mm512_load_pd(f1_t+c+x);
	    fce_vec = _mm512_load_pd(f1_t+c+x+1);
	    fcw_vec = _mm512_load_pd(f1_t+c+x-1);
	    fcs_vec = _mm512_load_pd(f1_t+c+x+s);
	    fcn_vec = _mm512_load_pd(f1_t+c+x+n);
	    fcb_vec = _mm512_load_pd(f1_t+c+x+b);
	    fct_vec = _mm512_load_pd(f1_t+c+x+t);
	    fc_vec  = _mm512_mul_pd(cc_vec,fc_vec);
	    fce_vec = _mm512_mul_pd(ce_vec,fce_vec);
	    fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec);
	    fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec);
	    fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec);
	    fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec);
	    fct_vec = _mm512_mul_pd(ct_vec,fct_vec);
	    tmp0 = _mm512_add_pd(fce_vec,fcw_vec);
	    tmp1 = _mm512_add_pd(fcn_vec,fcs_vec);
	    tmp2 = _mm512_add_pd(fct_vec,fcb_vec);
	    tmp0 = _mm512_add_pd(fc_vec, tmp0);
	    tmp1 = _mm512_add_pd(tmp1,   tmp2);
	    tmp0 = _mm512_add_pd(tmp0,   tmp1);
	    _mm512_store_pd(f2_t+c+x,tmp0);
	  }
	  REAL fcp1_arr[8] = {f1_t[c+x+1],f1_t[c+x+2],f1_t[c+x+3],f1_t[c+x+4],f1_t[c+x+5],f1_t[c+x+6],f1_t[c+x+7],f1_t[c+x+7]};
	  fc_vec  = _mm512_load_pd(f1_t+c+x);
	  fce_vec = _mm512_load_pd(fcp1_arr);
	  fcw_vec = _mm512_load_pd(f1_t+c+x-1);
	  fcs_vec = _mm512_load_pd(f1_t+c+x+s);
	  fcn_vec = _mm512_load_pd(f1_t+c+x+n);
	  fcb_vec = _mm512_load_pd(f1_t+c+x+b);
	  fct_vec = _mm512_load_pd(f1_t+c+x+t);
          fc_vec  = _mm512_mul_pd(cc_vec,fc_vec);
          fce_vec = _mm512_mul_pd(ce_vec,fce_vec);
          fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec);
          fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec);
          fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec);
          fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec);
          fct_vec = _mm512_mul_pd(ct_vec,fct_vec);
          tmp0 = _mm512_add_pd(fce_vec,fcw_vec);
          tmp1 = _mm512_add_pd(fcn_vec,fcs_vec);
          tmp2 = _mm512_add_pd(fct_vec,fcb_vec);
          tmp0 = _mm512_add_pd(fc_vec, tmp0);
          tmp1 = _mm512_add_pd(tmp1,   tmp2);
          tmp0 = _mm512_add_pd(tmp0,   tmp1);
	  _mm512_store_pd(f2_t+c+x,tmp0);
	}
      }
    }
    REAL *tmp = f1_t;
    f1_t = f2_t;
    f2_t = tmp;
    time += dt;
    count++;
    
  } while (time + 0.5*dt < 0.1);
  *f1_ret = f1_t; *f2_ret = f2_t;;
  *time_ret = time;      
  *count_ret = count;        
  
  return;
}
