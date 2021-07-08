#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#ifdef SVE
#include <arm_sve.h>
#endif /* SVE */
#include "diffusion_ker07.h"

#ifndef REAL
#define REAL double
#endif

#ifndef M_PI
#define M_PI (3.1415926535897932384626)
#endif

#ifndef MIN
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#endif

void allocate_ker07(REAL **buff_ret, const int nx, const int ny, const int nz) {

  posix_memalign((void**)buff_ret, 64, sizeof(REAL)*nx*ny*nz);

}

void init_ker07(REAL *buff1, const int nx, const int ny, const int nz,
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
    int yblock = 8;
    int yy;
    for (yy = ty*ychunk; yy < MIN((ty+1)*ychunk,ny); yy+= yblock) {
      for (jz = tz*zchunk; jz < MIN((tz+1)*zchunk,nz); jz++) {
	for (jy = yy; jy < MIN(yy+yblock,ny); jy++) {
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
}

void diffusion_ker07(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		     REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		     REAL cb, REAL cc, REAL dt,
		     REAL **f1_ret, REAL **f2_ret, REAL *time_ret, int *count_ret) {
  
#pragma omp parallel 
  {
    REAL time = 0.0;
    int count = 0;
    REAL *restrict f1_t = f1;
    REAL *restrict f2_t = f2;
    int c, n, s, b, t;
    int z, y, x, yy;
    int tid = omp_get_thread_num();
    int nth = omp_get_num_threads();
    int tz = tid/12;
    int ty = tid%12;
    int ychunk = (ny-1)/12 + 1;
    int zchunk = nz/((nth-1)/12+1);
    int yblock = 8;
    do {
      for (yy = ty*ychunk; yy < MIN((ty+1)*ychunk,ny); yy+= yblock) {
	for (z = tz*zchunk; z < MIN((tz+1)*zchunk,nz); z++) {
	  b = (z == 0)    ? 0 : - nx * ny;
	  t = (z == nz-1) ? 0 :   nx * ny;
	  for (y = yy; y < MIN(yy+yblock,ny); y++) {
	    n = (y == 0)    ? 0 : - nx;
	    s = (y == ny-1) ? 0 :   nx;
	    c =  y * nx + z * nx * ny;
	    f2_t[c] = cc * f1_t[c] + cw * f1_t[c] + ce * f1_t[c+1]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	    c++;
	    for (x = 1; x < nx-1; x++) {
	      f2_t[c] = cc * f1_t[c] + cw * f1_t[c-1] + ce * f1_t[c+1]
		+ cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	      c++;
	    }
	    f2_t[c] = cc * f1_t[c] + cw * f1_t[c-1] + ce * f1_t[c]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	  }
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

  return;
}
