#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifdef SVE
#include <arm_sve.h>
#endif /* SVE */
#include "diffusion_ker01.h"

#ifndef REAL
#define REAL double
#endif

#ifndef M_PI
#define M_PI (3.1415926535897932384626)
#endif

void allocate_ker01(REAL **buff_ret, const int nx, const int ny, const int nz) {

  posix_memalign((void**)buff_ret, 64, sizeof(REAL)*nx*ny*nz);

}

void init_ker01(REAL *buff1, const int nx, const int ny, const int nz,
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

void diffusion_ker01(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		     REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		     REAL cb, REAL cc, REAL dt,
		     REAL **f1_ret, REAL **f2_ret, REAL *time_ret, int *count_ret) {
  
  {
    REAL time = 0.0;
    int count = 0;
    REAL *restrict f1_t = f1;
    REAL *restrict f2_t = f2;
    int c, w, e, n, s, b, t;
    int z, y, x;
    
    do {
#pragma omp parallel for private(x,y,z,c,w,e,n,s,b,t)
      for (z = 0; z < nz; z++) {
	for (y = 0; y < ny; y++) {
	  for (x = 0; x < nx; x++) {
	    c =  x + y * nx + z * nx * ny;
	    w = (x == 0)    ? c : c - 1;
	    e = (x == nx-1) ? c : c + 1;
	    n = (y == 0)    ? c : c - nx;
	    s = (y == ny-1) ? c : c + nx;
	    b = (z == 0)    ? c : c - nx * ny;
	    t = (z == nz-1) ? c : c + nx * ny;
	    f2_t[c] = cc * f1_t[c] + cw * f1_t[w] + ce * f1_t[e]
	      + cs * f1_t[s] + cn * f1_t[n] + cb * f1_t[b] + ct * f1_t[t];
	  }
	}
      }
      /* if(count == 2){ */
      /* 	printf("f1(:,:,:)\n"); */
      /* 	for(z = 0; z < nz; z++){ */
      /* 	  printf("z = %d\n",z); */
      /* 	  for(y = 0; y < ny; y++){ */
      /* 	    for(x = 0; x < nx; x++){ */
      /* 	      printf("%f ",f1_t[x+y*nx+z*nx*ny]); */
      /* 	    } */
      /* 	    printf("\n"); */
      /* 	  } */
      /* 	} */
      /* 	printf("f2(:,:,:)\n"); */
      /* 	for(z = 0; z < nz; z++){ */
      /* 	  printf("z = %d\n",z); */
      /* 	  for(y = 0; y < ny; y++){ */
      /* 	    for(x = 0; x < nx; x++){ */
      /* 	      printf("%f ",f2_t[x+y*nx+z*nx*ny]); */
      /* 	    } */
      /* 	    printf("\n"); */
      /* 	  } */
      /* 	} */
      /* 	exit(0); */
      /* } */
      
      
      REAL *tmp = f1_t;
      f1_t = f2_t;
      f2_t = tmp;
      time += dt;
      //time += 1;
      count++;
    } while (time + 0.5*dt < 0.1);
    *f1_ret = f1_t; *f2_ret = f2_t;;
    *time_ret = time;      
    *count_ret = count;        
  }
  return;
}

