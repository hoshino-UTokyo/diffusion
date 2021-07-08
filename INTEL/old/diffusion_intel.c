#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#ifdef INTEL
#include "immintrin.h"
#endif
#ifdef _OPENACC
#include <openacc.h>
#endif

#ifndef REAL
#define REAL double
#endif
#define NX (256)
#ifndef M_PI
#define M_PI (3.1415926535897932384626)
#endif

void init(REAL *buff, const int nx, const int ny, const int nz,
          const REAL kx, const REAL ky, const REAL kz,
          const REAL dx, const REAL dy, const REAL dz,
          const REAL kappa, const REAL time) {
  REAL ax, ay, az;
  int jz, jy, jx;
  ax = exp(-kappa*time*(kx*kx));
  ay = exp(-kappa*time*(ky*ky));
  az = exp(-kappa*time*(kz*kz));
#pragma omp parallel for private(jy,jx)
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
        buff[j] = f0;
      }
    }
  }
}

REAL accuracy(const REAL *b1, REAL *b2, const int len) {
  REAL err = 0.0;
  int i;
  for (i = 0; i < len; i++) {
    err += (b1[i] - b2[i]) * (b1[i] - b2[i]);
  }
  return (REAL)sqrt(err/len);
}

typedef void (*diffusion_loop_t)(REAL *f1, REAL *f2, int nx, int ny, int nz,
                                 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
                                 REAL cb, REAL cc, REAL dt,
                                 REAL **f_ret, REAL *time_ret, int *count_ret);


// openacc
static void
diffusion_openacc(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
                   REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
                   REAL cb, REAL cc, REAL dt,
                   REAL **f_ret, REAL *time_ret, int *count_ret) {

  
  {
    REAL time = 0.0;
    int count = 0;
    REAL *restrict f1_t = f1;
    REAL *restrict f2_t = f2;
    int c, w, e, n, s, b, t;
    int z, y, x;
    
    do {
#pragma omp parallel for private(x,y,z,c,w,e,n,s,b,t)
#pragma acc kernels copy(f1_t[0:nx*ny*nz]) copy(f2_t[0:nx*ny*nz])
#pragma acc loop independent
      for (z = 0; z < nz; z++) {
#pragma acc loop independent
	for (y = 0; y < ny; y++) {
#pragma acc loop independent
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
      REAL *t = f1_t;
      f1_t = f2_t;
      f2_t = t;
      time += dt;
      //time += 1;
      count++;
    } while (time + 0.5*dt < 0.1);
    *f_ret = f1_t;
    *time_ret = time;      
    *count_ret = count;        
  }

  return;
}

// openmp
static void
diffusion_openmp(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		 REAL cb, REAL cc, REAL dt,
		 REAL **f_ret, REAL *time_ret, int *count_ret) {

  
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
      REAL *tmp = f1_t;
      f1_t = f2_t;
      f2_t = tmp;
      time += dt;
      //time += 1;
      count++;
    } while (time + 0.5*dt < 0.1);
    *f_ret = f1_t;
    *time_ret = time;      
    *count_ret = count;        
  }

  return;
}

#pragma omp declare simd uniform(cc,cw,ce,cs,cn,cb,ct) linear(c8,w8,e8,s8,n8,b8,t8)
static double calc(double cc,double cw,double ce,double cs,double cn,double cb,double ct,
		   double c8,double w8,double e8,double s8,double n8,double b8,double t8){
  return cc * c8 + cw * w8 + ce * e8 + cs * s8 + cn * n8 + cb * b8 + ct * t8;
}


// openmp
static void
diffusion_openmp_declare_simd(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		 REAL cb, REAL cc, REAL dt,
		 REAL **f_ret, REAL *time_ret, int *count_ret) {

  
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
	  double c8[8],w8[8],e8[8],s8[8],n8[8],b8[8],t8[8],ans[8];
	  for (x = 0; x < nx; x+=8) {
	    int xx; 
	    c = x + y * nx + z * nx * ny;
#pragma omp simd
	    for(xx = 0; xx < 8; xx++) {
	      c = c + xx;
	      c8[xx] =  f1_t[c];
	      w8[xx] = (x == 0)    ? f1_t[c] : f1_t[c - 1];
	      e8[xx] = (x == nx-1) ? f1_t[c] : f1_t[c + 1];
	      n8[xx] = (y == 0)    ? f1_t[c] : f1_t[c - nx];
	      s8[xx] = (y == ny-1) ? f1_t[c] : f1_t[c + nx];
	      b8[xx] = (z == 0)    ? f1_t[c] : f1_t[c - nx * ny];
	      t8[xx] = (z == nz-1) ? f1_t[c] : f1_t[c + nx * ny];
	    }
#pragma omp simd
	    for(xx = 0; xx < 8; xx++) {
	      ans[xx] = calc(cc,cw,ce,cs,cn,cb,ct,
			     c8[xx],w8[xx],e8[xx],s8[xx],
			     n8[xx],b8[xx],t8[xx]);
	    }
	    c = x + y * nx + z * nx * ny;
#pragma omp simd
	    for(xx = 0; xx < 8; xx++) {
	      f2_t[c + xx] = ans[xx];
	    }
	  }
	}
      }
      REAL *tmp = f1_t;
      f1_t = f2_t;
      f2_t = tmp;
      time += dt;
      //time += 1;
      count++;
    } while (time + 0.5*dt < 0.1);
    *f_ret = f1_t;
    *time_ret = time;      
    *count_ret = count;        
  }

  return;
}


static void
diffusion_openmp_old(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		 REAL cb, REAL cc, REAL dt,
		 REAL **f_ret, REAL *time_ret, int *count_ret) {

  
  {
    REAL time = 0.0;
    int count = 0;
    REAL *restrict f1_t = f1;
    REAL *restrict f2_t = f2;
    int c, w, e, n, s, b, t;
    int z, y, x;
    REAL ccc[NX], nnn[NX], sss[NX], ttt[NX], bbb[NX];
    
    
    /* REAL *ccc, *nnn, *sss, *ttt, *bbb; */
    /* posix_memalign((void**)&ccc, 64, sizeof(REAL)*nx); */
    /* posix_memalign((void**)&nnn, 64, sizeof(REAL)*nx); */
    /* posix_memalign((void**)&sss, 64, sizeof(REAL)*nx); */
    /* posix_memalign((void**)&ttt, 64, sizeof(REAL)*nx); */
    /* posix_memalign((void**)&bbb, 64, sizeof(REAL)*nx); */
    do {
#pragma omp parallel for private(x,y,z,c,w,e,n,s,b,t,ccc,nnn,sss,ttt,bbb)
      for (z = 0; z < nz; z++) {
	b = (z == 0)    ? 0 : -1;
	t = (z == nz-1) ? 0 :  1;
	for(x = 0; x < nx; x++){
	  ccc[x] = f1_t[x + 0 * nx + (z + 0) * nx * ny];
	  nnn[x] = ccc[x];
	  sss[x] = f1_t[x + 1 * nx + (z + 0) * nx * ny];
	  ttt[x] = f1_t[x + 0 * nx + (z + t) * nx * ny];
	  bbb[x] = f1_t[x + 0 * nx + (z + b) * nx * ny];
	}
	for (y = 0; y < ny; y++) {
	  for (x = 0; x < nx; x++) {
	    c =  x + y * nx + z * nx * ny;
	    w = (x == 0)    ? 0 : - 1;
	    e = (x == nx-1) ? 0 :   1;
	    f2_t[c] = cc * ccc[x] + cw * ccc[x+w] + ce * ccc[x+e]
	      + cs * sss[x] + cn * nnn[x] + cb * bbb[x] + ct * ttt[x];
	  }
	  if(y != ny-1){
	    for(x = 0; x < nx; x++){
	      nnn[x] = ccc[x];
	      ccc[x] = sss[x];
	      sss[x] = f1_t[x + (y + 1) * nx + (z + 0) * nx * ny];
	      ttt[x] = f1_t[x + (y + 0) * nx + (z + t) * nx * ny];
	      bbb[x] = f1_t[x + (y + 0) * nx + (z + b) * nx * ny];
	    }
	  }
	}
      }
      REAL *tmp = f1_t;
      f1_t = f2_t;
      f2_t = tmp;
      time += dt;
      //time += 1;
      count++;
    } while (time + 0.5*dt < 0.1);
    *f_ret = f1_t;
    *time_ret = time;      
    *count_ret = count;        
  }
  return;
}


static void
diffusion_openmp_y(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		 REAL cb, REAL cc, REAL dt,
		 REAL **f_ret, REAL *time_ret, int *count_ret) {

  
  {
    REAL time = 0.0;
    int count = 0;
    REAL *restrict f1_t = f1;
    REAL *restrict f2_t = f2;
    int c, w, e, n, s, b, t;
    int z, y, x;

    do {
#pragma omp parallel private(x,y,z,c,w,e,n,s,b,t)
      for (z = 0; z < nz; z++) {
#pragma omp for 
	for (y = 0; y < ny; y++) {
	  //#pragma omp simd private(c,w,e,n,s,b,t) linear(c:1,n:1,s:1,b:1,t:1) 
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
      REAL *tmp = f1_t;
      f1_t = f2_t;
      f2_t = tmp;
      time += dt;
      //time += 1;
      count++;
    } while (time + 0.5*dt < 0.1);
    *f_ret = f1_t;
    *time_ret = time;      
    *count_ret = count;        
  }

  return;
}



static void
diffusion_openmp_y_nowait(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		 REAL cb, REAL cc, REAL dt,
		 REAL **f_ret, REAL *time_ret, int *count_ret) {

  
  {
    REAL time = 0.0;
    int count = 0;
    REAL *restrict f1_t = f1;
    REAL *restrict f2_t = f2;
    int c, w, e, n, s, b, t;
    int z, y, x;

    do {
#pragma omp parallel private(x,y,z,c,w,e,n,s,b,t)
      for (z = 0; z < nz; z++) {
#pragma omp for nowait
	for (y = 0; y < ny; y++) {
	  //#pragma omp simd private(c,w,e,n,s,b,t) linear(c:1,n:1,s:1,b:1,t:1) 
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
      REAL *tmp = f1_t;
      f1_t = f2_t;
      f2_t = tmp;
      time += dt;
      //time += 1;
      count++;
    } while (time + 0.5*dt < 0.1);
    *f_ret = f1_t;
    *time_ret = time;      
    *count_ret = count;        
  }

  return;
}


static void
diffusion_openmp_y_nowait_simdwrong(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		 REAL cb, REAL cc, REAL dt,
		 REAL **f_ret, REAL *time_ret, int *count_ret) {

  
  {
    REAL time = 0.0;
    int count = 0;
    REAL *restrict f1_t = f1;
    REAL *restrict f2_t = f2;
    int c, w, e, n, s, b, t;
    int z, y, x;

    do {
#pragma omp parallel private(x,y,z,c,w,e,n,s,b,t)
      for (z = 0; z < nz; z++) {
#pragma omp for nowait
	for (y = 0; y < ny; y++) {
#pragma omp simd
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
      REAL *tmp = f1_t;
      f1_t = f2_t;
      f2_t = tmp;
      time += dt;
      //time += 1;
      count++;
    } while (time + 0.5*dt < 0.1);
    *f_ret = f1_t;
    *time_ret = time;      
    *count_ret = count;        
  }

  return;
}


static void
diffusion_openmp_y_nowait_simd(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		 REAL cb, REAL cc, REAL dt,
		 REAL **f_ret, REAL *time_ret, int *count_ret) {

  
  {
    REAL time = 0.0;
    int count = 0;
    REAL *restrict f1_t = f1;
    REAL *restrict f2_t = f2;
    int c, w, e, n, s, b, t;
    int z, y, x;

    do {
#pragma omp parallel private(x,y,z,c,w,e,n,s,b,t)
      for (z = 0; z < nz; z++) {
#pragma omp for nowait
	for (y = 0; y < ny; y++) {
#pragma omp simd private(w,e) linear(c:1,n:1,s:1,b:1,t:1) 
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
      REAL *tmp = f1_t;
      f1_t = f2_t;
      f2_t = tmp;
      time += dt;
      //time += 1;
      count++;
    } while (time + 0.5*dt < 0.1);
    *f_ret = f1_t;
    *time_ret = time;      
    *count_ret = count;        
  }

  return;
}



static void
diffusion_openmp_y_nowait_simd_peel(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		 REAL cb, REAL cc, REAL dt,
		 REAL **f_ret, REAL *time_ret, int *count_ret) {

  
  {
    REAL time = 0.0;
    int count = 0;
    REAL *restrict f1_t = f1;
    REAL *restrict f2_t = f2;
    int c, w, e, n, s, b, t;
    int z, y, x;

    do {
#pragma omp parallel private(x,y,z,c,w,e,n,s,b,t)
      for (z = 0; z < nz; z++) {
#pragma omp for nowait
	for (y = 0; y < ny; y++) {
	  n = (y == 0)    ? 0 : - nx;
	  s = (y == ny-1) ? 0 :   nx;
	  b = (z == 0)    ? 0 : - nx * ny;
	  t = (z == nz-1) ? 0 :   nx * ny;
	  c =  y * nx + z * nx * ny;
	  for(x = 0; x < ((8 < nx) ? 8:nx); x++,c++){
	    //	  for(x = 0; x < 8; x++,c++){
	    w = (x == 0)    ? 0 : - 1;
	    e = (x == nx-1) ? 0 :   1;
	    f2_t[c] = cc * f1_t[c] + cw * f1_t[c+w] + ce * f1_t[c+e]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	  }
	  /* __assume_aligned(f1_t, 64); */
	  /* __assume_aligned(f2_t, 64); */
	  /* __assume(nx%8==0); */
	  /* __assume(ny%8==0); */
	  /* __assume(s%8==0); */
	  /* __assume(n%8==0); */
	  /* __assume(b%8==0); */
	  /* __assume(t%8==0); */
#pragma omp simd 
	  for (x = 8; x < nx-8; x++,c++) {
	    f2_t[c] = cc * f1_t[c] + cw * f1_t[c-1] + ce * f1_t[c+1]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	  }
	  for (x = nx-8; x < nx; x++,c++) {
	    e = (x == nx-1) ? 0 :   1;
	    f2_t[c] = cc * f1_t[c] + cw * f1_t[c-1] + ce * f1_t[c+e]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	  }
	}
      }
      REAL *tmp = f1_t;
      f1_t = f2_t;
      f2_t = tmp;
      time += dt;
      //time += 1;
      count++;

    } while (time + 0.5*dt < 0.1);
    *f_ret = f1_t;
    *time_ret = time;      
    *count_ret = count;        
  }

  return;
}



static void
diffusion_openmp_y_nowait_simd_peel_aligned(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		 REAL cb, REAL cc, REAL dt,
		 REAL **f_ret, REAL *time_ret, int *count_ret) {

  
  {
    REAL time = 0.0;
    int count = 0;
    REAL *restrict f1_t = f1;
    REAL *restrict f2_t = f2;
    int c, w, e, n, s, b, t;
    int z, y, x;
    
    do {
#pragma omp parallel private(x,y,z,c,w,e,n,s,b,t)
      for (z = 0; z < nz; z++) {
#pragma omp for nowait
	for (y = 0; y < ny; y++) {
	  n = (y == 0)    ? 0 : - nx;
	  s = (y == ny-1) ? 0 :   nx;
	  b = (z == 0)    ? 0 : - nx * ny;
	  t = (z == nz-1) ? 0 :   nx * ny;
	  c =  y * nx + z * nx * ny;
	  for(x = 0; x < ((8 < nx) ? 8:nx); x++,c++){
	    //	  for(x = 0; x < 8; x++,c++){
	    w = (x == 0)    ? 0 : - 1;
	    e = (x == nx-1) ? 0 :   1;
	    f2_t[c] = cc * f1_t[c] + cw * f1_t[c+w] + ce * f1_t[c+e]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	  }
	  __assume_aligned(f1_t, 64);
	  __assume_aligned(f2_t, 64);
	  __assume(nx%8==0);
	  __assume(ny%8==0);
	  __assume(s%8==0);
	  __assume(n%8==0);
	  __assume(b%8==0);
	  __assume(t%8==0);
#pragma omp simd 
	  for (x = 8; x < nx-8; x++,c++) {
	    f2_t[c] = cc * f1_t[c] + cw * f1_t[c-1] + ce * f1_t[c+1]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	  }
	  for (x = nx-8; x < nx; x++,c++) {
	    e = (x == nx-1) ? 0 :   1;
	    f2_t[c] = cc * f1_t[c] + cw * f1_t[c-1] + ce * f1_t[c+e]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	  }
	}
      }
      REAL *tmp = f1_t;
      f1_t = f2_t;
      f2_t = tmp;
      time += dt;
      //time += 1;
      count++;
      
    } while (time + 0.5*dt < 0.1);
    *f_ret = f1_t;
    *time_ret = time;      
    *count_ret = count;        
  }
  
  return;
}



static void
diffusion_openmp_y_nowait_simd_peel_aligned_mvparallel(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		 REAL cb, REAL cc, REAL dt,
		 REAL **f_ret, REAL *time_ret, int *count_ret) {

  
  {

#pragma omp parallel 
    {
    REAL time;
    int count = 0;
    REAL *f1_t = f1;
    REAL *f2_t = f2;
    int c, w, e, n, s, b, t;
    int z, y, x;
    int i; 
    for(time = 0.0; time+0.5*dt < 0.1; time += dt,count++) {
      for (z = 0; z < nz; z++) {
#pragma omp for nowait
	for (y = 0; y < ny; y++) {
	  n = (y == 0)    ? 0 : - nx;
	  s = (y == ny-1) ? 0 :   nx;
	  b = (z == 0)    ? 0 : - nx * ny;
	  t = (z == nz-1) ? 0 :   nx * ny;
	  c =  y * nx + z * nx * ny;
	  for(x = 0; x < ((8 < nx) ? 8:nx); x++,c++){
	    //	  for(x = 0; x < 8; x++,c++){
	    w = (x == 0)    ? 0 : - 1;
	    e = (x == nx-1) ? 0 :   1;
	    f2_t[c] = cc * f1_t[c] + cw * f1_t[c+w] + ce * f1_t[c+e]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	  }
	  __assume_aligned(f1_t, 64);
	  __assume_aligned(f2_t, 64);
	  __assume(nx%8==0);
	  __assume(ny%8==0);
	  __assume(s%8==0);
	  __assume(n%8==0);
	  __assume(b%8==0);
	  __assume(t%8==0);
#pragma omp simd 
	  for (x = 8; x < nx-8; x++,c++) {
	    f2_t[c] = cc * f1_t[c] + cw * f1_t[c-1] + ce * f1_t[c+1]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	  }
	  for (x = nx-8; x < nx; x++,c++) {
	    e = (x == nx-1) ? 0 :   1;
	    f2_t[c] = cc * f1_t[c] + cw * f1_t[c-1] + ce * f1_t[c+e]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	  }
	}
      }
#pragma omp barrier
      REAL *tmp = f1_t;
      f1_t = f2_t;
      f2_t = tmp;
    }
    
#pragma omp master
    {
      *f_ret = f1_t;
      *time_ret = time;      
      *count_ret = count;
    }
    }

    return;
  }
}


static void
diffusion_openmp_KNC_book_asis(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		 REAL cb, REAL cc, REAL dt,
		 REAL **f_ret, REAL *time_ret, int *count_ret) {

  
  {

#pragma omp parallel 
    {
      REAL *f1_t = f1;
      REAL *f2_t = f2;
      int c, w, e, n, s, b, t;
      int z, y, x, yy;
      int i; 
      REAL time;
      int count = 0;
      for(time = 0.0; time+0.5*dt < 0.1; time += dt,count++) {
#define YBF 16
#pragma omp for collapse(2)
	for(yy = 0;yy < ny; yy+=YBF){
	  for (z = 0; z < nz; z++) {
	    int ymax = yy + YBF;
	    if(ymax >= ny) ymax = ny;
	    for (y = yy; y < ymax; y++) {
	      x = 0;
	      c = x + y * nx + z * nx * ny;
	      n = (y == 0)    ? c : c - nx;
	      s = (y == ny-1) ? c : c + nx;
	      b = (z == 0)    ? c : c - nx * ny;
	      t = (z == nz-1) ? c : c + nx * ny;
	      f2_t[c] = cc * f1_t[c] + cw * f1_t[c] + ce * f1_t[c+1]
		+ cs * f1_t[s] + cn * f1_t[n] + cb * f1_t[b] + ct * f1_t[t];
#pragma omp simd 
		for (x = 1; x < nx-1; x++) {
		  ++c;
		  ++n;
		  ++s;
		  ++b;
		  ++t;
		  f2_t[c] = cc * f1_t[c] + cw * f1_t[c-1] + ce * f1_t[c+1]
		    + cs * f1_t[s] + cn * f1_t[n] + cb * f1_t[b] + ct * f1_t[t];
		}
		++c;
		++n;
		++s;
		++b;
		++t;
		f2_t[c] = cc * f1_t[c] + cw * f1_t[c-1] + ce * f1_t[c]
		  + cs * f1_t[s] + cn * f1_t[n] + cb * f1_t[b] + ct * f1_t[t];
	    }
	  }
	}
	REAL *tmp = f1_t;
	f1_t = f2_t;
	f2_t = tmp;
      }
#undef YBF
#pragma omp master
      {
	*f_ret = f1_t;
	*time_ret = time;
	*count_ret = count;
      }
    }
    
    return;
  }
}


static void
diffusion_openmp_tiled_nowait_simd_peel_aligned(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		 REAL cb, REAL cc, REAL dt,
		 REAL **f_ret, REAL *time_ret, int *count_ret) {

  
  {
    REAL time = 0.0;
    int count = 0;
    REAL *restrict f1_t = f1;
    REAL *restrict f2_t = f2;
    int c, w, e, n, s, b, t;
    int z, y, x;

    do {
      int yy;
#define YBF 8
#pragma omp parallel private(x,y,z,c,w,e,n,s,b,t,yy)
#pragma omp for collapse(2)
      for (yy = 0; yy < ny; yy += YBF) {
      for (z = 0; z < nz; z++) {
	for (y = yy; y < (yy+YBF < ny ? yy+YBF:ny); y++) {
	  n = (y == 0)    ? 0 : - nx;
	  s = (y == ny-1) ? 0 :   nx;
	  b = (z == 0)    ? 0 : - nx * ny;
	  t = (z == nz-1) ? 0 :   nx * ny;
	  c =  y * nx + z * nx * ny;
	  __assume_aligned(f1_t, 64);
	  __assume_aligned(f2_t, 64);
	  __assume(nx%8==0);
	  __assume(ny%8==0);
	  __assume(s%8==0);
	  __assume(n%8==0);
	  __assume(b%8==0);
	  __assume(t%8==0);
	  __assume(c%8==0);
#pragma omp simd 
	  //	  for(x = 0; x < ((8 < nx) ? 8:nx); x++,c++){
	  for(x = 0; x < 8; x++,c++){
	    w = (x == 0)    ? 0 : - 1;
	    e = (x == nx-1) ? 0 :   1;
	    f2_t[c] = cc * f1_t[c] + cw * f1_t[c+w] + ce * f1_t[c+e]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	  }
	  __assume_aligned(f1_t, 64);
	  __assume_aligned(f2_t, 64);
	  __assume(nx%8==0);
	  __assume(ny%8==0);
	  __assume(s%8==0);
	  __assume(n%8==0);
	  __assume(b%8==0);
	  __assume(t%8==0);
	  __assume(c%8==0);
#pragma omp simd 
	  for (x = 8; x < nx-8; x++,c++) {
	    f2_t[c] = cc * f1_t[c] + cw * f1_t[c-1] + ce * f1_t[c+1]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	  }
	  __assume_aligned(f1_t, 64);
	  __assume_aligned(f2_t, 64);
	  __assume(nx%8==0);
	  __assume(ny%8==0);
	  __assume(s%8==0);
	  __assume(n%8==0);
	  __assume(b%8==0);
	  __assume(t%8==0);
	  __assume(c%8==0);
#pragma omp simd 
	  for (x = nx-8; x < nx; x++,c++) {
	    e = (x == nx-1) ? 0 :   1;
	    f2_t[c] = cc * f1_t[c] + cw * f1_t[c-1] + ce * f1_t[c+e]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	  }
	}
      }
      }
      REAL *tmp = f1_t;
      f1_t = f2_t;
      f2_t = tmp;
      time += dt;
      //time += 1;
      count++;

    } while (time + 0.5*dt < 0.1);
    *f_ret = f1_t;
    *time_ret = time;      
    *count_ret = count;        
  }
#undef YBF

  return;
}



static void
diffusion_openmp_tiled_nowait_simd_peel_aligned_mvparallel(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		 REAL cb, REAL cc, REAL dt,
		 REAL **f_ret, REAL *time_ret, int *count_ret) {

#pragma omp parallel 
  {
    REAL time = 0.0;
    int count = 0;
    REAL *restrict f1_t = f1;
    REAL *restrict f2_t = f2;
    int c, w, e, n, s, b, t;
    int z, y, x;
    int i; 

    do {
      int yy;
#define YBF 8
#pragma omp for collapse(2)
      for (yy = 0; yy < ny; yy += YBF) {
      for (z = 0; z < nz; z++) {
	for (y = yy; y < (yy+YBF < ny ? yy+YBF:ny); y++) {
	  n = (y == 0)    ? 0 : - nx;
	  s = (y == ny-1) ? 0 :   nx;
	  b = (z == 0)    ? 0 : - nx * ny;
	  t = (z == nz-1) ? 0 :   nx * ny;
	  c =  y * nx + z * nx * ny;
	  __assume_aligned(f1_t, 64);
	  __assume_aligned(f2_t, 64);
	  __assume(nx%8==0);
	  __assume(ny%8==0);
	  __assume(s%8==0);
	  __assume(n%8==0);
	  __assume(b%8==0);
	  __assume(t%8==0);
	  __assume(c%8==0);
#pragma omp simd 
	  //	  for(x = 0; x < ((8 < nx) ? 8:nx); x++,c++){
	  for(x = 0; x < 8; x++,c++){
	    w = (x == 0)    ? 0 : - 1;
	    e = (x == nx-1) ? 0 :   1;
	    f2_t[c] = cc * f1_t[c] + cw * f1_t[c+w] + ce * f1_t[c+e]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	  }
	  __assume_aligned(f1_t, 64);
	  __assume_aligned(f2_t, 64);
	  __assume(nx%8==0);
	  __assume(ny%8==0);
	  __assume(s%8==0);
	  __assume(n%8==0);
	  __assume(b%8==0);
	  __assume(t%8==0);
	  __assume(c%8==0);
#pragma omp simd 
	  for (x = 8; x < nx-8; x++,c++) {
	    f2_t[c] = cc * f1_t[c] + cw * f1_t[c-1] + ce * f1_t[c+1]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	  }
	  __assume_aligned(f1_t, 64);
	  __assume_aligned(f2_t, 64);
	  __assume(nx%8==0);
	  __assume(ny%8==0);
	  __assume(s%8==0);
	  __assume(n%8==0);
	  __assume(b%8==0);
	  __assume(t%8==0);
	  __assume(c%8==0);
#pragma omp simd 
	  for (x = nx-8; x < nx; x++,c++) {
	    e = (x == nx-1) ? 0 :   1;
	    f2_t[c] = cc * f1_t[c] + cw * f1_t[c-1] + ce * f1_t[c+e]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	  }
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
      *f_ret = f1_t;
      *time_ret = time;      
      *count_ret = count;        
    }
    
  }
#undef YBF

  return;
}


static void
diffusion_openmp_tiled_nowait_simd_peel_aligned_mvparallel_mvfor(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		 REAL cb, REAL cc, REAL dt,
		 REAL **f_ret, REAL *time_ret, int *count_ret) {

  /* { */
  /*   int tid; */
  /*   for(tid = 0;tid < 66;tid++){ */
  /*     int idy = tid % 32; */
  /*     if(idy % 8 == 0 || idy % 8 == 1) */
  /* 	idy = 2 * (idy / 8) + (idy % 2); */
  /*     else if(idy % 8 == 2 || idy % 8 == 3) */
  /* 	idy = 8 + 2 * (idy / 8) + (idy % 2); */
  /*     else if(idy % 8 == 4 || idy % 8 == 5) */
  /* 	idy = 16 + 2 * (idy / 8) + (idy % 2); */
  /*     else */
  /* 	idy = 24 + 2 * (idy / 8) + (idy % 2); */
  /*     idy = idy * YBF; */
  /*     int idz = tid / 32; */
  /*     printf("%d,(%d,%d)\n",tid,idy,idz); */
  /*   } */
  /* } */

#define YBF 8
#pragma omp parallel 
  {
    REAL time = 0.0;
    int count = 0;
    REAL *restrict f1_t = f1;
    REAL *restrict f2_t = f2;
    int c, w, e, n, s, b, t;
    int z, y, x;
    int tid = omp_get_thread_num();
    int nth = omp_get_num_threads();
    int nthpy = (ny-1)/YBF + 1;
    int nthpz = (nth-1)/nthpy + 1;
    int idy = tid % nthpy;
    idy = idy * YBF;
    int idz = tid / nthpy * nz / nthpz;
    do {
      int yy;
      for (yy = idy; yy < idy+1; yy++) {
	for (z = idz; z < idz + nz/nthpz; z++) {
	  for (y = yy; y < (yy+YBF < ny ? yy+YBF:ny); y++) {
	    n = (y == 0)    ? 0 : - nx;
	    s = (y == ny-1) ? 0 :   nx;
	    b = (z == 0)    ? 0 : - nx * ny;
	    t = (z == nz-1) ? 0 :   nx * ny;
	    c =  y * nx + z * nx * ny;
	    __assume_aligned(f1_t, 64);
	    __assume_aligned(f2_t, 64);
	    __assume(nx%8==0);
	    __assume(ny%8==0);
	    __assume(s%8==0);
	    __assume(n%8==0);
	    __assume(b%8==0);
	    __assume(t%8==0);
	    __assume(c%8==0);
#pragma omp simd 
	    //	  for(x = 0; x < ((8 < nx) ? 8:nx); x++,c++){
	    for(x = 0; x < 8; x++,c++){
	      w = (x == 0)    ? 0 : - 1;
	      e = (x == nx-1) ? 0 :   1;
	      f2_t[c] = cc * f1_t[c] + cw * f1_t[c+w] + ce * f1_t[c+e]
		+ cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	    }
	    __assume_aligned(f1_t, 64);
	    __assume_aligned(f2_t, 64);
	    __assume(nx%8==0);
	    __assume(ny%8==0);
	    __assume(s%8==0);
	    __assume(n%8==0);
	    __assume(b%8==0);
	    __assume(t%8==0);
	    __assume(c%8==0);
#pragma omp simd 
	    for (x = 8; x < nx-8; x++,c++) {
	      f2_t[c] = cc * f1_t[c] + cw * f1_t[c-1] + ce * f1_t[c+1]
		+ cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	    }
	    __assume_aligned(f1_t, 64);
	    __assume_aligned(f2_t, 64);
	    __assume(nx%8==0);
	    __assume(ny%8==0);
	    __assume(s%8==0);
	    __assume(n%8==0);
	    __assume(b%8==0);
	    __assume(t%8==0);
	    __assume(c%8==0);
#pragma omp simd 
	    for (x = nx-8; x < nx; x++,c++) {
	      e = (x == nx-1) ? 0 :   1;
	      f2_t[c] = cc * f1_t[c] + cw * f1_t[c-1] + ce * f1_t[c+e]
		+ cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	    }
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
      *f_ret = f1_t;
      *time_ret = time;      
      *count_ret = count;        
    }
    
  }
#undef YBF
  
  return;
}

static void
diffusion_openmp_tiled_nowait_simd_peel_aligned_mvparallel_mvfor2(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		 REAL cb, REAL cc, REAL dt,
		 REAL **f_ret, REAL *time_ret, int *count_ret) {

  /* { */
  /*   int tid; */
  /*   for(tid = 0;tid < 66;tid++){ */
  /*     int idy = tid % 32; */
  /*     if(idy % 8 == 0 || idy % 8 == 1) */
  /* 	idy = 2 * (idy / 8) + (idy % 2); */
  /*     else if(idy % 8 == 2 || idy % 8 == 3) */
  /* 	idy = 8 + 2 * (idy / 8) + (idy % 2); */
  /*     else if(idy % 8 == 4 || idy % 8 == 5) */
  /* 	idy = 16 + 2 * (idy / 8) + (idy % 2); */
  /*     else */
  /* 	idy = 24 + 2 * (idy / 8) + (idy % 2); */
  /*     idy = idy * YBF; */
  /*     int idz = tid / 32; */
  /*     printf("%d,(%d,%d)\n",tid,idy,idz); */
  /*   } */
  /* } */

#define YBF 8
#pragma omp parallel 
  {
    REAL time = 0.0;
    int count = 0;
    REAL *restrict f1_t = f1;
    REAL *restrict f2_t = f2;
    int c, w, e, n, s, b, t;
    int z, y, x;
    int tid = omp_get_thread_num();
    int nth = omp_get_num_threads();
    int nthpy = (ny-1)/YBF + 1;
    int nthpz = (nth-1)/nthpy + 1;
    int idy = tid % nthpy;
    if(idy % 8 == 0 || idy % 8 == 1)
      idy =             2 * (idy / 8) + (idy % 2);
    else if(idy % 8 == 2 || idy % 8 == 3)
      idy = nthpy/4   + 2 * (idy / 8) + (idy % 2);
    else if(idy % 8 == 4 || idy % 8 == 5)
      idy = nthpy/4*2 + 2 * (idy / 8) + (idy % 2);
    else
      idy = nthpy/4*3 + 2 * (idy / 8) + (idy % 2);
    idy = idy * YBF;
    int idz = tid / nthpy * nz / nthpz;
    do {
      int yy;
      for (yy = idy; yy < idy+1; yy++) {
	for (z = idz; z < idz + nz/nthpz; z++) {
	  for (y = yy; y < (yy+YBF < ny ? yy+YBF:ny); y++) {
	    n = (y == 0)    ? 0 : - nx;
	    s = (y == ny-1) ? 0 :   nx;
	    b = (z == 0)    ? 0 : - nx * ny;
	    t = (z == nz-1) ? 0 :   nx * ny;
	    c =  y * nx + z * nx * ny;
	    __assume_aligned(f1_t, 64);
	    __assume_aligned(f2_t, 64);
	    __assume(nx%8==0);
	    __assume(ny%8==0);
	    __assume(s%8==0);
	    __assume(n%8==0);
	    __assume(b%8==0);
	    __assume(t%8==0);
	    __assume(c%8==0);
#pragma omp simd 
	    //	  for(x = 0; x < ((8 < nx) ? 8:nx); x++,c++){
	    for(x = 0; x < 8; x++,c++){
	      w = (x == 0)    ? 0 : - 1;
	      e = (x == nx-1) ? 0 :   1;
	      f2_t[c] = cc * f1_t[c] + cw * f1_t[c+w] + ce * f1_t[c+e]
		+ cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	    }
	    __assume_aligned(f1_t, 64);
	    __assume_aligned(f2_t, 64);
	    __assume(nx%8==0);
	    __assume(ny%8==0);
	    __assume(s%8==0);
	    __assume(n%8==0);
	    __assume(b%8==0);
	    __assume(t%8==0);
	    __assume(c%8==0);
#pragma omp simd 
	    for (x = 8; x < nx-8; x++,c++) {
	      f2_t[c] = cc * f1_t[c] + cw * f1_t[c-1] + ce * f1_t[c+1]
		+ cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	    }
	    __assume_aligned(f1_t, 64);
	    __assume_aligned(f2_t, 64);
	    __assume(nx%8==0);
	    __assume(ny%8==0);
	    __assume(s%8==0);
	    __assume(n%8==0);
	    __assume(b%8==0);
	    __assume(t%8==0);
	    __assume(c%8==0);
#pragma omp simd 
	    for (x = nx-8; x < nx; x++,c++) {
	      e = (x == nx-1) ? 0 :   1;
	      f2_t[c] = cc * f1_t[c] + cw * f1_t[c-1] + ce * f1_t[c+e]
		+ cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	    }
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
      *f_ret = f1_t;
      *time_ret = time;      
      *count_ret = count;        
    }
    
  }
#undef YBF
  
  return;
}

#ifdef INTEL

static void
diffusion_openmp_intrin(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		 REAL cb, REAL cc, REAL dt,
		 REAL **f_ret, REAL *time_ret, int *count_ret) {

#define YBF 8
#pragma omp parallel 
  {
    REAL time = 0.0;
    int count = 0;
    REAL *restrict f1_t = f1;
    REAL *restrict f2_t = f2;
    int c, w, e, n, s, b, t;
    int z, y, x;
    int tid = omp_get_thread_num();
    int nth = omp_get_num_threads();
    int nthpy = (ny-1)/YBF + 1;
    int nthpz = (nth-1)/nthpy + 1;
    int idy = tid % nthpy;
    /* if(idy % 8 == 0 || idy % 8 == 1) */
    /*   idy =             2 * (idy / 8) + (idy % 2); */
    /* else if(idy % 8 == 2 || idy % 8 == 3) */
    /*   idy = nthpy/4   + 2 * (idy / 8) + (idy % 2); */
    /* else if(idy % 8 == 4 || idy % 8 == 5) */
    /*   idy = nthpy/4*2 + 2 * (idy / 8) + (idy % 2); */
    /* else */
    /*   idy = nthpy/4*3 + 2 * (idy / 8) + (idy % 2); */
    idy = idy * YBF;
    int zchunk = nz/nthpz;
    int idz = tid / nthpy * zchunk;
    const __m512d cc_vec = _mm512_set1_pd(cc);
    const __m512d cw_vec = _mm512_set1_pd(cw);
    const __m512d ce_vec = _mm512_set1_pd(ce);
    const __m512d cs_vec = _mm512_set1_pd(cs);
    const __m512d cn_vec = _mm512_set1_pd(cn);
    const __m512d cb_vec = _mm512_set1_pd(cb);
    const __m512d ct_vec = _mm512_set1_pd(ct);
    do {
      int yy;
      for (yy = idy; yy < idy+1; yy++) {
	for (z = idz; z < idz + zchunk; z++) {
	  b = (z == 0)    ? 0 : - nx * ny;
	  t = (z == nz-1) ? 0 :   nx * ny;
	  for (y = yy; y < (yy+YBF < ny ? yy+YBF:ny); y++) {
	    n = (y == 0)    ? 0 : - nx;
	    s = (y == ny-1) ? 0 :   nx;
	    c =  y * nx + z * nx * ny;
	    __m512d fc_vec   = _mm512_load_pd(f1_t+c);
	    __m512d fcp1_vec = _mm512_load_pd(f1_t+c+8);
	    __m512d fcm1_vec = _mm512_alignr_epi64(fc_vec,fc_vec,1);

	    __m512d fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
	    __m512d fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
	    __m512d fcs_vec = _mm512_load_pd(f1_t+c+s);
	    __m512d fcn_vec = _mm512_load_pd(f1_t+c+n);
	    __m512d fcb_vec = _mm512_load_pd(f1_t+c+b);
	    __m512d fct_vec = _mm512_load_pd(f1_t+c+t);
	    __m512d tmp = _mm512_mul_pd(cc_vec,fc_vec);
	    tmp = _mm512_fmadd_pd(cw_vec,fcw_vec,tmp);
	    tmp = _mm512_fmadd_pd(ce_vec,fce_vec,tmp);
	    tmp = _mm512_fmadd_pd(cs_vec,fcs_vec,tmp);
	    tmp = _mm512_fmadd_pd(cn_vec,fcn_vec,tmp);
	    tmp = _mm512_fmadd_pd(cb_vec,fcb_vec,tmp);
	    tmp = _mm512_fmadd_pd(ct_vec,fct_vec,tmp);
	    _mm512_store_pd(f2_t+c,tmp);
#pragma unroll 
	    for (x = 8; x < nx-8; x+=8) {
	      fcm1_vec = fc_vec;
	      fc_vec   = fcp1_vec;
	      fcp1_vec = _mm512_load_pd(f1_t+c+8+x);
	      
	      fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
	      fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
	      fcs_vec = _mm512_load_pd(f1_t+c+x+s);
	      fcn_vec = _mm512_load_pd(f1_t+c+x+n);
	      fcb_vec = _mm512_load_pd(f1_t+c+x+b);
	      fct_vec = _mm512_load_pd(f1_t+c+x+t);
	      tmp = _mm512_mul_pd(cc_vec,fc_vec);
	      tmp = _mm512_fmadd_pd(cw_vec,fcw_vec,tmp);
	      tmp = _mm512_fmadd_pd(ce_vec,fce_vec,tmp);
	      tmp = _mm512_fmadd_pd(cs_vec,fcs_vec,tmp);
	      tmp = _mm512_fmadd_pd(cn_vec,fcn_vec,tmp);
	      tmp = _mm512_fmadd_pd(cb_vec,fcb_vec,tmp);
	      tmp = _mm512_fmadd_pd(ct_vec,fct_vec,tmp);

	      /* tmp = _mm512_mul_pd(cc_vec,fc_vec); */
	      /* fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7); */
	      /* tmp = _mm512_fmadd_pd(cw_vec,fcw_vec,tmp); */
	      /* fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1); */
	      /* tmp = _mm512_fmadd_pd(ce_vec,fce_vec,tmp); */
	      /* fcs_vec = _mm512_load_pd(f1_t+c+x+s); */
	      /* tmp = _mm512_fmadd_pd(cs_vec,fcs_vec,tmp); */
	      /* fcn_vec = _mm512_load_pd(f1_t+c+x+n); */
	      /* tmp = _mm512_fmadd_pd(cn_vec,fcn_vec,tmp); */
	      /* fcb_vec = _mm512_load_pd(f1_t+c+x+b); */
	      /* tmp = _mm512_fmadd_pd(cb_vec,fcb_vec,tmp); */
	      /* fct_vec = _mm512_load_pd(f1_t+c+x+t); */
	      /* tmp = _mm512_fmadd_pd(ct_vec,fct_vec,tmp); */
	      _mm512_store_pd(f2_t+c+x,tmp);
	    }
	    fcm1_vec = fc_vec;
	    fc_vec   = fcp1_vec;
	    fcp1_vec = _mm512_alignr_epi64(fc_vec,fc_vec,7);
	    
	    fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
	    fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
	    fcs_vec = _mm512_load_pd(f1_t+c+x+s);
	    fcn_vec = _mm512_load_pd(f1_t+c+x+n);
	    fcb_vec = _mm512_load_pd(f1_t+c+x+b);
	    fct_vec = _mm512_load_pd(f1_t+c+x+t);
	    tmp = _mm512_mul_pd(cc_vec,fc_vec);
	    tmp = _mm512_fmadd_pd(cw_vec,fcw_vec,tmp);
	    tmp = _mm512_fmadd_pd(ce_vec,fce_vec,tmp);
	    tmp = _mm512_fmadd_pd(cs_vec,fcs_vec,tmp);
	    tmp = _mm512_fmadd_pd(cn_vec,fcn_vec,tmp);
	    tmp = _mm512_fmadd_pd(cb_vec,fcb_vec,tmp);
	    tmp = _mm512_fmadd_pd(ct_vec,fct_vec,tmp);
	    _mm512_store_pd(f2_t+c+x,tmp);
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
      *f_ret = f1_t;
      *time_ret = time;      
      *count_ret = count;        
    }
    
  }
#undef YBF
  
  return;
}

#endif

static void
transform(REAL *restrict f1_dst, REAL *restrict f2_dst, REAL *restrict f1_src, REAL *restrict f2_src, int nx, int ny, int nz){

#define YBF 8
#pragma omp parallel 
  {
    int tid = omp_get_thread_num();
    int nth = omp_get_num_threads();
    int nthpy = (ny-1)/YBF + 1;
    int nthpz = (nth-1)/nthpy + 1;
    int idy = tid % nthpy;
    idy = idy * YBF;
    int zchunk = nz/nthpz;
    int idz = tid / nthpy * zchunk;
    int str = tid * nx * YBF * zchunk;

    int z, y, x;
    int yy;
    for (yy = idy; yy < idy+1; yy++) {
      for (z = idz; z < idz + zchunk; z++) {
	for (y = yy; y < (yy+YBF < ny ? yy+YBF:ny); y++) {
	  for (x = 0; x < nx; x++) {
	    int c_src =  x + y * nx + z * nx * ny;
	    int c_dst =  str + x + (y-idy) * nx + (z-idz) * nx * YBF;
	    f1_dst[c_dst] = f1_src[c_src];
	    f2_dst[c_dst] = f2_src[c_src];
	  }
	}
      }
    }
  }
#undef YBF
  return;

}

static void
retransform(REAL *restrict f1_src, REAL *restrict f2_src, REAL *restrict f1_dst, REAL *restrict f2_dst, int nx, int ny, int nz){

#define YBF 8
#pragma omp parallel 
  {
    int tid = omp_get_thread_num();
    int nth = omp_get_num_threads();
    int nthpy = (ny-1)/YBF + 1;
    int nthpz = (nth-1)/nthpy + 1;
    int idy = tid % nthpy;
    idy = idy * YBF;
    int zchunk = nz/nthpz;
    int idz = tid / nthpy * zchunk;
    int str = tid * nx * YBF * zchunk;

    int z, y, x;
    int yy;
    for (yy = idy; yy < idy+1; yy++) {
      for (z = idz; z < idz + zchunk; z++) {
	for (y = yy; y < (yy+YBF < ny ? yy+YBF:ny); y++) {
	  for (x = 0; x < nx; x++) {
	    int c_src =  x + y * nx + z * nx * ny;
	    int c_dst =  str + x + (y-idy) * nx + (z-idz) * nx * YBF;
	    f1_src[c_src] = f1_dst[c_dst];
	    f2_src[c_src] = f2_dst[c_dst];
	  }
	}
      }
    }
  }
#undef YBF
  
  return;

}

#ifdef INTEL
static void
diffusion_openmp_intrin_independent(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		 REAL cb, REAL cc, REAL dt,
		 REAL **f_ret, REAL *time_ret, int *count_ret) {

#define YBF 8
#pragma omp parallel 
  {
    REAL time = 0.0;
    int count = 0;
    REAL *restrict f1_t = f1;
    REAL *restrict f2_t = f2;
    int c, w, e, n, s, b, t;
    int z, y, x;
    int tid = omp_get_thread_num();
    int nth = omp_get_num_threads();
    int nthpy = (ny-1)/YBF + 1;
    int nthpz = (nth-1)/nthpy + 1;
    int idy = tid % nthpy;
    //    idy = idy * YBF;
    int zchunk = nz/nthpz;
    int idz = tid / nthpy;
    int n_id,s_id,b_id,t_id;
    /* if(idy == 0) */
    /*   n_id = idy; */
    /* else */
    /*   n_id = idy-1; */
    /* if(idy == nthpy-1) */
    /*   s_id = idy; */
    /* else */
    /*   s_id = idy+1; */
    /* if(idz == 0) */
    /*   b_id = idz; */
    /* else */
    /*   b_id = idz-nthpy; */
    /* if(idz == nthpz-1) */
    /*   t_id = idz; */
    /* else */
    /*   t_id = idz+nthpy; */
    if(idy == 0)
      n_id = 0;
    else
      n_id = - nx * YBF * zchunk + nx * (YBF-1);
    if(idy == nthpy-1)
      s_id = 0;
    else
      s_id = nx * YBF * zchunk - nx * (YBF-1);
    if(idz == 0)
      b_id = 0;
    else
      b_id = -(nthpy * nx * YBF * zchunk) + nx * YBF * (zchunk-1);
    if(idz == nthpz-1)
      t_id = 0;
    else
      t_id = nthpy * nx * YBF * zchunk - nx * YBF * (zchunk-1);

    const int str = tid * nx * YBF * zchunk;

    const __m512d cc_vec = _mm512_set1_pd(cc);
    const __m512d cw_vec = _mm512_set1_pd(cw);
    const __m512d ce_vec = _mm512_set1_pd(ce);
    const __m512d cs_vec = _mm512_set1_pd(cs);
    const __m512d cn_vec = _mm512_set1_pd(cn);
    const __m512d cb_vec = _mm512_set1_pd(cb);
    const __m512d ct_vec = _mm512_set1_pd(ct);
    do {
      c = str;
      for (z = 0; z < zchunk; z++) {
	b = ( z == 0 ? b_id:-nx * YBF );
	t = ( z == zchunk-1 ? t_id:nx * YBF);
	for (y = 0; y < YBF; y++) {
	  n = ( y == 0 ? n_id:-nx);
	  s = ( y == YBF-1 ? s_id:nx);
 
	  __m512d fc_vec   = _mm512_load_pd(f1_t+c);
	  __m512d fcp1_vec = _mm512_load_pd(f1_t+c+8);
	  __m512d fcm1_vec = _mm512_alignr_epi64(fc_vec,fc_vec,1);

	  __m512d fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
	  __m512d fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
	  __m512d fcs_vec = _mm512_load_pd(f1_t+c+s);
	  __m512d fcn_vec = _mm512_load_pd(f1_t+c+n);
	  __m512d fcb_vec = _mm512_load_pd(f1_t+c+b);
	  __m512d fct_vec = _mm512_load_pd(f1_t+c+t);
	  __m512d tmp = _mm512_mul_pd(cc_vec,fc_vec);
	  tmp = _mm512_fmadd_pd(cw_vec,fcw_vec,tmp);
	  tmp = _mm512_fmadd_pd(ce_vec,fce_vec,tmp);
	  tmp = _mm512_fmadd_pd(cs_vec,fcs_vec,tmp);
	  tmp = _mm512_fmadd_pd(cn_vec,fcn_vec,tmp);
	  tmp = _mm512_fmadd_pd(cb_vec,fcb_vec,tmp);
	  tmp = _mm512_fmadd_pd(ct_vec,fct_vec,tmp);
	  _mm512_store_pd(f2_t+c,tmp);
	  c += 8;
#pragma unroll 
	  for (x = 8; x < nx-8; x+=8) {
	    fcm1_vec = fc_vec;
	    fc_vec   = fcp1_vec;
	    fcp1_vec = _mm512_load_pd(f1_t+c+8);
	    
	    fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
	    fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
	    fcs_vec = _mm512_load_pd(f1_t+c+s);
	    fcn_vec = _mm512_load_pd(f1_t+c+n);
	    fcb_vec = _mm512_load_pd(f1_t+c+b);
	    fct_vec = _mm512_load_pd(f1_t+c+t);
	    tmp = _mm512_mul_pd(cc_vec,fc_vec);
	    tmp = _mm512_fmadd_pd(cw_vec,fcw_vec,tmp);
	    tmp = _mm512_fmadd_pd(ce_vec,fce_vec,tmp);
	    tmp = _mm512_fmadd_pd(cs_vec,fcs_vec,tmp);
	    tmp = _mm512_fmadd_pd(cn_vec,fcn_vec,tmp);
	    tmp = _mm512_fmadd_pd(cb_vec,fcb_vec,tmp);
	    tmp = _mm512_fmadd_pd(ct_vec,fct_vec,tmp);
	    
	    _mm512_store_pd(f2_t+c,tmp);
	    c += 8;
	  }
	  fcm1_vec = fc_vec;
	  fc_vec   = fcp1_vec;
	  fcp1_vec = _mm512_alignr_epi64(fc_vec,fc_vec,7);
	  
	  fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
	  fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
	  fcs_vec = _mm512_load_pd(f1_t+c+s);
	  fcn_vec = _mm512_load_pd(f1_t+c+n);
	  fcb_vec = _mm512_load_pd(f1_t+c+b);
	  fct_vec = _mm512_load_pd(f1_t+c+t);
	  tmp = _mm512_mul_pd(cc_vec,fc_vec);
	  tmp = _mm512_fmadd_pd(cw_vec,fcw_vec,tmp);
	  tmp = _mm512_fmadd_pd(ce_vec,fce_vec,tmp);
	  tmp = _mm512_fmadd_pd(cs_vec,fcs_vec,tmp);
	  tmp = _mm512_fmadd_pd(cn_vec,fcn_vec,tmp);
	  tmp = _mm512_fmadd_pd(cb_vec,fcb_vec,tmp);
	  tmp = _mm512_fmadd_pd(ct_vec,fct_vec,tmp);
	  _mm512_store_pd(f2_t+c,tmp);
	  c += 8;
	}
      }
#pragma omp barrier
      REAL *tmp = f1_t;
      f1_t = f2_t;
      f2_t = tmp;
      time += dt;
      count++;
      
    } while (time + 0.5*dt < 0.1);
#pragma omp master
    {
      *f_ret = f1_t;
      *time_ret = time;      
      *count_ret = count;        
    }
    
  }
#undef YBF
  
  return;
}


static void
diffusion_openmp_intrin_independent2(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		 REAL cb, REAL cc, REAL dt,
		 REAL **f_ret, REAL *time_ret, int *count_ret) {

#define YBF 8
#pragma omp parallel 
  {
    REAL time = 0.0;
    int count = 0;
    REAL *restrict f1_t = f1;
    REAL *restrict f2_t = f2;
    int c, w, e, n, s, b, t;
    int z, y, x;
    int tid = omp_get_thread_num();
    int nth = omp_get_num_threads();
    int nthpy = (ny-1)/YBF + 1;
    int nthpz = (nth-1)/nthpy + 1;
    int idy = tid % nthpy;
    int zchunk = nz/nthpz;
    int idz = tid / nthpy;
    int n_id,s_id,b_id,t_id;
    if(idy == 0)
      n_id = 0;
    else
      n_id = - nx * YBF * zchunk + nx * (YBF-1);
    if(idy == nthpy-1)
      s_id = 0;
    else
      s_id = nx * YBF * zchunk - nx * (YBF-1);
    if(idz == 0)
      b_id = 0;
    else
      b_id = -(nthpy * nx * YBF * zchunk) + nx * YBF * (zchunk-1);
    if(idz == nthpz-1)
      t_id = 0;
    else
      t_id = nthpy * nx * YBF * zchunk - nx * YBF * (zchunk-1);

    const int str = tid * nx * YBF * zchunk;

    const __m512d cc_vec = _mm512_set1_pd(cc);
    const __m512d cw_vec = _mm512_set1_pd(cw);
    const __m512d ce_vec = _mm512_set1_pd(ce);
    const __m512d cs_vec = _mm512_set1_pd(cs);
    const __m512d cn_vec = _mm512_set1_pd(cn);
    const __m512d cb_vec = _mm512_set1_pd(cb);
    const __m512d ct_vec = _mm512_set1_pd(ct);
    do {
      for (c = str; c < str + nx * YBF * zchunk; c += 8){ 
	b = ( c < str + nx * YBF ? b_id:-nx * YBF);
	t = ( c >= str + nx * YBF * (zchunk-1) ? t_id:nx * YBF);
	n = ( (c / nx) % YBF == 0     ? n_id:-nx);
	s = ( (c / nx) % YBF == YBF-1 ? s_id: nx);
 
	__m512d fc_vec   = _mm512_load_pd(f1_t+c);
	__m512d fcp1_vec = _mm512_load_pd(f1_t+c+8);
	__m512d fcm1_vec = _mm512_alignr_epi64(fc_vec,fc_vec,1);
	
	__m512d fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
	__m512d fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
	__m512d fcs_vec = _mm512_load_pd(f1_t+c+s);
	__m512d fcn_vec = _mm512_load_pd(f1_t+c+n);
	__m512d fcb_vec = _mm512_load_pd(f1_t+c+b);
	__m512d fct_vec = _mm512_load_pd(f1_t+c+t);
	__m512d tmp = _mm512_mul_pd(cc_vec,fc_vec);
	tmp = _mm512_fmadd_pd(cw_vec,fcw_vec,tmp);
	tmp = _mm512_fmadd_pd(ce_vec,fce_vec,tmp);
	tmp = _mm512_fmadd_pd(cs_vec,fcs_vec,tmp);
	tmp = _mm512_fmadd_pd(cn_vec,fcn_vec,tmp);
	tmp = _mm512_fmadd_pd(cb_vec,fcb_vec,tmp);
	tmp = _mm512_fmadd_pd(ct_vec,fct_vec,tmp);
	_mm512_store_pd(f2_t+c,tmp);
	c += 8;
#pragma unroll 
	for (x = 8; x < nx-8; x+=8) {
	  fcm1_vec = fc_vec;
	  fc_vec   = fcp1_vec;
	  fcp1_vec = _mm512_load_pd(f1_t+c+8);
	  
	  fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
	  fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
	  fcs_vec = _mm512_load_pd(f1_t+c+s);
	  fcn_vec = _mm512_load_pd(f1_t+c+n);
	  fcb_vec = _mm512_load_pd(f1_t+c+b);
	  fct_vec = _mm512_load_pd(f1_t+c+t);
	  tmp = _mm512_mul_pd(cc_vec,fc_vec);
	  tmp = _mm512_fmadd_pd(cw_vec,fcw_vec,tmp);
	  tmp = _mm512_fmadd_pd(ce_vec,fce_vec,tmp);
	  tmp = _mm512_fmadd_pd(cs_vec,fcs_vec,tmp);
	  tmp = _mm512_fmadd_pd(cn_vec,fcn_vec,tmp);
	  tmp = _mm512_fmadd_pd(cb_vec,fcb_vec,tmp);
	  tmp = _mm512_fmadd_pd(ct_vec,fct_vec,tmp);
	  
	  _mm512_store_pd(f2_t+c,tmp);
	  c += 8;
	}
	fcm1_vec = fc_vec;
	fc_vec   = fcp1_vec;
	fcp1_vec = _mm512_alignr_epi64(fc_vec,fc_vec,7);
	
	fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
	fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
	fcs_vec = _mm512_load_pd(f1_t+c+s);
	fcn_vec = _mm512_load_pd(f1_t+c+n);
	fcb_vec = _mm512_load_pd(f1_t+c+b);
	fct_vec = _mm512_load_pd(f1_t+c+t);
	tmp = _mm512_mul_pd(cc_vec,fc_vec);
	tmp = _mm512_fmadd_pd(cw_vec,fcw_vec,tmp);
	tmp = _mm512_fmadd_pd(ce_vec,fce_vec,tmp);
	tmp = _mm512_fmadd_pd(cs_vec,fcs_vec,tmp);
	tmp = _mm512_fmadd_pd(cn_vec,fcn_vec,tmp);
	tmp = _mm512_fmadd_pd(cb_vec,fcb_vec,tmp);
	tmp = _mm512_fmadd_pd(ct_vec,fct_vec,tmp);
	_mm512_store_pd(f2_t+c,tmp);
      }
#pragma omp barrier
      REAL *tmp = f1_t;
      f1_t = f2_t;
      f2_t = tmp;
      time += dt;
      count++;
      
    } while (time + 0.5*dt < 0.1);
#pragma omp master
    {
      *f_ret = f1_t;
      *time_ret = time;      
      *count_ret = count;        
    }
    
  }
#undef YBF
  
  return;
}



static void
diffusion_like_stream_scale(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		 REAL cb, REAL cc, REAL dt,
		 REAL **f_ret, REAL *time_ret, int *count_ret) {

#define YBF 8
#pragma omp parallel 
  {
    REAL time = 0.0;
    int count = 0;
    REAL *restrict f1_t = f1;
    REAL *restrict f2_t = f2;
    int c, w, e, n, s, b, t;
    int z, y, x;
    int tid = omp_get_thread_num();
    int nth = omp_get_num_threads();
    int nthpy = (ny-1)/YBF + 1;
    int nthpz = (nth-1)/nthpy + 1;
    int idy = tid % nthpy * YBF;
    int zchunk = nz/nthpz;
    int idz = tid / nthpy * zchunk;

    const int str = tid * nx * YBF * zchunk;
    //    printf("tid,nth,nthpy,nthpz,idy,zchunk,idz,str = %d,%d,%d,%d,%d,%d,%d,%d\n",tid,nth,nthpy,nthpz,idy,zchunk,idz,str);

    const __m512d cc_vec = _mm512_set1_pd(cc);
    do {
      for (c = str; c < str + nx * YBF * zchunk; c += 8) {
      	  __m512d fc_vec   = _mm512_load_pd(f1_t+c);
      	  __m512d tmp = _mm512_mul_pd(cc_vec,fc_vec);
      	  _mm512_store_pd(f2_t+c,tmp);
      }
      /* if(count == 0) printf("tid,c,%d,%d\n",tid,c); */
      /* int yy; */
      /* for (yy = idy; yy < idy+1; yy++) { */
      /* 	for (z = idz; z < idz + zchunk; z++) { */
      /* 	  for (y = yy; y < (yy+YBF < ny ? yy+YBF:ny); y++) { */
      /* 	    c =  y * nx + z * nx * ny; */
      /* 	    for (x = 0; x < nx; x+=8,c+=8) { */
      /* 	      __m512d fc_vec   = _mm512_load_pd(f1_t+c); */
      /* 	      __m512d tmp = _mm512_mul_pd(cc_vec,fc_vec); */
      /* 	      _mm512_store_pd(f2_t+c,tmp); */
      /* 	    } */
      /* 	  } */
      /* 	} */
      /* } */
      /* if(count == 0) printf("tid,c,%d,%d\n",tid,c); */
      	    
#pragma omp barrier
      REAL *tmp = f1_t;
      f1_t = f2_t;
      f2_t = tmp;
      time += dt;
      count++;
      
    } while (time + 0.5*dt < 0.1);
#pragma omp master
    {
      *f_ret = f1_t;
      *time_ret = time;      
      *count_ret = count;        
    }
    
  }
#undef YBF
  
  return;
}
#endif


/* static void */
/* diffusion_openmp_intrin_independent(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz, */
/* 		 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct, */
/* 		 REAL cb, REAL cc, REAL dt, */
/* 		 REAL **f_ret, REAL *time_ret, int *count_ret) { */

/* #define YBF 8 */
/* #pragma omp parallel  */
/*   { */
/*     REAL time = 0.0; */
/*     int count = 0; */
/*     REAL *restrict f1_t = f1; */
/*     REAL *restrict f2_t = f2; */
/*     int c, w, e, n, s, b, t; */
/*     int z, y, x; */
/*     int tid = omp_get_thread_num(); */
/*     int nth = omp_get_num_threads(); */
/*     int nthpy = (ny-1)/YBF + 1; */
/*     int nthpz = (nth-1)/nthpy + 1; */
/*     int idy = tid % nthpy; */
/*     //    idy = idy * YBF; */
/*     int zchunk = nz/nthpz; */
/*     int idz = tid / nthpy; */
/*     int n_id,s_id,b_id,t_id; */
/*     /\* if(idy == 0) *\/ */
/*     /\*   n_id = idy; *\/ */
/*     /\* else *\/ */
/*     /\*   n_id = idy-1; *\/ */
/*     /\* if(idy == nthpy-1) *\/ */
/*     /\*   s_id = idy; *\/ */
/*     /\* else *\/ */
/*     /\*   s_id = idy+1; *\/ */
/*     /\* if(idz == 0) *\/ */
/*     /\*   b_id = idz; *\/ */
/*     /\* else *\/ */
/*     /\*   b_id = idz-nthpy; *\/ */
/*     /\* if(idz == nthpz-1) *\/ */
/*     /\*   t_id = idz; *\/ */
/*     /\* else *\/ */
/*     /\*   t_id = idz+nthpy; *\/ */
/*     if(idy == 0) */
/*       n_id = 0; */
/*     else */
/*       n_id = - nx * YBF * zchunk + nx * (YBF-1); */
/*     if(idy == nthpy-1) */
/*       s_id = 0; */
/*     else */
/*       s_id = nx * YBF * zchunk - nx * (YBF-1); */
/*     if(idz == 0) */
/*       b_id = 0; */
/*     else */
/*       b_id = -(nthpy * nx * YBF * zchunk) + nx * YBF * (zchunk-1); */
/*     if(idz == nthpz-1) */
/*       t_id = 0; */
/*     else */
/*       t_id = nthpy * nx * YBF * zchunk - nx * YBF * (zchunk-1); */

/*     int str = tid * nx * YBF * zchunk; */

/*     const __m512d cc_vec = _mm512_set1_pd(cc); */
/*     const __m512d cw_vec = _mm512_set1_pd(cw); */
/*     const __m512d ce_vec = _mm512_set1_pd(ce); */
/*     const __m512d cs_vec = _mm512_set1_pd(cs); */
/*     const __m512d cn_vec = _mm512_set1_pd(cn); */
/*     const __m512d cb_vec = _mm512_set1_pd(cb); */
/*     const __m512d ct_vec = _mm512_set1_pd(ct); */
/*     do { */
/*       for (z = 0; z < zchunk; z++) { */
/* 	b = ( z == 0 ? b_id:-nx * YBF ); */
/* 	t = ( z == zchunk-1 ? t_id:nx * YBF); */
/* 	for (y = 0; y < YBF; y++) { */
/* 	  n = ( y == 0 ? n_id:-nx); */
/* 	  s = ( y == YBF-1 ? s_id:nx); */
 
/* 	  c =  str + y * nx + z * nx * YBF; */
/* 	  __m512d fc_vec   = _mm512_load_pd(f1_t+c); */
/* 	  __m512d fcp1_vec = _mm512_load_pd(f1_t+c+8); */
/* 	  __m512d fcm1_vec = _mm512_alignr_epi64(fc_vec,fc_vec,1); */

/* 	  __m512d fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7); */
/* 	  __m512d fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1); */
/* 	  __m512d fcs_vec = _mm512_load_pd(f1_t+c+s); */
/* 	  __m512d fcn_vec = _mm512_load_pd(f1_t+c+n); */
/* 	  __m512d fcb_vec = _mm512_load_pd(f1_t+c+b); */
/* 	  __m512d fct_vec = _mm512_load_pd(f1_t+c+t); */
/* 	  __m512d tmp = _mm512_mul_pd(cc_vec,fc_vec); */
/* 	  tmp = _mm512_fmadd_pd(cw_vec,fcw_vec,tmp); */
/* 	  tmp = _mm512_fmadd_pd(ce_vec,fce_vec,tmp); */
/* 	  tmp = _mm512_fmadd_pd(cs_vec,fcs_vec,tmp); */
/* 	  tmp = _mm512_fmadd_pd(cn_vec,fcn_vec,tmp); */
/* 	  tmp = _mm512_fmadd_pd(cb_vec,fcb_vec,tmp); */
/* 	  tmp = _mm512_fmadd_pd(ct_vec,fct_vec,tmp); */
/* 	  _mm512_store_pd(f2_t+c,tmp); */
/* #pragma unroll  */
/* 	  for (x = 8; x < nx-8; x+=8) { */
/* 	    fcm1_vec = fc_vec; */
/* 	    fc_vec   = fcp1_vec; */
/* 	    fcp1_vec = _mm512_load_pd(f1_t+c+8+x); */
	    
/* 	    fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7); */
/* 	    fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1); */
/* 	    fcs_vec = _mm512_load_pd(f1_t+c+x+s); */
/* 	    fcn_vec = _mm512_load_pd(f1_t+c+x+n); */
/* 	    fcb_vec = _mm512_load_pd(f1_t+c+x+b); */
/* 	    fct_vec = _mm512_load_pd(f1_t+c+x+t); */
/* 	    tmp = _mm512_mul_pd(cc_vec,fc_vec); */
/* 	    tmp = _mm512_fmadd_pd(cw_vec,fcw_vec,tmp); */
/* 	    tmp = _mm512_fmadd_pd(ce_vec,fce_vec,tmp); */
/* 	    tmp = _mm512_fmadd_pd(cs_vec,fcs_vec,tmp); */
/* 	    tmp = _mm512_fmadd_pd(cn_vec,fcn_vec,tmp); */
/* 	    tmp = _mm512_fmadd_pd(cb_vec,fcb_vec,tmp); */
/* 	    tmp = _mm512_fmadd_pd(ct_vec,fct_vec,tmp); */
	    
/* 	    /\* tmp = _mm512_mul_pd(cc_vec,fc_vec); *\/ */
/* 	    /\* fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7); *\/ */
/* 	    /\* tmp = _mm512_fmadd_pd(cw_vec,fcw_vec,tmp); *\/ */
/* 	    /\* fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1); *\/ */
/* 	    /\* tmp = _mm512_fmadd_pd(ce_vec,fce_vec,tmp); *\/ */
/* 	    /\* fcs_vec = _mm512_load_pd(f1_t+c+x+s); *\/ */
/* 	    /\* tmp = _mm512_fmadd_pd(cs_vec,fcs_vec,tmp); *\/ */
/* 	    /\* fcn_vec = _mm512_load_pd(f1_t+c+x+n); *\/ */
/* 	    /\* tmp = _mm512_fmadd_pd(cn_vec,fcn_vec,tmp); *\/ */
/* 	    /\* fcb_vec = _mm512_load_pd(f1_t+c+x+b); *\/ */
/* 	    /\* tmp = _mm512_fmadd_pd(cb_vec,fcb_vec,tmp); *\/ */
/* 	    /\* fct_vec = _mm512_load_pd(f1_t+c+x+t); *\/ */
/* 	    /\* tmp = _mm512_fmadd_pd(ct_vec,fct_vec,tmp); *\/ */
/* 	    _mm512_store_pd(f2_t+c+x,tmp); */
/* 	  } */
/* 	  fcm1_vec = fc_vec; */
/* 	  fc_vec   = fcp1_vec; */
/* 	  fcp1_vec = _mm512_alignr_epi64(fc_vec,fc_vec,7); */
	  
/* 	  fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7); */
/* 	  fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1); */
/* 	  fcs_vec = _mm512_load_pd(f1_t+c+x+s); */
/* 	  fcn_vec = _mm512_load_pd(f1_t+c+x+n); */
/* 	  fcb_vec = _mm512_load_pd(f1_t+c+x+b); */
/* 	  fct_vec = _mm512_load_pd(f1_t+c+x+t); */
/* 	  tmp = _mm512_mul_pd(cc_vec,fc_vec); */
/* 	  tmp = _mm512_fmadd_pd(cw_vec,fcw_vec,tmp); */
/* 	  tmp = _mm512_fmadd_pd(ce_vec,fce_vec,tmp); */
/* 	  tmp = _mm512_fmadd_pd(cs_vec,fcs_vec,tmp); */
/* 	  tmp = _mm512_fmadd_pd(cn_vec,fcn_vec,tmp); */
/* 	  tmp = _mm512_fmadd_pd(cb_vec,fcb_vec,tmp); */
/* 	  tmp = _mm512_fmadd_pd(ct_vec,fct_vec,tmp); */
/* 	  _mm512_store_pd(f2_t+c+x,tmp); */
/* 	} */
/*       } */
/* #pragma omp barrier */
/*       REAL *tmp = f1_t; */
/*       f1_t = f2_t; */
/*       f2_t = tmp; */
/*       time += dt; */
/*       //time += 1; */
/*       count++; */
      
/*     } while (time + 0.5*dt < 0.1); */
/* #pragma omp master */
/*     { */
/*       *f_ret = f1_t; */
/*       *time_ret = time;       */
/*       *count_ret = count;         */
/*     } */
    
/*   } */
/* #undef YBF */
  
/*   return; */
/* } */


#ifdef INTEL
static void
diffusion_openmp_tiled2_nowait_simd_peel_aligned(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		 REAL cb, REAL cc, REAL dt,
		 REAL **f_ret, REAL *time_ret, int *count_ret) {

  
  {
    REAL time = 0.0;
    int count = 0;
    REAL *restrict f1_t = f1;
    REAL *restrict f2_t = f2;
    int c, w, e, n, s, b, t;
    int z, y, x;

    do {
      int xx,yy;
#define YBF 8
#define XBF 128
#pragma omp parallel private(x,y,z,c,w,e,n,s,b,t,yy)
#pragma omp for collapse(3)
      for (yy = 0; yy < ny; yy += YBF) {
	for (xx = 0; xx < nx; xx += XBF) {
	  for (z = 0; z < nz; z++) {
	    for (y = yy; y < (yy+YBF < ny ? yy+YBF:ny); y++) {
	      n = (y == 0)    ? 0 : - nx;
	      s = (y == ny-1) ? 0 :   nx;
	      b = (z == 0)    ? 0 : - nx * ny;
	      t = (z == nz-1) ? 0 :   nx * ny;
	      c =  y * nx + z * nx * ny;
	      int xmax = xx+XBF < nx ? xx+XBF:nx;
	      x = 0;
	      if(xx == 0){
		__assume_aligned(f1_t, 64);
		__assume_aligned(f2_t, 64);
		__assume(nx%8==0);
		__assume(ny%8==0);
		__assume(s%8==0);
		__assume(n%8==0);
		__assume(b%8==0);
		__assume(t%8==0);
		__assume(x%8==0);
#pragma omp simd private(w,e)
		for(x = x; x < ((8 < xmax) ? 8:xmax); x++,c++){
		  //	  for(x = 0; x < 8; x++,c++){
		  w = (x == 0)    ? 0 : - 1;
		  e = (x == nx-1) ? 0 :   1;
		  f2_t[c] = cc * f1_t[c] + cw * f1_t[c+w] + ce * f1_t[c+e]
		    + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
		}
	      }
	      __assume_aligned(f1_t, 64);
	      __assume_aligned(f2_t, 64);
	      __assume(nx%8==0);
	      __assume(ny%8==0);
	      __assume(s%8==0);
	      __assume(n%8==0);
	      __assume(b%8==0);
	      __assume(t%8==0);
		__assume(x%8==0);
#pragma omp simd 
	      for (x = x; x < xmax; x++,c++) {
		f2_t[c] = cc * f1_t[c] + cw * f1_t[c-1] + ce * f1_t[c+1]
		  + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	      }
	      if(xmax == nx){
		__assume_aligned(f1_t, 64);
		__assume_aligned(f2_t, 64);
		__assume(nx%8==0);
		__assume(ny%8==0);
		__assume(s%8==0);
		__assume(n%8==0);
		__assume(b%8==0);
		__assume(t%8==0);
#pragma omp simd 
		for (x = xx; x < xmax-8; x++,c++) {
		  f2_t[c] = cc * f1_t[c] + cw * f1_t[c-1] + ce * f1_t[c+1]
		    + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
		}
#pragma omp simd private(e)
		for (x = xmax-8; x < xmax; x++,c++) {
		  e = (x == nx-1) ? 0 :   1;
		  f2_t[c] = cc * f1_t[c] + cw * f1_t[c-1] + ce * f1_t[c+e]
		    + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
		}
	      }
	      else{
		__assume_aligned(f1_t, 64);
		__assume_aligned(f2_t, 64);
		__assume(nx%8==0);
		__assume(ny%8==0);
		__assume(s%8==0);
		__assume(n%8==0);
		__assume(b%8==0);
		__assume(t%8==0);
#pragma omp simd 
		for (x = xx; x < xmax; x++,c++) {
		  f2_t[c] = cc * f1_t[c] + cw * f1_t[c-1] + ce * f1_t[c+1]
		    + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
		}
	      }
	    }
	  }
	}
      }
    
      REAL *tmp = f1_t;
      f1_t = f2_t;
      f2_t = tmp;
      time += dt;
      //time += 1;
      count++;
      
    } while (time + 0.5*dt < 0.1);
    *f_ret = f1_t;
    *time_ret = time;      
    *count_ret = count;        
  }
#undef XBF
#undef YBF
  
  return;
}

#endif

static void
diffusion_openmp_z_simd_peel_aligned(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		 REAL cb, REAL cc, REAL dt,
		 REAL **f_ret, REAL *time_ret, int *count_ret) {

  
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
	  n = (y == 0)    ? 0 : - nx;
	  s = (y == ny-1) ? 0 :   nx;
	  b = (z == 0)    ? 0 : - nx * ny;
	  t = (z == nz-1) ? 0 :   nx * ny;
	  c =  y * nx + z * nx * ny;
	  for(x = 0; x < ((8 < nx) ? 8:nx); x++,c++){
	    //	  for(x = 0; x < 8; x++,c++){
	    w = (x == 0)    ? 0 : - 1;
	    e = (x == nx-1) ? 0 :   1;
	    f2_t[c] = cc * f1_t[c] + cw * f1_t[c+w] + ce * f1_t[c+e]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	  }
	  __assume_aligned(f1_t, 64);
	  __assume_aligned(f2_t, 64);
	  __assume(nx%8==0);
	  __assume(ny%8==0);
	  __assume(s%8==0);
	  __assume(n%8==0);
	  __assume(b%8==0);
	  __assume(t%8==0);
#pragma omp simd 
	  for (x = 8; x < nx-8; x++,c++) {
	    f2_t[c] = cc * f1_t[c] + cw * f1_t[c-1] + ce * f1_t[c+1]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	  }
	  for (x = nx-8; x < nx; x++,c++) {
	    e = (x == nx-1) ? 0 :   1;
	    f2_t[c] = cc * f1_t[c] + cw * f1_t[c-1] + ce * f1_t[c+e]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	  }
	}
      }
      REAL *tmp = f1_t;
      f1_t = f2_t;
      f2_t = tmp;
      time += dt;
      //time += 1;
      count++;

    } while (time + 0.5*dt < 0.1);
    *f_ret = f1_t;
    *time_ret = time;      
    *count_ret = count;        
  }

  return;
}

#ifdef INTEL

static void
diffusion_openmp3(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		 REAL cb, REAL cc, REAL dt,
		 REAL **f_ret, REAL *time_ret, int *count_ret) {

  
  {
    REAL time = 0.0;
    int count = 0;
    int c, w, e, n, s, b, t;
    int z, y, x;
    REAL *restrict f1_t = f1;
    REAL *restrict f2_t = f2;
    /* REAL *restrict L2_b; */
    /* REAL *restrict L2_c; */
    /* REAL *restrict L2_t; */

    //#pragma omp parallel private(x,y,z,c,w,e,n,s,b,t)
    //    {
      /* REAL *restrict f1_t = f1; */
      /* REAL *restrict f2_t = f2; */

    do {
#pragma omp parallel private(x,y,z,c,w,e,n,s,b,t)
      for (z = 0; z < nz; z++) {
#pragma omp for nowait
	for (y = 0; y < ny; y++) {
/* #pragma omp simd  */
/* #pragma vector aligned */
#if 0
	  __assume_aligned(f1_t, 64);
	  __assume_aligned(f2_t, 64);
	  __assume(nx%8==0);
	  __assume(ny%8==0);
#pragma omp simd private(c,w,e,n,s,b,t) linear(c:1,n:1,s:1,b:1,t:1) 
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
#else
	  n = (y == 0)    ? 0 : - nx;
	  s = (y == ny-1) ? 0 :   nx;
	  b = (z == 0)    ? 0 : - nx * ny;
	  t = (z == nz-1) ? 0 :   nx * ny;
	  c =  y * nx + z * nx * ny;
	  for(x = 0; x < ((8 < nx) ? 8:nx); x++,c++){
	    //	  for(x = 0; x < 8; x++,c++){
	    w = (x == 0)    ? 0 : - 1;
	    e = (x == nx-1) ? 0 :   1;
	    f2_t[c] = cc * f1_t[c] + cw * f1_t[c+w] + ce * f1_t[c+e]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	  }
	  __assume_aligned(f1_t, 64);
	  __assume_aligned(f2_t, 64);
	  __assume(nx%8==0);
	  __assume(ny%8==0);
	  __assume(s%8==0);
	  __assume(n%8==0);
	  __assume(b%8==0);
	  __assume(t%8==0);
#pragma omp simd 
	  for (x = 8; x < nx-8; x++,c++) {
	    f2_t[c] = cc * f1_t[c] + cw * f1_t[c-1] + ce * f1_t[c+1]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	  }
	  for (x = nx-8; x < nx; x++,c++) {
	    e = (x == nx-1) ? 0 :   1;
	    f2_t[c] = cc * f1_t[c] + cw * f1_t[c-1] + ce * f1_t[c+e]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	  }
#endif
	}
      }
	REAL *tmp = f1_t;
	f1_t = f2_t;
	f2_t = tmp;
/* #pragma omp master */
/*       { */
	time += dt;
	//time += 1;
	count++;
      /* } */

    } while (time + 0.5*dt < 0.1);
/* #pragma omp master */
/*       { */
    *f_ret = f1_t;
    *time_ret = time;      
    *count_ret = count;        
      /* } */

  }

  return;
}


static void
diffusion_openmp4(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		 REAL cb, REAL cc, REAL dt,
		 REAL **f_ret, REAL *time_ret, int *count_ret) {

  
  {
    REAL time = 0.0;
    int count = 0;
    REAL *restrict f1_t = f1;
    REAL *restrict f2_t = f2;
    int c, w, e, n, s, b, t;
    int z, y, x;
    /* REAL *restrict L2_b; */
    /* REAL *restrict L2_c; */
    /* REAL *restrict L2_t; */
    REAL ccc[NX], nnn[NX], sss[NX], ttt[NX], bbb[NX];
    int step;
    __assume_aligned(f1_t, 64);
    __assume_aligned(f2_t, 64);
    __assume(nx%64==0);
    __assume(ny%64==0);

    
    do {
#pragma omp parallel private(x,y,z,c,w,e,n,s,b,t,ccc,nnn,sss,ttt,bbb,step)
      for (z = 0; z < nz; z++) {
	b = (z == 0)    ? 0 : - 1;
	t = (z == nz-1) ? 0 :   1;
	step = 0;
#pragma omp for nowait
	for (y = 0; y < ny; y++) {
	  n = (y == 0)    ? 0 : - 1;
	  s = (y == ny-1) ? 0 :   1;
	  if(step == 0){
	    for(x = 0; x < nx; x++){
	      ccc[x] = f1_t[x + (y + 0) * nx + (z + 0) * nx * ny];
	      nnn[x] = f1_t[x + (y + n) * nx + (z + 0) * nx * ny];
	      sss[x] = f1_t[x + (y + s) * nx + (z + 0) * nx * ny];
	      ttt[x] = f1_t[x + (y + 0) * nx + (z + t) * nx * ny];
	      bbb[x] = f1_t[x + (y + 0) * nx + (z + b) * nx * ny];
	    }
	  }else{
	    /* nnn = ccc; */
	    /* ccc = sss; */
	    for(x = 0; x < nx; x++){
	      nnn[x] = ccc[x];
	      ccc[x] = sss[x];
	      sss[x] = f1_t[x + (y + s) * nx + (z + 0) * nx * ny];
	      ttt[x] = f1_t[x + (y + 0) * nx + (z + t) * nx * ny];
	      bbb[x] = f1_t[x + (y + 0) * nx + (z + b) * nx * ny];
	    }
	  }
	  step++;
#pragma omp simd
#pragma vector aligned	  
	  for (x = 0; x < nx; x++) {
	    c =  x + y * nx + z * nx * ny;
	    /* w = (x == 0)    ? c : c - 1; */
	    /* e = (x == nx-1) ? c : c + 1; */
	    w = (x == 0)    ? 0 : - 1;
	    e = (x == nx-1) ? 0 :   1;
	    /* f2_t[c] = cc * f1_t[c] + cw * f1_t[w] + ce * f1_t[e] */
	    /*   + cs * f1_t[s] + cn * f1_t[n] + cb * f1_t[b] + ct * f1_t[t]; */
	    f2_t[c] = cc * ccc[x] + cw * ccc[x+w] + ce * ccc[x+e]
	      + cs * sss[x] + cn * nnn[x] + cb * bbb[x] + ct * ttt[x];
	  }
	}
      }
      REAL *tmp = f1_t;
      f1_t = f2_t;
      f2_t = tmp;
      time += dt;
      //time += 1;
      count++;
    } while (time + 0.5*dt < 0.1);
    *f_ret = f1_t;
    *time_ret = time;      
    *count_ret = count;        
  }

  return;
}
#endif

#if 0
int main(int argc, char *argv[]) 
{

  struct timeval time_begin, time_end;

  int i = 1;
  int    nx    = NX;
  int    ny    = NX;
  int    nz    = NX;

  /* REAL   time  = 0.0; */
  /* int    count = 0;   */
  
  REAL l, dx, dy, dz, kx, ky, kz, kappa, dt;
  REAL ce, cw, cn, cs, ct, cb, cc;

#ifdef _OPENACC
  acc_init(0);
#endif

  l = 1.0;
  kappa = 0.1;
  dx = dy = dz = l / nx;
  kx = ky = kz = 2.0 * M_PI;
  dt = 0.1*dx*dx / kappa;
  ce = cw = kappa*dt/(dx*dx);
  cn = cs = kappa*dt/(dy*dy);
  ct = cb = kappa*dt/(dz*dz);
  cc = 1.0 - (ce + cw + cn + cs + ct + cb);

  /* REAL *f1 = (REAL *)malloc(sizeof(REAL)*nx*ny*nz); */
  /* REAL *f2 = (REAL *)malloc(sizeof(REAL)*nx*ny*nz);   */
  REAL *f1, *f2;
  posix_memalign((void**)&f1, 2*1024*1024, sizeof(REAL)*nx*ny*nz);
  posix_memalign((void**)&f2, 2*1024*1024, sizeof(REAL)*nx*ny*nz);


  // print data
  printf("(nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);

  REAL *answer = (REAL *)malloc(sizeof(REAL) * nx*ny*nz);

  /* void (*diffusion[])(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz, */
  /* 		      REAL ce, REAL cw, REAL cn, REAL cs, REAL ct, */
  /* 		      REAL cb, REAL cc, REAL dt, */
  /* 		      REAL **f_ret, REAL *time_ret, int *count_ret)  */
  /*   = {diffusion_openmp3, */
  /*      diffusion_openmp, */
  /*      diffusion_openmp_y, */
  /*      diffusion_openmp_y_nowait, */
  /*      diffusion_openmp_y_nowait_simdwrong, */
  /*      diffusion_openmp_y_nowait_simd, */
  /*      diffusion_openmp_y_nowait_simd_peel, */
  /*      diffusion_openmp_y_nowait_simd_peel_aligned}; */
  /* char *name[] = {"diffusion_openmp3", */
  /* 		 "diffusion_openmp", */
  /* 		 "diffusion_openmp_y", */
  /* 		 "diffusion_openmp_y_nowait", */
  /* 		 "diffusion_openmp_y_nowait_simdwrong", */
  /* 		 "diffusion_openmp_y_nowait_simd", */
  /* 		 "diffusion_openmp_y_nowait_simd_peel", */
  /* 		 "diffusion_openmp_y_nowait_simd_peel_aligned"}; */

  /* int args; */
  /* for(args = 0;args < 8;args++){ */
    REAL   time  = 0.0;
    int    count = 0;  
    init(f1, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);
    gettimeofday(&time_begin, NULL);
    /* diffusion_openacc(f1, f2, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt, */
    /* 		    &f1, &time, &count); */
    diffusion_openmp_y_nowait_simdwrong(f1, f2, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt,
		       &f1, &time, &count);
    gettimeofday(&time_end, NULL);
    
    init(answer, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);
    REAL err = accuracy(f1, answer, nx*ny*nz);
    double elapsed_time = (time_end.tv_sec - time_begin.tv_sec)
      + (time_end.tv_usec - time_begin.tv_usec)*1.0e-6;
    REAL gflops = (nx*ny*nz)*13.0*count/elapsed_time * 1.0e-09;
    double thput = (nx * ny * nz) * sizeof(REAL) * 2.0 * count
      / elapsed_time * 1.0e-09;

    
    /* fprintf(stdout, "\n %s\n", name[args]); */
    fprintf(stdout, "elapsed time : %.3f (s)\n", elapsed_time);
    fprintf(stdout, "flops        : %.3f (GFlops)\n", gflops);
    fprintf(stdout, "throughput   : %.3f (GB/s)\n", thput);  
    fprintf(stdout, "accuracy     : %e\n", err);
    fprintf(stdout, "count        : %.3d\n", count);
  /* } */
    
  free(answer);
  free(f1);
  free(f2);
  return 0;
}

#else

int main(int argc, char *argv[]) 
{

  struct timeval time_begin, time_end;

  int i = 1;
  int    nx    = NX;
  int    ny    = NX;
  int    nz    = NX;

  /* REAL   time  = 0.0; */
  /* int    count = 0;   */
  
  REAL l, dx, dy, dz, kx, ky, kz, kappa, dt;
  REAL ce, cw, cn, cs, ct, cb, cc;

#ifdef _OPENACC
  acc_init(0);
#endif

  l = 1.0;
  kappa = 0.1;
  dx = dy = dz = l / nx;
  kx = ky = kz = 2.0 * M_PI;
  dt = 0.1*dx*dx / kappa;
  ce = cw = kappa*dt/(dx*dx);
  cn = cs = kappa*dt/(dy*dy);
  ct = cb = kappa*dt/(dz*dz);
  cc = 1.0 - (ce + cw + cn + cs + ct + cb);

  /* REAL *f1 = (REAL *)malloc(sizeof(REAL)*nx*ny*nz); */
  /* REAL *f2 = (REAL *)malloc(sizeof(REAL)*nx*ny*nz); */
  REAL *f1, *f2;
  posix_memalign((void**)&f1, 64, sizeof(REAL)*nx*ny*nz);
  posix_memalign((void**)&f2, 64, sizeof(REAL)*nx*ny*nz);


  // print data
  printf("(nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);

  REAL *answer = (REAL *)malloc(sizeof(REAL) * nx*ny*nz);

  void (*diffusion[])(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		      REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		      REAL cb, REAL cc, REAL dt,
		      REAL **f_ret, REAL *time_ret, int *count_ret) 
    = {diffusion_openmp,
       //       diffusion_openmp_declare_simd,
       diffusion_openmp_y,
       diffusion_openmp_y_nowait,
       diffusion_openmp_y_nowait_simdwrong,
       diffusion_openmp_y_nowait_simd,
       diffusion_openmp_y_nowait_simd_peel,
       diffusion_openmp_y_nowait_simd_peel_aligned,
       diffusion_openmp_y_nowait_simd_peel_aligned_mvparallel,
       diffusion_openmp_tiled_nowait_simd_peel_aligned,
       diffusion_openmp_tiled_nowait_simd_peel_aligned_mvparallel,
       diffusion_openmp_tiled_nowait_simd_peel_aligned_mvparallel_mvfor,
       diffusion_openmp_tiled_nowait_simd_peel_aligned_mvparallel_mvfor2,
       //       diffusion_openmp_tiled2_nowait_simd_peel_aligned,
       diffusion_openmp_KNC_book_asis,
       diffusion_openmp_intrin,
       diffusion_like_stream_scale,
       //       diffusion_openmp_intrin_independent,
       diffusion_openmp_z_simd_peel_aligned};
  char *name[] = {"diffusion_openmp",
		  //		  "diffusion_openmp_declare_simd",
		  "diffusion_openmp_y",
		  "diffusion_openmp_y_nowait",
		  "diffusion_openmp_y_nowait_simdwrong",
		  "diffusion_openmp_y_nowait_simd",
		  "diffusion_openmp_y_nowait_simd_peel",
		  "diffusion_openmp_y_nowait_simd_peel_aligned",
		  "diffusion_openmp_y_nowait_simd_peel_aligned_mvparallel",
		  "diffusion_openmp_tiled_nowait_simd_peel_aligned",
		  "diffusion_openmp_tiled_nowait_simd_peel_aligned_mvparallel",
		  "diffusion_openmp_tiled_nowait_simd_peel_aligned_mvparallel_mvfor",
		  "diffusion_openmp_tiled_nowait_simd_peel_aligned_mvparallel_mvfor2",
		  //		  "diffusion_openmp_tiled2_nowait_simd_peel_aligned",
		  "diffusion_openmp_KNC_book_asis",
		  "diffusion_openmp_intrin",
		  "diffusion_like_stream_scale",
		  //		  "diffusion_openmp_intrin_independent",
		  "diffusion_openmp_z_simd_peel_aligned"};

  int args;
  for(args = 0;args < 15;args++){
    REAL   time  = 0.0;
    int    count = 0;  
    init(f1, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);
    gettimeofday(&time_begin, NULL);
    /* diffusion_openacc(f1, f2, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt, */
    /* 		    &f1, &time, &count); */
    (*diffusion[args])(f1, f2, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt,
		       &f1, &time, &count);
    gettimeofday(&time_end, NULL);
    
    init(answer, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);
    REAL err = accuracy(f1, answer, nx*ny*nz);
    double elapsed_time = (time_end.tv_sec - time_begin.tv_sec)
      + (time_end.tv_usec - time_begin.tv_usec)*1.0e-6;
    REAL gflops = (nx*ny*nz)*13.0*count/elapsed_time * 1.0e-09;
    double thput = (nx * ny * nz) * sizeof(REAL) * 2.0 * count
      / elapsed_time * 1.0e-09;

    
    fprintf(stdout, "\n %s\n", name[args]);
    fprintf(stdout, "elapsed time : %.3f (s)\n", elapsed_time);
    fprintf(stdout, "flops        : %.3f (GFlops)\n", gflops);
    fprintf(stdout, "throughput   : %.3f (GB/s)\n", thput);  
    fprintf(stdout, "accuracy     : %e\n", err);
    fprintf(stdout, "count        : %.3d\n", count);
  }
  REAL   time  = 0.0;
  int    count = 0;  
  init(f1, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);
  REAL *f1_dis, *f2_dis;
  posix_memalign((void**)&f1_dis, 64, sizeof(REAL)*nx*ny*nz);
  posix_memalign((void**)&f2_dis, 64, sizeof(REAL)*nx*ny*nz);
  transform(f1_dis, f2_dis, f1, f2, nx, ny, nz);
  gettimeofday(&time_begin, NULL);
  /* diffusion_openmp_intrin_independent(f1_dis, f2_dis, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt, */
  /* 		     &f1_dis, &time, &count); */
  diffusion_openmp_intrin_independent2(f1_dis, f2_dis, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt,
  		     &f1_dis, &time, &count);
  /* diffusion_like_stream_scale(f1_dis, f2_dis, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt, */
  /* 		     &f1_dis, &time, &count); */
  gettimeofday(&time_end, NULL);
  retransform(f1, f2, f1_dis, f2_dis, nx, ny, nz);
  
  init(answer, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);
  REAL err = accuracy(f1, answer, nx*ny*nz);
  double elapsed_time = (time_end.tv_sec - time_begin.tv_sec)
    + (time_end.tv_usec - time_begin.tv_usec)*1.0e-6;
  REAL gflops = (nx*ny*nz)*13.0*count/elapsed_time * 1.0e-09;
  double thput = (nx * ny * nz) * sizeof(REAL) * 2.0 * count
    / elapsed_time * 1.0e-09;
  
  fprintf(stdout, "\n independent\n");
  fprintf(stdout, "elapsed time : %.3f (s)\n", elapsed_time);
  fprintf(stdout, "flops        : %.3f (GFlops)\n", gflops);
  fprintf(stdout, "throughput   : %.3f (GB/s)\n", thput);  
  fprintf(stdout, "accuracy     : %e\n", err);
  fprintf(stdout, "count        : %.3d\n", count);
  
  free(answer);
  free(f1);
  free(f2);
  return 0;
}
#endif
