#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#ifdef SVE
#include <arm_sve.h>
#endif /* SVE */
#include "misc.h"

#ifndef REAL
#define REAL double
#endif
#ifndef NX
#define NX (256)
#endif
#ifndef NY
#define NY (384)
#endif
#ifndef NZ
#define NZ (384)
#endif

#ifndef M_PI
#define M_PI (3.1415926535897932384626)
#endif


typedef void (*diffusion_loop_t)(REAL *f1, REAL *f2, int nx, int ny, int nz,
                                 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
                                 REAL cb, REAL cc, REAL dt,
                                 REAL **f1_ret, REAL **f2_ret, REAL *time_ret, int *count_ret);

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
    //#pragma omp for 
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


void init_y(REAL *buff, const int nx, const int ny, const int nz,
          const REAL kx, const REAL ky, const REAL kz,
          const REAL dx, const REAL dy, const REAL dz,
          const REAL kappa, const REAL time) {
  REAL ax, ay, az;
  int jz, jy, jx;
  ax = exp(-kappa*time*(kx*kx));
  ay = exp(-kappa*time*(ky*ky));
  az = exp(-kappa*time*(kz*kz));
#pragma omp parallel private(jy,jx)
  for (jz = 0; jz < nz; jz++) {
#pragma omp for 
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

void init_knc(REAL *buff, const int nx, const int ny, const int nz,
          const REAL kx, const REAL ky, const REAL kz,
          const REAL dx, const REAL dy, const REAL dz,
          const REAL kappa, const REAL time) {
  REAL ax, ay, az;
  int jz, jy, jx, yy;
  ax = exp(-kappa*time*(kx*kx));
  ay = exp(-kappa*time*(ky*ky));
  az = exp(-kappa*time*(kz*kz));
#define YBF 16
#pragma omp for collapse(2)
  for(yy = 0;yy < ny; yy+=YBF){
    for (jz = 0; jz < nz; jz++) {
      int ymax = yy + YBF;
      if(ymax >= ny) ymax = ny;
      for (jy = yy; jy < ymax; jy++) {
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
}
#undef YBF


REAL accuracy(const REAL *b1, REAL *b2, const int len) {
  REAL err = 0.0;
  int i;
  for (i = 0; i < len; i++) {
    err += (b1[i] - b2[i]) * (b1[i] - b2[i]);
  }
  return (REAL)sqrt(err/len);
}


// openmp
static void
diffusion_openmp(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
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


// openmp
static void
diffusion_openmp_peeling(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
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
	b = (z == 0)    ? 0 : - nx * ny;
	t = (z == nz-1) ? 0 :   nx * ny;
	for (y = 0; y < ny; y++) {
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


static void
diffusion_stream(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
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
#pragma omp parallel for 
      for (c = 0; c < nz*ny*nx; c++) {
	f2_t[c] = f1_t[c];
      }
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


static void
diffusion_stream3D(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
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
	    f2_t[c] = f1_t[c];
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
    *f1_ret = f1_t; *f2_ret = f2_t;;
    *time_ret = time;      
    *count_ret = count;        
  }

  return;
}


static void
diffusion_stream3D_collapse(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
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
#pragma omp parallel for collapse(3) private(x,y,z,c,w,e,n,s,b,t)
      for (z = 0; z < nz; z++) {
	for (y = 0; y < ny; y++) {
	  for (x = 0; x < nx; x++) {
	    c =  x + y * nx + z * nx * ny;
	    f2_t[c] = f1_t[c];
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
    *f1_ret = f1_t; *f2_ret = f2_t;;
    *time_ret = time;      
    *count_ret = count;        
  }

  return;
}


static void
diffusion_stream3D_wofor(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
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
#pragma omp parallel private(x,y,z,c)
      {
	int id = omp_get_thread_num();
	int nth = omp_get_num_threads();
	int length = (nz-1) / nth + 1;
	int str = length * id;
	int end = length * (id+1);
	c = str * nx * ny;
	for (z = str; z < end; z++) {
	  for (y = 0; y < ny; y++) {
	    for (x = 0; x < nx; x++) {
	      f2_t[c] = f1_t[c];
	      c++;
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
    *f1_ret = f1_t; *f2_ret = f2_t;;
    *time_ret = time;      
    *count_ret = count;        
  }

  return;
}

static void
diffusion_stream3D_wofor_diff1(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
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
#pragma omp parallel private(x,y,z,c,w,e,n,s,b,t)
      {
	int id = omp_get_thread_num();
	int nth = omp_get_num_threads();
	int length = (nz-1) / nth + 1;
	int str = length * id;
	int end = length * (id+1);
	c = str * nx * ny;
	for (z = str; z < end; z++) {
	  for (y = 0; y < ny; y++) {
	    for (x = 0; x < nx; x++) {
	      w = (x == 0)    ? c : c - 1;
	      e = (x == nx-1) ? c : c + 1;
	      n = (y == 0)    ? c : c - nx;
	      s = (y == ny-1) ? c : c + nx;
	      b = (z == 0)    ? c : c - nx * ny;
	      t = (z == nz-1) ? c : c + nx * ny;
	      f2_t[c] = cc*f1_t[c] + cw*f1_t[w] + ce*f1_t[e] + cn*f1_t[n] + cs*f1_t[s] + cb*f1_t[b] + ct*f1_t[t];
	      c++;
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
    *f1_ret = f1_t; *f2_ret = f2_t;;
    *time_ret = time;      
    *count_ret = count;        
  }

  return;
}


static void
diffusion_openmp_y(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
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
#pragma omp parallel private(x,y,z,c,w,e,n,s,b,t) 
      for (z = 0; z < nz; z++) {
#pragma omp for 
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
    *f1_ret = f1_t; *f2_ret = f2_t;;
    *time_ret = time;      
    *count_ret = count;        
  }

  return;
}


static void
diffusion_openmp_y_nowait(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
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
#pragma omp parallel private(x,y,z,c,w,e,n,s,b,t)
      for (z = 0; z < nz; z++) {
#pragma omp for nowait
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
    *f1_ret = f1_t; *f2_ret = f2_t;;
    *time_ret = time;      
    *count_ret = count;        
  }

  return;
}

static void
diffusion_openmp_KNC_book_asis(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		 REAL cb, REAL cc, REAL dt,
		 REAL **f1_ret, REAL **f2_ret, REAL *time_ret, int *count_ret) {

  
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
	*f1_ret = f1_t; *f2_ret = f2_t;;
	*time_ret = time;
	*count_ret = count;
      }
    }
    
    return;
  }
}


static void
diffusion_openmp_intrin(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		 REAL cb, REAL cc, REAL dt,
		 REAL **f1_ret, REAL **f2_ret, REAL *time_ret, int *count_ret) {

  REAL time = 0.0;
  int count = 0;
  REAL *restrict f1_t = f1;
  REAL *restrict f2_t = f2;
  int c, w, e, n, s, b, t;
  int z, y, x;
  
  do {
#pragma omp parallel private(x,y,z,c,w,e,n,s,b,t)
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
	  //	  float64_t fcm1_arr[8] = {0.0};
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

	  /* for(x = 0; x < 8; x++){ */
	  /*   w = (x == 0)    ? 0 : - 1; */
	  /*   e = (x == nx-1) ? 0 :   1; */
	  /*   f2_t[c+x] = cc * f1_t[c+x] + cw * f1_t[c+x+w] + ce * f1_t[c+x+e] */
	  /*     + cs * f1_t[c+x+s] + cn * f1_t[c+x+n] + cb * f1_t[c+x+b] + ct * f1_t[c+x+t]; */
	  /* } */

	  /* svfloat64_t fc_vec ; */
	  /* svfloat64_t fce_vec; */
	  /* svfloat64_t fcw_vec; */
	  /* svfloat64_t fcs_vec; */
	  /* svfloat64_t fcn_vec; */
	  /* svfloat64_t fcb_vec; */
	  /* svfloat64_t fct_vec; */
	  /* svfloat64_t tmp; */
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
	  //	  float64_t fcp1_arr[8] = {0.0};
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
	  /* for(; x < nx; x++){ */
	  /*   w = (x == 0)    ? 0 : - 1; */
	  /*   e = (x == nx-1) ? 0 :   1; */
	  /*   f2_t[c+x] = cc * f1_t[c+x] + cw * f1_t[c+x+w] + ce * f1_t[c+x+e] */
	  /*     + cs * f1_t[c+x+s] + cn * f1_t[c+x+n] + cb * f1_t[c+x+b] + ct * f1_t[c+x+t]; */
	  /* } */

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
  *f1_ret = f1_t; *f2_ret = f2_t;;
  *time_ret = time;      
  *count_ret = count;        
  
  return;
}



static void
diffusion_openmp_intrin_tile(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		 REAL cb, REAL cc, REAL dt,
		 REAL **f1_ret, REAL **f2_ret, REAL *time_ret, int *count_ret) {

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
    const svfloat64_t cc_vec = svdup_f64(cc);
    const svfloat64_t cw_vec = svdup_f64(cw);
    const svfloat64_t ce_vec = svdup_f64(ce);
    const svfloat64_t cs_vec = svdup_f64(cs);
    const svfloat64_t cn_vec = svdup_f64(cn);
    const svfloat64_t cb_vec = svdup_f64(cb);
    const svfloat64_t ct_vec = svdup_f64(ct);
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
      *f1_ret = f1_t; *f2_ret = f2_t;;
      *time_ret = time;      
      *count_ret = count;        
    }
    
  }
#undef YBF
  
  return;
}


static void
diffusion_openmp_intrin_tile_pipeline(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		 REAL cb, REAL cc, REAL dt,
		 REAL **f1_ret, REAL **f2_ret, REAL *time_ret, int *count_ret) {

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
    int zchunk = nz/nthpz;
    int idz = tid / nthpy * zchunk;
    const svfloat64_t cc_vec = svdup_f64(cc);
    const svfloat64_t cw_vec = svdup_f64(cw);
    const svfloat64_t ce_vec = svdup_f64(ce);
    const svfloat64_t cs_vec = svdup_f64(cs);
    const svfloat64_t cn_vec = svdup_f64(cn);
    const svfloat64_t cb_vec = svdup_f64(cb);
    const svfloat64_t ct_vec = svdup_f64(ct);
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
      *f1_ret = f1_t; *f2_ret = f2_t;;
      *time_ret = time;      
      *count_ret = count;        
    }
    
  }
#undef YBF
  
  return;
}

int main(int argc, char *argv[]) 
{

  struct timeval time_begin, time_end;

  int i = 1;
  int    nx    = NX;
  int    ny    = NY;
  int    nz    = NZ;

  /* REAL   time  = 0.0; */
  /* int    count = 0;   */
  
  REAL l, dx, dy, dz, kx, ky, kz, kappa, dt;
  REAL ce, cw, cn, cs, ct, cb, cc;

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
		      REAL **f1_ret, REAL **f2_ret, REAL *time_ret, int *count_ret) 
    = {diffusion_openmp
       /* ,diffusion_openmp_y */
       /* ,diffusion_openmp_y_nowait */
       ,diffusion_openmp_peeling
       ,diffusion_openmp_KNC_book_asis
       ,diffusion_openmp_intrin
       ,diffusion_openmp_intrin_tile
       ,diffusion_openmp_intrin_tile_pipeline
       /* ,diffusion_stream */
       /* ,diffusion_stream3D */
       /* //       ,diffusion_stream3D_collapse */
       /* ,diffusion_stream3D_wofor */
       /* ,diffusion_stream3D_wofor_diff1 */
  };
  char *name[] = {"diffusion_openmp",
		  //		  "diffusion_openmp_declare_simd",
		  /* "diffusion_openmp_y", */
		  /* "diffusion_openmp_y_nowait", */
		  /* /\* "diffusion_openmp_y_nowait_simdwrong", *\/ */
		  /* /\* "diffusion_openmp_y_nowait_simd", *\/ */
		  /* /\* "diffusion_openmp_y_nowait_simd_peel", *\/ */
		  /* /\* "diffusion_openmp_y_nowait_simd_peel_aligned", *\/ */
		  /* /\* "diffusion_openmp_y_nowait_simd_peel_aligned_mvparallel", *\/ */
		  /* /\* "diffusion_openmp_tiled_nowait_simd_peel_aligned", *\/ */
		  /* /\* "diffusion_openmp_tiled_nowait_simd_peel_aligned_mvparallel", *\/ */
		  /* /\* "diffusion_openmp_tiled_nowait_simd_peel_aligned_mvparallel_mvfor", *\/ */
		  /* /\* "diffusion_openmp_tiled_nowait_simd_peel_aligned_mvparallel_mvfor2", *\/ */
		  /* /\* //		  "diffusion_openmp_tiled2_nowait_simd_peel_aligned", *\/ */
		  "diffusion_openmp_peel",
		  "diffusion_openmp_KNC_book_asis",
		  "diffusion_openmp_intrin",
		  "diffusion_openmp_intrin_tile",
		  "diffusion_openmp_intrin_tile_pipeline",
		  /* "diffusion_stream", */
		  /* "diffusion_stream3D", */
		  /* //		  "diffusion_stream3D_collapse", */
		  /* "diffusion_stream3D_wofor", */
		  /* "diffusion_stream3D_wofor_diff1", */
		  /* "diffusion_like_stream_scale", */
		  //		  "diffusion_openmp_intrin_independent",
		  "diffusion_openmp_z_simd_peel_aligned"};

  int args;
  for(args = 0;args < 5;args++){
    REAL   time  = 0.0;
    int    count = 0;  
    //    init(f1, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);
    init_knc(f1, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);
    //    printf("init %d\n",__LINE__);
    gettimeofday(&time_begin, NULL);
    /* diffusion_openacc(f1, f2, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt, */
    /* 		    &f1, &time, &count); */
    (*diffusion[args])(f1, f2, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt,
		       &f1, &time, &count);
    gettimeofday(&time_end, NULL);
    //    printf("fin %d\n",__LINE__);
    
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
    double sum = 0.0;
    int i;
    for(i = 0;i < nx*ny*nz;i++)
      sum += f1[i];
    fprintf(stdout, "sum          : %f\n", sum);
  }
  
  free(answer);
  free(f1);
  free(f2);
  return 0;
}
