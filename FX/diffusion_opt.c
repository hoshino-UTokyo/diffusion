#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include <openacc.h>

#ifndef REAL
#define REAL double
#endif
#define NX (128)
//#define NX (256)
#ifndef M_PI
#define M_PI (3.1415926535897932384626)
#endif

extern void diffusion_cuda_host(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
				REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
				REAL cb, REAL cc, REAL dt,
				REAL **f_ret, REAL *time_ret, int *count_ret);

void init(REAL *buff, const int nx, const int ny, const int nz,
          const REAL kx, const REAL ky, const REAL kz,
          const REAL dx, const REAL dy, const REAL dz,
          const REAL kappa, const REAL time) {
  REAL ax, ay, az;
  int jz, jy, jx;
  ax = exp(-kappa*time*(kx*kx));
  ay = exp(-kappa*time*(ky*ky));
  az = exp(-kappa*time*(kz*kz));
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

typedef void (*diffusion_loop_t)(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
                                 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
                                 REAL cb, REAL cc, REAL dt,
                                 REAL **f_ret, REAL *time_ret, int *count_ret);


// openacc
static void
diffusion_openacc_baseline(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
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
#pragma acc kernels present(f1_t) present(f2_t)
      {
#pragma acc loop seq
	for (z = 0; z < nz; z++) {
#pragma acc loop independent gang 
	  for (y = 0; y < ny; y++) {
#pragma acc loop independent vector(128)
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


// openacc
static void
diffusion_openacc_async(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
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
#pragma acc kernels async(0) present(f1_t) present(f2_t)
      {
#pragma acc loop seq
	for (z = 0; z < nz; z++) {
#pragma acc loop independent gang 
	  for (y = 0; y < ny; y++) {
#pragma acc loop independent vector(128)
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

static void
diffusion_openacc_loop_interchange(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
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
#pragma acc kernels async(0) present(f1_t) present(f2_t)
      {
#pragma acc loop independent gang 
	for (y = 0; y < ny; y++) {
#pragma acc loop independent vector(128)
	  for (x = 0; x < nx; x++) {
#pragma acc loop seq
	    for (z = 0; z < nz; z++) {
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


static void
diffusion_openacc_loop_peeling(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
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
#pragma acc kernels async(0) present(f1_t) present(f2_t)
      {
	#pragma acc loop independent gang 
	for (y = 0; y < ny; y++) {
	  #pragma acc loop independent vector(128)
	  for (x = 0; x < nx; x++) {
	    z = 0;
	    c =  x + y * nx;
	    w = (x == 0)    ? 0 : - 1;
	    e = (x == nx-1) ? 0 :   1;
	    n = (y == 0)    ? 0 : - nx;
	    s = (y == ny-1) ? 0 :   nx;
	    b = 0;
	    t = nx * ny;
	    f2_t[c] = cc * f1_t[c] + cw * f1_t[c+w] + ce * f1_t[c+e]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	    b = - nx * ny;
#pragma acc loop seq
	    for (z = 1; z < nz-1; z++) {
	      c +=  nx * ny;
	      f2_t[c] = cc * f1_t[c] + cw * f1_t[c+w] + ce * f1_t[c+e]
		+ cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
	    }
	    c +=  nx * ny;
	    t = 0;
	    f2_t[c] = cc * f1_t[c] + cw * f1_t[c+w] + ce * f1_t[c+e]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * f1_t[c+b] + ct * f1_t[c+t];
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


static void
diffusion_openacc_register_blocking(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
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
    REAL ft, fc, fb;
    
    do {
#pragma acc kernels async(0) present(f1_t) present(f2_t)
      {
	#pragma acc loop independent gang 
	for (y = 0; y < ny; y++) {
	  #pragma acc loop independent vector(128)
	  for (x = 0; x < nx; x++) {
	    z = 0;
	    c =  x + y * nx;
	    w = (x == 0)    ? 0 : - 1;
	    e = (x == nx-1) ? 0 :   1;
	    n = (y == 0)    ? 0 : - nx;
	    s = (y == ny-1) ? 0 :   nx;
	    ft = f1_t[c+nx*ny];
	    fc = f1_t[c];
	    fb = fc;
	    f2_t[c] = cc * fc + cw * f1_t[c+w] + ce * f1_t[c+e]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * fb + ct * ft;
#pragma acc loop seq
	    for (z = 1; z < nz-1; z++) {
	      c +=  nx * ny;
	      fb = fc;
	      fc = ft;
	      ft = f1_t[c+nx*ny];
	      f2_t[c] = cc * fc + cw * f1_t[c+w] + ce * f1_t[c+e]
		+ cs * f1_t[c+s] + cn * f1_t[c+n] + cb * fb + ct * ft;
	    }
	    c +=  nx * ny;
	    fb = fc;
	    fc = ft;
	    f2_t[c] = cc * fc + cw * f1_t[c+w] + ce * f1_t[c+e]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * fb + ct * ft;
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


static void
diffusion_openacc_tile(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
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
    REAL ft, fc, fb;
    
    do {
#pragma acc kernels async(0) present(f1_t) present(f2_t)
      {
#pragma acc loop independent tile(64,2) gang vector
	for (y = 0; y < ny; y++) {
	  for (x = 0; x < nx; x++) {
	    z = 0;
	    c =  x + y * nx;
	    w = (x == 0)    ? 0 : - 1;
	    e = (x == nx-1) ? 0 :   1;
	    n = (y == 0)    ? 0 : - nx;
	    s = (y == ny-1) ? 0 :   nx;
	    ft = f1_t[c+nx*ny];
	    fc = f1_t[c];
	    fb = fc;
	    f2_t[c] = cc * fc + cw * f1_t[c+w] + ce * f1_t[c+e]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * fb + ct * ft;
#pragma acc loop seq
	    for (z = 1; z < nz-1; z++) {
	      c +=  nx * ny;
	      fb = fc;
	      fc = ft;
	      ft = f1_t[c+nx*ny];
	      f2_t[c] = cc * fc + cw * f1_t[c+w] + ce * f1_t[c+e]
		+ cs * f1_t[c+s] + cn * f1_t[c+n] + cb * fb + ct * ft;
	    }
	    c +=  nx * ny;
	    fb = fc;
	    fc = ft;
	    f2_t[c] = cc * fc + cw * f1_t[c+w] + ce * f1_t[c+e]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * fb + ct * ft;
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

static void
diffusion_openacc_cache(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
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
    REAL ft, fc, fb;
    
    do {
#pragma acc kernels async(0) present(f1_t) present(f2_t)
      {
#pragma acc loop independent gang
	for (y = 0; y < ny; y++) {
#pragma acc loop independent vector(128)
	  for (x = 0; x < nx; x++) {
	    z = 0;
	    c =  x + y * nx;
	    w = (x == 0)    ? 0 : - 1;
	    e = (x == nx-1) ? 0 :   1;
	    n = (y == 0)    ? 0 : - nx;
	    s = (y == ny-1) ? 0 :   nx;
	    ft = f1_t[c+nx*ny];
	    fc = f1_t[c];
	    fb = fc;
	    f2_t[c] = cc * fc + cw * f1_t[c+w] + ce * f1_t[c+e]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * fb + ct * ft;
#pragma acc loop seq
	    for (z = 1; z < nz-1; z++) {
#pragma acc cache(f1_t[c])
	      c +=  nx * ny;
	      fb = fc;
	      fc = ft;
	      ft = f1_t[c+nx*ny];
	      f2_t[c] = cc * fc + cw * f1_t[c+w] + ce * f1_t[c+e]
		+ cs * f1_t[c+s] + cn * f1_t[c+n] + cb * fb + ct * ft;
	    }
	    c +=  nx * ny;
	    fb = fc;
	    fc = ft;
	    f2_t[c] = cc * fc + cw * f1_t[c+w] + ce * f1_t[c+e]
	      + cs * f1_t[c+s] + cn * f1_t[c+n] + cb * fb + ct * ft;
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


static void
diffusion_cuda(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
	       REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
	       REAL cb, REAL cc, REAL dt,
	       REAL **f_ret, REAL *time_ret, int *count_ret) {
  
#pragma acc host_data use_device(f1, f2)
  {
    diffusion_cuda_host(f1, f2, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt,
			f_ret, time_ret, count_ret);
  }
}


int main(int argc, char *argv[]) 
{
  
  struct timeval time_begin, time_end;

  int i = 1;
  int    nx    = NX;
  int    ny    = NX;
  int    nz    = NX;

  REAL   time  = 0.0;
  int    count = 0;  
  
  REAL l, dx, dy, dz, kx, ky, kz, kappa, dt;
  REAL ce, cw, cn, cs, ct, cb, cc;

#ifdef _OPENACC
  acc_init(0);
#endif
  // print data
  printf("(nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);

  diffusion_loop_t diffusion_loop = diffusion_openacc_baseline;
  int opt = 0;
  if(argc > 1) opt = atoi(argv[1]);
  switch(opt) {
  case 0: 
    diffusion_loop = diffusion_openacc_baseline;
    fprintf(stdout, "kernel type  : baseline\n");
    break;
  case 1: 
    diffusion_loop = diffusion_openacc_async;
    fprintf(stdout, "kernel type  : async\n");
    break;
  case 2: 
    diffusion_loop = diffusion_openacc_loop_interchange;
    fprintf(stdout, "kernel type  : loop interchange\n");
    break;
  case 3: 
    diffusion_loop = diffusion_openacc_loop_peeling;
    fprintf(stdout, "kernel type  : loop peeling\n");
    break;
  case 4: 
    diffusion_loop = diffusion_openacc_register_blocking;
    fprintf(stdout, "kernel type  : register blocking\n");
    break;
  case 5: 
    diffusion_loop = diffusion_cuda;
    fprintf(stdout, "kernel type  : cuda\n");
    break;
  case 6: 
    diffusion_loop = diffusion_openacc_tile;
    fprintf(stdout, "kernel type  : tile\n");
    break;
  case 7: 
    diffusion_loop = diffusion_openacc_cache;
    fprintf(stdout, "kernel type  : cache\n");
    break;
  default: 
    fprintf(stderr,"unknown option\n");
    break;
  }

  l = 1.0;
  kappa = 0.1;
  dx = dy = dz = l / nx;
  kx = ky = kz = 2.0 * M_PI;
  dt = 0.1*dx*dx / kappa;
  ce = cw = kappa*dt/(dx*dx);
  cn = cs = kappa*dt/(dy*dy);
  ct = cb = kappa*dt/(dz*dz);
  cc = 1.0 - (ce + cw + cn + cs + ct + cb);

  REAL *f1 = (REAL *)malloc(sizeof(REAL)*nx*ny*nz);
  REAL *f2 = (REAL *)malloc(sizeof(REAL)*nx*ny*nz);  
  init(f1, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);

#pragma acc data copy(f1[0:nx*ny*nz]) copy(f2[0:nx*ny*nz])
  {
#pragma acc wait
    gettimeofday(&time_begin, NULL);
    diffusion_loop(f1, f2, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, dt,
		      &f1, &time, &count);
#pragma acc wait
    gettimeofday(&time_end, NULL);
  }
    
  REAL *answer = (REAL *)malloc(sizeof(REAL) * nx*ny*nz);
  init(answer, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);

  REAL err = accuracy(f1, answer, nx*ny*nz);
  double elapsed_time = (time_end.tv_sec - time_begin.tv_sec)
      + (time_end.tv_usec - time_begin.tv_usec)*1.0e-6;
  REAL gflops = (nx*ny*nz)*13.0*count/elapsed_time * 1.0e-09;
  double thput = (nx * ny * nz) * sizeof(REAL) * 2.0 * count
      / elapsed_time * 1.0e-09;

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
