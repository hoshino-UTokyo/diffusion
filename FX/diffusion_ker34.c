#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#ifdef SVE
#include <arm_sve.h>
#endif /* SVE */
#include "diffusion_ker34.h"

#ifndef REAL
#define REAL double
#endif

#ifndef M_PI
#define M_PI (3.1415926535897932384626)
#endif

#ifndef MIN
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#endif

void allocate_ker34(REAL **buff_ret, const int nx, const int ny, const int nz) {

  posix_memalign((void**)buff_ret, 64, sizeof(REAL)*nx*ny*nz);

}

void init_ker34(REAL *buff1, const int nx, const int ny, const int nz,
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


void diffusion_ker34(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		     REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		     REAL cb, REAL cc, REAL dt,
		     REAL **f1_ret, REAL **f2_ret, REAL *time_ret, int *count_ret) {

#pragma omp parallel
  {
    REAL time = 0.0;
    int count = 0;
    REAL *restrict f1_t = f1;
    REAL *restrict f2_t = f2;
    int c, w, e, n, s, b, t, b0, t0, c2;
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
    REAL *zb_o,*zc_o,*zt_o;
    REAL *zb,*zc,*zt;
/* #pragma omp critical */
/*     { */
/*       posix_memalign((void**)zb_o, 64, sizeof(REAL)*nx*(ychunk+2)); */
/*       posix_memalign((void**)zc_o, 64, sizeof(REAL)*nx*(ychunk+2)); */
/*       posix_memalign((void**)zt_o, 64, sizeof(REAL)*nx*(ychunk+2)); */
/*     } */

    zb_o = (REAL*)malloc(sizeof(REAL)*nx*(ychunk+2));
    zc_o = (REAL*)malloc(sizeof(REAL)*nx*(ychunk+2));
    zt_o = (REAL*)malloc(sizeof(REAL)*nx*(ychunk+2));
    zb = zb_o;
    zc = zc_o;
    zt = zt_o;
    /* for(x = 0; x < nx*(ychunk+2); x++){ */
    /*   zb[x] = 0; */
    /*   zc[x] = 0; */
    /*   zt[x] = 0; */
    /* } */
    const svfloat64_t cc_vec = svdup_f64(cc);
    const svfloat64_t cw_vec = svdup_f64(cw);
    const svfloat64_t ce_vec = svdup_f64(ce);
    const svfloat64_t cs_vec = svdup_f64(cs);
    const svfloat64_t cn_vec = svdup_f64(cn);
    const svfloat64_t cb_vec = svdup_f64(cb);
    const svfloat64_t ct_vec = svdup_f64(ct);
    const svbool_t pg = svptrue_b64();
#pragma statement cache_sector_size 4 0 8 8
#pragma statement cache_subsector_assign zb, zc, zt
    do {
      int iy,iystr,iyend;
      iystr = (ystr == 0)  ? 1 : 0;
      iyend = (yend == ny) ? ychunk+1: ychunk+2;

      b = (zstr == 0) ? 0 : -1;
      /* if(zstr != 0){ */
	y = ystr + iystr -1;
	for(iy = iystr;iy < iyend;iy++,y++){
	  for(x = 0; x < nx; x++){
	    //	    zb[x+iy*nx] = f1_t[x+y*nx+(zstr-1)*nx*ny];
	    zb[x+iy*nx] = f1_t[x+y*nx+(zstr+b)*nx*ny];
	  }
	}
      /* } */
      y = ystr + iystr -1; 
      for(iy = iystr;iy < iyend;iy++,y++){
	for(x = 0; x < nx; x++){
	  zc[x+iy*nx] = f1_t[x+y*nx+(zstr)*nx*ny];
	}
      }
      for (z = zstr; z < zend; z++) {
	t = (z == nz-1) ? 0 : 1;
	
	/* if(z == 0)    zb = zc; */
	/* if(z == nz-1){ */
	/*   zt = zc; */
	/* }else{ */
	  y = ystr + iystr -1; 
	  for(iy = iystr;iy < iyend;iy++,y++){
	    for(x = 0; x < nx; x++){
	      //	      zt[x+iy*nx] = f1_t[x+y*nx+(z+1)*nx*ny];
	      zt[x+iy*nx] = f1_t[x+y*nx+(z+t)*nx*ny];
	    }
	  }
	/* } */
	for (y = ystr,iy = 1; y < yend; y+=2, iy+=2) {
	  n = (y == 0)    ? 0 : - nx;
	  s = (y == ny-2) ? 0 :   nx;
	  c2 =  y * nx + z * nx * ny;
	  c =  iy * nx;

	  svfloat64_t fc_vec00;
	  svfloat64_t fc_vec10;
	  svfloat64_t tmp00;
	  svfloat64_t tmp10;
	  // n
	  fc_vec00  = svld1(pg,(float64_t*)&zc[c+n      ]);
	  fc_vec10  = svld1(pg,(float64_t*)&zc[c        ]);
	  tmp00     = svmul_x(pg,cn_vec,fc_vec00);
	  tmp10     = svmul_x(pg,cn_vec,fc_vec10);
	  // c
	  fc_vec00  = fc_vec10;
	  fc_vec10  = svld1(pg,(float64_t*)&zc[c     +nx]);
	  tmp00     = svmad_x(pg,cc_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,cc_vec,fc_vec10,tmp10);
	  // s
	  fc_vec00  = fc_vec10;
	  fc_vec10  = svld1(pg,(float64_t*)&zc[c+s   +nx]);
	  tmp00     = svmad_x(pg,cs_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,cs_vec,fc_vec10,tmp10);
	  // w
	  float64_t fcm1[8] = {zc[c],zc[c],zc[c+1],zc[c+2],zc[c+3],zc[c+4],zc[c+5],zc[c+6]};
	  float64_t fcm2[8] = {zc[c+nx],zc[c+nx],zc[c+1+nx],zc[c+2+nx],zc[c+3+nx],zc[c+4+nx],zc[c+5+nx],zc[c+6+nx]};
	  fc_vec00  = svld1(pg,(float64_t*)&fcm1[0        ]);
	  fc_vec10  = svld1(pg,(float64_t*)&fcm2[0        ]);
	  tmp00     = svmad_x(pg,cw_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,cw_vec,fc_vec10,tmp10);
	  // e
	  fc_vec00  = svld1(pg,(float64_t*)&zc[c+1      ]);
	  fc_vec10  = svld1(pg,(float64_t*)&zc[c+1   +nx]);
	  tmp00     = svmad_x(pg,ce_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,ce_vec,fc_vec10,tmp10);
	  // b
	  fc_vec00  = svld1(pg,(float64_t*)&zb[c        ]);
	  fc_vec10  = svld1(pg,(float64_t*)&zb[c     +nx]);
	  tmp00     = svmad_x(pg,cb_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,cb_vec,fc_vec10,tmp10);
	  // t
	  fc_vec00  = svld1(pg,(float64_t*)&zt[c        ]);
	  fc_vec10  = svld1(pg,(float64_t*)&zt[c     +nx]);
	  tmp00     = svmad_x(pg,ct_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,ct_vec,fc_vec10,tmp10);
	  
	  svst1(pg,(float64_t*)&f2_t[c2      ],tmp00);
	  svst1(pg,(float64_t*)&f2_t[c2   +nx],tmp10);

	  c += 8;
	  c2 += 8;
	  /* w += 32; */
	  /* e += 32; */
	  /* n += 32; */
	  /* s += 32; */
	  /* b += 32; */
	  /* t += 32; */

	  for (x = 8; x < nx-8; x+=8) {
	    
	    // n
	    fc_vec00  = svld1(pg,(float64_t*)&zc[c+n      ]);
	    fc_vec10  = svld1(pg,(float64_t*)&zc[c        ]);
	    tmp00     = svmul_x(pg,cn_vec,fc_vec00);
	    tmp10     = svmul_x(pg,cn_vec,fc_vec10);
	    // c
	    fc_vec00  = fc_vec10;
	    fc_vec10  = svld1(pg,(float64_t*)&zc[c     +nx]);
	    tmp00     = svmad_x(pg,cc_vec,fc_vec00,tmp00);
	    tmp10     = svmad_x(pg,cc_vec,fc_vec10,tmp10);
	    // s
	    fc_vec00  = fc_vec10;
	    fc_vec10  = svld1(pg,(float64_t*)&zc[c+s   +nx]);
	    tmp00     = svmad_x(pg,cs_vec,fc_vec00,tmp00);
	    tmp10     = svmad_x(pg,cs_vec,fc_vec10,tmp10);
	    // w
	    fc_vec00  = svld1(pg,(float64_t*)&zc[c-1      ]);
	    fc_vec10  = svld1(pg,(float64_t*)&zc[c-1   +nx]);
	    tmp00     = svmad_x(pg,cw_vec,fc_vec00,tmp00);
	    tmp10     = svmad_x(pg,cw_vec,fc_vec10,tmp10);
	    // e
	    fc_vec00  = svld1(pg,(float64_t*)&zc[c+1      ]);
	    fc_vec10  = svld1(pg,(float64_t*)&zc[c+1   +nx]);
	    tmp00     = svmad_x(pg,ce_vec,fc_vec00,tmp00);
	    tmp10     = svmad_x(pg,ce_vec,fc_vec10,tmp10);
	    // b
	    fc_vec00  = svld1(pg,(float64_t*)&zb[c        ]);
	    fc_vec10  = svld1(pg,(float64_t*)&zb[c     +nx]);
	    tmp00     = svmad_x(pg,cb_vec,fc_vec00,tmp00);
	    tmp10     = svmad_x(pg,cb_vec,fc_vec10,tmp10);
	    // t
	    fc_vec00  = svld1(pg,(float64_t*)&zt[c        ]);
	    fc_vec10  = svld1(pg,(float64_t*)&zt[c     +nx]);
	    tmp00     = svmad_x(pg,ct_vec,fc_vec00,tmp00);
	    tmp10     = svmad_x(pg,ct_vec,fc_vec10,tmp10);
	    
	    svst1(pg,(float64_t*)&f2_t[c2      ],tmp00);
	    svst1(pg,(float64_t*)&f2_t[c2   +nx],tmp10);
	    
	    c += 8;
	    c2 += 8;
	    /* w += 32; */
	    /* e += 32; */
	    /* n += 32; */
	    /* s += 32; */
	    /* b += 32; */
	    /* t += 32; */

	  }
	  
	  // n
	  fc_vec00  = svld1(pg,(float64_t*)&zc[c+n      ]);
	  fc_vec10  = svld1(pg,(float64_t*)&zc[c        ]);
	  tmp00     = svmul_x(pg,cn_vec,fc_vec00);
	  tmp10     = svmul_x(pg,cn_vec,fc_vec10);
	  // c
	  fc_vec00  = fc_vec10;
	  fc_vec10  = svld1(pg,(float64_t*)&zc[c     +nx]);
	  tmp00     = svmad_x(pg,cc_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,cc_vec,fc_vec10,tmp10);
	  // s
	  fc_vec00  = fc_vec10;
	  fc_vec10  = svld1(pg,(float64_t*)&zc[c+s   +nx]);
	  tmp00     = svmad_x(pg,cs_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,cs_vec,fc_vec10,tmp10);
	  // w
	  fc_vec00  = svld1(pg,(float64_t*)&zc[c-1      ]);
	  fc_vec10  = svld1(pg,(float64_t*)&zc[c-1   +nx]);
	  tmp00     = svmad_x(pg,cw_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,cw_vec,fc_vec10,tmp10);
	  // e
	  float64_t fcp1[8] = {zc[c+1],zc[c+2],zc[c+3],zc[c+4],zc[c+5],zc[c+6],zc[c+7],zc[c+7]};
	  float64_t fcp2[8] = {zc[c+1+nx],zc[c+2+nx],zc[c+3+nx],zc[c+4+nx],zc[c+5+nx],zc[c+6+nx],zc[c+7+nx],zc[c+7+nx]};
	  fc_vec00  = svld1(pg,(float64_t*)&fcp1[0        ]);
	  fc_vec10  = svld1(pg,(float64_t*)&fcp2[0        ]);
	  tmp00     = svmad_x(pg,ce_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,ce_vec,fc_vec10,tmp10);
	  // b
	  fc_vec00  = svld1(pg,(float64_t*)&zb[c        ]);
	  fc_vec10  = svld1(pg,(float64_t*)&zb[c     +nx]);
	  tmp00     = svmad_x(pg,cb_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,cb_vec,fc_vec10,tmp10);
	  // t
	  fc_vec00  = svld1(pg,(float64_t*)&zt[c        ]);
	  fc_vec10  = svld1(pg,(float64_t*)&zt[c     +nx]);
	  tmp00     = svmad_x(pg,ct_vec,fc_vec00,tmp00);
	  tmp10     = svmad_x(pg,ct_vec,fc_vec10,tmp10);
	  
	  svst1(pg,(float64_t*)&f2_t[c2      ],tmp00);
	  svst1(pg,(float64_t*)&f2_t[c2   +nx],tmp10);

	}
	REAL *ztmp = zb;
	zb = zc;
	zc = zt;
	zt = ztmp;
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
#pragma statement end_cache_subsector
#pragma statement end_cache_sector_size
    free(zb_o);
    free(zc_o);
    free(zt_o);
  }
}
  

