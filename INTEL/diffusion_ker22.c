#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#ifdef SVE
#include <arm_sve.h>
#endif /* SVE */
#include "diffusion_ker22.h"

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

#define XBF 24
#define YBF 8
#define HALO 1
#define ZB (3+HALO-1)
#define STEP 3
#define TB (HALO+STEP-2)

void allocate_ker22(REAL **buff_ret, const int nx, const int ny, const int nz) {

  posix_memalign((void**)buff_ret, 64, sizeof(REAL)*nx*ny*nz);

}

void init_ker22(REAL *buff1, const int nx, const int ny, const int nz,
		const REAL kx, const REAL ky, const REAL kz,
		const REAL dx, const REAL dy, const REAL dz,
		const REAL kappa, const REAL time) {

  REAL ax, ay, az;
  int jz, jy, jx;
  ax = exp(-kappa*time*(kx*kx));
  ay = exp(-kappa*time*(ky*ky));
  az = exp(-kappa*time*(kz*kz));
/* #pragma omp parallel private(jx,jy,jz) */
/*   { */
/*     int tid = omp_get_thread_num(); */
/*     int nth = omp_get_num_threads(); */
/*     int tz = tid/12; */
/*     int ty = tid%12; */
/*     int zchunk = nz/((nth-1)/12+1); */
/*     int yblock = YBF; */
/*     int ychunk = yblock * 12; */
    int yy;
/*     int yystr = ty*yblock; */
    /* for (yy = yystr; yy < ny; yy+= ychunk) { */
    /*   for (jz = tz*zchunk; jz < MIN((tz+1)*zchunk,nz); jz++) { */
    /* 	for (jy = yy; jy < MIN(yy+yblock,ny); jy++) { */
    /* 	  for (jx = 0; jx < nx; jx++) { */
#pragma omp parallel for private(yy,jx,jy,jz)
    for (yy = 0; yy < ny; yy+= YBF) {
      for (jz = 0; jz < nz; jz++) {
    	for (jy = yy; jy < MIN(yy+YBF,ny); jy++) {
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
  /* } */
}


void diffusion_ker22(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		     REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		     REAL cb, REAL cc, REAL dt,
		     REAL **f1_ret, REAL **f2_ret, REAL *time_ret, int *count_ret) {

#pragma omp parallel
  {
    REAL time = 0.0;
    int count = 0;
    REAL *restrict f1_t = f1;
    REAL *restrict f2_t = f2;
    int c, w, e, n, s, b, t, b0, t0, n0, s0;
    int z, y, x, xx, yy, zz, h;
    /* int tid = omp_get_thread_num(); */
    /* int nth = omp_get_num_threads(); */
    /* int tz = tid/12; */
    /* int ty = tid%12; */
    int xblock = XBF;
    int yblock = YBF;
    int hst = TB;
    /* int ychunk = YBF*12; */
    /* int zchunk = nz/((nth-1)/12+1); */
    /* int zstr = tz*zchunk; */
    /* int zend = MIN((tz+1)*zchunk,nz); */
    /* int yystr = ty*yblock; */
    int hight = 3;
    int izm,izc,izp,iy,ix;
    int step;
    //    REAL temporal[2][ZB][YBF+2*HALO][XBF+2*HALO] = {0};
    REAL temporal[STEP-1][3][YBF+2*TB][XBF+2*TB] = {0};

    
    //    int prefetch_flag; 
    /* const svfloat64_t cc_vec = svdup_f64(cc); */
    /* const svfloat64_t cw_vec = svdup_f64(cw); */
    /* const svfloat64_t ce_vec = svdup_f64(ce); */
    /* const svfloat64_t cs_vec = svdup_f64(cs); */
    /* const svfloat64_t cn_vec = svdup_f64(cn); */
    /* const svfloat64_t cb_vec = svdup_f64(cb); */
    /* const svfloat64_t ct_vec = svdup_f64(ct); */
    /* const svbool_t pg = svptrue_b64(); */
    do {
      //      for (yy = yystr; yy < ny; yy+=ychunk) {
      //#pragma omp parallel for private(c,w,e,n,s,b,t,b0,t0,n0,s0,z,y,x,xx,yy,zz,h,izm,izc,izp,iy,ix,step)
#pragma omp for
      for (yy = 0; yy < ny; yy+=yblock) {
	for (xx = 0; xx < nx; xx+=xblock) {

	  izm = 0; 
	  izc = 1; 
	  izp = 2; 
	  for (h = 0; h < hst; h++){  // ????
	    z = h;
	    step = 0;

	    int halo = hst-step;
	    b0 = (z == 0)    ? 0 : - nx * ny;
	    t0 = (z == nz-1) ? 0 :   nx * ny;
	    //	    for (y = MAX(yy-hst,0),iy = (yy-hst < 0) ? hst : 0; y < MIN(yy+yblock+hst,ny); y++,iy++) {
	    for (y = MAX(yy-halo,0),iy = (yy-hst < 0) ? hst : step; y < MIN(yy+yblock+halo,ny); y++,iy++) {
	      n0 = (y == 0)    ? 0 : - nx;
	      s0 = (y == ny-1) ? 0 :   nx;
	      //	      for (x = MAX(xx-hst,0),ix = (xx-hst < 0) ? hst : 0; x < MIN(xx+xblock+hst,nx); x++,ix++) {
	      for (x = MAX(xx-halo,0),ix = (xx-hst < 0) ? hst : step; x < MIN(xx+xblock+halo,nx); x++,ix++) {
		w = (x == 0)    ? 0 : - 1;
		e = (x == nx-1) ? 0 :   1;
		c = x + y*nx + z*nx*ny;
		e = c + e;
		w = c + w;
		n = c + n0;
		s = c + s0;
		t = c + t0;
		b = c + b0;
		//		printf("(yy,ix,iy,iz,step,x,y,z,c) = (%d,%d,%d,%d,%d,%d,%d,%d,%d)\n",yy,ix,iy,izp,step,x,y,z,c);
		temporal[step][izp][iy][ix] = cc * f1_t[c] + cw * f1_t[w] + ce * f1_t[e]
		  + cs * f1_t[s] + cn * f1_t[n] + cb * f1_t[b] + ct * f1_t[t];
		//		printf("(yy,ix,iy,iz,step,x,y,z,c,f1_t[c],f1_t[s]) = (%d,%d,%d,%d,%d,%d,%d,%d,%d,%f,%f)\n",yy,ix,iy,izp,step,x,y,z,c,f1_t[c],f1_t[s]);
	      }
	    }

	    for( z = h-1; z >= 0; z--){
	      b0 = (z == 0)    ? 0 : - 1;
	      t0 = (z == nz-1) ? 0 :   1;
	      b = (z == 0)    ? izc : izm;
	      t = (z == nz-1) ? izc : izp;

	      halo = hst-(step+1);
	      //	      for (y = MAX(yy-hst,0),iy = (yy-hst < 0) ? hst : 0; y < MIN(yy+yblock+hst,ny); y++,iy++) {
	      for (y = MAX(yy-halo,0),iy = (yy-hst < 0) ? hst : step+1; y < MIN(yy+yblock+halo,ny); y++,iy++) {
		n = (y == 0)    ? 0 : - 1;
		s = (y == ny-1) ? 0 :   1;
		//		for (x = MAX(xx-hst,0),ix = (xx-hst < 0) ? hst : 0; x < MIN(xx+xblock+hst,nx); x++,ix++) {
		for (x = MAX(xx-halo,0),ix = (xx-hst < 0) ? hst : step+1; x < MIN(xx+xblock+halo,nx); x++,ix++) {
		  w = (x == 0)    ? 0 : - 1;
		  e = (x == nx-1) ? 0 :   1;
		  temporal[step+1][izp][iy][ix] = cc * temporal[step][izc][iy][ix] + cw * temporal[step][izc][iy][ix+w] + ce * temporal[step][izc][iy][ix+e]
		    + cs * temporal[step][izc][iy+s][ix] + cn * temporal[step][izc][iy+n][ix] + cb * temporal[step][b][iy][ix] + ct * temporal[step][t][iy][ix];
		}
	      }
	      step++;
	    }
	    int tmp = izm;
	    izm = izc;
	    izc = izp;
	    izp = tmp;

	  }


      /* printf("temporal(:,:,:)\n"); */
      /* for(step = 0; step < STEP-1; step++){ */
      /* 	printf("step = %d\n",step); */
      /* 	for(z = 0; z < HALO+2; z++){ */
      /* 	  printf("z = %d\n",z); */
      /* 	  for(y = 0; y < YBF+2*TB; y++){ */
      /* 	    for(x = 0; x < XBF+2*TB; x++){ */
      /* 	      printf("%f ",temporal[step][z][y][x]); */
      /* 	    } */
      /* 	    printf("\n"); */
      /* 	  } */
      /* 	} */
      /* } */
	  
	  for (zz = 0; zz < nz-hst; zz++) {
	    h = hst;
	    z = zz+h;
	    b0 = (z == 0)    ? 0 : - nx * ny;
	    t0 = (z == nz-1) ? 0 :   nx * ny;

	    step = 0;
	    int halo = hst-step;
	    //	    for (y = MAX(yy-hst,0),iy = (yy-hst < 0) ? hst : 0; y < MIN(yy+yblock+hst,ny); y++,iy++) {
	    for (y = MAX(yy-halo,0),iy = (yy-hst < 0) ? hst : step; y < MIN(yy+yblock+halo,ny); y++,iy++) {
	      n0 = (y == 0)    ? 0 : - nx;
	      s0 = (y == ny-1) ? 0 :   nx;
	      
	      //	      for (x = MAX(xx-hst,0),ix = (xx-hst < 0) ? hst : 0; x < MIN(xx+xblock+hst,nx); x++,ix++) {
	      for (x = MAX(xx-halo,0),ix = (xx-hst < 0) ? hst : step; x < MIN(xx+xblock+halo,nx); x++,ix++) {
		w = (x == 0)    ? 0 : - 1;
		e = (x == nx-1) ? 0 :   1;
		c = x + y*nx + z*nx*ny;
		e = c + e;
		w = c + w;
		n = c + n0;
		s = c + s0;
		t = c + t0;
		b = c + b0;
		temporal[step][izp][iy][ix] = cc * f1_t[c] + cw * f1_t[w] + ce * f1_t[e]
		  + cs * f1_t[s] + cn * f1_t[n] + cb * f1_t[b] + ct * f1_t[t];
	      }
	    }
	    
	    for( z = z-1; z >= zz+1; z--){
	      b0 = (z == 0)    ? 0 : - 1;
	      t0 = (z == nz-1) ? 0 :   1;
	      b = (z == 0)    ? izc : izm;
	      t = (z == nz-1) ? izc : izp;

	      int halo = hst-(step+1);
	      //	      for (y = MAX(yy-hst,0),iy = (yy-hst < 0) ? hst : 0; y < MIN(yy+yblock+hst,ny); y++,iy++) {
	      for (y = MAX(yy-halo,0),iy = (yy-hst < 0) ? hst : step+1; y < MIN(yy+yblock+halo,ny); y++,iy++) {
		n = (y == 0)    ? 0 : - 1;
		s = (y == ny-1) ? 0 :   1;
		//		for (x = MAX(xx-hst,0),ix = (xx-hst < 0) ? hst : 0; x < MIN(xx+xblock+hst,nx); x++,ix++) {
		for (x = MAX(xx-halo,0),ix = (xx-hst < 0) ? hst : step+1; x < MIN(xx+xblock+halo,nx); x++,ix++) {
		  w = (x == 0)    ? 0 : - 1;
		  e = (x == nx-1) ? 0 :   1;
		  temporal[step+1][izp][iy][ix] = cc * temporal[step][izc][iy][ix] + cw * temporal[step][izc][iy][ix+w] + ce * temporal[step][izc][iy][ix+e]
		    + cs * temporal[step][izc][iy+s][ix] + cn * temporal[step][izc][iy+n][ix] + cb * temporal[step][b][iy][ix] + ct * temporal[step][t][iy][ix];
		}
	      }
	      step++;
	    }

	    z = zz;
	    b0 = (z == 0)    ? 0 : - 1;
	    t0 = (z == nz-1) ? 0 :   1;
	    b = (z == 0)    ? izc : izm;
	    t = (z == nz-1) ? izc : izp;

	    halo = hst-(step+1);
	    //	    for (y = MAX(yy-hst,0),iy = (yy-hst < 0) ? hst : 0; y < MIN(yy+yblock+hst,ny); y++,iy++) {
	    for (y = MAX(yy-halo,0),iy = (yy-hst < 0) ? hst : step+1; y < MIN(yy+yblock+halo,ny); y++,iy++) {
	      n = (y == 0)    ? 0 : - 1;
	      s = (y == ny-1) ? 0 :   1;
	      //	      for (x = MAX(xx-hst,0),ix = (xx-hst < 0) ? hst : 0; x < MIN(xx+xblock+hst,nx); x++,ix++) {
	      for (x = MAX(xx-halo,0),ix = (xx-hst < 0) ? hst : step+1; x < MIN(xx+xblock+halo,nx); x++,ix++) {
		w = (x == 0)    ? 0 : - 1;
		e = (x == nx-1) ? 0 :   1;
		c = x + y*nx + z*nx*ny;
		f2_t[c] = cc * temporal[step][izc][iy][ix] + cw * temporal[step][izc][iy][ix+w] + ce * temporal[step][izc][iy][ix+e]
		  + cs * temporal[step][izc][iy+s][ix] + cn * temporal[step][izc][iy+n][ix] + cb * temporal[step][b][iy][ix] + ct * temporal[step][t][iy][ix];
		//		printf("(x,y,z) = (%d,%d,%d)\n",x,y,z);
	      }
	    }
	    int tmp = izm;
	    izm = izc;
	    izc = izp;
	    izp = tmp;


	    /* printf("\n\n\n\n\n\n\n\n\n zz = %d temporal(:,:,:)\n", zz); */
	    /* for(step = 0; step < STEP-1;step++){ */
	    /*   printf("step = %d\n",step); */
	    /*   for(z = 0; z < HALO+2; z++){ */
	    /* 	printf("z = %d\n",z); */
	    /* 	for(y = 0; y < YBF+2*TB; y++){ */
	    /* 	  for(x = 0; x < XBF+2*TB; x++){ */
	    /* 	    printf("%f ",temporal[step][z][y][x]); */
	    /* 	  } */
	    /* 	  printf("\n"); */
	    /* 	} */
	    /*   } */
	    /* } */

	    
	  }

	  for (zz = nz-hst, h = hst-1; zz < nz; zz++, h--) {

	    step = hst-1-h;
	    
	    for(z = zz+h ; z >= zz+1; z--){
	      b0 = (z == 0)    ? 0 : - 1;
	      t0 = (z == nz-1) ? 0 :   1;
	      t = (z == nz-1) ? izc : izp;
	      int halo = hst-(step+1);
	      //	      for (y = MAX(yy-hst,0),iy = (yy-hst < 0) ? hst : 0; y < MIN(yy+yblock+hst,ny); y++,iy++) {
	      for (y = MAX(yy-halo,0),iy = (yy-hst < 0) ? hst : step+1; y < MIN(yy+yblock+halo,ny); y++,iy++) {
		n = (y == 0)    ? 0 : - 1;
		s = (y == ny-1) ? 0 :   1;
		//		for (x = MAX(xx-hst,0),ix = (xx-hst < 0) ? hst : 0; x < MIN(xx+xblock+hst,nx); x++,ix++) {
		for (x = MAX(xx-halo,0),ix = (xx-hst < 0) ? hst : step+1; x < MIN(xx+xblock+halo,nx); x++,ix++) {
		  w = (x == 0)    ? 0 : - 1;
		  e = (x == nx-1) ? 0 :   1;
		  temporal[step+1][izp][iy][ix] = cc * temporal[step][izc][iy][ix] + cw * temporal[step][izc][iy][ix+w] + ce * temporal[step][izc][iy][ix+e]
		    + cs * temporal[step][izc][iy+s][ix] + cn * temporal[step][izc][iy+n][ix] + cb * temporal[step][izm][iy][ix] + ct * temporal[step][t][iy][ix];
		}
	      }
	      step++;
	    }

	    z = zz;
	    b0 = (z == 0)    ? 0 : - 1;
	    t0 = (z == nz-1) ? 0 :   1;
	    t = (z == nz-1) ? izc : izp;
	    b = (z == 0)    ? izc : izm;
	    int halo = hst-(step+1);
	    //	    for (y = MAX(yy-hst,0),iy = (yy-hst < 0) ? hst : 0; y < MIN(yy+yblock+hst,ny); y++,iy++) {
	    for (y = MAX(yy-halo,0),iy = (yy-hst < 0) ? hst : step+1; y < MIN(yy+yblock+halo,ny); y++,iy++) {
	      n = (y == 0)    ? 0 : - 1;
	      s = (y == ny-1) ? 0 :   1;
	      //	      for (x = MAX(xx-hst,0),ix = (xx-hst < 0) ? hst : 0; x < MIN(xx+xblock+hst,nx); x++,ix++) {
	      for (x = MAX(xx-halo,0),ix = (xx-hst < 0) ? hst : step+1; x < MIN(xx+xblock+halo,nx); x++,ix++) {
		w = (x == 0)    ? 0 : - 1;
		e = (x == nx-1) ? 0 :   1;
		c = x + y*nx + z*nx*ny;
		f2_t[c] = cc * temporal[step][izc][iy][ix] + cw * temporal[step][izc][iy][ix+w] + ce * temporal[step][izc][iy][ix+e]
		  + cs * temporal[step][izc][iy+s][ix] + cn * temporal[step][izc][iy+n][ix] + cb * temporal[step][b][iy][ix] + ct * temporal[step][t][iy][ix];
		//		printf("(x,y,z) = (%d,%d,%d)\n",x,y,z);
	      }
	    }
	    int tmp = izm;
	    izm = izc;
	    izc = izp;
	    izp = tmp;
	  }
	}
      }
 
      /* printf("temporal(:,:,:)\n"); */
      /* for(step = 0; step < STEP-1; step++){ */
      /* 	printf("step = %d\n",step); */
      /* 	for(z = 0; z < HALO+2; z++){ */
      /* 	  printf("z = %d\n",z); */
      /* 	  for(y = 0; y < YBF+2*TB; y++){ */
      /* 	    for(x = 0; x < XBF+2*TB; x++){ */
      /* 	      printf("%f ",temporal[step][z][y][x]); */
      /* 	    } */
      /* 	    printf("\n"); */
      /* 	  } */
      /* 	} */
      /* } */
      /* printf("f1(:,:,:)\n"); */
      /* for(z = 0; z < nz; z++){ */
      /* 	printf("z = %d\n",z); */
      /* 	for(y = 0; y < ny; y++){ */
      /* 	  for(x = 0; x < nx; x++){ */
      /* 	    printf("%f ",f1_t[x+y*nx+z*nx*ny]); */
      /* 	  } */
      /* 	  printf("\n"); */
      /* 	} */
      /* } */
      /* printf("f2(:,:,:)\n"); */
      /* for(z = 0; z < nz; z++){ */
      /* 	printf("z = %d\n",z); */
      /* 	for(y = 0; y < ny; y++){ */
      /* 	  for(x = 0; x < nx; x++){ */
      /* 	    printf("%f ",f2_t[x+y*nx+z*nx*ny]); */
      /* 	  } */
      /* 	  printf("\n"); */
      /* 	} */
      /* } */
      /* exit(0); */
      
#pragma omp barrier
      REAL *tmp = f1_t;
      f1_t = f2_t;
      f2_t = tmp;
      time += STEP*dt;
      //time += 1;
      count+=STEP;
      
    } while (time + 0.5*dt < 0.1);
#pragma omp master
    {
      *f1_ret = f1_t; *f2_ret = f2_t;;
      *time_ret = time;      
      *count_ret = count;        
    }
    
  }
}

#undef YBF
#undef XBF
#undef HALO 
#undef ZB 
#undef STEP 
#undef TB 
