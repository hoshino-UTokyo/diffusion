#ifndef DIFFUSION_H
#define DIFFUSION_H

#ifndef REAL
#define REAL double
#endif
#ifndef NX
#define NX (256)
#endif

void init(REAL *buff, const int nx, const int ny, const int nz,
          const REAL kx, const REAL ky, const REAL kz,
          const REAL dx, const REAL dy, const REAL dz,
          const REAL kappa, const REAL time);

REAL accuracy(const REAL *b1, REAL *b2, const int len);


void diffusion_openmp(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		      REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		      REAL cb, REAL cc, REAL dt,
		      REAL **f_ret, REAL *time_ret, int *count_ret);

void diffusion_openmp_declare_simd(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
				   REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
				   REAL cb, REAL cc, REAL dt,
				   REAL **f_ret, REAL *time_ret, int *count_ret);

void diffusion_openmp_y(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
			REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
			REAL cb, REAL cc, REAL dt,
			REAL **f_ret, REAL *time_ret, int *count_ret);

void diffusion_openmp_y_nowait(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
			       REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
			       REAL cb, REAL cc, REAL dt,
			       REAL **f_ret, REAL *time_ret, int *count_ret);



#endif /* DIFFUSION_H */
