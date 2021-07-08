#ifndef DIFFUSION_KER19_H
#define DIFFUSION_KER19_H

#ifndef REAL
#define REAL double
#endif

void allocate_ker19(REAL **buff_ret, const int nx, const int ny, const int nz);

void init_ker19(REAL *buff, const int nx, const int ny, const int nz,
		const REAL kx, const REAL ky, const REAL kz,
		const REAL dx, const REAL dy, const REAL dz,
		const REAL kappa, const REAL time);

void diffusion_ker19(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
		     REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		     REAL cb, REAL cc, REAL dt,
		     REAL **f1_ret, REAL **f2_ret, REAL *time_ret, int *count_ret);


#endif /* DIFFUSION_KER19_H */
