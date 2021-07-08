#define REAL double

__global__ void diffusion_kernel(
				 const REAL * __restrict__ gf1, REAL *gf2, int nx, int ny, int nz,
				 REAL ce, REAL cw, REAL cn, REAL cs, REAL ct, REAL cb, REAL cc){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int c = i + j*nx;
  int xy = nx*ny;
  int w = (i == 0)      ? c : c - 1;
  int e = (i == nx-1)   ? c : c + 1;
  int n = (j == 0)      ? c : c - nx;
  int s = (j == ny-1)   ? c : c + nx;
  REAL t1, t2, t3;
  
  t1 = t2 = gf1[c];
  t3 = gf1[c + xy];
  gf2[c] = cc*t2 + cw*gf1[w] + ce*gf1[e] + cs*gf1[s]
    + cn*gf1[n] + cb*t1 + ct*t3;
  c += xy;
  w += xy;
  e += xy;
  n += xy;
  s += xy;
  
  for(int k=1; k<nz-1; k++){
    t1 = t2;
    t2 = t3;
    t3 = gf1[c+xy];
    gf2[c] = cc*t2 + cw*gf1[w] + ce*gf1[e] + cs*gf1[s]
      + cn*gf1[n] + cb*t1 + ct*t3;
    c += xy;
    w += xy;
    e += xy;
    n += xy;
    s += xy;
  }
  t1 = t2;
  t2 = t3;
  gf2[c] = cc*t2 + cw*gf1[w] + ce*gf1[e] + cs*gf1[s]
    + cn*gf1[n] + cb*t1 + ct*t3;
  return;
}


extern "C" {
  void diffusion_cuda_host(REAL *f1, REAL *f2, int nx, int ny, int nz,
		      REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
		      REAL cb, REAL cc, REAL dt,
		      REAL **f_ret, REAL *time_ret, int *count_ret) {
    REAL time = 0.0;
    int count = 0;
    
    int blockDim_x = 128;
    int blockDim_y = 1;
    int blockDim_z = 1;
    int grid_x = (nx-1) / blockDim_x + 1;
    int grid_y = (ny-1) / blockDim_y + 1;
    int grid_z = 1;
    
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 threads(blockDim_x, blockDim_y, blockDim_z);
    
    //cudaFuncSetCacheConfig(diffusion_kernel, cudaFuncCachePreferL1);
    
    do {
      diffusion_kernel<<<grid, threads>>>(f1, f2, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc);
      REAL *t = f1;
      f1 = f2;
      f2 = t;
      time += dt;
      count++;
    } while (time + 0.5*dt < 0.1);

    //*f_ret = f1;
    *time_ret = time;      
    *count_ret = count;        
    return;
  }
}
