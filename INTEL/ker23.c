
{
  int izm = 0; 
  int izc = 1; 
  int izp = 2; 

  for (int h = 0; h < TB; h++){  // ????
    int z = h;
    int step = 0;
    
    int halo = TB-step;
    int b0 = (z == 0)    ? 0 : - nx * ny;
    int t0 = (z == nz-1) ? 0 :   nx * ny;

    int ystr = yy-halo;
    int yend = yy+YBF+halo;
    int xstr = xx-halo;
    int xend = xx+XBF+halo;
    int iy = step;
    for (y = ystr; y < yend; y++) {
      int ix = step;
      for (x = xstr; x < xend; x++) {
	int c = x + y*nx + z*nx*ny;
	int w = c - 1;
	int e = c + 1;
	int n = c - nx;
	int s = c + nx;
	int t = c + t0;
	int b = c + b0;
	temporal[step][izp][iy][ix] = cc * f1_t[c] + cw * f1_t[w] + ce * f1_t[e]
	  + cs * f1_t[s] + cn * f1_t[n] + cb * f1_t[b] + ct * f1_t[t];
	ix++;
      }
      iy++;
    }
    
    for( z = h-1; z >= 0; z--){
      int b0 = (z == 0)    ? 0 : - 1;
      int t0 = (z == nz-1) ? 0 :   1;
      int b = (z == 0)    ? izc : izm;
      int t = (z == nz-1) ? izc : izp;
      
      int halo = TB-(step+1);
      int ystr = yy-halo;
      int yend = yy+YBF+halo;
      int xstr = xx-halo;
      int xend = xx+XBF+halo;
      int iy = step+1;
      for (y = ystr; y < yend; y++) {
	int ix = step+1;
	for (x = xstr; x < xend; x++) {
	  temporal[step+1][izp][iy][ix] = cc * temporal[step][izc][iy][ix] + cw * temporal[step][izc][iy][ix-1] + ce * temporal[step][izc][iy][ix+1]
	    + cs * temporal[step][izc][iy+1][ix] + cn * temporal[step][izc][iy-1][ix] + cb * temporal[step][b][iy][ix] + ct * temporal[step][t][iy][ix];
	  ix++;
	}
	iy++;
      }
      step++;
    }
    int tmp = izm;
    izm = izc;
    izc = izp;
    izp = tmp;
    
  }
  
  for (zz = 0; zz < nz-TB; zz++) {
    h = TB;
    z = zz+h;
    b0 = (z == 0)    ? 0 : - nx * ny;
    t0 = (z == nz-1) ? 0 :   nx * ny;
    
    step = 0;
    int halo = TB-step;
    for (y = yy-halo,iy = (yy-TB < 0) ? TB : step; y < yy+YBF+halo; y++,iy++) {
      for (x = xx-halo,ix = (xx-TB < 0) ? TB : step; x < xx+XBF+halo; x++,ix++) {
	c = x + y*nx + z*nx*ny;
	w = c - 1;
	e = c + 1;
	n = c - nx;
	s = c + nx;
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
      
      int halo = TB-(step+1);
      for (y = yy-halo,iy = (yy-TB < 0) ? TB : step+1; y < yy+YBF+halo; y++,iy++) {
	for (x = xx-halo,ix = (xx-TB < 0) ? TB : step+1; x < xx+XBF+halo; x++,ix++) {
	  temporal[step+1][izp][iy][ix] = cc * temporal[step][izc][iy][ix] + cw * temporal[step][izc][iy][ix-1] + ce * temporal[step][izc][iy][ix+1]
	    + cs * temporal[step][izc][iy+1][ix] + cn * temporal[step][izc][iy-1][ix] + cb * temporal[step][b][iy][ix] + ct * temporal[step][t][iy][ix];
	}
      }
      step++;
    }
    
    z = zz;
    b0 = (z == 0)    ? 0 : - 1;
    t0 = (z == nz-1) ? 0 :   1;
    b = (z == 0)    ? izc : izm;
    t = (z == nz-1) ? izc : izp;
    
    halo = TB-(step+1);
    for (y = yy-halo,iy = (yy-TB < 0) ? TB : step+1; y < yy+YBF+halo; y++,iy++) {
      for (x = xx-halo,ix = (xx-TB < 0) ? TB : step+1; x < xx+XBF+halo; x++,ix++) {
	c = x + y*nx + z*nx*ny;
	f2_t[c] = cc * temporal[step][izc][iy][ix] + cw * temporal[step][izc][iy][ix-1] + ce * temporal[step][izc][iy][ix+1]
	  + cs * temporal[step][izc][iy+1][ix] + cn * temporal[step][izc][iy-1][ix] + cb * temporal[step][b][iy][ix] + ct * temporal[step][t][iy][ix];
      }
    }
    int tmp = izm;
    izm = izc;
    izc = izp;
    izp = tmp;
    
  }
  
  for (zz = nz-TB, h = TB-1; zz < nz; zz++, h--) {
    
    step = TB-1-h;
    
    for(z = zz+h ; z >= zz+1; z--){
      b0 = (z == 0)    ? 0 : - 1;
      t0 = (z == nz-1) ? 0 :   1;
      t = (z == nz-1) ? izc : izp;
      int halo = TB-(step+1);
      for (y = yy-halo,iy = (yy-TB < 0) ? TB : step+1; y < yy+YBF+halo; y++,iy++) {
	for (x = xx-halo,ix = (xx-TB < 0) ? TB : step+1; x < xx+XBF+halo; x++,ix++) {
	  temporal[step+1][izp][iy][ix] = cc * temporal[step][izc][iy][ix] + cw * temporal[step][izc][iy][ix-1] + ce * temporal[step][izc][iy][ix+1]
	    + cs * temporal[step][izc][iy+1][ix] + cn * temporal[step][izc][iy-1][ix] + cb * temporal[step][izm][iy][ix] + ct * temporal[step][t][iy][ix];
	}
      }
      step++;
    }
    
    z = zz;
    b0 = (z == 0)    ? 0 : - 1;
    t0 = (z == nz-1) ? 0 :   1;
    t = (z == nz-1) ? izc : izp;
    b = (z == 0)    ? izc : izm;
    int halo = TB-(step+1);
    for (y = yy-halo,iy = (yy-TB < 0) ? TB : step+1; y < yy+YBF+halo; y++,iy++) {
      for (x = xx-halo,ix = (xx-TB < 0) ? TB : step+1; x < xx+XBF+halo; x++,ix++) {
	c = x + y*nx + z*nx*ny;
	f2_t[c] = cc * temporal[step][izc][iy][ix] + cw * temporal[step][izc][iy][ix-1] + ce * temporal[step][izc][iy][ix+1]
	  + cs * temporal[step][izc][iy+1][ix] + cn * temporal[step][izc][iy-1][ix] + cb * temporal[step][b][iy][ix] + ct * temporal[step][t][iy][ix];
      }
    }
    
    int tmp = izm;
    izm = izc;
    izc = izp;
    izp = tmp;
  }
}
