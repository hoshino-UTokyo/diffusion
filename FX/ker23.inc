
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
    int n0 = -nx;
    int s0 =  nx;

#if (YY == 0) || (YY == 3)
    int ystr = 0;
    int iy = TB;
#else
    int ystr = yy-halo;
    int iy = step;
#endif
#if (YY == 2) || (YY == 3)
    int yend = ny;
#else
    int yend = yy+YBF+halo;
#endif
#if (XX == 0) || (XX == 3)
    int xstr = 1;
#else
    int xstr = xx-halo;
#endif
#if (XX == 2) || (XX == 3)
    int xend = nx-1;
#else
    int xend = xx+XBF+halo;
#endif
    for (int y = ystr; y < yend; y++) {
#if (YY == 0) || (YY == 3)
      n0 = (y == 0)    ? 0 : - nx;
#endif
#if (YY == 2) || (YY == 3)
      s0 = (y == ny-1) ? 0 :   nx;
#endif
#if (XX == 0) || (XX == 3)
      int ix = TB;
      {
	int x = 0;
	int c = x + y*nx + z*nx*ny;
	int w = c;
	int e = c + 1;
	int n = c + n0;
	int s = c + s0;
	int t = c + t0;
	int b = c + b0;
	temporal[step][izp][iy][ix] = cc * f1_t[c] + cw * f1_t[w] + ce * f1_t[e]
	  + cs * f1_t[s] + cn * f1_t[n] + cb * f1_t[b] + ct * f1_t[t];
      }
      ix++;
#else
      int ix = step;
#endif
      for (int x = xstr; x < xend; x++) {
	int c = x + y*nx + z*nx*ny;
	int w = c - 1;
	int e = c + 1;
	int n = c + n0;
	int s = c + s0;
	int t = c + t0;
	int b = c + b0;
	temporal[step][izp][iy][ix] = cc * f1_t[c] + cw * f1_t[w] + ce * f1_t[e]
	  + cs * f1_t[s] + cn * f1_t[n] + cb * f1_t[b] + ct * f1_t[t];
	ix++;
      }
#if (XX == 2) || (XX == 3)
      {
	int x = nx-1;
	int c = x + y*nx + z*nx*ny;
	int w = c - 1;
	int e = c;
	int n = c + n0;
	int s = c + s0;
	int t = c + t0;
	int b = c + b0;
	temporal[step][izp][iy][ix] = cc * f1_t[c] + cw * f1_t[w] + ce * f1_t[e]
	  + cs * f1_t[s] + cn * f1_t[n] + cb * f1_t[b] + ct * f1_t[t];
      }
#endif
      iy++;
    }
    
    for(z = h-1; z >= 0; z--){
      int b = (z == 0)    ? izc : izm;
      int t = (z == nz-1) ? izc : izp;
      int n = -1;
      int s =  1;
      
      int halo = TB-(step+1);
#if (YY == 0) || (YY == 3)
      int ystr = 0;
      int iy = TB;
#else
      int ystr = yy-halo;
      int iy = step+1;
#endif
#if (YY == 2) || (YY == 3)
      int yend = ny;
#else
      int yend = yy+YBF+halo;
#endif
#if (XX == 0) || (XX == 3)
      int xstr = 1;
#else
      int xstr = xx-halo;
#endif
#if (XX == 2) || (XX == 3)
      int xend = nx-1;
#else
      int xend = xx+XBF+halo;
#endif
      for (int y = ystr; y < yend; y++) {
#if (YY == 0) || (YY == 3)
	n = (y == 0)    ? 0 : -1;
#endif
#if (YY == 2) || (YY == 3)
	s = (y == ny-1) ? 0 :  1;
#endif
#if (XX == 0) || (XX == 3)
	int ix = TB;
	{
	  temporal[step+1][izp][iy][ix] = cc * temporal[step][izc][iy][ix] + cw * temporal[step][izc][iy][ix] + ce * temporal[step][izc][iy][ix+1]
	    + cs * temporal[step][izc][iy+s][ix] + cn * temporal[step][izc][iy+n][ix] + cb * temporal[step][b][iy][ix] + ct * temporal[step][t][iy][ix];
	}
	ix++;
#else
	int ix = step+1;
#endif
	for (int x = xstr; x < xend; x++) {
	  temporal[step+1][izp][iy][ix] = cc * temporal[step][izc][iy][ix] + cw * temporal[step][izc][iy][ix-1] + ce * temporal[step][izc][iy][ix+1]
	    + cs * temporal[step][izc][iy+s][ix] + cn * temporal[step][izc][iy+n][ix] + cb * temporal[step][b][iy][ix] + ct * temporal[step][t][iy][ix];
	  ix++;
	}
#if (XX == 2) || (XX == 3)
	{
	  temporal[step+1][izp][iy][ix] = cc * temporal[step][izc][iy][ix] + cw * temporal[step][izc][iy][ix-1] + ce * temporal[step][izc][iy][ix]
	    + cs * temporal[step][izc][iy+s][ix] + cn * temporal[step][izc][iy+n][ix] + cb * temporal[step][b][iy][ix] + ct * temporal[step][t][iy][ix];
	}
#endif
	iy++;
      }
      step++;
    }
    int tmp = izm;
    izm = izc;
    izc = izp;
    izp = tmp;
    
  }
  
  for (int zz = 0; zz < nz-TB; zz++) {
    int z = zz+TB;
    int b0 = (z == 0)    ? 0 : - nx * ny;
    int t0 = (z == nz-1) ? 0 :   nx * ny;
    int n0 = -nx;
    int s0 =  nx;
    
    int step = 0;
    int halo = TB-step;
    
#if (YY == 0) || (YY == 3)
    int ystr = 0;
    int iy = TB;
#else
    int ystr = yy-halo;
    int iy = step;
#endif
#if (YY == 2) || (YY == 3)
    int yend = ny;
#else
    int yend = yy+YBF+halo;
#endif
    #if (XX == 0) || (XX == 3)
    int xstr = 1;
#else
    int xstr = xx-halo;
#endif
#if (XX == 2) || (XX == 3)
    int xend = nx-1;
#else
    int xend = xx+XBF+halo;
#endif
    for (int y = ystr; y < yend; y++) {
#if (YY == 0) || (YY == 3)
      n0 = (y == 0)    ? 0 : - nx;
#endif
#if (YY == 2) || (YY == 3)
      s0 = (y == ny-1) ? 0 :   nx;
#endif
#if (XX == 0) || (XX == 3)
      int ix = TB;
      {
	int x = 0;
	int c = x + y*nx + z*nx*ny;
	int w = c;
	int e = c + 1;
	int n = c + n0;
	int s = c + s0;
	int t = c + t0;
	int b = c + b0;
	temporal[step][izp][iy][ix] = cc * f1_t[c] + cw * f1_t[w] + ce * f1_t[e]
	  + cs * f1_t[s] + cn * f1_t[n] + cb * f1_t[b] + ct * f1_t[t];
      }
      ix++;
#else
      int ix = step;
#endif
      for (int x = xstr; x < xend; x++) {
	int c = x + y*nx + z*nx*ny;
	int w = c - 1;
	int e = c + 1;
	int n = c + n0;
	int s = c + s0;
	int t = c + t0;
	int b = c + b0;
	temporal[step][izp][iy][ix] = cc * f1_t[c] + cw * f1_t[w] + ce * f1_t[e]
	  + cs * f1_t[s] + cn * f1_t[n] + cb * f1_t[b] + ct * f1_t[t];
	ix++;
      }
#if (XX == 2) || (XX == 3)
      {
	int x = nx-1;
	int c = x + y*nx + z*nx*ny;
	int w = c - 1;
	int e = c;
	int n = c + n0;
	int s = c + s0;
	int t = c + t0;
	int b = c + b0;
	temporal[step][izp][iy][ix] = cc * f1_t[c] + cw * f1_t[w] + ce * f1_t[e]
	  + cs * f1_t[s] + cn * f1_t[n] + cb * f1_t[b] + ct * f1_t[t];
      }
#endif
      iy++;
    }

    
    for(int z = zz+TB-1; z >= zz+1; z--){
      int b  = (z == 0)    ? izc : izm;
      int t  = (z == nz-1) ? izc : izp;
      int n = -1;
      int s =  1;

      int halo = TB-(step+1);
#if (YY == 0) || (YY == 3)
      int ystr = 0;
      int iy = TB;
#else
      int ystr = yy-halo;
      int iy = step+1;
#endif
#if (YY == 2) || (YY == 3)
      int yend = ny;
#else
      int yend = yy+YBF+halo;
#endif
      
#if (XX == 0) || (XX == 3)
      int xstr = 1;
#else
      int xstr = xx-halo;
#endif
#if (XX == 2) || (XX == 3)
      int xend = nx-1;
#else
      int xend = xx+XBF+halo;
#endif
      for (int y = ystr; y < yend; y++) {
#if (YY == 0) || (YY == 3)
	n = (y == 0)    ? 0 : -1;
#endif
#if (YY == 2) || (YY == 3)
	s = (y == ny-1) ? 0 :  1;
#endif
#if (XX == 0) || (XX == 3)
	int ix = TB;
	{
	  temporal[step+1][izp][iy][ix] = cc * temporal[step][izc][iy][ix] + cw * temporal[step][izc][iy][ix] + ce * temporal[step][izc][iy][ix+1]
	    + cs * temporal[step][izc][iy+s][ix] + cn * temporal[step][izc][iy+n][ix] + cb * temporal[step][b][iy][ix] + ct * temporal[step][t][iy][ix];
	}
	ix++;
#else
	int ix = step+1;
#endif
	for (int x = xstr; x < xend; x++) {
	  temporal[step+1][izp][iy][ix] = cc * temporal[step][izc][iy][ix] + cw * temporal[step][izc][iy][ix-1] + ce * temporal[step][izc][iy][ix+1]
	    + cs * temporal[step][izc][iy+s][ix] + cn * temporal[step][izc][iy+n][ix] + cb * temporal[step][b][iy][ix] + ct * temporal[step][t][iy][ix];
	  ix++;
	}
#if (XX == 2) || (XX == 3)
	{
	  temporal[step+1][izp][iy][ix] = cc * temporal[step][izc][iy][ix] + cw * temporal[step][izc][iy][ix-1] + ce * temporal[step][izc][iy][ix]
	    + cs * temporal[step][izc][iy+s][ix] + cn * temporal[step][izc][iy+n][ix] + cb * temporal[step][b][iy][ix] + ct * temporal[step][t][iy][ix];
	}
#endif
	iy++;
      }
      step++;

    }

    {
      int z = zz;
      int b = (z == 0)    ? izc : izm;
      int t = (z == nz-1) ? izc : izp;
      int n = -1;
      int s =  1;
      
      int halo = TB-(step+1);
#if (YY == 0) || (YY == 3)
      int ystr = 0;
      int iy = TB;
#else
      int ystr = yy-halo;
      int iy = step+1;
#endif
#if (YY == 2) || (YY == 3)
      int yend = ny;
#else
      int yend = yy+YBF+halo;
#endif
#if (XX == 0) || (XX == 3)
      int xstr = 1;
#else
      int xstr = xx-halo;
#endif
#if (XX == 2) || (XX == 3)
      int xend = nx-1;
#else
      int xend = xx+XBF+halo;
#endif
      for (int y = ystr; y < yend; y++) {
#if (YY == 0) || (YY == 3)
	n = (y == 0)    ? 0 : -1;
#endif
#if (YY == 2) || (YY == 3)
	s = (y == ny-1) ? 0 :  1;
#endif
#if (XX == 0) || (XX == 3)
	int ix = TB;
	{
	  int x = 0;
	  int c = x + y*nx + z*nx*ny;
	  f2_t[c] = cc * temporal[step][izc][iy][ix] + cw * temporal[step][izc][iy][ix] + ce * temporal[step][izc][iy][ix+1]
	    + cs * temporal[step][izc][iy+s][ix] + cn * temporal[step][izc][iy+n][ix] + cb * temporal[step][b][iy][ix] + ct * temporal[step][t][iy][ix];
	}
	ix++;
#else
	int ix = step+1;
#endif
	for (int x = xstr; x < xend; x++) {
	  int c = x + y*nx + z*nx*ny;
	  f2_t[c] = cc * temporal[step][izc][iy][ix] + cw * temporal[step][izc][iy][ix-1] + ce * temporal[step][izc][iy][ix+1]
	    + cs * temporal[step][izc][iy+s][ix] + cn * temporal[step][izc][iy+n][ix] + cb * temporal[step][b][iy][ix] + ct * temporal[step][t][iy][ix];
	  ix++;
	}
#if (XX == 2) || (XX == 3)
	int x = nx-1; 
	int c = x + y*nx + z*nx*ny;
	f2_t[c] = cc * temporal[step][izc][iy][ix] + cw * temporal[step][izc][iy][ix-1] + ce * temporal[step][izc][iy][ix]
	  + cs * temporal[step][izc][iy+s][ix] + cn * temporal[step][izc][iy+n][ix] + cb * temporal[step][b][iy][ix] + ct * temporal[step][t][iy][ix];
#endif
	iy++;
      }
    }
    int tmp = izm;
    izm = izc;
    izc = izp;
    izp = tmp;
    
  }
  
  for (int zz = nz-TB, h = TB-1; zz < nz; zz++, h--) {
    
    int step = TB-1-h;
    
    for(int z = zz+h ; z >= zz+1; z--){
      int t = (z == nz-1) ? izc : izp;
      int n = -1;
      int s =  1;
      int halo = TB-(step+1);
#if (YY == 0) || (YY == 3)
      int ystr = 0;
      int iy = TB;
#else
      int ystr = yy-halo;
      int iy = step+1;
#endif
#if (YY == 2) || (YY == 3)
      int yend = ny;
#else
      int yend = yy+YBF+halo;
#endif
#if (XX == 0) || (XX == 3)
      int xstr = 1;
#else
      int xstr = xx-halo;
#endif
#if (XX == 2) || (XX == 3)
      int xend = nx-1;
#else
      int xend = xx+XBF+halo;
#endif
      for (int y = ystr; y < yend; y++) {
#if (YY == 0) || (YY == 3)
	n = (y == 0)    ? 0 : -1;
#endif
#if (YY == 2) || (YY == 3)
	s = (y == ny-1) ? 0 :  1;
#endif
#if (XX == 0) || (XX == 3)
	int ix = TB;
	{
	  temporal[step+1][izp][iy][ix] = cc * temporal[step][izc][iy][ix] + cw * temporal[step][izc][iy][ix] + ce * temporal[step][izc][iy][ix+1]
	    + cs * temporal[step][izc][iy+s][ix] + cn * temporal[step][izc][iy+n][ix] + cb * temporal[step][izm][iy][ix] + ct * temporal[step][t][iy][ix];
	}
	ix++;
#else
	int ix = step+1;
#endif
	for (int x = xstr; x < xend; x++) {
	  temporal[step+1][izp][iy][ix] = cc * temporal[step][izc][iy][ix] + cw * temporal[step][izc][iy][ix-1] + ce * temporal[step][izc][iy][ix+1]
	    + cs * temporal[step][izc][iy+s][ix] + cn * temporal[step][izc][iy+n][ix] + cb * temporal[step][izm][iy][ix] + ct * temporal[step][t][iy][ix];
	  ix++;
	}
#if (XX == 2) || (XX == 3)
	{
	  temporal[step+1][izp][iy][ix] = cc * temporal[step][izc][iy][ix] + cw * temporal[step][izc][iy][ix-1] + ce * temporal[step][izc][iy][ix]
	    + cs * temporal[step][izc][iy+s][ix] + cn * temporal[step][izc][iy+n][ix] + cb * temporal[step][izm][iy][ix] + ct * temporal[step][t][iy][ix];
	}
#endif
	iy++;
      }
      step++;

    }

    {
      int z = zz;
      int t = (z == nz-1) ? izc : izp;
      int b = (z == 0)    ? izc : izm;
      int n = -1;
      int s =  1;
      int halo = TB-(step+1);
      
#if (YY == 0) || (YY == 3)
      int ystr = 0;
      int iy = TB;
#else
      int ystr = yy-halo;
      int iy = step+1;
#endif
#if (YY == 2) || (YY == 3)
      int yend = ny;
#else
      int yend = yy+YBF+halo;
#endif
#if (XX == 0) || (XX == 3)
      int xstr = 1;
#else
      int xstr = xx-halo;
#endif
#if (XX == 2) || (XX == 3)
      int xend = nx-1;
#else
      int xend = xx+XBF+halo;
#endif
      for (int y = ystr; y < yend; y++) {
#if (YY == 0) || (YY == 3)
	n = (y == 0)    ? 0 : -1;
#endif
#if (YY == 2) || (YY == 3)
	s = (y == ny-1) ? 0 :  1;
#endif
#if (XX == 0) || (XX == 3)
	int ix = TB;
	{
	  int x = 0;
	  int c = x + y*nx + z*nx*ny;
	  f2_t[c] = cc * temporal[step][izc][iy][ix] + cw * temporal[step][izc][iy][ix] + ce * temporal[step][izc][iy][ix+1]
	    + cs * temporal[step][izc][iy+s][ix] + cn * temporal[step][izc][iy+n][ix] + cb * temporal[step][b][iy][ix] + ct * temporal[step][t][iy][ix];
	}
	ix++;
#else
	int ix = step+1;
#endif
	for (int x = xstr; x < xend; x++) {
	  int c = x + y*nx + z*nx*ny;
	  f2_t[c] = cc * temporal[step][izc][iy][ix] + cw * temporal[step][izc][iy][ix-1] + ce * temporal[step][izc][iy][ix+1]
	    + cs * temporal[step][izc][iy+s][ix] + cn * temporal[step][izc][iy+n][ix] + cb * temporal[step][b][iy][ix] + ct * temporal[step][t][iy][ix];
	  ix++;
	}
#if (XX == 2) || (XX == 3)
	int x = nx-1; 
	int c = x + y*nx + z*nx*ny;
	f2_t[c] = cc * temporal[step][izc][iy][ix] + cw * temporal[step][izc][iy][ix-1] + ce * temporal[step][izc][iy][ix]
	  + cs * temporal[step][izc][iy+s][ix] + cn * temporal[step][izc][iy+n][ix] + cb * temporal[step][b][iy][ix] + ct * temporal[step][t][iy][ix];
#endif
	iy++;
      }
    }
    
    int tmp = izm;
    izm = izc;
    izc = izp;
    izp = tmp;
  }
}
