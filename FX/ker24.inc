
{
  izm = 0; 
  izc = 1; 
  izp = 2; 

  for (h = 0; h < TB; h++){  // ????
    z = h;
    step = 0;
    
    halo = TB-step;
    b0 = (z == 0)    ? 0 : - nx * ny;
    t0 = (z == nz-1) ? 0 :   nx * ny;
    n0 = -nx;
    s0 =  nx;

#if (YY == 0) || (YY == 3)
    ystr = 0;
    iy = TB;
#else
    ystr = yy-halo;
    iy = step;
#endif
#if (YY == 2) || (YY == 3)
    yend = ny;
#else
    yend = yy+YBF+halo;
#endif
#if (XX == 0) || (XX == 3)
    xstr = 1;
#else
    xstr = xx-halo;
#endif
#if (XX == 2) || (XX == 3)
    xend = nx-1;
#else
    xend = xx+XBF+halo;
#endif
    for (y = ystr; y < yend; y++) {
#if (YY == 0) || (YY == 3)
      n0 = (y == 0)    ? 0 : - nx;
#endif
#if (YY == 2) || (YY == 3)
      s0 = (y == ny-1) ? 0 :   nx;
#endif
#if (XX == 0) || (XX == 3)
      ix = TB;
      id2 = ix+iy*tbx+izp*tbx*tby+step*tbx*tby*tbz;
      {
	x = 0;
	c = x + y*nx + z*nx*ny;
	w = c;
	e = c + 1;
	n = c + n0;
	s = c + s0;
	t = c + t0;
	b = c + b0;
	temporal[id2] = cc * f1_t[c] + cw * f1_t[w] + ce * f1_t[e]
	  + cs * f1_t[s] + cn * f1_t[n] + cb * f1_t[b] + ct * f1_t[t];
      }
      id2++;
#else
      ix = step;
      id2 = ix+iy*tbx+izp*tbx*tby+step*tbx*tby*tbz;
#endif
      for (x = xstr; x < xend; x++) {
	c = x + y*nx + z*nx*ny;
	w = c - 1;
	e = c + 1;
	n = c + n0;
	s = c + s0;
	t = c + t0;
	b = c + b0;
	temporal[id2] = cc * f1_t[c] + cw * f1_t[w] + ce * f1_t[e]
	  + cs * f1_t[s] + cn * f1_t[n] + cb * f1_t[b] + ct * f1_t[t];
	id2++;
      }
#if (XX == 2) || (XX == 3)
      {
	x = nx-1;
	c = x + y*nx + z*nx*ny;
	w = c - 1;
	e = c;
	n = c + n0;
	s = c + s0;
	t = c + t0;
	b = c + b0;
	temporal[id2] = cc * f1_t[c] + cw * f1_t[w] + ce * f1_t[e]
	  + cs * f1_t[s] + cn * f1_t[n] + cb * f1_t[b] + ct * f1_t[t];
      }
#endif
      iy++;
    }
    
    for(z = h-1; z >= 0; z--){
      b = (z == 0)    ? 0 : (izm-izc)*tbx*tby;
      t = (z == nz-1) ? 0 : (izp-izc)*tbx*tby;
      n = -tbx;
      s =  tbx;
      
      halo = TB-(step+1);
#if (YY == 0) || (YY == 3)
      ystr = 0;
      iy = TB;
#else
      ystr = yy-halo;
      iy = step+1;
#endif
#if (YY == 2) || (YY == 3)
      yend = ny;
#else
      yend = yy+YBF+halo;
#endif
#if (XX == 0) || (XX == 3)
      xstr = 1;
#else
      xstr = xx-halo;
#endif
#if (XX == 2) || (XX == 3)
      xend = nx-1;
#else
      xend = xx+XBF+halo;
#endif
      for (y = ystr; y < yend; y++) {
#if (YY == 0) || (YY == 3)
	n = (y == 0)    ? 0 : -tbx;
#endif
#if (YY == 2) || (YY == 3)
	s = (y == ny-1) ? 0 :  tbx;
#endif
#if (XX == 0) || (XX == 3)
	ix = TB;
	id2 = ix+iy*tbx+izp*tbx*tby+(step+1)*tbx*tby*tbz;
	id1 = ix+iy*tbx+izc*tbx*tby+step*tbx*tby*tbz;
	{
	  temporal[id2] = cc * temporal[id1] + cw * temporal[id1] + ce * temporal[id1+1]
	    + cs * temporal[id1+s] + cn * temporal[id1+n] + cb * temporal[id1+b] + ct * temporal[id1+t];
	}
	id2++;
	id1++;
#else
	ix = step+1;
	id2 = ix+iy*tbx+izp*tbx*tby+(step+1)*tbx*tby*tbz;
	id1 = ix+iy*tbx+izc*tbx*tby+step*tbx*tby*tbz;
#endif
	for (x = xstr; x < xend; x++) {
	  temporal[id2] = cc * temporal[id1] + cw * temporal[id1-1] + ce * temporal[id1+1]
	    + cs * temporal[id1+s] + cn * temporal[id1+n] + cb * temporal[id1+b] + ct * temporal[id1+t];
	  id2++;
	  id1++;
	}
#if (XX == 2) || (XX == 3)
	{
	  temporal[id2] = cc * temporal[id1] + cw * temporal[id1-1] + ce * temporal[id1]
	    + cs * temporal[id1+s] + cn * temporal[id1+n] + cb * temporal[id1+b] + ct * temporal[id1+t];
	}
#endif
	iy++;
      }
      step++;
    }
    tmp = izm;
    izm = izc;
    izc = izp;
    izp = tmp;
    
  }
  
  for (zz = 0; zz < nz-TB; zz++) {
    z = zz+TB;
    b0 = (z == 0)    ? 0 : - nx * ny;
    t0 = (z == nz-1) ? 0 :   nx * ny;
    n0 = -nx;
    s0 =  nx;
    
    step = 0;
    halo = TB-step;
    
#if (YY == 0) || (YY == 3)
    ystr = 0;
    iy = TB;
#else
    ystr = yy-halo;
    iy = step;
#endif
#if (YY == 2) || (YY == 3)
    yend = ny;
#else
    yend = yy+YBF+halo;
#endif
    #if (XX == 0) || (XX == 3)
    xstr = 1;
#else
    xstr = xx-halo;
#endif
#if (XX == 2) || (XX == 3)
    xend = nx-1;
#else
    xend = xx+XBF+halo;
#endif
    for (y = ystr; y < yend; y++) {
#if (YY == 0) || (YY == 3)
      n0 = (y == 0)    ? 0 : - nx;
#endif
#if (YY == 2) || (YY == 3)
      s0 = (y == ny-1) ? 0 :   nx;
#endif
#if (XX == 0) || (XX == 3)
      ix = TB;
      id2 = ix+iy*tbx+izp*tbx*tby+step*tbx*tby*tbz;
      {
	x = 0;
	c = x + y*nx + z*nx*ny;
	w = c;
	e = c + 1;
	n = c + n0;
	s = c + s0;
	t = c + t0;
	b = c + b0;
	temporal[id2] = cc * f1_t[c] + cw * f1_t[w] + ce * f1_t[e]
	  + cs * f1_t[s] + cn * f1_t[n] + cb * f1_t[b] + ct * f1_t[t];
      }
      id2++;
#else
      ix = step;
      id2 = ix+iy*tbx+izp*tbx*tby+step*tbx*tby*tbz;
#endif
      for (x = xstr; x < xend; x++) {
	c = x + y*nx + z*nx*ny;
	w = c - 1;
	e = c + 1;
	n = c + n0;
	s = c + s0;
	t = c + t0;
	b = c + b0;
	temporal[id2] = cc * f1_t[c] + cw * f1_t[w] + ce * f1_t[e]
	  + cs * f1_t[s] + cn * f1_t[n] + cb * f1_t[b] + ct * f1_t[t];
	id2++;
      }
#if (XX == 2) || (XX == 3)
      {
	x = nx-1;
	c = x + y*nx + z*nx*ny;
	w = c - 1;
	e = c;
	n = c + n0;
	s = c + s0;
	t = c + t0;
	b = c + b0;
	temporal[id2] = cc * f1_t[c] + cw * f1_t[w] + ce * f1_t[e]
	  + cs * f1_t[s] + cn * f1_t[n] + cb * f1_t[b] + ct * f1_t[t];
      }
#endif
      iy++;
    }

    
    for(z = zz+TB-1; z >= zz+1; z--){
      b = (z == 0)    ? 0 : (izm-izc)*tbx*tby;
      t = (z == nz-1) ? 0 : (izp-izc)*tbx*tby;
      n = -tbx;
      s =  tbx;

      halo = TB-(step+1);
#if (YY == 0) || (YY == 3)
      ystr = 0;
      iy = TB;
#else
      ystr = yy-halo;
      iy = step+1;
#endif
#if (YY == 2) || (YY == 3)
      yend = ny;
#else
      yend = yy+YBF+halo;
#endif
      
#if (XX == 0) || (XX == 3)
      //      xstr = 1;
      xstr = 0;
#else
      xstr = xx-halo;
#endif
#if (XX == 2) || (XX == 3)
      xend = nx-1;
#else
      xend = xx+XBF+halo;
#endif
#if (XX == 0) || (XX == 3)
      ix = TB;
#else
      ix = step+1;
#endif
      id2 = ix+iy*tbx+izp*tbx*tby+(step+1)*tbx*tby*tbz;
      id1 = ix+iy*tbx+izc*tbx*tby+step*tbx*tby*tbz;

      for (y = ystr; y < yend; y++) {
#if (YY == 0) || (YY == 3)
	n = (y == 0)    ? 0 : -tbx;
#endif
#if (YY == 2) || (YY == 3)
	s = (y == ny-1) ? 0 :  tbx;
#endif
#if (XX == 0) || (XX == 3)
	/* ix = TB; */
	/* id2 = ix+iy*tbx+izp*tbx*tby+(step+1)*tbx*tby*tbz; */
	/* id1 = ix+iy*tbx+izc*tbx*tby+step*tbx*tby*tbz; */
	{
	  temporal[id2] = cc * temporal[id1] + cw * temporal[id1] + ce * temporal[id1+1]
	    + cs * temporal[id1+s] + cn * temporal[id1+n] + cb * temporal[id1+b] + ct * temporal[id1+t];
	}
	id2++;
	id1++;
	x = xstr+1;
#else
	/* ix = step+1; */
	/* id2 = ix+iy*tbx+izp*tbx*tby+(step+1)*tbx*tby*tbz; */
	/* id1 = ix+iy*tbx+izc*tbx*tby+step*tbx*tby*tbz; */
	x = xstr;
#endif
	for (; x < xend; x++) {
	  temporal[id2] = cc * temporal[id1] + cw * temporal[id1-1] + ce * temporal[id1+1]
	    + cs * temporal[id1+s] + cn * temporal[id1+n] + cb * temporal[id1+b] + ct * temporal[id1+t];
	  id2++;
	  id1++;
	}
#if (XX == 2) || (XX == 3)
	{
	  temporal[id2] = cc * temporal[id1] + cw * temporal[id1-1] + ce * temporal[id1]
	    + cs * temporal[id1+s] + cn * temporal[id1+n] + cb * temporal[id1+b] + ct * temporal[id1+t];
	}
#endif
	/* iy++; */
	id2 = id2 - (xend-xstr) + tbx;
	id1 = id1 - (xend-xstr) + tbx;
      }
      step++;

    }

    {
      z = zz;
      b = (z == 0)    ? 0 : (izm-izc)*tbx*tby;
      t = (z == nz-1) ? 0 : (izp-izc)*tbx*tby;
      n = -tbx;
      s =  tbx;
      
      halo = TB-(step+1);
#if (YY == 0) || (YY == 3)
      ystr = 0;
      iy = TB;
#else
      ystr = yy-halo;
      iy = step+1;
#endif
#if (YY == 2) || (YY == 3)
      yend = ny;
#else
      yend = yy+YBF+halo;
#endif
#if (XX == 0) || (XX == 3)
      xstr = 1;
#else
      xstr = xx-halo;
#endif
#if (XX == 2) || (XX == 3)
      xend = nx-1;
#else
      xend = xx+XBF+halo;
#endif
      for (y = ystr; y < yend; y++) {
#if (YY == 0) || (YY == 3)
	n = (y == 0)    ? 0 : -tbx;
#endif
#if (YY == 2) || (YY == 3)
	s = (y == ny-1) ? 0 :  tbx;
#endif
#if (XX == 0) || (XX == 3)
	ix = TB;
	id1 = ix+iy*tbx+izc*tbx*tby+step*tbx*tby*tbz;
	{
	  x = 0;
	  c = x + y*nx + z*nx*ny;
	  f2_t[c] = cc * temporal[id1] + cw * temporal[id1] + ce * temporal[id1+1]
	    + cs * temporal[id1+s] + cn * temporal[id1+n] + cb * temporal[id1+b] + ct * temporal[id1+t];
	}
	id1++;
#else
	ix = step+1;
	id1 = ix+iy*tbx+izc*tbx*tby+step*tbx*tby*tbz;
#endif
	for (x = xstr; x < xend; x++) {
	  c = x + y*nx + z*nx*ny;
	  f2_t[c] = cc * temporal[id1] + cw * temporal[id1-1] + ce * temporal[id1+1]
	    + cs * temporal[id1+s] + cn * temporal[id1+n] + cb * temporal[id1+b] + ct * temporal[id1+t];
	  id1++;
	}
#if (XX == 2) || (XX == 3)
	x = nx-1; 
	c = x + y*nx + z*nx*ny;
	f2_t[c] = cc * temporal[id1] + cw * temporal[id1-1] + ce * temporal[id1]
	    + cs * temporal[id1+s] + cn * temporal[id1+n] + cb * temporal[id1+b] + ct * temporal[id1+t];
#endif
	iy++;
      }
    }
    tmp = izm;
    izm = izc;
    izc = izp;
    izp = tmp;
    
  }
  
  for (zz = nz-TB, h = TB-1; zz < nz; zz++, h--) {
    
    step = TB-1-h;
    
    for(z = zz+h ; z >= zz+1; z--){
      b = (z == 0)    ? 0 : (izm-izc)*tbx*tby;
      t = (z == nz-1) ? 0 : (izp-izc)*tbx*tby;
      n = -tbx;
      s =  tbx;

      halo = TB-(step+1);
#if (YY == 0) || (YY == 3)
      ystr = 0;
      iy = TB;
#else
      ystr = yy-halo;
      iy = step+1;
#endif
#if (YY == 2) || (YY == 3)
      yend = ny;
#else
      yend = yy+YBF+halo;
#endif
#if (XX == 0) || (XX == 3)
      xstr = 1;
#else
      xstr = xx-halo;
#endif
#if (XX == 2) || (XX == 3)
      xend = nx-1;
#else
      xend = xx+XBF+halo;
#endif
      for (y = ystr; y < yend; y++) {
#if (YY == 0) || (YY == 3)
	n = (y == 0)    ? 0 : -tbx;
#endif
#if (YY == 2) || (YY == 3)
	s = (y == ny-1) ? 0 :  tbx;
#endif
#if (XX == 0) || (XX == 3)
	ix = TB;
	id2  = ix+iy*tbx+izp*tbx*tby+(step+1)*tbx*tby*tbz;
	id1 = ix+iy*tbx+izc*tbx*tby+step*tbx*tby*tbz;
	{
	  temporal[id2] = cc * temporal[id1] + cw * temporal[id1] + ce * temporal[id1+1]
	    + cs * temporal[id1+s] + cn * temporal[id1+n] + cb * temporal[id1+b] + ct * temporal[id1+t];
	}
	id2++;
	id1++;
#else
	ix = step+1;
	id2  = ix+iy*tbx+izp*tbx*tby+(step+1)*tbx*tby*tbz;
	id1 = ix+iy*tbx+izc*tbx*tby+step*tbx*tby*tbz;
#endif
	for (x = xstr; x < xend; x++) {
	  temporal[id2] = cc * temporal[id1] + cw * temporal[id1-1] + ce * temporal[id1+1]
	    + cs * temporal[id1+s] + cn * temporal[id1+n] + cb * temporal[id1+b] + ct * temporal[id1+t];
	  id2++;
	  id1++;
	}
#if (XX == 2) || (XX == 3)
	{
	  temporal[id2] = cc * temporal[id1] + cw * temporal[id1-1] + ce * temporal[id1]
	    + cs * temporal[id1+s] + cn * temporal[id1+n] + cb * temporal[id1+b] + ct * temporal[id1+t];
	}
#endif
	iy++;
      }
      step++;

    }

    {
      z = zz;
      b = (z == 0)    ? 0 : (izm-izc)*tbx*tby;
      t = (z == nz-1) ? 0 : (izp-izc)*tbx*tby;
      n = -tbx;
      s =  tbx;
      halo = TB-(step+1);
      
#if (YY == 0) || (YY == 3)
      ystr = 0;
      iy = TB;
#else
      ystr = yy-halo;
      iy = step+1;
#endif
#if (YY == 2) || (YY == 3)
      yend = ny;
#else
      yend = yy+YBF+halo;
#endif
#if (XX == 0) || (XX == 3)
      xstr = 1;
#else
      xstr = xx-halo;
#endif
#if (XX == 2) || (XX == 3)
      xend = nx-1;
#else
      xend = xx+XBF+halo;
#endif
      for (y = ystr; y < yend; y++) {
#if (YY == 0) || (YY == 3)
	n = (y == 0)    ? 0 : -tbx;
#endif
#if (YY == 2) || (YY == 3)
	s = (y == ny-1) ? 0 :  tbx;
#endif
#if (XX == 0) || (XX == 3)
	ix = TB;
	id1 = ix+iy*tbx+izc*tbx*tby+step*tbx*tby*tbz;
	{
	  x = 0;
	  c = x + y*nx + z*nx*ny;
	  f2_t[c] = cc * temporal[id1] + cw * temporal[id1] + ce * temporal[id1+1]
	    + cs * temporal[id1+s] + cn * temporal[id1+n] + cb * temporal[id1+b] + ct * temporal[id1+t];
	}
	id1++;
#else
	ix = step+1;
	id1 = ix+iy*tbx+izc*tbx*tby+step*tbx*tby*tbz;
#endif
	for (x = xstr; x < xend; x++) {
	  c = x + y*nx + z*nx*ny;
	  f2_t[c] = cc * temporal[id1] + cw * temporal[id1-1] + ce * temporal[id1+1]
	    + cs * temporal[id1+s] + cn * temporal[id1+n] + cb * temporal[id1+b] + ct * temporal[id1+t];
	  id1++;
	}
#if (XX == 2) || (XX == 3)
	x = nx-1; 
	c = x + y*nx + z*nx*ny;
	f2_t[c] = cc * temporal[id1] + cw * temporal[id1-1] + ce * temporal[id1]
	    + cs * temporal[id1+s] + cn * temporal[id1+n] + cb * temporal[id1+b] + ct * temporal[id1+t];
#endif
	iy++;
      }
    }
    
    tmp = izm;
    izm = izc;
    izc = izp;
    izp = tmp;
  }
}
