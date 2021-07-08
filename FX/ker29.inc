
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
    xstr = 0;
#else
    xstr = xx-halo;
#endif
#if (XX == 2) || (XX == 3)
    xend = nx;
#else
    xend = xx+XBF+halo;
#endif
    int length = xend-xstr;
    int remainder = length & (SIMDLENGTH-1);
    length = length - remainder;
#if (XX == 2) || (XX == 3)
    if(remainder == 0){
      length = length - SIMDLENGTH;
      remainder = SIMDLENGTH;
    }
#endif
    xend = xstr + length;

#if (XX == 0) || (XX == 3)
    ix = TB;
#else
    ix = step;
#endif
    id2 = ix+iy*tbx+izp*tbx*tby+step*tbx*tby*tbz;
    c   = xstr + ystr*nx + z*nx*ny;

    for (y = ystr; y < yend; y++) {
      int id2p = id2 + tbx;
      int cp   = c   + nx;
#if (YY == 0) || (YY == 3)
      n0 = (y == 0)    ? 0 : - nx;
#endif
#if (YY == 2) || (YY == 3)
      s0 = (y == ny-1) ? 0 :   nx;
#endif
      x = xstr;
#if (XX == 0) || (XX == 3)
      {
	w = c - 1;
	e = c + 1;
	n = c + n0;
	s = c + s0;
	t = c + t0;
	b = c + b0;
	float64_t fcm1_arr[SIMDLENGTH] = {f1_t[c],f1_t[c],f1_t[c+1],f1_t[c+2],f1_t[c+3],f1_t[c+4],f1_t[c+5],f1_t[c+6]};
	svfloat64_t fc_vec  = svld1(pg,(float64_t*)&f1_t[c]);
	svfloat64_t fce_vec = svld1(pg,(float64_t*)&f1_t[e]);
	svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
	svfloat64_t fcs_vec = svld1(pg,(float64_t*)&f1_t[s]);
	svfloat64_t fcn_vec = svld1(pg,(float64_t*)&f1_t[n]);
	svfloat64_t fcb_vec = svld1(pg,(float64_t*)&f1_t[b]);
	svfloat64_t fct_vec = svld1(pg,(float64_t*)&f1_t[t]);
	svfloat64_t tmp0,tmp1,tmp2;
	fc_vec  = svmul_x(pg,cc_vec,fc_vec);
	fce_vec = svmul_x(pg,ce_vec,fce_vec);
	fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
	fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
	fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
	fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
	fct_vec = svmul_x(pg,ct_vec,fct_vec);
	tmp0 = svadd_x(pg,fce_vec,fcw_vec);
	tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
	tmp2 = svadd_x(pg,fct_vec,fcb_vec);
	tmp0 = svadd_x(pg,fc_vec, tmp0);
	tmp1 = svadd_x(pg,tmp1,   tmp2);
	tmp0 = svadd_x(pg,tmp0,   tmp1);
	svst1(pg,(float64_t*)&temporal[id2],tmp0);
      }
      id2+=SIMDLENGTH;
      x  +=SIMDLENGTH;
      c  +=SIMDLENGTH;
#endif
      for (; x < xend; x+=SIMDLENGTH) {
      	w = c - 1;
      	e = c + 1;
      	n = c + n0;
      	s = c + s0;
      	t = c + t0;
      	b = c + b0;
	//svprfd(pg, &f1_t[c+nx*ny], SV_PLDL1STRM);
svprfd(pg, &f1_t[c+nx*ny*4], SV_PLDL2STRM);
      	svfloat64_t fc_vec  = svld1(pg,(float64_t*)&f1_t[c]);
      	svfloat64_t fce_vec = svld1(pg,(float64_t*)&f1_t[e]);
      	svfloat64_t fcw_vec = svld1(pg,(float64_t*)&f1_t[w]);
      	svfloat64_t fcs_vec = svld1(pg,(float64_t*)&f1_t[s]);
      	svfloat64_t fcn_vec = svld1(pg,(float64_t*)&f1_t[n]);
      	svfloat64_t fcb_vec = svld1(pg,(float64_t*)&f1_t[b]);
      	svfloat64_t fct_vec = svld1(pg,(float64_t*)&f1_t[t]);
      	svfloat64_t tmp0,tmp1,tmp2;
      	fc_vec  = svmul_x(pg,cc_vec,fc_vec);
      	fce_vec = svmul_x(pg,ce_vec,fce_vec);
      	fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
      	fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
      	fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
      	fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
      	fct_vec = svmul_x(pg,ct_vec,fct_vec);
      	tmp0 = svadd_x(pg,fce_vec,fcw_vec);
      	tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
      	tmp2 = svadd_x(pg,fct_vec,fcb_vec);
      	tmp0 = svadd_x(pg,fc_vec, tmp0);
      	tmp1 = svadd_x(pg,tmp1,   tmp2);
      	tmp0 = svadd_x(pg,tmp0,   tmp1);
      	svst1(pg,(float64_t*)&temporal[id2],tmp0);
      	id2+=SIMDLENGTH;
	c  +=SIMDLENGTH;
      }
      int i = 0;
      svbool_t pgt = svwhilelt_b64(i,remainder);
      {
      	w = c - 1;
      	e = c + 1;
      	n = c + n0;
      	s = c + s0;
      	t = c + t0;
      	b = c + b0;
#if (XX == 2) || (XX == 3)
	
	float64_t fcp1_arr[SIMDLENGTH];
	for(int i = 0; i < remainder; i++){
	  if(i == remainder-1){
	    fcp1_arr[i] = f1_t[c+i];
	  }else{
	    fcp1_arr[i] = f1_t[e+i];
	  }
	}
	svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
#else
      	svfloat64_t fce_vec = svld1(pgt,(float64_t*)&f1_t[e]);
#endif
      	svfloat64_t fc_vec  = svld1(pgt,(float64_t*)&f1_t[c]);
      	svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&f1_t[w]);
      	svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&f1_t[s]);
      	svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&f1_t[n]);
      	svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&f1_t[b]);
      	svfloat64_t fct_vec = svld1(pgt,(float64_t*)&f1_t[t]);
	svfloat64_t tmp0,tmp1,tmp2;
	fc_vec  = svmul_x(pgt,cc_vec,fc_vec);
	fce_vec = svmul_x(pgt,ce_vec,fce_vec);
	fcw_vec = svmul_x(pgt,cw_vec,fcw_vec);
	fcn_vec = svmul_x(pgt,cn_vec,fcn_vec);
	fcs_vec = svmul_x(pgt,cs_vec,fcs_vec);
	fcb_vec = svmul_x(pgt,cb_vec,fcb_vec);
	fct_vec = svmul_x(pgt,ct_vec,fct_vec);
	tmp0 = svadd_x(pgt,fce_vec,fcw_vec);
	tmp1 = svadd_x(pgt,fcn_vec,fcs_vec);
	tmp2 = svadd_x(pgt,fct_vec,fcb_vec);
	tmp0 = svadd_x(pgt,fc_vec, tmp0);
	tmp1 = svadd_x(pgt,tmp1,   tmp2);
	tmp0 = svadd_x(pgt,tmp0,   tmp1);
	svst1(pgt,(float64_t*)&temporal[id2],tmp0);
      }
      id2 = id2p;
      c   = cp;
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
      xstr = 0;
#else
      xstr = xx-halo;
#endif
#if (XX == 2) || (XX == 3)
      xend = nx;
#else
      xend = xx+XBF+halo;
#endif
      int length = xend-xstr;
      int remainder = length & (SIMDLENGTH-1);
      length = length - remainder;
#if (XX == 0) || (XX == 3)
      length = length - SIMDLENGTH;
#endif
#if (XX == 2) || (XX == 3)
      if(remainder == 0){
	length = length - SIMDLENGTH;
	remainder = SIMDLENGTH;
      }
#endif
#if (XX == 0) || (XX == 3)
      ix = TB;
#else
      ix = step+1;
#endif
      id2 = ix+iy*tbx+izp*tbx*tby+(step+1)*tbx*tby*tbz;
      id1 = ix+iy*tbx+izc*tbx*tby+step*tbx*tby*tbz;

      for (y = ystr; y < yend; y++) {
	int id2p = id2+tbx;
	int id1p = id1+tbx;
#if (YY == 0) || (YY == 3)
	n = (y == 0)    ? 0 : -tbx;
#endif
#if (YY == 2) || (YY == 3)
	s = (y == ny-1) ? 0 :  tbx;
#endif
#if (XX == 0) || (XX == 3)
	{
	  float64_t fcm1_arr[SIMDLENGTH] = {temporal[id1],temporal[id1],temporal[id1+1],temporal[id1+2],temporal[id1+3],temporal[id1+4],temporal[id1+5],temporal[id1+6]};
	  svfloat64_t fc_vec  = svld1(pg,(float64_t*)&temporal[id1]);
	  svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+1]);
	  svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
	  svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+s]);
	  svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+n]);
	  svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+b]);
	  svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+t]);
	  svfloat64_t tmp0,tmp1,tmp2;
	  fc_vec  = svmul_x(pg,cc_vec,fc_vec);
	  fce_vec = svmul_x(pg,ce_vec,fce_vec);
	  fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
	  fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
	  fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
	  fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
	  fct_vec = svmul_x(pg,ct_vec,fct_vec);
	  tmp0 = svadd_x(pg,fce_vec,fcw_vec);
	  tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
	  tmp2 = svadd_x(pg,fct_vec,fcb_vec);
	  tmp0 = svadd_x(pg,fc_vec, tmp0);
	  tmp1 = svadd_x(pg,tmp1,   tmp2);
	  tmp0 = svadd_x(pg,tmp0,   tmp1);
	  svst1(pg,(float64_t*)&temporal[id2],tmp0);
	  id2+=SIMDLENGTH;
	  id1+=SIMDLENGTH;
	}
#endif

	for (x = 0; x < length; x+=SIMDLENGTH) {
	  svfloat64_t fc_vec  = svld1(pg,(float64_t*)&temporal[id1]);
	  svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+1]);
	  svfloat64_t fcw_vec = svld1(pg,(float64_t*)&temporal[id1-1]);
	  svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+s]);
	  svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+n]);
	  svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+b]);
	  svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+t]);
	  svfloat64_t tmp0,tmp1,tmp2;
	  fc_vec  = svmul_x(pg,cc_vec,fc_vec);
	  fce_vec = svmul_x(pg,ce_vec,fce_vec);
	  fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
	  fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
	  fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
	  fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
	  fct_vec = svmul_x(pg,ct_vec,fct_vec);
	  tmp0 = svadd_x(pg,fce_vec,fcw_vec);
	  tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
	  tmp2 = svadd_x(pg,fct_vec,fcb_vec);
	  tmp0 = svadd_x(pg,fc_vec, tmp0);
	  tmp1 = svadd_x(pg,tmp1,   tmp2);
	  tmp0 = svadd_x(pg,tmp0,   tmp1);
	  svst1(pg,(float64_t*)&temporal[id2],tmp0);
	  id2+=SIMDLENGTH;
	  id1+=SIMDLENGTH;
	}
	x = 0;
	svbool_t pgt = svwhilelt_b64(x,remainder);
	{
#if (XX == 2) || (XX == 3)

	  float64_t fcp1_arr[SIMDLENGTH];
	  for(int i = 0; i < remainder; i++){
	    if(i == remainder-1){
	      fcp1_arr[i] = temporal[id1+i];
	    }else{
	      fcp1_arr[i] = temporal[id1+1+i];
	    }
	  }
	  svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
#else
	  svfloat64_t fce_vec = svld1(pgt,(float64_t*)&temporal[id1+1]);
#endif
	  svfloat64_t fc_vec  = svld1(pgt,(float64_t*)&temporal[id1]);
	  svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&temporal[id1-1]);
	  svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&temporal[id1+s]);
	  svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&temporal[id1+n]);
	  svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&temporal[id1+b]);
	  svfloat64_t fct_vec = svld1(pgt,(float64_t*)&temporal[id1+t]);
	  svfloat64_t tmp0,tmp1,tmp2;
	  fc_vec  = svmul_x(pgt,cc_vec,fc_vec);
	  fce_vec = svmul_x(pgt,ce_vec,fce_vec);
	  fcw_vec = svmul_x(pgt,cw_vec,fcw_vec);
	  fcn_vec = svmul_x(pgt,cn_vec,fcn_vec);
	  fcs_vec = svmul_x(pgt,cs_vec,fcs_vec);
	  fcb_vec = svmul_x(pgt,cb_vec,fcb_vec);
	  fct_vec = svmul_x(pgt,ct_vec,fct_vec);
	  tmp0 = svadd_x(pgt,fce_vec,fcw_vec);
	  tmp1 = svadd_x(pgt,fcn_vec,fcs_vec);
	  tmp2 = svadd_x(pgt,fct_vec,fcb_vec);
	  tmp0 = svadd_x(pgt,fc_vec, tmp0);
	  tmp1 = svadd_x(pgt,tmp1,   tmp2);
	  tmp0 = svadd_x(pgt,tmp0,   tmp1);
	  svst1(pgt,(float64_t*)&temporal[id2],tmp0);
	}
	id2 = id2p;
	id1 = id1p;
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
    xstr = 0;
#else
    xstr = xx-halo;
#endif
#if (XX == 2) || (XX == 3)
    xend = nx;
#else
    xend = xx+XBF+halo;
#endif
    int length = xend-xstr;
    int remainder = length & (SIMDLENGTH-1);
    length = length - remainder;
#if (XX == 2) || (XX == 3)
    if(remainder == 0){
      length = length - SIMDLENGTH;
      remainder = SIMDLENGTH;
    }
#endif
    xend = xstr + length;

#if (XX == 0) || (XX == 3)
    ix = TB;
#else
    ix = step;
#endif
    id2 = ix+iy*tbx+izp*tbx*tby+step*tbx*tby*tbz;
    c   = xstr + ystr*nx + z*nx*ny;

    for (y = ystr; y < yend; y++) {
      int id2p = id2 + tbx;
      int cp   = c + nx;
#if (YY == 0) || (YY == 3)
      n0 = (y == 0)    ? 0 : - nx;
#endif
#if (YY == 2) || (YY == 3)
      s0 = (y == ny-1) ? 0 :   nx;
#endif
      x = xstr;
#if (XX == 0) || (XX == 3)
      {
	w = c - 1;
	e = c + 1;
	n = c + n0;
	s = c + s0;
	t = c + t0;
	b = c + b0;
	float64_t fcm1_arr[SIMDLENGTH] = {f1_t[c],f1_t[c],f1_t[c+1],f1_t[c+2],f1_t[c+3],f1_t[c+4],f1_t[c+5],f1_t[c+6]};
	svfloat64_t fc_vec  = svld1(pg,(float64_t*)&f1_t[c]);
	svfloat64_t fce_vec = svld1(pg,(float64_t*)&f1_t[e]);
	svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
	svfloat64_t fcs_vec = svld1(pg,(float64_t*)&f1_t[s]);
	svfloat64_t fcn_vec = svld1(pg,(float64_t*)&f1_t[n]);
	svfloat64_t fcb_vec = svld1(pg,(float64_t*)&f1_t[b]);
	svfloat64_t fct_vec = svld1(pg,(float64_t*)&f1_t[t]);
	svfloat64_t tmp0,tmp1,tmp2;
	fc_vec  = svmul_x(pg,cc_vec,fc_vec);
	fce_vec = svmul_x(pg,ce_vec,fce_vec);
	fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
	fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
	fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
	fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
	fct_vec = svmul_x(pg,ct_vec,fct_vec);
	tmp0 = svadd_x(pg,fce_vec,fcw_vec);
	tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
	tmp2 = svadd_x(pg,fct_vec,fcb_vec);
	tmp0 = svadd_x(pg,fc_vec, tmp0);
	tmp1 = svadd_x(pg,tmp1,   tmp2);
	tmp0 = svadd_x(pg,tmp0,   tmp1);
	svst1(pg,(float64_t*)&temporal[id2],tmp0);
      }
      id2+=SIMDLENGTH;
      x  +=SIMDLENGTH;
      c  +=SIMDLENGTH;
#endif
      for (; x < xend; x+=SIMDLENGTH) {
      	w = c - 1;
      	e = c + 1;
      	n = c + n0;
      	s = c + s0;
      	t = c + t0;
      	b = c + b0;
	//svprfd(pg, &f1_t[c+nx*ny], SV_PLDL1STRM);
svprfd(pg, &f1_t[c+nx*ny*4], SV_PLDL2STRM);
      	svfloat64_t fc_vec  = svld1(pg,(float64_t*)&f1_t[c]);
      	svfloat64_t fce_vec = svld1(pg,(float64_t*)&f1_t[e]);
      	svfloat64_t fcw_vec = svld1(pg,(float64_t*)&f1_t[w]);
      	svfloat64_t fcs_vec = svld1(pg,(float64_t*)&f1_t[s]);
      	svfloat64_t fcn_vec = svld1(pg,(float64_t*)&f1_t[n]);
      	svfloat64_t fcb_vec = svld1(pg,(float64_t*)&f1_t[b]);
      	svfloat64_t fct_vec = svld1(pg,(float64_t*)&f1_t[t]);
      	svfloat64_t tmp0,tmp1,tmp2;
      	fc_vec  = svmul_x(pg,cc_vec,fc_vec);
      	fce_vec = svmul_x(pg,ce_vec,fce_vec);
      	fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
      	fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
      	fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
      	fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
      	fct_vec = svmul_x(pg,ct_vec,fct_vec);
      	tmp0 = svadd_x(pg,fce_vec,fcw_vec);
      	tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
      	tmp2 = svadd_x(pg,fct_vec,fcb_vec);
      	tmp0 = svadd_x(pg,fc_vec, tmp0);
      	tmp1 = svadd_x(pg,tmp1,   tmp2);
      	tmp0 = svadd_x(pg,tmp0,   tmp1);
      	svst1(pg,(float64_t*)&temporal[id2],tmp0);
      	id2+=SIMDLENGTH;
	c  +=SIMDLENGTH;
      }
      int i = 0;
      svbool_t pgt = svwhilelt_b64(i,remainder);
      {
      	w = c - 1;
      	e = c + 1;
      	n = c + n0;
      	s = c + s0;
      	t = c + t0;
      	b = c + b0;
#if (XX == 2) || (XX == 3)
	
	float64_t fcp1_arr[SIMDLENGTH];
	for(int i = 0; i < remainder; i++){
	  if(i == remainder-1){
	    fcp1_arr[i] = f1_t[c+i];
	  }else{
	    fcp1_arr[i] = f1_t[e+i];
	  }
	}
	svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
#else
      	svfloat64_t fce_vec = svld1(pgt,(float64_t*)&f1_t[e]);
#endif
      	svfloat64_t fc_vec  = svld1(pgt,(float64_t*)&f1_t[c]);
      	svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&f1_t[w]);
      	svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&f1_t[s]);
      	svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&f1_t[n]);
      	svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&f1_t[b]);
      	svfloat64_t fct_vec = svld1(pgt,(float64_t*)&f1_t[t]);
	svfloat64_t tmp0,tmp1,tmp2;
	fc_vec  = svmul_x(pgt,cc_vec,fc_vec);
	fce_vec = svmul_x(pgt,ce_vec,fce_vec);
	fcw_vec = svmul_x(pgt,cw_vec,fcw_vec);
	fcn_vec = svmul_x(pgt,cn_vec,fcn_vec);
	fcs_vec = svmul_x(pgt,cs_vec,fcs_vec);
	fcb_vec = svmul_x(pgt,cb_vec,fcb_vec);
	fct_vec = svmul_x(pgt,ct_vec,fct_vec);
	tmp0 = svadd_x(pgt,fce_vec,fcw_vec);
	tmp1 = svadd_x(pgt,fcn_vec,fcs_vec);
	tmp2 = svadd_x(pgt,fct_vec,fcb_vec);
	tmp0 = svadd_x(pgt,fc_vec, tmp0);
	tmp1 = svadd_x(pgt,tmp1,   tmp2);
	tmp0 = svadd_x(pgt,tmp0,   tmp1);
	svst1(pgt,(float64_t*)&temporal[id2],tmp0);
      }
      id2 = id2p;
      c = cp;
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
      xstr = 0;
#else
      xstr = xx-halo;
#endif
#if (XX == 2) || (XX == 3)
      xend = nx;
#else
      xend = xx+XBF+halo;
#endif
      int length = xend-xstr;
      int remainder = length & (SIMDLENGTH-1);
      length = length - remainder;
#if (XX == 0) || (XX == 3)
      length = length - SIMDLENGTH;
#endif
#if (XX == 2) || (XX == 3)
      if(remainder == 0){
	length = length - SIMDLENGTH;
	remainder = SIMDLENGTH;
      }
#endif
#if (XX == 0) || (XX == 3)
      ix = TB;
#else
      ix = step+1;
#endif
      id2 = ix+iy*tbx+izp*tbx*tby+(step+1)*tbx*tby*tbz;
      id1 = ix+iy*tbx+izc*tbx*tby+step*tbx*tby*tbz;

      for (y = ystr; y < yend; y++) {
	int id2p = id2+tbx;
	int id1p = id1+tbx;
#if (YY == 0) || (YY == 3)
	n = (y == 0)    ? 0 : -tbx;
#endif
#if (YY == 2) || (YY == 3)
	s = (y == ny-1) ? 0 :  tbx;
#endif
#if (XX == 0) || (XX == 3)
	{
	  float64_t fcm1_arr[SIMDLENGTH] = {temporal[id1],temporal[id1],temporal[id1+1],temporal[id1+2],temporal[id1+3],temporal[id1+4],temporal[id1+5],temporal[id1+6]};
	  svfloat64_t fc_vec  = svld1(pg,(float64_t*)&temporal[id1]);
	  svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+1]);
	  svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
	  svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+s]);
	  svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+n]);
	  svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+b]);
	  svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+t]);
	  svfloat64_t tmp0,tmp1,tmp2;
	  fc_vec  = svmul_x(pg,cc_vec,fc_vec);
	  fce_vec = svmul_x(pg,ce_vec,fce_vec);
	  fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
	  fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
	  fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
	  fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
	  fct_vec = svmul_x(pg,ct_vec,fct_vec);
	  tmp0 = svadd_x(pg,fce_vec,fcw_vec);
	  tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
	  tmp2 = svadd_x(pg,fct_vec,fcb_vec);
	  tmp0 = svadd_x(pg,fc_vec, tmp0);
	  tmp1 = svadd_x(pg,tmp1,   tmp2);
	  tmp0 = svadd_x(pg,tmp0,   tmp1);
	  svst1(pg,(float64_t*)&temporal[id2],tmp0);
	  id2+=SIMDLENGTH;
	  id1+=SIMDLENGTH;
	}
#endif

	for (x = 0; x < length; x+=SIMDLENGTH) {
	  svfloat64_t fc_vec  = svld1(pg,(float64_t*)&temporal[id1]);
	  svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+1]);
	  svfloat64_t fcw_vec = svld1(pg,(float64_t*)&temporal[id1-1]);
	  svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+s]);
	  svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+n]);
	  svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+b]);
	  svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+t]);
	  svfloat64_t tmp0,tmp1,tmp2;
	  fc_vec  = svmul_x(pg,cc_vec,fc_vec);
	  fce_vec = svmul_x(pg,ce_vec,fce_vec);
	  fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
	  fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
	  fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
	  fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
	  fct_vec = svmul_x(pg,ct_vec,fct_vec);
	  tmp0 = svadd_x(pg,fce_vec,fcw_vec);
	  tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
	  tmp2 = svadd_x(pg,fct_vec,fcb_vec);
	  tmp0 = svadd_x(pg,fc_vec, tmp0);
	  tmp1 = svadd_x(pg,tmp1,   tmp2);
	  tmp0 = svadd_x(pg,tmp0,   tmp1);
	  svst1(pg,(float64_t*)&temporal[id2],tmp0);
	  id2+=SIMDLENGTH;
	  id1+=SIMDLENGTH;
	}
	x = 0;
	svbool_t pgt = svwhilelt_b64(x,remainder);
	{
#if (XX == 2) || (XX == 3)

	  float64_t fcp1_arr[SIMDLENGTH];
	  for(int i = 0; i < remainder; i++){
	    if(i == remainder-1){
	      fcp1_arr[i] = temporal[id1+i];
	    }else{
	      fcp1_arr[i] = temporal[id1+1+i];
	    }
	  }
	  svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
#else
	  svfloat64_t fce_vec = svld1(pgt,(float64_t*)&temporal[id1+1]);
#endif
	  svfloat64_t fc_vec  = svld1(pgt,(float64_t*)&temporal[id1]);
	  svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&temporal[id1-1]);
	  svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&temporal[id1+s]);
	  svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&temporal[id1+n]);
	  svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&temporal[id1+b]);
	  svfloat64_t fct_vec = svld1(pgt,(float64_t*)&temporal[id1+t]);
	  svfloat64_t tmp0,tmp1,tmp2;
	  fc_vec  = svmul_x(pgt,cc_vec,fc_vec);
	  fce_vec = svmul_x(pgt,ce_vec,fce_vec);
	  fcw_vec = svmul_x(pgt,cw_vec,fcw_vec);
	  fcn_vec = svmul_x(pgt,cn_vec,fcn_vec);
	  fcs_vec = svmul_x(pgt,cs_vec,fcs_vec);
	  fcb_vec = svmul_x(pgt,cb_vec,fcb_vec);
	  fct_vec = svmul_x(pgt,ct_vec,fct_vec);
	  tmp0 = svadd_x(pgt,fce_vec,fcw_vec);
	  tmp1 = svadd_x(pgt,fcn_vec,fcs_vec);
	  tmp2 = svadd_x(pgt,fct_vec,fcb_vec);
	  tmp0 = svadd_x(pgt,fc_vec, tmp0);
	  tmp1 = svadd_x(pgt,tmp1,   tmp2);
	  tmp0 = svadd_x(pgt,tmp0,   tmp1);
	  svst1(pgt,(float64_t*)&temporal[id2],tmp0);
	}
	id2 = id2p;
	id1 = id1p;
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
      xstr = 0;
#else
      xstr = xx-halo;
#endif
#if (XX == 2) || (XX == 3)
      xend = nx;
#else
      xend = xx+XBF+halo;
#endif
      int length = xend-xstr;
      int remainder = length & (SIMDLENGTH-1);
      length = length - remainder;
#if (XX == 2) || (XX == 3)
      if(remainder == 0){
	length = length - SIMDLENGTH;
	remainder = SIMDLENGTH;
      }
#endif
      xend = xstr + length;
#if (XX == 0) || (XX == 3)
      ix = TB;
#else
      ix = step+1;
#endif
      id1 = ix+iy*tbx+izc*tbx*tby+step*tbx*tby*tbz;
      for (y = ystr; y < yend; y++) {
	int id1p = id1 + tbx;
#if (YY == 0) || (YY == 3)
	n = (y == 0)    ? 0 : -tbx;
#endif
#if (YY == 2) || (YY == 3)
	s = (y == ny-1) ? 0 :  tbx;
#endif
	x = xstr;
	c = x + y*nx + z*nx*ny;
#if (XX == 0) || (XX == 3)
	{
	  float64_t fcm1_arr[SIMDLENGTH] = {temporal[id1],temporal[id1],temporal[id1+1],temporal[id1+2],temporal[id1+3],temporal[id1+4],temporal[id1+5],temporal[id1+6]};
	  svfloat64_t fc_vec  = svld1(pg,(float64_t*)&temporal[id1]);
	  svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+1]);
	  svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
	  svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+s]);
	  svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+n]);
	  svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+b]);
	  svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+t]);
	  svfloat64_t tmp0,tmp1,tmp2;
	  fc_vec  = svmul_x(pg,cc_vec,fc_vec);
	  fce_vec = svmul_x(pg,ce_vec,fce_vec);
	  fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
	  fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
	  fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
	  fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
	  fct_vec = svmul_x(pg,ct_vec,fct_vec);
	  tmp0 = svadd_x(pg,fce_vec,fcw_vec);
	  tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
	  tmp2 = svadd_x(pg,fct_vec,fcb_vec);
	  tmp0 = svadd_x(pg,fc_vec, tmp0);
	  tmp1 = svadd_x(pg,tmp1,   tmp2);
	  tmp0 = svadd_x(pg,tmp0,   tmp1);
	  svst1(pg,(float64_t*)&f2_t[c],tmp0);

	}
	id1+=SIMDLENGTH;
	x  +=SIMDLENGTH;
	c  +=SIMDLENGTH;
#endif

	for (; x < xend; x+=SIMDLENGTH) {
	  svfloat64_t fc_vec  = svld1(pg,(float64_t*)&temporal[id1]);
	  svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+1]);
	  svfloat64_t fcw_vec = svld1(pg,(float64_t*)&temporal[id1-1]);
	  svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+s]);
	  svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+n]);
	  svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+b]);
	  svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+t]);
	  svfloat64_t tmp0,tmp1,tmp2;
	  fc_vec  = svmul_x(pg,cc_vec,fc_vec);
	  fce_vec = svmul_x(pg,ce_vec,fce_vec);
	  fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
	  fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
	  fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
	  fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
	  fct_vec = svmul_x(pg,ct_vec,fct_vec);
	  tmp0 = svadd_x(pg,fce_vec,fcw_vec);
	  tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
	  tmp2 = svadd_x(pg,fct_vec,fcb_vec);
	  tmp0 = svadd_x(pg,fc_vec, tmp0);
	  tmp1 = svadd_x(pg,tmp1,   tmp2);
	  tmp0 = svadd_x(pg,tmp0,   tmp1);
	  svst1(pg,(float64_t*)&f2_t[c],tmp0);
	  id1+=SIMDLENGTH;
	  c  +=SIMDLENGTH;
	}
	int i = 0;
	svbool_t pgt = svwhilelt_b64(i,remainder);
	{
#if (XX == 2) || (XX == 3)

	  float64_t fcp1_arr[SIMDLENGTH];
	  for(int i = 0; i < remainder; i++){
	    if(i == remainder-1){
	      fcp1_arr[i] = temporal[id1+i];
	    }else{
	      fcp1_arr[i] = temporal[id1+1+i];
	    }
	  }
	  svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
#else
	  svfloat64_t fce_vec = svld1(pgt,(float64_t*)&temporal[id1+1]);
#endif
	  svfloat64_t fc_vec  = svld1(pgt,(float64_t*)&temporal[id1]);
	  svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&temporal[id1-1]);
	  svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&temporal[id1+s]);
	  svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&temporal[id1+n]);
	  svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&temporal[id1+b]);
	  svfloat64_t fct_vec = svld1(pgt,(float64_t*)&temporal[id1+t]);
	  svfloat64_t tmp0,tmp1,tmp2;
	  fc_vec  = svmul_x(pgt,cc_vec,fc_vec);
	  fce_vec = svmul_x(pgt,ce_vec,fce_vec);
	  fcw_vec = svmul_x(pgt,cw_vec,fcw_vec);
	  fcn_vec = svmul_x(pgt,cn_vec,fcn_vec);
	  fcs_vec = svmul_x(pgt,cs_vec,fcs_vec);
	  fcb_vec = svmul_x(pgt,cb_vec,fcb_vec);
	  fct_vec = svmul_x(pgt,ct_vec,fct_vec);
	  tmp0 = svadd_x(pgt,fce_vec,fcw_vec);
	  tmp1 = svadd_x(pgt,fcn_vec,fcs_vec);
	  tmp2 = svadd_x(pgt,fct_vec,fcb_vec);
	  tmp0 = svadd_x(pgt,fc_vec, tmp0);
	  tmp1 = svadd_x(pgt,tmp1,   tmp2);
	  tmp0 = svadd_x(pgt,tmp0,   tmp1);
	  svst1(pgt,(float64_t*)&f2_t[c],tmp0);

	}
	
	id1 = id1p;
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
      xstr = 0;
#else
      xstr = xx-halo;
#endif
#if (XX == 2) || (XX == 3)
      xend = nx;
#else
      xend = xx+XBF+halo;
#endif
      int length = xend-xstr;
      int remainder = length & (SIMDLENGTH-1);
      length = length - remainder;
#if (XX == 0) || (XX == 3)
      length = length - SIMDLENGTH;
#endif
#if (XX == 2) || (XX == 3)
      if(remainder == 0){
	length = length - SIMDLENGTH;
	remainder = SIMDLENGTH;
      }
#endif
#if (XX == 0) || (XX == 3)
      ix = TB;
#else
      ix = step+1;
#endif
      id2 = ix+iy*tbx+izp*tbx*tby+(step+1)*tbx*tby*tbz;
      id1 = ix+iy*tbx+izc*tbx*tby+step*tbx*tby*tbz;

      for (y = ystr; y < yend; y++) {
	int id2p = id2+tbx;
	int id1p = id1+tbx;
#if (YY == 0) || (YY == 3)
	n = (y == 0)    ? 0 : -tbx;
#endif
#if (YY == 2) || (YY == 3)
	s = (y == ny-1) ? 0 :  tbx;
#endif
#if (XX == 0) || (XX == 3)
	{
	  float64_t fcm1_arr[SIMDLENGTH] = {temporal[id1],temporal[id1],temporal[id1+1],temporal[id1+2],temporal[id1+3],temporal[id1+4],temporal[id1+5],temporal[id1+6]};
	  svfloat64_t fc_vec  = svld1(pg,(float64_t*)&temporal[id1]);
	  svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+1]);
	  svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
	  svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+s]);
	  svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+n]);
	  svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+b]);
	  svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+t]);
	  svfloat64_t tmp0,tmp1,tmp2;
	  fc_vec  = svmul_x(pg,cc_vec,fc_vec);
	  fce_vec = svmul_x(pg,ce_vec,fce_vec);
	  fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
	  fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
	  fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
	  fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
	  fct_vec = svmul_x(pg,ct_vec,fct_vec);
	  tmp0 = svadd_x(pg,fce_vec,fcw_vec);
	  tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
	  tmp2 = svadd_x(pg,fct_vec,fcb_vec);
	  tmp0 = svadd_x(pg,fc_vec, tmp0);
	  tmp1 = svadd_x(pg,tmp1,   tmp2);
	  tmp0 = svadd_x(pg,tmp0,   tmp1);
	  svst1(pg,(float64_t*)&temporal[id2],tmp0);
	  id2+=SIMDLENGTH;
	  id1+=SIMDLENGTH;
	}
#endif

	for (x = 0; x < length; x+=SIMDLENGTH) {
	  svfloat64_t fc_vec  = svld1(pg,(float64_t*)&temporal[id1]);
	  svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+1]);
	  svfloat64_t fcw_vec = svld1(pg,(float64_t*)&temporal[id1-1]);
	  svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+s]);
	  svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+n]);
	  svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+b]);
	  svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+t]);
	  svfloat64_t tmp0,tmp1,tmp2;
	  fc_vec  = svmul_x(pg,cc_vec,fc_vec);
	  fce_vec = svmul_x(pg,ce_vec,fce_vec);
	  fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
	  fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
	  fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
	  fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
	  fct_vec = svmul_x(pg,ct_vec,fct_vec);
	  tmp0 = svadd_x(pg,fce_vec,fcw_vec);
	  tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
	  tmp2 = svadd_x(pg,fct_vec,fcb_vec);
	  tmp0 = svadd_x(pg,fc_vec, tmp0);
	  tmp1 = svadd_x(pg,tmp1,   tmp2);
	  tmp0 = svadd_x(pg,tmp0,   tmp1);
	  svst1(pg,(float64_t*)&temporal[id2],tmp0);
	  id2+=SIMDLENGTH;
	  id1+=SIMDLENGTH;
	}
	x = 0;
	svbool_t pgt = svwhilelt_b64(x,remainder);
	{
#if (XX == 2) || (XX == 3)

	  float64_t fcp1_arr[SIMDLENGTH];
	  for(int i = 0; i < remainder; i++){
	    if(i == remainder-1){
	      fcp1_arr[i] = temporal[id1+i];
	    }else{
	      fcp1_arr[i] = temporal[id1+1+i];
	    }
	  }
	  svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
#else
	  svfloat64_t fce_vec = svld1(pgt,(float64_t*)&temporal[id1+1]);
#endif
	  svfloat64_t fc_vec  = svld1(pgt,(float64_t*)&temporal[id1]);
	  svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&temporal[id1-1]);
	  svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&temporal[id1+s]);
	  svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&temporal[id1+n]);
	  svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&temporal[id1+b]);
	  svfloat64_t fct_vec = svld1(pgt,(float64_t*)&temporal[id1+t]);
	  svfloat64_t tmp0,tmp1,tmp2;
	  fc_vec  = svmul_x(pgt,cc_vec,fc_vec);
	  fce_vec = svmul_x(pgt,ce_vec,fce_vec);
	  fcw_vec = svmul_x(pgt,cw_vec,fcw_vec);
	  fcn_vec = svmul_x(pgt,cn_vec,fcn_vec);
	  fcs_vec = svmul_x(pgt,cs_vec,fcs_vec);
	  fcb_vec = svmul_x(pgt,cb_vec,fcb_vec);
	  fct_vec = svmul_x(pgt,ct_vec,fct_vec);
	  tmp0 = svadd_x(pgt,fce_vec,fcw_vec);
	  tmp1 = svadd_x(pgt,fcn_vec,fcs_vec);
	  tmp2 = svadd_x(pgt,fct_vec,fcb_vec);
	  tmp0 = svadd_x(pgt,fc_vec, tmp0);
	  tmp1 = svadd_x(pgt,tmp1,   tmp2);
	  tmp0 = svadd_x(pgt,tmp0,   tmp1);
	  svst1(pgt,(float64_t*)&temporal[id2],tmp0);
	}
	id2 = id2p;
	id1 = id1p;
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
      xstr = 0;
#else
      xstr = xx-halo;
#endif
#if (XX == 2) || (XX == 3)
      xend = nx;
#else
      xend = xx+XBF+halo;
#endif
      int length = xend-xstr;
      int remainder = length & (SIMDLENGTH-1);
      length = length - remainder;
#if (XX == 2) || (XX == 3)
      if(remainder == 0){
	length = length - SIMDLENGTH;
	remainder = SIMDLENGTH;
      }
#endif
      xend = xstr + length;
#if (XX == 0) || (XX == 3)
      ix = TB;
#else
      ix = step+1;
#endif
      id1 = ix+iy*tbx+izc*tbx*tby+step*tbx*tby*tbz;
      for (y = ystr; y < yend; y++) {
	int id1p = id1 + tbx;
#if (YY == 0) || (YY == 3)
	n = (y == 0)    ? 0 : -tbx;
#endif
#if (YY == 2) || (YY == 3)
	s = (y == ny-1) ? 0 :  tbx;
#endif
	x = xstr;
	c = x + y*nx + z*nx*ny;
#if (XX == 0) || (XX == 3)
	{
	  float64_t fcm1_arr[SIMDLENGTH] = {temporal[id1],temporal[id1],temporal[id1+1],temporal[id1+2],temporal[id1+3],temporal[id1+4],temporal[id1+5],temporal[id1+6]};
	  svfloat64_t fc_vec  = svld1(pg,(float64_t*)&temporal[id1]);
	  svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+1]);
	  svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
	  svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+s]);
	  svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+n]);
	  svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+b]);
	  svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+t]);
	  svfloat64_t tmp0,tmp1,tmp2;
	  fc_vec  = svmul_x(pg,cc_vec,fc_vec);
	  fce_vec = svmul_x(pg,ce_vec,fce_vec);
	  fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
	  fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
	  fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
	  fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
	  fct_vec = svmul_x(pg,ct_vec,fct_vec);
	  tmp0 = svadd_x(pg,fce_vec,fcw_vec);
	  tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
	  tmp2 = svadd_x(pg,fct_vec,fcb_vec);
	  tmp0 = svadd_x(pg,fc_vec, tmp0);
	  tmp1 = svadd_x(pg,tmp1,   tmp2);
	  tmp0 = svadd_x(pg,tmp0,   tmp1);
	  svst1(pg,(float64_t*)&f2_t[c],tmp0);

	}
	id1+=SIMDLENGTH;
	x  +=SIMDLENGTH;
	c  +=SIMDLENGTH;
#endif

	for (; x < xend; x+=SIMDLENGTH) {
	  svfloat64_t fc_vec  = svld1(pg,(float64_t*)&temporal[id1]);
	  svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+1]);
	  svfloat64_t fcw_vec = svld1(pg,(float64_t*)&temporal[id1-1]);
	  svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+s]);
	  svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+n]);
	  svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+b]);
	  svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+t]);
	  svfloat64_t tmp0,tmp1,tmp2;
	  fc_vec  = svmul_x(pg,cc_vec,fc_vec);
	  fce_vec = svmul_x(pg,ce_vec,fce_vec);
	  fcw_vec = svmul_x(pg,cw_vec,fcw_vec);
	  fcn_vec = svmul_x(pg,cn_vec,fcn_vec);
	  fcs_vec = svmul_x(pg,cs_vec,fcs_vec);
	  fcb_vec = svmul_x(pg,cb_vec,fcb_vec);
	  fct_vec = svmul_x(pg,ct_vec,fct_vec);
	  tmp0 = svadd_x(pg,fce_vec,fcw_vec);
	  tmp1 = svadd_x(pg,fcn_vec,fcs_vec);
	  tmp2 = svadd_x(pg,fct_vec,fcb_vec);
	  tmp0 = svadd_x(pg,fc_vec, tmp0);
	  tmp1 = svadd_x(pg,tmp1,   tmp2);
	  tmp0 = svadd_x(pg,tmp0,   tmp1);
	  svst1(pg,(float64_t*)&f2_t[c],tmp0);
	  id1+=SIMDLENGTH;
	  c  +=SIMDLENGTH;
	}
	int i = 0;
	svbool_t pgt = svwhilelt_b64(i,remainder);
	{
#if (XX == 2) || (XX == 3)

	  float64_t fcp1_arr[SIMDLENGTH];
	  for(int i = 0; i < remainder; i++){
	    if(i == remainder-1){
	      fcp1_arr[i] = temporal[id1+i];
	    }else{
	      fcp1_arr[i] = temporal[id1+1+i];
	    }
	  }
	  svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
#else
	  svfloat64_t fce_vec = svld1(pgt,(float64_t*)&temporal[id1+1]);
#endif
	  svfloat64_t fc_vec  = svld1(pgt,(float64_t*)&temporal[id1]);
	  svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&temporal[id1-1]);
	  svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&temporal[id1+s]);
	  svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&temporal[id1+n]);
	  svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&temporal[id1+b]);
	  svfloat64_t fct_vec = svld1(pgt,(float64_t*)&temporal[id1+t]);
	  svfloat64_t tmp0,tmp1,tmp2;
	  fc_vec  = svmul_x(pgt,cc_vec,fc_vec);
	  fce_vec = svmul_x(pgt,ce_vec,fce_vec);
	  fcw_vec = svmul_x(pgt,cw_vec,fcw_vec);
	  fcn_vec = svmul_x(pgt,cn_vec,fcn_vec);
	  fcs_vec = svmul_x(pgt,cs_vec,fcs_vec);
	  fcb_vec = svmul_x(pgt,cb_vec,fcb_vec);
	  fct_vec = svmul_x(pgt,ct_vec,fct_vec);
	  tmp0 = svadd_x(pgt,fce_vec,fcw_vec);
	  tmp1 = svadd_x(pgt,fcn_vec,fcs_vec);
	  tmp2 = svadd_x(pgt,fct_vec,fcb_vec);
	  tmp0 = svadd_x(pgt,fc_vec, tmp0);
	  tmp1 = svadd_x(pgt,tmp1,   tmp2);
	  tmp0 = svadd_x(pgt,tmp0,   tmp1);
	  svst1(pgt,(float64_t*)&f2_t[c],tmp0);

	}
	id1 = id1p;
      }
    }
    
    tmp = izm;
    izm = izc;
    izc = izp;
    izp = tmp;
  }
}
