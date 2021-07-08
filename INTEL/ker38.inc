
{
  izm = 0; 
  izc = 1; 
  izp = 2; 

  for (h = 0; h < TB; h++){  
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

    id2 = 0+iy*nx+izp*nx*tby+step*nx*tby*tbz;

    for (y = ystr; y < yend; y++) {
#if (YY == 0) || (YY == 3)
      n0 = (y == 0)    ? 0 : - nx;
#endif
#if (YY == 2) || (YY == 3)
      s0 = (y == ny-1) ? 0 :   nx;
#endif
      x = 0;
      c = x + y*nx + z*nx*ny;
      __m512d fc_vec  = _mm512_load_pd(f1_t+c+x);
      __m512d fcp1_vec= _mm512_load_pd(f1_t+c+x+8);
      __m512d fcm1_vec= _mm512_alignr_epi64(fc_vec,fc_vec,1);
      __m512d fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
      __m512d fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
      __m512d fcs_vec = _mm512_load_pd(f1_t+c+x+s0);
      __m512d fcn_vec = _mm512_load_pd(f1_t+c+x+n0);
      __m512d fcb_vec = _mm512_load_pd(f1_t+c+x+b0);
      __m512d fct_vec = _mm512_load_pd(f1_t+c+x+t0);
      __m512d tmp0,tmp1,tmp2,tmp3;
      tmp3    = _mm512_mul_pd(cc_vec,fc_vec);
      fce_vec = _mm512_mul_pd(ce_vec,fce_vec);
      fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec);
      fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec);
      fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec);
      fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec);
      fct_vec = _mm512_mul_pd(ct_vec,fct_vec);
      tmp0 = _mm512_add_pd(fce_vec,fcw_vec);
      tmp1 = _mm512_add_pd(fcn_vec,fcs_vec);
      tmp2 = _mm512_add_pd(fct_vec,fcb_vec);
      tmp0 = _mm512_add_pd(tmp3,   tmp0);
      tmp1 = _mm512_add_pd(tmp1,   tmp2);
      tmp0 = _mm512_add_pd(tmp0,   tmp1);
      _mm512_store_pd(temporal+id2+x,tmp0);
      for (x = SIMDLENGTH; x < nx-SIMDLENGTH; x+=SIMDLENGTH) {

	fcm1_vec = fc_vec;
	fc_vec   = fcp1_vec;
	fcp1_vec = _mm512_load_pd(f1_t+c+8+x);

	fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
	fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
	/* fcm1_vec = fc_vec; */
	/* fc_vec   = fcp1_vec; */
	/* fcp1_vec = _mm512_load_pd(f1_t+c+x+8); */

	/* fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7); */
	/* fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1); */
	fcs_vec = _mm512_load_pd(f1_t+c+x+s0);
	fcn_vec = _mm512_load_pd(f1_t+c+x+n0);
	fcb_vec = _mm512_load_pd(f1_t+c+x+b0);
	fct_vec = _mm512_load_pd(f1_t+c+x+t0);
	tmp3    = _mm512_mul_pd(cc_vec,fc_vec);
	fce_vec = _mm512_mul_pd(ce_vec,fce_vec);
	fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec);
	fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec);
	fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec);
	fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec);
	fct_vec = _mm512_mul_pd(ct_vec,fct_vec);
	tmp0 = _mm512_add_pd(fce_vec,fcw_vec);
	tmp1 = _mm512_add_pd(fcn_vec,fcs_vec);
	tmp2 = _mm512_add_pd(fct_vec,fcb_vec);
	tmp0 = _mm512_add_pd(tmp3,   tmp0);
	tmp1 = _mm512_add_pd(tmp1,   tmp2);
	tmp0 = _mm512_add_pd(tmp0,   tmp1);
	_mm512_store_pd(temporal+id2+x,tmp0);


	/* __m512d fc_vec  = _mm512_load_pd(f1_t+c+x); */
	/* __m512d fce_vec = _mm512_load_pd(f1_t+c+x+1); */
	/* __m512d fcw_vec = _mm512_load_pd(f1_t+c+x-1); */
	/* __m512d fcs_vec = _mm512_load_pd(f1_t+c+x+s0); */
	/* __m512d fcn_vec = _mm512_load_pd(f1_t+c+x+n0); */
	/* __m512d fcb_vec = _mm512_load_pd(f1_t+c+x+b0); */
	/* __m512d fct_vec = _mm512_load_pd(f1_t+c+x+t0); */
	/* __m512d tmp0,tmp1,tmp2; */
	/* fc_vec  = _mm512_mul_pd(cc_vec,fc_vec); */
	/* fce_vec = _mm512_mul_pd(ce_vec,fce_vec); */
	/* fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec); */
	/* fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec); */
	/* fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec); */
	/* fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec); */
	/* fct_vec = _mm512_mul_pd(ct_vec,fct_vec); */
	/* tmp0 = _mm512_add_pd(fce_vec,fcw_vec); */
	/* tmp1 = _mm512_add_pd(fcn_vec,fcs_vec); */
	/* tmp2 = _mm512_add_pd(fct_vec,fcb_vec); */
	/* tmp0 = _mm512_add_pd(fc_vec, tmp0); */
	/* tmp1 = _mm512_add_pd(tmp1,   tmp2); */
	/* tmp0 = _mm512_add_pd(tmp0,   tmp1); */
	/* _mm512_store_pd(temporal+id2+x,tmp0); */

      }
      x = nx-SIMDLENGTH;

      fcm1_vec = fc_vec;
      fc_vec   = fcp1_vec;
      fcp1_vec = _mm512_alignr_epi64(fc_vec,fc_vec,7);

      fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
      fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
      fcs_vec = _mm512_load_pd(f1_t+c+x+s0);
      fcn_vec = _mm512_load_pd(f1_t+c+x+n0);
      fcb_vec = _mm512_load_pd(f1_t+c+x+b0);
      fct_vec = _mm512_load_pd(f1_t+c+x+t0);
      
      tmp3    = _mm512_mul_pd(cc_vec,fc_vec);
      fce_vec = _mm512_mul_pd(ce_vec,fce_vec);
      fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec);
      fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec);
      fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec);
      fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec);
      fct_vec = _mm512_mul_pd(ct_vec,fct_vec);
      tmp0 = _mm512_add_pd(fce_vec,fcw_vec);
      tmp1 = _mm512_add_pd(fcn_vec,fcs_vec);
      tmp2 = _mm512_add_pd(fct_vec,fcb_vec);
      tmp0 = _mm512_add_pd(tmp3,   tmp0);
      tmp1 = _mm512_add_pd(tmp1,   tmp2);
      tmp0 = _mm512_add_pd(tmp0,   tmp1);
      _mm512_store_pd(temporal+id2+x,tmp0);
      /* { */
      /* 	REAL fcp1_arr[8] = {f1_t[c+x+1],f1_t[c+x+2],f1_t[c+x+3],f1_t[c+x+4],f1_t[c+x+5],f1_t[c+x+6],f1_t[c+x+7],f1_t[c+x+7]}; */
      /* 	__m512d fc_vec  = _mm512_load_pd(f1_t+c+x); */
      /* 	__m512d fce_vec = _mm512_load_pd(fcp1_arr); */
      /* 	__m512d fcw_vec = _mm512_load_pd(f1_t+c+x-1); */
      /* 	__m512d fcs_vec = _mm512_load_pd(f1_t+c+x+s0); */
      /* 	__m512d fcn_vec = _mm512_load_pd(f1_t+c+x+n0); */
      /* 	__m512d fcb_vec = _mm512_load_pd(f1_t+c+x+b0); */
      /* 	__m512d fct_vec = _mm512_load_pd(f1_t+c+x+t0); */
      /* 	__m512d tmp0,tmp1,tmp2; */
      /* 	fc_vec  = _mm512_mul_pd(cc_vec,fc_vec); */
      /* 	fce_vec = _mm512_mul_pd(ce_vec,fce_vec); */
      /* 	fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec); */
      /* 	fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec); */
      /* 	fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec); */
      /* 	fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec); */
      /* 	fct_vec = _mm512_mul_pd(ct_vec,fct_vec); */
      /* 	tmp0 = _mm512_add_pd(fce_vec,fcw_vec); */
      /* 	tmp1 = _mm512_add_pd(fcn_vec,fcs_vec); */
      /* 	tmp2 = _mm512_add_pd(fct_vec,fcb_vec); */
      /* 	tmp0 = _mm512_add_pd(fc_vec, tmp0); */
      /* 	tmp1 = _mm512_add_pd(tmp1,   tmp2); */
      /* 	tmp0 = _mm512_add_pd(tmp0,   tmp1); */
      /* 	_mm512_store_pd(temporal+id2+x,tmp0); */
      /* } */

      id2 += nx;
    }

    for(z = h-1; z >= 0; z--){
      b = (z == 0)    ? 0 : (izm-izc)*nx*tby;
      t = (z == nz-1) ? 0 : (izp-izc)*nx*tby;
      n = -nx;
      s =  nx;
      
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

      id2 = 0+iy*nx+izp*nx*tby+(step+1)*nx*tby*tbz;
      id1 = 0+iy*nx+izc*nx*tby+step*nx*tby*tbz;

      for (y = ystr; y < yend; y++) {
#if (YY == 0) || (YY == 3)
	n = (y == 0)    ? 0 : -nx;
#endif
#if (YY == 2) || (YY == 3)
	s = (y == ny-1) ? 0 :  nx;
#endif
	x = 0;

	__m512d fc_vec  = _mm512_load_pd(temporal+id1+x);
	__m512d fcp1_vec= _mm512_load_pd(temporal+id1+x+8);
	__m512d fcm1_vec= _mm512_alignr_epi64(fc_vec,fc_vec,1);
	__m512d fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
	__m512d fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
	__m512d fcs_vec = _mm512_load_pd(temporal+id1+x+s);
	__m512d fcn_vec = _mm512_load_pd(temporal+id1+x+n);
	__m512d fcb_vec = _mm512_load_pd(temporal+id1+x+b);
	__m512d fct_vec = _mm512_load_pd(temporal+id1+x+t);
	__m512d tmp0,tmp1,tmp2,tmp3;
	tmp3    = _mm512_mul_pd(cc_vec,fc_vec);
	fce_vec = _mm512_mul_pd(ce_vec,fce_vec);
	fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec);
	fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec);
	fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec);
	fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec);
	fct_vec = _mm512_mul_pd(ct_vec,fct_vec);
	tmp0 = _mm512_add_pd(fce_vec,fcw_vec);
	tmp1 = _mm512_add_pd(fcn_vec,fcs_vec);
	tmp2 = _mm512_add_pd(fct_vec,fcb_vec);
	tmp0 = _mm512_add_pd(tmp3,   tmp0);
	tmp1 = _mm512_add_pd(tmp1,   tmp2);
	tmp0 = _mm512_add_pd(tmp0,   tmp1);
	_mm512_store_pd(temporal+id2+x,tmp0);

	/* { */
	/*   REAL fcm1_arr[SIMDLENGTH] = {temporal[id1+x],temporal[id1+x],temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6]}; */
	/*   __m512d fc_vec  = _mm512_load_pd(temporal+id1+x); */
	/*   __m512d fce_vec = _mm512_load_pd(temporal+id1+x+1); */
	/*   __m512d fcw_vec = _mm512_load_pd(fcm1_arr); */
	/*   __m512d fcs_vec = _mm512_load_pd(temporal+id1+x+s); */
	/*   __m512d fcn_vec = _mm512_load_pd(temporal+id1+x+n); */
	/*   __m512d fcb_vec = _mm512_load_pd(temporal+id1+x+b); */
	/*   __m512d fct_vec = _mm512_load_pd(temporal+id1+x+t); */
	/*   __m512d tmp0,tmp1,tmp2; */
	/*   fc_vec  = _mm512_mul_pd(cc_vec,fc_vec); */
	/*   fce_vec = _mm512_mul_pd(ce_vec,fce_vec); */
	/*   fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec); */
	/*   fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec); */
	/*   fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec); */
	/*   fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec); */
	/*   fct_vec = _mm512_mul_pd(ct_vec,fct_vec); */
	/*   tmp0 = _mm512_add_pd(fce_vec,fcw_vec); */
	/*   tmp1 = _mm512_add_pd(fcn_vec,fcs_vec); */
	/*   tmp2 = _mm512_add_pd(fct_vec,fcb_vec); */
	/*   tmp0 = _mm512_add_pd(fc_vec, tmp0); */
	/*   tmp1 = _mm512_add_pd(tmp1,   tmp2); */
	/*   tmp0 = _mm512_add_pd(tmp0,   tmp1); */
	/*   _mm512_store_pd(temporal+id2+x,tmp0); */
	/* } */
	for (x = SIMDLENGTH; x < nx-SIMDLENGTH; x+=SIMDLENGTH) {
	  fcm1_vec = fc_vec;
	  fc_vec   = fcp1_vec;
	  fcp1_vec = _mm512_load_pd(temporal+id1+8+x);
	  
	  fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
	  fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
	  fcs_vec = _mm512_load_pd(temporal+id1+x+s);
	  fcn_vec = _mm512_load_pd(temporal+id1+x+n);
	  fcb_vec = _mm512_load_pd(temporal+id1+x+b);
	  fct_vec = _mm512_load_pd(temporal+id1+x+t);
	  tmp3    = _mm512_mul_pd(cc_vec,fc_vec);
	  fce_vec = _mm512_mul_pd(ce_vec,fce_vec);
	  fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec);
	  fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec);
	  fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec);
	  fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec);
	  fct_vec = _mm512_mul_pd(ct_vec,fct_vec);
	  tmp0 = _mm512_add_pd(fce_vec,fcw_vec);
	  tmp1 = _mm512_add_pd(fcn_vec,fcs_vec);
	  tmp2 = _mm512_add_pd(fct_vec,fcb_vec);
	  tmp0 = _mm512_add_pd(tmp3,   tmp0);
	  tmp1 = _mm512_add_pd(tmp1,   tmp2);
	  tmp0 = _mm512_add_pd(tmp0,   tmp1);
	  _mm512_store_pd(temporal+id2+x,tmp0);

	  /* __m512d fc_vec  = _mm512_load_pd(temporal+id1+x); */
	  /* __m512d fce_vec = _mm512_load_pd(temporal+id1+x+1); */
	  /* __m512d fcw_vec = _mm512_load_pd(temporal+id1+x-1); */
	  /* __m512d fcs_vec = _mm512_load_pd(temporal+id1+x+s); */
	  /* __m512d fcn_vec = _mm512_load_pd(temporal+id1+x+n); */
	  /* __m512d fcb_vec = _mm512_load_pd(temporal+id1+x+b); */
	  /* __m512d fct_vec = _mm512_load_pd(temporal+id1+x+t); */
	  /* __m512d tmp0,tmp1,tmp2; */
	  /* fc_vec  = _mm512_mul_pd(cc_vec,fc_vec); */
	  /* fce_vec = _mm512_mul_pd(ce_vec,fce_vec); */
	  /* fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec); */
	  /* fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec); */
	  /* fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec); */
	  /* fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec); */
	  /* fct_vec = _mm512_mul_pd(ct_vec,fct_vec); */
	  /* tmp0 = _mm512_add_pd(fce_vec,fcw_vec); */
	  /* tmp1 = _mm512_add_pd(fcn_vec,fcs_vec); */
	  /* tmp2 = _mm512_add_pd(fct_vec,fcb_vec); */
	  /* tmp0 = _mm512_add_pd(fc_vec, tmp0); */
	  /* tmp1 = _mm512_add_pd(tmp1,   tmp2); */
	  /* tmp0 = _mm512_add_pd(tmp0,   tmp1); */
	  /* _mm512_store_pd(temporal+id2+x,tmp0); */
	}
	x = nx-SIMDLENGTH;
	fcm1_vec = fc_vec;
	fc_vec   = fcp1_vec;
	fcp1_vec = _mm512_alignr_epi64(fc_vec,fc_vec,7);
	
	fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
	fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
	fcs_vec = _mm512_load_pd(temporal+id1+x+s);
	fcn_vec = _mm512_load_pd(temporal+id1+x+n);
	fcb_vec = _mm512_load_pd(temporal+id1+x+b);
	fct_vec = _mm512_load_pd(temporal+id1+x+t);
	
	tmp3    = _mm512_mul_pd(cc_vec,fc_vec);
	fce_vec = _mm512_mul_pd(ce_vec,fce_vec);
	fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec);
	fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec);
	fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec);
	fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec);
	fct_vec = _mm512_mul_pd(ct_vec,fct_vec);
	tmp0 = _mm512_add_pd(fce_vec,fcw_vec);
	tmp1 = _mm512_add_pd(fcn_vec,fcs_vec);
	tmp2 = _mm512_add_pd(fct_vec,fcb_vec);
	tmp0 = _mm512_add_pd(tmp3,   tmp0);
	tmp1 = _mm512_add_pd(tmp1,   tmp2);
	tmp0 = _mm512_add_pd(tmp0,   tmp1);
	_mm512_store_pd(temporal+id2+x,tmp0);
	/* { */

	/*   REAL fcp1_arr[8] = {temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6],temporal[id1+x+7],temporal[id1+x+7]}; */
	/*   __m512d fc_vec  = _mm512_load_pd(temporal+id1+x); */
	/*   __m512d fce_vec = _mm512_load_pd(fcp1_arr); */
	/*   __m512d fcw_vec = _mm512_load_pd(temporal+id1+x-1); */
	/*   __m512d fcs_vec = _mm512_load_pd(temporal+id1+x+s); */
	/*   __m512d fcn_vec = _mm512_load_pd(temporal+id1+x+n); */
	/*   __m512d fcb_vec = _mm512_load_pd(temporal+id1+x+b); */
	/*   __m512d fct_vec = _mm512_load_pd(temporal+id1+x+t); */
	/*   __m512d tmp0,tmp1,tmp2; */
	/*   fc_vec  = _mm512_mul_pd(cc_vec,fc_vec); */
	/*   fce_vec = _mm512_mul_pd(ce_vec,fce_vec); */
	/*   fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec); */
	/*   fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec); */
	/*   fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec); */
	/*   fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec); */
	/*   fct_vec = _mm512_mul_pd(ct_vec,fct_vec); */
	/*   tmp0 = _mm512_add_pd(fce_vec,fcw_vec); */
	/*   tmp1 = _mm512_add_pd(fcn_vec,fcs_vec); */
	/*   tmp2 = _mm512_add_pd(fct_vec,fcb_vec); */
	/*   tmp0 = _mm512_add_pd(fc_vec, tmp0); */
	/*   tmp1 = _mm512_add_pd(tmp1,   tmp2); */
	/*   tmp0 = _mm512_add_pd(tmp0,   tmp1); */
	/*   _mm512_store_pd(temporal+id2+x,tmp0); */
	/* } */
	id2 += nx;
	id1 += nx;
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
    id2 = 0+iy*nx+izp*nx*tby+step*nx*tby*tbz;

    for (y = ystr; y < yend; y++) {
#if (YY == 0) || (YY == 3)
      n0 = (y == 0)    ? 0 : - nx;
#endif
#if (YY == 2) || (YY == 3)
      s0 = (y == ny-1) ? 0 :   nx;
#endif
      x = 0;
      c = x + y*nx + z*nx*ny;
      __m512d fc_vec  = _mm512_load_pd(f1_t+c+x);
      __m512d fcp1_vec= _mm512_load_pd(f1_t+c+x+8);
      __m512d fcm1_vec= _mm512_alignr_epi64(fc_vec,fc_vec,1);
      __m512d fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
      __m512d fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
      __m512d fcs_vec = _mm512_load_pd(f1_t+c+x+s0);
      __m512d fcn_vec = _mm512_load_pd(f1_t+c+x+n0);
      __m512d fcb_vec = _mm512_load_pd(f1_t+c+x+b0);
      __m512d fct_vec = _mm512_load_pd(f1_t+c+x+t0);
      __m512d tmp0,tmp1,tmp2,tmp3;
      tmp3    = _mm512_mul_pd(cc_vec,fc_vec);
      fce_vec = _mm512_mul_pd(ce_vec,fce_vec);
      fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec);
      fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec);
      fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec);
      fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec);
      fct_vec = _mm512_mul_pd(ct_vec,fct_vec);
      tmp0 = _mm512_add_pd(fce_vec,fcw_vec);
      tmp1 = _mm512_add_pd(fcn_vec,fcs_vec);
      tmp2 = _mm512_add_pd(fct_vec,fcb_vec);
      tmp0 = _mm512_add_pd(tmp3,   tmp0);
      tmp1 = _mm512_add_pd(tmp1,   tmp2);
      tmp0 = _mm512_add_pd(tmp0,   tmp1);
      _mm512_store_pd(temporal+id2+x,tmp0);
      for (x = SIMDLENGTH; x < nx-SIMDLENGTH; x+=SIMDLENGTH) {

	fcm1_vec = fc_vec;
	fc_vec   = fcp1_vec;
	fcp1_vec = _mm512_load_pd(f1_t+c+8+x);

	fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
	fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
	fcs_vec = _mm512_load_pd(f1_t+c+x+s0);
	fcn_vec = _mm512_load_pd(f1_t+c+x+n0);
	fcb_vec = _mm512_load_pd(f1_t+c+x+b0);
	fct_vec = _mm512_load_pd(f1_t+c+x+t0);
	tmp3    = _mm512_mul_pd(cc_vec,fc_vec);
	fce_vec = _mm512_mul_pd(ce_vec,fce_vec);
	fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec);
	fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec);
	fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec);
	fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec);
	fct_vec = _mm512_mul_pd(ct_vec,fct_vec);
	tmp0 = _mm512_add_pd(fce_vec,fcw_vec);
	tmp1 = _mm512_add_pd(fcn_vec,fcs_vec);
	tmp2 = _mm512_add_pd(fct_vec,fcb_vec);
	tmp0 = _mm512_add_pd(tmp3,   tmp0);
	tmp1 = _mm512_add_pd(tmp1,   tmp2);
	tmp0 = _mm512_add_pd(tmp0,   tmp1);
	_mm512_store_pd(temporal+id2+x,tmp0);

      }
      x = nx-SIMDLENGTH;

      fcm1_vec = fc_vec;
      fc_vec   = fcp1_vec;
      fcp1_vec = _mm512_alignr_epi64(fc_vec,fc_vec,7);

      fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
      fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
      fcs_vec = _mm512_load_pd(f1_t+c+x+s0);
      fcn_vec = _mm512_load_pd(f1_t+c+x+n0);
      fcb_vec = _mm512_load_pd(f1_t+c+x+b0);
      fct_vec = _mm512_load_pd(f1_t+c+x+t0);
      
      tmp3    = _mm512_mul_pd(cc_vec,fc_vec);
      fce_vec = _mm512_mul_pd(ce_vec,fce_vec);
      fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec);
      fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec);
      fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec);
      fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec);
      fct_vec = _mm512_mul_pd(ct_vec,fct_vec);
      tmp0 = _mm512_add_pd(fce_vec,fcw_vec);
      tmp1 = _mm512_add_pd(fcn_vec,fcs_vec);
      tmp2 = _mm512_add_pd(fct_vec,fcb_vec);
      tmp0 = _mm512_add_pd(tmp3,   tmp0);
      tmp1 = _mm512_add_pd(tmp1,   tmp2);
      tmp0 = _mm512_add_pd(tmp0,   tmp1);
      _mm512_store_pd(temporal+id2+x,tmp0);
      /* { */
      /* 	REAL fcm1_arr[SIMDLENGTH] = {f1_t[c+x],f1_t[c+x],f1_t[c+x+1],f1_t[c+x+2],f1_t[c+x+3],f1_t[c+x+4],f1_t[c+x+5],f1_t[c+x+6]}; */
      /* 	__m512d fc_vec  = _mm512_load_pd(f1_t+c+x); */
      /* 	__m512d fce_vec = _mm512_load_pd(f1_t+c+x+1); */
      /* 	__m512d fcw_vec = _mm512_load_pd(fcm1_arr); */
      /* 	__m512d fcs_vec = _mm512_load_pd(f1_t+c+x+s0); */
      /* 	__m512d fcn_vec = _mm512_load_pd(f1_t+c+x+n0); */
      /* 	__m512d fcb_vec = _mm512_load_pd(f1_t+c+x+b0); */
      /* 	__m512d fct_vec = _mm512_load_pd(f1_t+c+x+t0); */
      /* 	__m512d tmp0,tmp1,tmp2; */
      /* 	fc_vec  = _mm512_mul_pd(cc_vec,fc_vec); */
      /* 	fce_vec = _mm512_mul_pd(ce_vec,fce_vec); */
      /* 	fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec); */
      /* 	fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec); */
      /* 	fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec); */
      /* 	fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec); */
      /* 	fct_vec = _mm512_mul_pd(ct_vec,fct_vec); */
      /* 	tmp0 = _mm512_add_pd(fce_vec,fcw_vec); */
      /* 	tmp1 = _mm512_add_pd(fcn_vec,fcs_vec); */
      /* 	tmp2 = _mm512_add_pd(fct_vec,fcb_vec); */
      /* 	tmp0 = _mm512_add_pd(fc_vec, tmp0); */
      /* 	tmp1 = _mm512_add_pd(tmp1,   tmp2); */
      /* 	tmp0 = _mm512_add_pd(tmp0,   tmp1); */
      /* 	_mm512_store_pd(temporal+id2+x,tmp0); */
      /* } */
      /* for (x = SIMDLENGTH; x < nx-SIMDLENGTH; x+=SIMDLENGTH) { */
      /* 	__m512d fc_vec  = _mm512_load_pd(f1_t+c+x); */
      /* 	__m512d fce_vec = _mm512_load_pd(f1_t+c+x+1); */
      /* 	__m512d fcw_vec = _mm512_load_pd(f1_t+c+x-1); */
      /* 	__m512d fcs_vec = _mm512_load_pd(f1_t+c+x+s0); */
      /* 	__m512d fcn_vec = _mm512_load_pd(f1_t+c+x+n0); */
      /* 	__m512d fcb_vec = _mm512_load_pd(f1_t+c+x+b0); */
      /* 	__m512d fct_vec = _mm512_load_pd(f1_t+c+x+t0); */
      /* 	__m512d tmp0,tmp1,tmp2; */
      /* 	fc_vec  = _mm512_mul_pd(cc_vec,fc_vec); */
      /* 	fce_vec = _mm512_mul_pd(ce_vec,fce_vec); */
      /* 	fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec); */
      /* 	fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec); */
      /* 	fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec); */
      /* 	fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec); */
      /* 	fct_vec = _mm512_mul_pd(ct_vec,fct_vec); */
      /* 	tmp0 = _mm512_add_pd(fce_vec,fcw_vec); */
      /* 	tmp1 = _mm512_add_pd(fcn_vec,fcs_vec); */
      /* 	tmp2 = _mm512_add_pd(fct_vec,fcb_vec); */
      /* 	tmp0 = _mm512_add_pd(fc_vec, tmp0); */
      /* 	tmp1 = _mm512_add_pd(tmp1,   tmp2); */
      /* 	tmp0 = _mm512_add_pd(tmp0,   tmp1); */
      /* 	_mm512_store_pd(temporal+id2+x,tmp0); */
      /* } */
      /* x = nx-SIMDLENGTH; */
      /* { */
      /* 	REAL fcp1_arr[8] = {f1_t[c+x+1],f1_t[c+x+2],f1_t[c+x+3],f1_t[c+x+4],f1_t[c+x+5],f1_t[c+x+6],f1_t[c+x+7],f1_t[c+x+7]}; */
      /* 	__m512d fc_vec  = _mm512_load_pd(f1_t+c+x); */
      /* 	__m512d fce_vec = _mm512_load_pd(fcp1_arr); */
      /* 	__m512d fcw_vec = _mm512_load_pd(f1_t+c+x-1); */
      /* 	__m512d fcs_vec = _mm512_load_pd(f1_t+c+x+s0); */
      /* 	__m512d fcn_vec = _mm512_load_pd(f1_t+c+x+n0); */
      /* 	__m512d fcb_vec = _mm512_load_pd(f1_t+c+x+b0); */
      /* 	__m512d fct_vec = _mm512_load_pd(f1_t+c+x+t0); */
      /* 	__m512d tmp0,tmp1,tmp2; */
      /* 	fc_vec  = _mm512_mul_pd(cc_vec,fc_vec); */
      /* 	fce_vec = _mm512_mul_pd(ce_vec,fce_vec); */
      /* 	fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec); */
      /* 	fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec); */
      /* 	fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec); */
      /* 	fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec); */
      /* 	fct_vec = _mm512_mul_pd(ct_vec,fct_vec); */
      /* 	tmp0 = _mm512_add_pd(fce_vec,fcw_vec); */
      /* 	tmp1 = _mm512_add_pd(fcn_vec,fcs_vec); */
      /* 	tmp2 = _mm512_add_pd(fct_vec,fcb_vec); */
      /* 	tmp0 = _mm512_add_pd(fc_vec, tmp0); */
      /* 	tmp1 = _mm512_add_pd(tmp1,   tmp2); */
      /* 	tmp0 = _mm512_add_pd(tmp0,   tmp1); */
      /* 	_mm512_store_pd(temporal+id2+x,tmp0); */
      /* } */
      id2 += nx;
    }

    for(z = zz+TB-1; z >= zz+1; z--){
      b = (z == 0)    ? 0 : (izm-izc)*nx*tby;
      t = (z == nz-1) ? 0 : (izp-izc)*nx*tby;
      n = -nx;
      s =  nx;

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
      
      id2 = 0+iy*nx+izp*nx*tby+(step+1)*nx*tby*tbz;
      id1 = 0+iy*nx+izc*nx*tby+step*nx*tby*tbz;

      for (y = ystr; y < yend; y++) {
#if (YY == 0) || (YY == 3)
	n = (y == 0)    ? 0 : -nx;
#endif
#if (YY == 2) || (YY == 3)
	s = (y == ny-1) ? 0 :  nx;
#endif
	x = 0;
	__m512d fc_vec  = _mm512_load_pd(temporal+id1+x);
	__m512d fcp1_vec= _mm512_load_pd(temporal+id1+x+8);
	__m512d fcm1_vec= _mm512_alignr_epi64(fc_vec,fc_vec,1);
	__m512d fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
	__m512d fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
	__m512d fcs_vec = _mm512_load_pd(temporal+id1+x+s);
	__m512d fcn_vec = _mm512_load_pd(temporal+id1+x+n);
	__m512d fcb_vec = _mm512_load_pd(temporal+id1+x+b);
	__m512d fct_vec = _mm512_load_pd(temporal+id1+x+t);
	__m512d tmp0,tmp1,tmp2,tmp3;
	tmp3    = _mm512_mul_pd(cc_vec,fc_vec);
	fce_vec = _mm512_mul_pd(ce_vec,fce_vec);
	fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec);
	fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec);
	fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec);
	fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec);
	fct_vec = _mm512_mul_pd(ct_vec,fct_vec);
	tmp0 = _mm512_add_pd(fce_vec,fcw_vec);
	tmp1 = _mm512_add_pd(fcn_vec,fcs_vec);
	tmp2 = _mm512_add_pd(fct_vec,fcb_vec);
	tmp0 = _mm512_add_pd(tmp3,   tmp0);
	tmp1 = _mm512_add_pd(tmp1,   tmp2);
	tmp0 = _mm512_add_pd(tmp0,   tmp1);
	_mm512_store_pd(temporal+id2+x,tmp0);

	for (x = SIMDLENGTH; x < nx-SIMDLENGTH; x+=SIMDLENGTH) {
	  fcm1_vec = fc_vec;
	  fc_vec   = fcp1_vec;
	  fcp1_vec = _mm512_load_pd(temporal+id1+8+x);
	  
	  fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
	  fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
	  fcs_vec = _mm512_load_pd(temporal+id1+x+s);
	  fcn_vec = _mm512_load_pd(temporal+id1+x+n);
	  fcb_vec = _mm512_load_pd(temporal+id1+x+b);
	  fct_vec = _mm512_load_pd(temporal+id1+x+t);
	  tmp3    = _mm512_mul_pd(cc_vec,fc_vec);
	  fce_vec = _mm512_mul_pd(ce_vec,fce_vec);
	  fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec);
	  fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec);
	  fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec);
	  fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec);
	  fct_vec = _mm512_mul_pd(ct_vec,fct_vec);
	  tmp0 = _mm512_add_pd(fce_vec,fcw_vec);
	  tmp1 = _mm512_add_pd(fcn_vec,fcs_vec);
	  tmp2 = _mm512_add_pd(fct_vec,fcb_vec);
	  tmp0 = _mm512_add_pd(tmp3,   tmp0);
	  tmp1 = _mm512_add_pd(tmp1,   tmp2);
	  tmp0 = _mm512_add_pd(tmp0,   tmp1);
	  _mm512_store_pd(temporal+id2+x,tmp0);

	}
	x = nx-SIMDLENGTH;
	fcm1_vec = fc_vec;
	fc_vec   = fcp1_vec;
	fcp1_vec = _mm512_alignr_epi64(fc_vec,fc_vec,7);
	
	fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
	fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
	fcs_vec = _mm512_load_pd(temporal+id1+x+s);
	fcn_vec = _mm512_load_pd(temporal+id1+x+n);
	fcb_vec = _mm512_load_pd(temporal+id1+x+b);
	fct_vec = _mm512_load_pd(temporal+id1+x+t);
	
	tmp3    = _mm512_mul_pd(cc_vec,fc_vec);
	fce_vec = _mm512_mul_pd(ce_vec,fce_vec);
	fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec);
	fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec);
	fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec);
	fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec);
	fct_vec = _mm512_mul_pd(ct_vec,fct_vec);
	tmp0 = _mm512_add_pd(fce_vec,fcw_vec);
	tmp1 = _mm512_add_pd(fcn_vec,fcs_vec);
	tmp2 = _mm512_add_pd(fct_vec,fcb_vec);
	tmp0 = _mm512_add_pd(tmp3,   tmp0);
	tmp1 = _mm512_add_pd(tmp1,   tmp2);
	tmp0 = _mm512_add_pd(tmp0,   tmp1);
	_mm512_store_pd(temporal+id2+x,tmp0);
	/* { */
	/*   REAL fcm1_arr[SIMDLENGTH] = {temporal[id1+x],temporal[id1+x],temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6]}; */
	/*   __m512d fc_vec  = _mm512_load_pd(temporal+id1+x); */
	/*   __m512d fce_vec = _mm512_load_pd(temporal+id1+x+1); */
	/*   __m512d fcw_vec = _mm512_load_pd(fcm1_arr); */
	/*   __m512d fcs_vec = _mm512_load_pd(temporal+id1+x+s); */
	/*   __m512d fcn_vec = _mm512_load_pd(temporal+id1+x+n); */
	/*   __m512d fcb_vec = _mm512_load_pd(temporal+id1+x+b); */
	/*   __m512d fct_vec = _mm512_load_pd(temporal+id1+x+t); */
	/*   __m512d tmp0,tmp1,tmp2; */
	/*   fc_vec  = _mm512_mul_pd(cc_vec,fc_vec); */
	/*   fce_vec = _mm512_mul_pd(ce_vec,fce_vec); */
	/*   fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec); */
	/*   fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec); */
	/*   fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec); */
	/*   fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec); */
	/*   fct_vec = _mm512_mul_pd(ct_vec,fct_vec); */
	/*   tmp0 = _mm512_add_pd(fce_vec,fcw_vec); */
	/*   tmp1 = _mm512_add_pd(fcn_vec,fcs_vec); */
	/*   tmp2 = _mm512_add_pd(fct_vec,fcb_vec); */
	/*   tmp0 = _mm512_add_pd(fc_vec, tmp0); */
	/*   tmp1 = _mm512_add_pd(tmp1,   tmp2); */
	/*   tmp0 = _mm512_add_pd(tmp0,   tmp1); */
	/*   _mm512_store_pd(temporal+id2+x,tmp0); */
	/* } */
	/* for (x = SIMDLENGTH; x < nx-SIMDLENGTH; x+=SIMDLENGTH) { */
	/*   __m512d fc_vec  = _mm512_load_pd(temporal+id1+x); */
	/*   __m512d fce_vec = _mm512_load_pd(temporal+id1+x+1); */
	/*   __m512d fcw_vec = _mm512_load_pd(temporal+id1+x-1); */
	/*   __m512d fcs_vec = _mm512_load_pd(temporal+id1+x+s); */
	/*   __m512d fcn_vec = _mm512_load_pd(temporal+id1+x+n); */
	/*   __m512d fcb_vec = _mm512_load_pd(temporal+id1+x+b); */
	/*   __m512d fct_vec = _mm512_load_pd(temporal+id1+x+t); */
	/*   __m512d tmp0,tmp1,tmp2; */
	/*   fc_vec  = _mm512_mul_pd(cc_vec,fc_vec); */
	/*   fce_vec = _mm512_mul_pd(ce_vec,fce_vec); */
	/*   fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec); */
	/*   fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec); */
	/*   fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec); */
	/*   fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec); */
	/*   fct_vec = _mm512_mul_pd(ct_vec,fct_vec); */
	/*   tmp0 = _mm512_add_pd(fce_vec,fcw_vec); */
	/*   tmp1 = _mm512_add_pd(fcn_vec,fcs_vec); */
	/*   tmp2 = _mm512_add_pd(fct_vec,fcb_vec); */
	/*   tmp0 = _mm512_add_pd(fc_vec, tmp0); */
	/*   tmp1 = _mm512_add_pd(tmp1,   tmp2); */
	/*   tmp0 = _mm512_add_pd(tmp0,   tmp1); */
	/*   _mm512_store_pd(temporal+id2+x,tmp0); */
	/* } */
	/* x = nx-SIMDLENGTH; */
	/* { */

	/*   REAL fcp1_arr[8] = {temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6],temporal[id1+x+7],temporal[id1+x+7]}; */
	/*   __m512d fc_vec  = _mm512_load_pd(temporal+id1+x); */
	/*   __m512d fce_vec = _mm512_load_pd(fcp1_arr); */
	/*   __m512d fcw_vec = _mm512_load_pd(temporal+id1+x-1); */
	/*   __m512d fcs_vec = _mm512_load_pd(temporal+id1+x+s); */
	/*   __m512d fcn_vec = _mm512_load_pd(temporal+id1+x+n); */
	/*   __m512d fcb_vec = _mm512_load_pd(temporal+id1+x+b); */
	/*   __m512d fct_vec = _mm512_load_pd(temporal+id1+x+t); */
	/*   __m512d tmp0,tmp1,tmp2; */
	/*   fc_vec  = _mm512_mul_pd(cc_vec,fc_vec); */
	/*   fce_vec = _mm512_mul_pd(ce_vec,fce_vec); */
	/*   fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec); */
	/*   fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec); */
	/*   fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec); */
	/*   fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec); */
	/*   fct_vec = _mm512_mul_pd(ct_vec,fct_vec); */
	/*   tmp0 = _mm512_add_pd(fce_vec,fcw_vec); */
	/*   tmp1 = _mm512_add_pd(fcn_vec,fcs_vec); */
	/*   tmp2 = _mm512_add_pd(fct_vec,fcb_vec); */
	/*   tmp0 = _mm512_add_pd(fc_vec, tmp0); */
	/*   tmp1 = _mm512_add_pd(tmp1,   tmp2); */
	/*   tmp0 = _mm512_add_pd(tmp0,   tmp1); */
	/*   _mm512_store_pd(temporal+id2+x,tmp0); */
	/* } */
	id1 += nx;
	id2 += nx;
      }
      step++;

    }

    {
      z = zz;
      b = (z == 0)    ? 0 : (izm-izc)*nx*tby;
      t = (z == nz-1) ? 0 : (izp-izc)*nx*tby;
      n = -nx;
      s =  nx;
      
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
      id1 = 0+iy*nx+izc*nx*tby+step*nx*tby*tbz;
      for (y = ystr; y < yend; y++) {
	//	int id1p = id1 + nx;
#if (YY == 0) || (YY == 3)
	n = (y == 0)    ? 0 : -nx;
#endif
#if (YY == 2) || (YY == 3)
	s = (y == ny-1) ? 0 :  nx;
#endif
	x = 0;
	c = y*nx + z*nx*ny;
	__m512d fc_vec  = _mm512_load_pd(temporal+id1+x);
	__m512d fcp1_vec= _mm512_load_pd(temporal+id1+x+8);
	__m512d fcm1_vec= _mm512_alignr_epi64(fc_vec,fc_vec,1);
	__m512d fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
	__m512d fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
	__m512d fcs_vec = _mm512_load_pd(temporal+id1+x+s);
	__m512d fcn_vec = _mm512_load_pd(temporal+id1+x+n);
	__m512d fcb_vec = _mm512_load_pd(temporal+id1+x+b);
	__m512d fct_vec = _mm512_load_pd(temporal+id1+x+t);
	__m512d tmp0,tmp1,tmp2,tmp3;
	tmp3    = _mm512_mul_pd(cc_vec,fc_vec);
	fce_vec = _mm512_mul_pd(ce_vec,fce_vec);
	fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec);
	fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec);
	fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec);
	fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec);
	fct_vec = _mm512_mul_pd(ct_vec,fct_vec);
	tmp0 = _mm512_add_pd(fce_vec,fcw_vec);
	tmp1 = _mm512_add_pd(fcn_vec,fcs_vec);
	tmp2 = _mm512_add_pd(fct_vec,fcb_vec);
	tmp0 = _mm512_add_pd(tmp3,   tmp0);
	tmp1 = _mm512_add_pd(tmp1,   tmp2);
	tmp0 = _mm512_add_pd(tmp0,   tmp1);
	_mm512_store_pd(f2_t+c+x,tmp0);

	for (x = SIMDLENGTH; x < nx-SIMDLENGTH; x+=SIMDLENGTH) {
	  fcm1_vec = fc_vec;
	  fc_vec   = fcp1_vec;
	  fcp1_vec = _mm512_load_pd(temporal+id1+8+x);
	  
	  fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
	  fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
	  fcs_vec = _mm512_load_pd(temporal+id1+x+s);
	  fcn_vec = _mm512_load_pd(temporal+id1+x+n);
	  fcb_vec = _mm512_load_pd(temporal+id1+x+b);
	  fct_vec = _mm512_load_pd(temporal+id1+x+t);
	  tmp3    = _mm512_mul_pd(cc_vec,fc_vec);
	  fce_vec = _mm512_mul_pd(ce_vec,fce_vec);
	  fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec);
	  fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec);
	  fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec);
	  fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec);
	  fct_vec = _mm512_mul_pd(ct_vec,fct_vec);
	  tmp0 = _mm512_add_pd(fce_vec,fcw_vec);
	  tmp1 = _mm512_add_pd(fcn_vec,fcs_vec);
	  tmp2 = _mm512_add_pd(fct_vec,fcb_vec);
	  tmp0 = _mm512_add_pd(tmp3,   tmp0);
	  tmp1 = _mm512_add_pd(tmp1,   tmp2);
	  tmp0 = _mm512_add_pd(tmp0,   tmp1);
	  _mm512_store_pd(f2_t+c+x,tmp0);

	}
	x = nx-SIMDLENGTH;
	fcm1_vec = fc_vec;
	fc_vec   = fcp1_vec;
	fcp1_vec = _mm512_alignr_epi64(fc_vec,fc_vec,7);
	
	fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
	fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
	fcs_vec = _mm512_load_pd(temporal+id1+x+s);
	fcn_vec = _mm512_load_pd(temporal+id1+x+n);
	fcb_vec = _mm512_load_pd(temporal+id1+x+b);
	fct_vec = _mm512_load_pd(temporal+id1+x+t);
	
	tmp3    = _mm512_mul_pd(cc_vec,fc_vec);
	fce_vec = _mm512_mul_pd(ce_vec,fce_vec);
	fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec);
	fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec);
	fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec);
	fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec);
	fct_vec = _mm512_mul_pd(ct_vec,fct_vec);
	tmp0 = _mm512_add_pd(fce_vec,fcw_vec);
	tmp1 = _mm512_add_pd(fcn_vec,fcs_vec);
	tmp2 = _mm512_add_pd(fct_vec,fcb_vec);
	tmp0 = _mm512_add_pd(tmp3,   tmp0);
	tmp1 = _mm512_add_pd(tmp1,   tmp2);
	tmp0 = _mm512_add_pd(tmp0,   tmp1);
	_mm512_store_pd(f2_t+c+x,tmp0);
	/* { */
	/*   REAL fcm1_arr[SIMDLENGTH] = {temporal[id1+x],temporal[id1+x],temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6]}; */
	/*   __m512d fc_vec  = _mm512_load_pd(temporal+id1+x); */
	/*   __m512d fce_vec = _mm512_load_pd(temporal+id1+x+1); */
	/*   __m512d fcw_vec = _mm512_load_pd(fcm1_arr); */
	/*   __m512d fcs_vec = _mm512_load_pd(temporal+id1+x+s); */
	/*   __m512d fcn_vec = _mm512_load_pd(temporal+id1+x+n); */
	/*   __m512d fcb_vec = _mm512_load_pd(temporal+id1+x+b); */
	/*   __m512d fct_vec = _mm512_load_pd(temporal+id1+x+t); */
	/*   __m512d tmp0,tmp1,tmp2; */
	/*   fc_vec  = _mm512_mul_pd(cc_vec,fc_vec); */
	/*   fce_vec = _mm512_mul_pd(ce_vec,fce_vec); */
	/*   fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec); */
	/*   fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec); */
	/*   fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec); */
	/*   fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec); */
	/*   fct_vec = _mm512_mul_pd(ct_vec,fct_vec); */
	/*   tmp0 = _mm512_add_pd(fce_vec,fcw_vec); */
	/*   tmp1 = _mm512_add_pd(fcn_vec,fcs_vec); */
	/*   tmp2 = _mm512_add_pd(fct_vec,fcb_vec); */
	/*   tmp0 = _mm512_add_pd(fc_vec, tmp0); */
	/*   tmp1 = _mm512_add_pd(tmp1,   tmp2); */
	/*   tmp0 = _mm512_add_pd(tmp0,   tmp1); */
	/*   _mm512_store_pd(f2_t+c+x,tmp0); */
	/* } */
	/* for (x = SIMDLENGTH; x < nx-SIMDLENGTH; x+=SIMDLENGTH) { */
	/*   __m512d fc_vec  = _mm512_load_pd(temporal+id1+x); */
	/*   __m512d fce_vec = _mm512_load_pd(temporal+id1+x+1); */
	/*   __m512d fcw_vec = _mm512_load_pd(temporal+id1+x-1); */
	/*   __m512d fcs_vec = _mm512_load_pd(temporal+id1+x+s); */
	/*   __m512d fcn_vec = _mm512_load_pd(temporal+id1+x+n); */
	/*   __m512d fcb_vec = _mm512_load_pd(temporal+id1+x+b); */
	/*   __m512d fct_vec = _mm512_load_pd(temporal+id1+x+t); */
	/*   __m512d tmp0,tmp1,tmp2; */
	/*   fc_vec  = _mm512_mul_pd(cc_vec,fc_vec); */
	/*   fce_vec = _mm512_mul_pd(ce_vec,fce_vec); */
	/*   fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec); */
	/*   fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec); */
	/*   fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec); */
	/*   fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec); */
	/*   fct_vec = _mm512_mul_pd(ct_vec,fct_vec); */
	/*   tmp0 = _mm512_add_pd(fce_vec,fcw_vec); */
	/*   tmp1 = _mm512_add_pd(fcn_vec,fcs_vec); */
	/*   tmp2 = _mm512_add_pd(fct_vec,fcb_vec); */
	/*   tmp0 = _mm512_add_pd(fc_vec, tmp0); */
	/*   tmp1 = _mm512_add_pd(tmp1,   tmp2); */
	/*   tmp0 = _mm512_add_pd(tmp0,   tmp1); */
	/*   _mm512_store_pd(f2_t+c+x,tmp0); */
	/* } */
	/* x = nx-SIMDLENGTH; */
	/* { */

	/*   REAL fcp1_arr[8] = {temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6],temporal[id1+x+7],temporal[id1+x+7]}; */
	/*   __m512d fc_vec  = _mm512_load_pd(temporal+id1+x); */
	/*   __m512d fce_vec = _mm512_load_pd(fcp1_arr); */
	/*   __m512d fcw_vec = _mm512_load_pd(temporal+id1+x-1); */
	/*   __m512d fcs_vec = _mm512_load_pd(temporal+id1+x+s); */
	/*   __m512d fcn_vec = _mm512_load_pd(temporal+id1+x+n); */
	/*   __m512d fcb_vec = _mm512_load_pd(temporal+id1+x+b); */
	/*   __m512d fct_vec = _mm512_load_pd(temporal+id1+x+t); */
	/*   __m512d tmp0,tmp1,tmp2; */
	/*   fc_vec  = _mm512_mul_pd(cc_vec,fc_vec); */
	/*   fce_vec = _mm512_mul_pd(ce_vec,fce_vec); */
	/*   fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec); */
	/*   fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec); */
	/*   fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec); */
	/*   fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec); */
	/*   fct_vec = _mm512_mul_pd(ct_vec,fct_vec); */
	/*   tmp0 = _mm512_add_pd(fce_vec,fcw_vec); */
	/*   tmp1 = _mm512_add_pd(fcn_vec,fcs_vec); */
	/*   tmp2 = _mm512_add_pd(fct_vec,fcb_vec); */
	/*   tmp0 = _mm512_add_pd(fc_vec, tmp0); */
	/*   tmp1 = _mm512_add_pd(tmp1,   tmp2); */
	/*   tmp0 = _mm512_add_pd(tmp0,   tmp1); */
	/*   _mm512_store_pd(f2_t+c+x,tmp0); */
	/* } */
	id1 += nx;
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
      b = (z == 0)    ? 0 : (izm-izc)*nx*tby;
      t = (z == nz-1) ? 0 : (izp-izc)*nx*tby;
      n = -nx;
      s =  nx;

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

      id2 = 0+iy*nx+izp*nx*tby+(step+1)*nx*tby*tbz;
      id1 = 0+iy*nx+izc*nx*tby+step*nx*tby*tbz;

      for (y = ystr; y < yend; y++) {
#if (YY == 0) || (YY == 3)
	n = (y == 0)    ? 0 : -nx;
#endif
#if (YY == 2) || (YY == 3)
	s = (y == ny-1) ? 0 :  nx;
#endif
	x = 0;
	__m512d fc_vec  = _mm512_load_pd(temporal+id1+x);
	__m512d fcp1_vec= _mm512_load_pd(temporal+id1+x+8);
	__m512d fcm1_vec= _mm512_alignr_epi64(fc_vec,fc_vec,1);
	__m512d fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
	__m512d fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
	__m512d fcs_vec = _mm512_load_pd(temporal+id1+x+s);
	__m512d fcn_vec = _mm512_load_pd(temporal+id1+x+n);
	__m512d fcb_vec = _mm512_load_pd(temporal+id1+x+b);
	__m512d fct_vec = _mm512_load_pd(temporal+id1+x+t);
	__m512d tmp0,tmp1,tmp2,tmp3;
	tmp3    = _mm512_mul_pd(cc_vec,fc_vec);
	fce_vec = _mm512_mul_pd(ce_vec,fce_vec);
	fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec);
	fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec);
	fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec);
	fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec);
	fct_vec = _mm512_mul_pd(ct_vec,fct_vec);
	tmp0 = _mm512_add_pd(fce_vec,fcw_vec);
	tmp1 = _mm512_add_pd(fcn_vec,fcs_vec);
	tmp2 = _mm512_add_pd(fct_vec,fcb_vec);
	tmp0 = _mm512_add_pd(tmp3,   tmp0);
	tmp1 = _mm512_add_pd(tmp1,   tmp2);
	tmp0 = _mm512_add_pd(tmp0,   tmp1);
	_mm512_store_pd(temporal+id2+x,tmp0);

	for (x = SIMDLENGTH; x < nx-SIMDLENGTH; x+=SIMDLENGTH) {
	  fcm1_vec = fc_vec;
	  fc_vec   = fcp1_vec;
	  fcp1_vec = _mm512_load_pd(temporal+id1+8+x);
	  
	  fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
	  fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
	  fcs_vec = _mm512_load_pd(temporal+id1+x+s);
	  fcn_vec = _mm512_load_pd(temporal+id1+x+n);
	  fcb_vec = _mm512_load_pd(temporal+id1+x+b);
	  fct_vec = _mm512_load_pd(temporal+id1+x+t);
	  tmp3    = _mm512_mul_pd(cc_vec,fc_vec);
	  fce_vec = _mm512_mul_pd(ce_vec,fce_vec);
	  fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec);
	  fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec);
	  fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec);
	  fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec);
	  fct_vec = _mm512_mul_pd(ct_vec,fct_vec);
	  tmp0 = _mm512_add_pd(fce_vec,fcw_vec);
	  tmp1 = _mm512_add_pd(fcn_vec,fcs_vec);
	  tmp2 = _mm512_add_pd(fct_vec,fcb_vec);
	  tmp0 = _mm512_add_pd(tmp3,   tmp0);
	  tmp1 = _mm512_add_pd(tmp1,   tmp2);
	  tmp0 = _mm512_add_pd(tmp0,   tmp1);
	  _mm512_store_pd(temporal+id2+x,tmp0);

	}
	x = nx-SIMDLENGTH;
	fcm1_vec = fc_vec;
	fc_vec   = fcp1_vec;
	fcp1_vec = _mm512_alignr_epi64(fc_vec,fc_vec,7);
	
	fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
	fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
	fcs_vec = _mm512_load_pd(temporal+id1+x+s);
	fcn_vec = _mm512_load_pd(temporal+id1+x+n);
	fcb_vec = _mm512_load_pd(temporal+id1+x+b);
	fct_vec = _mm512_load_pd(temporal+id1+x+t);
	
	tmp3    = _mm512_mul_pd(cc_vec,fc_vec);
	fce_vec = _mm512_mul_pd(ce_vec,fce_vec);
	fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec);
	fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec);
	fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec);
	fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec);
	fct_vec = _mm512_mul_pd(ct_vec,fct_vec);
	tmp0 = _mm512_add_pd(fce_vec,fcw_vec);
	tmp1 = _mm512_add_pd(fcn_vec,fcs_vec);
	tmp2 = _mm512_add_pd(fct_vec,fcb_vec);
	tmp0 = _mm512_add_pd(tmp3,   tmp0);
	tmp1 = _mm512_add_pd(tmp1,   tmp2);
	tmp0 = _mm512_add_pd(tmp0,   tmp1);
	_mm512_store_pd(temporal+id2+x,tmp0);
	/* { */
	/*   REAL fcm1_arr[SIMDLENGTH] = {temporal[id1+x],temporal[id1+x],temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6]}; */
	/*   __m512d fc_vec  = _mm512_load_pd(temporal+id1+x); */
	/*   __m512d fce_vec = _mm512_load_pd(temporal+id1+x+1); */
	/*   __m512d fcw_vec = _mm512_load_pd(fcm1_arr); */
	/*   __m512d fcs_vec = _mm512_load_pd(temporal+id1+x+s); */
	/*   __m512d fcn_vec = _mm512_load_pd(temporal+id1+x+n); */
	/*   __m512d fcb_vec = _mm512_load_pd(temporal+id1+x+b); */
	/*   __m512d fct_vec = _mm512_load_pd(temporal+id1+x+t); */
	/*   __m512d tmp0,tmp1,tmp2; */
	/*   fc_vec  = _mm512_mul_pd(cc_vec,fc_vec); */
	/*   fce_vec = _mm512_mul_pd(ce_vec,fce_vec); */
	/*   fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec); */
	/*   fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec); */
	/*   fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec); */
	/*   fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec); */
	/*   fct_vec = _mm512_mul_pd(ct_vec,fct_vec); */
	/*   tmp0 = _mm512_add_pd(fce_vec,fcw_vec); */
	/*   tmp1 = _mm512_add_pd(fcn_vec,fcs_vec); */
	/*   tmp2 = _mm512_add_pd(fct_vec,fcb_vec); */
	/*   tmp0 = _mm512_add_pd(fc_vec, tmp0); */
	/*   tmp1 = _mm512_add_pd(tmp1,   tmp2); */
	/*   tmp0 = _mm512_add_pd(tmp0,   tmp1); */
	/*   _mm512_store_pd(temporal+id2+x,tmp0); */
	/* } */
	/* for (x = SIMDLENGTH; x < nx-SIMDLENGTH; x+=SIMDLENGTH) { */
	/*   __m512d fc_vec  = _mm512_load_pd(temporal+id1+x); */
	/*   __m512d fce_vec = _mm512_load_pd(temporal+id1+x+1); */
	/*   __m512d fcw_vec = _mm512_load_pd(temporal+id1+x-1); */
	/*   __m512d fcs_vec = _mm512_load_pd(temporal+id1+x+s); */
	/*   __m512d fcn_vec = _mm512_load_pd(temporal+id1+x+n); */
	/*   __m512d fcb_vec = _mm512_load_pd(temporal+id1+x+b); */
	/*   __m512d fct_vec = _mm512_load_pd(temporal+id1+x+t); */
	/*   __m512d tmp0,tmp1,tmp2; */
	/*   fc_vec  = _mm512_mul_pd(cc_vec,fc_vec); */
	/*   fce_vec = _mm512_mul_pd(ce_vec,fce_vec); */
	/*   fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec); */
	/*   fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec); */
	/*   fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec); */
	/*   fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec); */
	/*   fct_vec = _mm512_mul_pd(ct_vec,fct_vec); */
	/*   tmp0 = _mm512_add_pd(fce_vec,fcw_vec); */
	/*   tmp1 = _mm512_add_pd(fcn_vec,fcs_vec); */
	/*   tmp2 = _mm512_add_pd(fct_vec,fcb_vec); */
	/*   tmp0 = _mm512_add_pd(fc_vec, tmp0); */
	/*   tmp1 = _mm512_add_pd(tmp1,   tmp2); */
	/*   tmp0 = _mm512_add_pd(tmp0,   tmp1); */
	/*   _mm512_store_pd(temporal+id2+x,tmp0); */
	/* } */
	/* x = nx-SIMDLENGTH; */
	/* { */

	/*   REAL fcp1_arr[8] = {temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6],temporal[id1+x+7],temporal[id1+x+7]}; */
	/*   __m512d fc_vec  = _mm512_load_pd(temporal+id1+x); */
	/*   __m512d fce_vec = _mm512_load_pd(fcp1_arr); */
	/*   __m512d fcw_vec = _mm512_load_pd(temporal+id1+x-1); */
	/*   __m512d fcs_vec = _mm512_load_pd(temporal+id1+x+s); */
	/*   __m512d fcn_vec = _mm512_load_pd(temporal+id1+x+n); */
	/*   __m512d fcb_vec = _mm512_load_pd(temporal+id1+x+b); */
	/*   __m512d fct_vec = _mm512_load_pd(temporal+id1+x+t); */
	/*   __m512d tmp0,tmp1,tmp2; */
	/*   fc_vec  = _mm512_mul_pd(cc_vec,fc_vec); */
	/*   fce_vec = _mm512_mul_pd(ce_vec,fce_vec); */
	/*   fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec); */
	/*   fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec); */
	/*   fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec); */
	/*   fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec); */
	/*   fct_vec = _mm512_mul_pd(ct_vec,fct_vec); */
	/*   tmp0 = _mm512_add_pd(fce_vec,fcw_vec); */
	/*   tmp1 = _mm512_add_pd(fcn_vec,fcs_vec); */
	/*   tmp2 = _mm512_add_pd(fct_vec,fcb_vec); */
	/*   tmp0 = _mm512_add_pd(fc_vec, tmp0); */
	/*   tmp1 = _mm512_add_pd(tmp1,   tmp2); */
	/*   tmp0 = _mm512_add_pd(tmp0,   tmp1); */
	/*   _mm512_store_pd(temporal+id2+x,tmp0); */
	/* } */
	id2 += nx;
	id1 += nx;
      }

      step++;

    }

    {
      z = zz;
      b = (z == 0)    ? 0 : (izm-izc)*nx*tby;
      t = (z == nz-1) ? 0 : (izp-izc)*nx*tby;
      n = -nx;
      s =  nx;
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

      id1 = 0+iy*nx+izc*nx*tby+step*nx*tby*tbz;
      for (y = ystr; y < yend; y++) {
#if (YY == 0) || (YY == 3)
	n = (y == 0)    ? 0 : -nx;
#endif
#if (YY == 2) || (YY == 3)
	s = (y == ny-1) ? 0 :  nx;
#endif
	x = 0;
	c = x + y*nx + z*nx*ny;
	__m512d fc_vec  = _mm512_load_pd(temporal+id1+x);
	__m512d fcp1_vec= _mm512_load_pd(temporal+id1+x+8);
	__m512d fcm1_vec= _mm512_alignr_epi64(fc_vec,fc_vec,1);
	__m512d fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
	__m512d fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
	__m512d fcs_vec = _mm512_load_pd(temporal+id1+x+s);
	__m512d fcn_vec = _mm512_load_pd(temporal+id1+x+n);
	__m512d fcb_vec = _mm512_load_pd(temporal+id1+x+b);
	__m512d fct_vec = _mm512_load_pd(temporal+id1+x+t);
	__m512d tmp0,tmp1,tmp2,tmp3;
	tmp3    = _mm512_mul_pd(cc_vec,fc_vec);
	fce_vec = _mm512_mul_pd(ce_vec,fce_vec);
	fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec);
	fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec);
	fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec);
	fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec);
	fct_vec = _mm512_mul_pd(ct_vec,fct_vec);
	tmp0 = _mm512_add_pd(fce_vec,fcw_vec);
	tmp1 = _mm512_add_pd(fcn_vec,fcs_vec);
	tmp2 = _mm512_add_pd(fct_vec,fcb_vec);
	tmp0 = _mm512_add_pd(tmp3,   tmp0);
	tmp1 = _mm512_add_pd(tmp1,   tmp2);
	tmp0 = _mm512_add_pd(tmp0,   tmp1);
	_mm512_store_pd(f2_t+c+x,tmp0);

	for (x = SIMDLENGTH; x < nx-SIMDLENGTH; x+=SIMDLENGTH) {
	  fcm1_vec = fc_vec;
	  fc_vec   = fcp1_vec;
	  fcp1_vec = _mm512_load_pd(temporal+id1+8+x);
	  
	  fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
	  fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
	  fcs_vec = _mm512_load_pd(temporal+id1+x+s);
	  fcn_vec = _mm512_load_pd(temporal+id1+x+n);
	  fcb_vec = _mm512_load_pd(temporal+id1+x+b);
	  fct_vec = _mm512_load_pd(temporal+id1+x+t);
	  tmp3    = _mm512_mul_pd(cc_vec,fc_vec);
	  fce_vec = _mm512_mul_pd(ce_vec,fce_vec);
	  fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec);
	  fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec);
	  fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec);
	  fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec);
	  fct_vec = _mm512_mul_pd(ct_vec,fct_vec);
	  tmp0 = _mm512_add_pd(fce_vec,fcw_vec);
	  tmp1 = _mm512_add_pd(fcn_vec,fcs_vec);
	  tmp2 = _mm512_add_pd(fct_vec,fcb_vec);
	  tmp0 = _mm512_add_pd(tmp3,   tmp0);
	  tmp1 = _mm512_add_pd(tmp1,   tmp2);
	  tmp0 = _mm512_add_pd(tmp0,   tmp1);
	  _mm512_store_pd(f2_t+c+x,tmp0);

	}
	x = nx-SIMDLENGTH;
	fcm1_vec = fc_vec;
	fc_vec   = fcp1_vec;
	fcp1_vec = _mm512_alignr_epi64(fc_vec,fc_vec,7);
	
	fcw_vec = _mm512_alignr_epi64(fc_vec,fcm1_vec,7);
	fce_vec = _mm512_alignr_epi64(fcp1_vec,fc_vec,1);
	fcs_vec = _mm512_load_pd(temporal+id1+x+s);
	fcn_vec = _mm512_load_pd(temporal+id1+x+n);
	fcb_vec = _mm512_load_pd(temporal+id1+x+b);
	fct_vec = _mm512_load_pd(temporal+id1+x+t);
	
	tmp3    = _mm512_mul_pd(cc_vec,fc_vec);
	fce_vec = _mm512_mul_pd(ce_vec,fce_vec);
	fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec);
	fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec);
	fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec);
	fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec);
	fct_vec = _mm512_mul_pd(ct_vec,fct_vec);
	tmp0 = _mm512_add_pd(fce_vec,fcw_vec);
	tmp1 = _mm512_add_pd(fcn_vec,fcs_vec);
	tmp2 = _mm512_add_pd(fct_vec,fcb_vec);
	tmp0 = _mm512_add_pd(tmp3,   tmp0);
	tmp1 = _mm512_add_pd(tmp1,   tmp2);
	tmp0 = _mm512_add_pd(tmp0,   tmp1);
	_mm512_store_pd(f2_t+c+x,tmp0);
	/* { */
	/*   REAL fcm1_arr[SIMDLENGTH] = {temporal[id1+x],temporal[id1+x],temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6]}; */
	/*   __m512d fc_vec  = _mm512_load_pd(temporal+id1+x); */
	/*   __m512d fce_vec = _mm512_load_pd(temporal+id1+x+1); */
	/*   __m512d fcw_vec = _mm512_load_pd(fcm1_arr); */
	/*   __m512d fcs_vec = _mm512_load_pd(temporal+id1+x+s); */
	/*   __m512d fcn_vec = _mm512_load_pd(temporal+id1+x+n); */
	/*   __m512d fcb_vec = _mm512_load_pd(temporal+id1+x+b); */
	/*   __m512d fct_vec = _mm512_load_pd(temporal+id1+x+t); */
	/*   __m512d tmp0,tmp1,tmp2; */
	/*   fc_vec  = _mm512_mul_pd(cc_vec,fc_vec); */
	/*   fce_vec = _mm512_mul_pd(ce_vec,fce_vec); */
	/*   fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec); */
	/*   fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec); */
	/*   fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec); */
	/*   fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec); */
	/*   fct_vec = _mm512_mul_pd(ct_vec,fct_vec); */
	/*   tmp0 = _mm512_add_pd(fce_vec,fcw_vec); */
	/*   tmp1 = _mm512_add_pd(fcn_vec,fcs_vec); */
	/*   tmp2 = _mm512_add_pd(fct_vec,fcb_vec); */
	/*   tmp0 = _mm512_add_pd(fc_vec, tmp0); */
	/*   tmp1 = _mm512_add_pd(tmp1,   tmp2); */
	/*   tmp0 = _mm512_add_pd(tmp0,   tmp1); */
	/*   _mm512_store_pd(f2_t+c+x,tmp0); */
	/* } */
	/* for (x = SIMDLENGTH; x < nx-SIMDLENGTH; x+=SIMDLENGTH) { */
	/*   __m512d fc_vec  = _mm512_load_pd(temporal+id1+x); */
	/*   __m512d fce_vec = _mm512_load_pd(temporal+id1+x+1); */
	/*   __m512d fcw_vec = _mm512_load_pd(temporal+id1+x-1); */
	/*   __m512d fcs_vec = _mm512_load_pd(temporal+id1+x+s); */
	/*   __m512d fcn_vec = _mm512_load_pd(temporal+id1+x+n); */
	/*   __m512d fcb_vec = _mm512_load_pd(temporal+id1+x+b); */
	/*   __m512d fct_vec = _mm512_load_pd(temporal+id1+x+t); */
	/*   __m512d tmp0,tmp1,tmp2; */
	/*   fc_vec  = _mm512_mul_pd(cc_vec,fc_vec); */
	/*   fce_vec = _mm512_mul_pd(ce_vec,fce_vec); */
	/*   fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec); */
	/*   fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec); */
	/*   fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec); */
	/*   fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec); */
	/*   fct_vec = _mm512_mul_pd(ct_vec,fct_vec); */
	/*   tmp0 = _mm512_add_pd(fce_vec,fcw_vec); */
	/*   tmp1 = _mm512_add_pd(fcn_vec,fcs_vec); */
	/*   tmp2 = _mm512_add_pd(fct_vec,fcb_vec); */
	/*   tmp0 = _mm512_add_pd(fc_vec, tmp0); */
	/*   tmp1 = _mm512_add_pd(tmp1,   tmp2); */
	/*   tmp0 = _mm512_add_pd(tmp0,   tmp1); */
	/*   _mm512_store_pd(f2_t+c+x,tmp0); */
	/* } */
	/* x = nx-SIMDLENGTH; */
	/* { */

	/*   REAL fcp1_arr[8] = {temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6],temporal[id1+x+7],temporal[id1+x+7]}; */
	/*   __m512d fc_vec  = _mm512_load_pd(temporal+id1+x); */
	/*   __m512d fce_vec = _mm512_load_pd(fcp1_arr); */
	/*   __m512d fcw_vec = _mm512_load_pd(temporal+id1+x-1); */
	/*   __m512d fcs_vec = _mm512_load_pd(temporal+id1+x+s); */
	/*   __m512d fcn_vec = _mm512_load_pd(temporal+id1+x+n); */
	/*   __m512d fcb_vec = _mm512_load_pd(temporal+id1+x+b); */
	/*   __m512d fct_vec = _mm512_load_pd(temporal+id1+x+t); */
	/*   __m512d tmp0,tmp1,tmp2; */
	/*   fc_vec  = _mm512_mul_pd(cc_vec,fc_vec); */
	/*   fce_vec = _mm512_mul_pd(ce_vec,fce_vec); */
	/*   fcw_vec = _mm512_mul_pd(cw_vec,fcw_vec); */
	/*   fcn_vec = _mm512_mul_pd(cn_vec,fcn_vec); */
	/*   fcs_vec = _mm512_mul_pd(cs_vec,fcs_vec); */
	/*   fcb_vec = _mm512_mul_pd(cb_vec,fcb_vec); */
	/*   fct_vec = _mm512_mul_pd(ct_vec,fct_vec); */
	/*   tmp0 = _mm512_add_pd(fce_vec,fcw_vec); */
	/*   tmp1 = _mm512_add_pd(fcn_vec,fcs_vec); */
	/*   tmp2 = _mm512_add_pd(fct_vec,fcb_vec); */
	/*   tmp0 = _mm512_add_pd(fc_vec, tmp0); */
	/*   tmp1 = _mm512_add_pd(tmp1,   tmp2); */
	/*   tmp0 = _mm512_add_pd(tmp0,   tmp1); */
	/*   _mm512_store_pd(f2_t+c+x,tmp0); */
	/* } */
	id1 += nx;
      }
    }
    
    tmp = izm;
    izm = izc;
    izc = izp;
    izp = tmp;
  }
}
