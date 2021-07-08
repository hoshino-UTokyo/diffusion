
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
      {
	c = x + y*nx + z*nx*ny;
	float64_t fcm1_arr[SIMDLENGTH] = {f1_t[c+x],f1_t[c+x],f1_t[c+x+1],f1_t[c+x+2],f1_t[c+x+3],f1_t[c+x+4],f1_t[c+x+5],f1_t[c+x+6]};
	svfloat64_t fc_vec  = svld1(pg,(float64_t*)&f1_t[c+x]);
	svfloat64_t fce_vec = svld1(pg,(float64_t*)&f1_t[c+x+1]);
	svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
	svfloat64_t fcs_vec = svld1(pg,(float64_t*)&f1_t[c+x+s0]);
	svfloat64_t fcn_vec = svld1(pg,(float64_t*)&f1_t[c+x+n0]);
	svfloat64_t fcb_vec = svld1(pg,(float64_t*)&f1_t[c+x+b0]);
	svfloat64_t fct_vec = svld1(pg,(float64_t*)&f1_t[c+x+t0]);
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
	svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
      }
      for (x = SIMDLENGTH; x < nx-SIMDLENGTH; x+=SIMDLENGTH) {
      	svfloat64_t fc_vec  = svld1(pg,(float64_t*)&f1_t[c+x]);
      	svfloat64_t fce_vec = svld1(pg,(float64_t*)&f1_t[c+x+1]);
      	svfloat64_t fcw_vec = svld1(pg,(float64_t*)&f1_t[c+x-1]);
      	svfloat64_t fcs_vec = svld1(pg,(float64_t*)&f1_t[c+x+s0]);
      	svfloat64_t fcn_vec = svld1(pg,(float64_t*)&f1_t[c+x+n0]);
      	svfloat64_t fcb_vec = svld1(pg,(float64_t*)&f1_t[c+x+b0]);
      	svfloat64_t fct_vec = svld1(pg,(float64_t*)&f1_t[c+x+t0]);
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
      	svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
      }
      int i = 0;
      //      int remainder = (nx & (SIMDLENGTH-1) == 0) ? SIMDLENGTH : nx & (SIMDLENGTH-1);
      int remainder = SIMDLENGTH;
      svbool_t pgt = svwhilelt_b64(i,remainder);
      {
	float64_t fcp1_arr[SIMDLENGTH];
	for(int i = 0; i < remainder; i++){
	  if(i == remainder-1){
	    fcp1_arr[i] = f1_t[c+x+i];
	  }else{
	    fcp1_arr[i] = f1_t[c+x+1+i];
	  }
	}
	svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
      	svfloat64_t fc_vec  = svld1(pgt,(float64_t*)&f1_t[c+x]);
      	svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&f1_t[c+x-1]);
      	svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&f1_t[c+x+s0]);
      	svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&f1_t[c+x+n0]);
      	svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&f1_t[c+x+b0]);
      	svfloat64_t fct_vec = svld1(pgt,(float64_t*)&f1_t[c+x+t0]);
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
	svst1(pgt,(float64_t*)&temporal[id2+x],tmp0);
      }
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
	{
	  float64_t fcm1_arr[SIMDLENGTH] = {temporal[id1+x],temporal[id1+x],temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6]};
	  svfloat64_t fc_vec  = svld1(pg,(float64_t*)&temporal[id1+x]);
	  svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
	  svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
	  svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
	  svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
	  svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
	  svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
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
	  svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
	}
	for (x = SIMDLENGTH; x < nx-SIMDLENGTH; x+=SIMDLENGTH) {
	  svfloat64_t fc_vec  = svld1(pg,(float64_t*)&temporal[id1+x]);
	  svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
	  svfloat64_t fcw_vec = svld1(pg,(float64_t*)&temporal[id1+x-1]);
	  svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
	  svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
	  svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
	  svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
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
	  svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
	}
	x = nx-SIMDLENGTH;
	int i = 0;
	//	int remainder = (nx & (SIMDLENGTH-1) == 0) ? SIMDLENGTH : nx & (SIMDLENGTH-1);
	int remainder = SIMDLENGTH;
	svbool_t pgt = svwhilelt_b64(i,remainder);
	{

	  float64_t fcp1_arr[SIMDLENGTH];
	  for(int i = 0; i < remainder; i++){
	    if(i == remainder-1){
	      fcp1_arr[i] = temporal[id1+x+i];
	    }else{
	      fcp1_arr[i] = temporal[id1+x+1+i];
	    }
	  }
	  svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
	  svfloat64_t fc_vec  = svld1(pgt,(float64_t*)&temporal[id1+x]);
	  svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&temporal[id1+x-1]);
	  svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&temporal[id1+x+s]);
	  svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&temporal[id1+x+n]);
	  svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&temporal[id1+x+b]);
	  svfloat64_t fct_vec = svld1(pgt,(float64_t*)&temporal[id1+x+t]);
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
	  svst1(pgt,(float64_t*)&temporal[id2+x],tmp0);
	}
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
      {
	c = x + y*nx + z*nx*ny;
	float64_t fcm1_arr[SIMDLENGTH] = {f1_t[c+x],f1_t[c+x],f1_t[c+x+1],f1_t[c+x+2],f1_t[c+x+3],f1_t[c+x+4],f1_t[c+x+5],f1_t[c+x+6]};
	svfloat64_t fc_vec  = svld1(pg,(float64_t*)&f1_t[c+x]);
	svfloat64_t fce_vec = svld1(pg,(float64_t*)&f1_t[c+x+1]);
	svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
	svfloat64_t fcs_vec = svld1(pg,(float64_t*)&f1_t[c+x+s0]);
	svfloat64_t fcn_vec = svld1(pg,(float64_t*)&f1_t[c+x+n0]);
	svfloat64_t fcb_vec = svld1(pg,(float64_t*)&f1_t[c+x+b0]);
	svfloat64_t fct_vec = svld1(pg,(float64_t*)&f1_t[c+x+t0]);
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
	svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
      }
      for (x = SIMDLENGTH; x < nx-SIMDLENGTH; x+=SIMDLENGTH) {
      	svfloat64_t fc_vec  = svld1(pg,(float64_t*)&f1_t[c+x]);
      	svfloat64_t fce_vec = svld1(pg,(float64_t*)&f1_t[c+x+1]);
      	svfloat64_t fcw_vec = svld1(pg,(float64_t*)&f1_t[c+x-1]);
      	svfloat64_t fcs_vec = svld1(pg,(float64_t*)&f1_t[c+x+s0]);
      	svfloat64_t fcn_vec = svld1(pg,(float64_t*)&f1_t[c+x+n0]);
      	svfloat64_t fcb_vec = svld1(pg,(float64_t*)&f1_t[c+x+b0]);
      	svfloat64_t fct_vec = svld1(pg,(float64_t*)&f1_t[c+x+t0]);
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
      	svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
      }
      x = nx-SIMDLENGTH;
      int i = 0;
      int remainder = SIMDLENGTH;
      //      int remainder = (nx & (SIMDLENGTH-1) == 0) ? SIMDLENGTH : nx & (SIMDLENGTH-1);
      svbool_t pgt = svwhilelt_b64(i,remainder);
      {
	float64_t fcp1_arr[SIMDLENGTH];
	for(int i = 0; i < remainder; i++){
	  if(i == remainder-1){
	    fcp1_arr[i] = f1_t[c+x+i];
	  }else{
	    fcp1_arr[i] = f1_t[c+x+1+i];
	  }
	}
	svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
      	svfloat64_t fc_vec  = svld1(pgt,(float64_t*)&f1_t[c+x]);
      	svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&f1_t[c+x-1]);
      	svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&f1_t[c+x+s0]);
      	svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&f1_t[c+x+n0]);
      	svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&f1_t[c+x+b0]);
      	svfloat64_t fct_vec = svld1(pgt,(float64_t*)&f1_t[c+x+t0]);
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
	svst1(pgt,(float64_t*)&temporal[id2+x],tmp0);
      }
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
	{
	  float64_t fcm1_arr[SIMDLENGTH] = {temporal[id1+x],temporal[id1+x],temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6]};
	  svfloat64_t fc_vec  = svld1(pg,(float64_t*)&temporal[id1+x]);
	  svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
	  svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
	  svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
	  svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
	  svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
	  svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
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
	  svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
	  /* id2+=SIMDLENGTH; */
	  /* id1+=SIMDLENGTH; */
	}

	for (x = SIMDLENGTH; x < nx-SIMDLENGTH; x+=SIMDLENGTH) {
	  svfloat64_t fc_vec  = svld1(pg,(float64_t*)&temporal[id1+x]);
	  svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
	  svfloat64_t fcw_vec = svld1(pg,(float64_t*)&temporal[id1+x-1]);
	  svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
	  svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
	  svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
	  svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
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
	  svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
	  /* id2+=SIMDLENGTH; */
	  /* id1+=SIMDLENGTH; */
	}
	x = nx-SIMDLENGTH;
	int remainder = SIMDLENGTH;
	int i = 0;
	//	int remainder = (nx & (SIMDLENGTH-1) == 0) ? SIMDLENGTH : nx & (SIMDLENGTH-1);
	svbool_t pgt = svwhilelt_b64(i,remainder);
	{
	  float64_t fcp1_arr[SIMDLENGTH];
	  for(int i = 0; i < remainder; i++){
	    if(i == remainder-1){
	      fcp1_arr[i] = temporal[id1+x+i];
	    }else{
	      fcp1_arr[i] = temporal[id1+x+1+i];
	    }
	  }
	  svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
	  svfloat64_t fc_vec  = svld1(pgt,(float64_t*)&temporal[id1+x]);
	  svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&temporal[id1+x-1]);
	  svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&temporal[id1+x+s]);
	  svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&temporal[id1+x+n]);
	  svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&temporal[id1+x+b]);
	  svfloat64_t fct_vec = svld1(pgt,(float64_t*)&temporal[id1+x+t]);
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
	  svst1(pgt,(float64_t*)&temporal[id2+x],tmp0);
	}
	/* id2 = id2p; */
	/* id1 = id1p; */
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
	{
	  float64_t fcm1_arr[SIMDLENGTH] = {temporal[id1+x],temporal[id1+x],temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6]};
	  svfloat64_t fc_vec  = svld1(pg,(float64_t*)&temporal[id1+x]);
	  svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
	  svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
	  svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
	  svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
	  svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
	  svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
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
	  svst1(pg,(float64_t*)&f2_t[c+x],tmp0);

	}

	for (x = SIMDLENGTH; x < nx-SIMDLENGTH; x+=SIMDLENGTH) {
	  svfloat64_t fc_vec  = svld1(pg,(float64_t*)&temporal[id1+x]);
	  svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
	  svfloat64_t fcw_vec = svld1(pg,(float64_t*)&temporal[id1+x-1]);
	  svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
	  svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
	  svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
	  svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
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
	  svst1(pg,(float64_t*)&f2_t[c+x],tmp0);
	}
	x = nx-SIMDLENGTH;
	int i = 0;
	int remainder = SIMDLENGTH;
	//	int remainder = (nx & (SIMDLENGTH-1) == 0) ? SIMDLENGTH : nx & (SIMDLENGTH-1);
	svbool_t pgt = svwhilelt_b64(i,remainder);
	{
	  float64_t fcp1_arr[SIMDLENGTH];
	  for(int i = 0; i < remainder; i++){
	    if(i == remainder-1){
	      fcp1_arr[i] = temporal[id1+x+i];
	    }else{
	      fcp1_arr[i] = temporal[id1+x+1+i];
	    }
	  }
	  svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
	  svfloat64_t fc_vec  = svld1(pgt,(float64_t*)&temporal[id1+x]);
	  svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&temporal[id1+x-1]);
	  svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&temporal[id1+x+s]);
	  svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&temporal[id1+x+n]);
	  svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&temporal[id1+x+b]);
	  svfloat64_t fct_vec = svld1(pgt,(float64_t*)&temporal[id1+x+t]);
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
	  svst1(pgt,(float64_t*)&f2_t[c+x],tmp0);

	}
	
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
	{
	  float64_t fcm1_arr[SIMDLENGTH] = {temporal[id1+x],temporal[id1+x],temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6]};
	  svfloat64_t fc_vec  = svld1(pg,(float64_t*)&temporal[id1+x]);
	  svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
	  svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
	  svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
	  svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
	  svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
	  svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
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
	  svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
	}

	for (x = SIMDLENGTH; x < nx-SIMDLENGTH; x+=SIMDLENGTH) {
	  svfloat64_t fc_vec  = svld1(pg,(float64_t*)&temporal[id1+x]);
	  svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
	  svfloat64_t fcw_vec = svld1(pg,(float64_t*)&temporal[id1+x-1]);
	  svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
	  svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
	  svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
	  svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
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
	  svst1(pg,(float64_t*)&temporal[id2+x],tmp0);
	}
	x = nx-SIMDLENGTH;
	int i = 0;
	int remainder = SIMDLENGTH;
	//	int remainder = (nx & (SIMDLENGTH-1) == 0) ? SIMDLENGTH : nx & (SIMDLENGTH-1);
	svbool_t pgt = svwhilelt_b64(i,remainder);
	{
	  float64_t fcp1_arr[SIMDLENGTH];
	  for(int i = 0; i < remainder; i++){
	    if(i == remainder-1){
	      fcp1_arr[i] = temporal[id1+x+i];
	    }else{
	      fcp1_arr[i] = temporal[id1+x+1+i];
	    }
	  }
	  svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
	  svfloat64_t fc_vec  = svld1(pgt,(float64_t*)&temporal[id1+x]);
	  svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&temporal[id1+x-1]);
	  svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&temporal[id1+x+s]);
	  svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&temporal[id1+x+n]);
	  svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&temporal[id1+x+b]);
	  svfloat64_t fct_vec = svld1(pgt,(float64_t*)&temporal[id1+x+t]);
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
	  svst1(pgt,(float64_t*)&temporal[id2+x],tmp0);
	}
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
	{
	  float64_t fcm1_arr[SIMDLENGTH] = {temporal[id1+x],temporal[id1+x],temporal[id1+x+1],temporal[id1+x+2],temporal[id1+x+3],temporal[id1+x+4],temporal[id1+x+5],temporal[id1+x+6]};
	  svfloat64_t fc_vec  = svld1(pg,(float64_t*)&temporal[id1+x]);
	  svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
	  svfloat64_t fcw_vec = svld1(pg,(float64_t*)&fcm1_arr[0]);
	  svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
	  svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
	  svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
	  svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
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
	  svst1(pg,(float64_t*)&f2_t[c+x],tmp0);

	}

	for (x = SIMDLENGTH; x < nx-SIMDLENGTH; x+=SIMDLENGTH) {
	  svfloat64_t fc_vec  = svld1(pg,(float64_t*)&temporal[id1+x]);
	  svfloat64_t fce_vec = svld1(pg,(float64_t*)&temporal[id1+x+1]);
	  svfloat64_t fcw_vec = svld1(pg,(float64_t*)&temporal[id1+x-1]);
	  svfloat64_t fcs_vec = svld1(pg,(float64_t*)&temporal[id1+x+s]);
	  svfloat64_t fcn_vec = svld1(pg,(float64_t*)&temporal[id1+x+n]);
	  svfloat64_t fcb_vec = svld1(pg,(float64_t*)&temporal[id1+x+b]);
	  svfloat64_t fct_vec = svld1(pg,(float64_t*)&temporal[id1+x+t]);
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
	  svst1(pg,(float64_t*)&f2_t[c+x],tmp0);
	}
	x = nx-SIMDLENGTH;
	int i = 0;
	int remainder = SIMDLENGTH;
	//	int remainder = (nx & (SIMDLENGTH-1) == 0) ? SIMDLENGTH : nx & (SIMDLENGTH-1);
	svbool_t pgt = svwhilelt_b64(i,remainder);
	{

	  float64_t fcp1_arr[SIMDLENGTH];
	  for(int i = 0; i < remainder; i++){
	    if(i == remainder-1){
	      fcp1_arr[i] = temporal[id1+x+i];
	    }else{
	      fcp1_arr[i] = temporal[id1+x+1+i];
	    }
	  }
	  svfloat64_t fce_vec = svld1(pgt,(float64_t*)&fcp1_arr[0]);
	  svfloat64_t fc_vec  = svld1(pgt,(float64_t*)&temporal[id1+x]);
	  svfloat64_t fcw_vec = svld1(pgt,(float64_t*)&temporal[id1+x-1]);
	  svfloat64_t fcs_vec = svld1(pgt,(float64_t*)&temporal[id1+x+s]);
	  svfloat64_t fcn_vec = svld1(pgt,(float64_t*)&temporal[id1+x+n]);
	  svfloat64_t fcb_vec = svld1(pgt,(float64_t*)&temporal[id1+x+b]);
	  svfloat64_t fct_vec = svld1(pgt,(float64_t*)&temporal[id1+x+t]);
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
	  svst1(pgt,(float64_t*)&f2_t[c+x],tmp0);

	}
	id1 += nx;
      }
    }
    
    tmp = izm;
    izm = izc;
    izc = izp;
    izp = tmp;
  }
}
