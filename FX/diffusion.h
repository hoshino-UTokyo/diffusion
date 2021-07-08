#ifndef DIFFUSION_H
#define DIFFUSION_H

#ifndef REAL
#define REAL double
#endif

#include "diffusion_ker00.h"
#include "diffusion_ker01.h"
#include "diffusion_ker02.h"
#include "diffusion_ker03.h"
#include "diffusion_ker04.h"
#include "diffusion_ker05.h"
#include "diffusion_ker06.h"
#include "diffusion_ker07.h"
#include "diffusion_ker08.h"
#include "diffusion_ker09.h"
#include "diffusion_ker10.h"
#include "diffusion_ker11.h"
#include "diffusion_ker12.h"
#include "diffusion_ker13.h"
#include "diffusion_ker14.h"
#include "diffusion_ker15.h"
#include "diffusion_ker16.h"
#include "diffusion_ker17.h"
#include "diffusion_ker18.h"
#include "diffusion_ker19.h"
#include "diffusion_ker20.h"
#include "diffusion_ker21.h"
#include "diffusion_ker22.h"
#include "diffusion_ker23.h"
#include "diffusion_ker24.h"
#include "diffusion_ker25.h"
#include "diffusion_ker26.h"
#include "diffusion_ker27.h"
#include "diffusion_ker28.h"
#include "diffusion_ker29.h"
#include "diffusion_ker30.h"
#include "diffusion_ker31.h"
#include "diffusion_ker32.h"
#include "diffusion_ker33.h"
#include "diffusion_ker34.h"
#include "diffusion_ker35.h"
#include "diffusion_ker36.h"
#include "diffusion_ker37.h"
#include "diffusion_ker38.h"
#include "diffusion_ker39.h"

static void (*allocate[])(REAL **buff_ret, const int nx, const int ny, const int nz)
  = {allocate_ker00
     ,allocate_ker01
     ,allocate_ker02
     ,allocate_ker03
     ,allocate_ker04
     ,allocate_ker05
     ,allocate_ker06
     ,allocate_ker07
     ,allocate_ker08
     ,allocate_ker09
     ,allocate_ker10
     ,allocate_ker11
     ,allocate_ker12
     ,allocate_ker13
     ,allocate_ker14
     ,allocate_ker15
     ,allocate_ker16
     ,allocate_ker17
     ,allocate_ker18
     ,allocate_ker19
     ,allocate_ker20
     ,allocate_ker21
     ,allocate_ker22
     ,allocate_ker23
     ,allocate_ker24
     ,allocate_ker25
     ,allocate_ker26
     ,allocate_ker27
     ,allocate_ker28
     ,allocate_ker29
     ,allocate_ker30
     ,allocate_ker31
     ,allocate_ker32
     ,allocate_ker33
     ,allocate_ker34
     ,allocate_ker35
     ,allocate_ker36
     ,allocate_ker37
     ,allocate_ker38
     ,allocate_ker39
};

static void (*init[])(REAL *buff1, const int nx, const int ny, const int nz,
		      const REAL kx, const REAL ky, const REAL kz,
		      const REAL dx, const REAL dy, const REAL dz,
		      const REAL kappa, const REAL time)
  = {init_ker00
     ,init_ker01
     ,init_ker02
     ,init_ker03
     ,init_ker04
     ,init_ker05
     ,init_ker06
     ,init_ker07
     ,init_ker08
     ,init_ker09
     ,init_ker10
     ,init_ker11
     ,init_ker12
     ,init_ker13
     ,init_ker14
     ,init_ker15
     ,init_ker16
     ,init_ker17
     ,init_ker18
     ,init_ker19
     ,init_ker20
     ,init_ker21
     ,init_ker22
     ,init_ker23
     ,init_ker24
     ,init_ker25
     ,init_ker26
     ,init_ker27
     ,init_ker28
     ,init_ker29
     ,init_ker30
     ,init_ker31
     ,init_ker32
     ,init_ker33
     ,init_ker34
     ,init_ker35
     ,init_ker36
     ,init_ker37
     ,init_ker38
     ,init_ker39
};

static void (*diffusion[])(REAL *restrict f1, REAL *restrict f2, int nx, int ny, int nz,
			   REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
			   REAL cb, REAL cc, REAL dt,
			   REAL **f1_ret, REAL **f2_ret, REAL *time_ret, int *count_ret) 
  = {diffusion_ker00
     ,diffusion_ker01
     ,diffusion_ker02
     ,diffusion_ker03
     ,diffusion_ker04
     ,diffusion_ker05
     ,diffusion_ker06
     ,diffusion_ker07
     ,diffusion_ker08
     ,diffusion_ker09
     ,diffusion_ker10
     ,diffusion_ker11
     ,diffusion_ker12
     ,diffusion_ker13
     ,diffusion_ker14
     ,diffusion_ker15
     ,diffusion_ker16
     ,diffusion_ker17
     ,diffusion_ker18
     ,diffusion_ker19
     ,diffusion_ker20
     ,diffusion_ker21
     ,diffusion_ker22
     ,diffusion_ker23
     ,diffusion_ker24
     ,diffusion_ker25
     ,diffusion_ker26
     ,diffusion_ker27
     ,diffusion_ker28
     ,diffusion_ker29
     ,diffusion_ker30
     ,diffusion_ker31
     ,diffusion_ker32
     ,diffusion_ker33
     ,diffusion_ker34
     ,diffusion_ker35
     ,diffusion_ker36
     ,diffusion_ker37
     ,diffusion_ker38
     ,diffusion_ker39
};

static char *available[] =
  {"00"
   ,"01"
   ,"02"
   ,"03"
   ,"04"
   ,"05"
   ,"06"
   ,"07"
   ,"08"
   ,"09"
   ,"10"
   ,"11"
   ,"12"
   ,"13"
   ,"14"
   ,"15"
   ,"16"
   ,"17"
   ,"18"
   ,"19"
   ,"20"
   ,"21"
   ,"22"
   ,"23"
   ,"24"
   ,"25"
   ,"26"
   ,"27"
   ,"28"
   ,"29"
   ,"30"
   ,"31"
   ,"32"
   ,"33"
   ,"34"
   ,"35"
   ,"36"
   ,"37"
   ,"38"
  };

static char *opt_list[] =
  {"firstTouch"
   ,"memAlign"
   ,"peeling"
   ,"intrinsic"
   ,"yDim"
   ,"zyDim"
   ,"xyDim"
   ,"L1blocking"
   ,"L2blocking"
   ,"unrole"
   ,"reduceInt"
   ,"woFMA"
   ,"binFMA"
   ,"accessChange"
   ,"registerBlocking"
   ,"registerBlocking2"
   ,"load2vec(cancel)"
   ,"SWprefetch"
   ,"TB"
   ,"Yunroll"
   ,"local"
   ,"Zunroll"
   ,"Z4unroll"
  };

static int opt_flags[][30] =
  {{ 0,1,0,0}                 // 0
   ,{1,1,0,0}                 // 1
   ,{1,1,1,0}                 // 2
   ,{1,1,1,1}                 // 3
   ,{1,1,1,1,1}               // 4
   ,{1,1,1,1,0,1}             // 5
   ,{1,1,1,0,0,1}             // 6
   ,{1,1,1,0,0,1,0,1}           // 7
   ,{1,1,1,0,0,1,0,1,1}         // 8
   ,{1,1,1,1,0,1,0,0,0,1}       // 9
   ,{1,1,1,1,0,1,0,0,0,1,1}     //10
   ,{1,1,1,1,0,1,0,0,0,1,1,1}   //11
   ,{1,1,1,1,0,1,0,0,0,0,0,1}   //12
   ,{1,1,1,1,0,1,0,0,0,0,1,1}   //13
   ,{1,1,1,1,0,1,0,0,0,0,1,0,1} //14
   ,{1,1,1,1,0,1,0,0,0,1,1,1,0,1}   //15
   ,{1,1,1,1,0,1,0,0,0,1,1,1,0,1,1}   //16
   ,{1,1,1,1,0,1,0,0,0,1,1,1,0,1,0,1}   //17
   ,{1,1,1,1,0,1,0,0,0,1,1,1,0,1,0,0,1}   //18
   ,{1,1,1,1,0,1,0,0,0,0,1,1,0,0.0,1}   //19
   ,{1,1,1,1,0,1,0,1,1,0,1,1,0,0.0,1}   //20
   ,{1,1,1,1,0,1,0,1,1,0,1,1,0,0.0,1,0,0,1}   //21
   ,{1,1,0,0,1,0,0,0,0,0,0,0,0,0.0,0,0,0,0,1}   //22
   ,{1,1,1,0,1,0,0,0,0,0,0,0,0,0.0,0,0,0,0,1}   //23
   ,{1,1,1,0,1,0,0,0,0,0,1,0,0,0.0,0,0,0,0,1}   //24
   ,{1,1,1,0,0,0,1,0,0,0,0,0,0,0.0,0,0,0,0,1}   //25
   ,{1,1,1,1,0,0,1,0,0,0,0,0,0,0.0,0,0,0,0,1}   //26
   ,{1,1,1,0,1}                 // 27
   ,{1,1,1,0,0,1}                 // 28
   ,{1,1,1,1,0,0,1,0,0,0,0,0,0,0.0,0,0,0,0,1}   //29
   ,{1,1,1,0,0,1}                 // 30
   ,{1,1,1,1,0,1,0,0,0,1,1,0,0,1}   //31
   ,{1,1,1,1,0,1,0,0,0,1,1,0,0,1}   //32
   ,{1,1,1,1,0,1,0,0,0,1,1,0,0,1,0,0,0,0,0,1,1}   //33
   ,{1,1,1,1,0,1,0,0,0,1,1,0,0,1,0,0,0,0,0,0,1,1}   //34
   ,{1,1,1,1,0,1,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,1,0}   //35
   //   ,{1,1,1,1,0,1,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0}   //36 
   ,{1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,1}   //36
   ,{1,1,1,1,0,1,0,1,1,0,1,1,0,0.0,1}   //20
   ,{1,1,1,1,0,0,1,0,0,0,0,0,0,0.0,0,0,0,0,1}   //26
   ,{1,1,1,1,0,1,0,0,0,1,1,1}   //11
  };

#endif /* DIFFUSION_H */
