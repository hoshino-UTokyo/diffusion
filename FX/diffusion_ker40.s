..text.b:
	.ident	"$Options: Fujitsu C/C++ Compiler 4.2.0 (Jun 15 2020 13:54:08) --preinclude /opt/FJSVstclanga/v1.1.0/bin/../lib/fcc.pre --gcc -Dunix -Dlinux -D__FUJITSU -D__FCC_major__=4 -D__FCC_minor__=2 -D__FCC_patchlevel__=0 -D__FCC_version__=\"4.2.0\" -D__aarch64__ -D__unix -D_OPENMP=201107 -D__fcc_version__=0x800 -D__fcc_version=800 -D__USER_LABEL_PREFIX__= -D__OPTIMIZE__ -D__ARM_ARCH=8 -D__ARM_FEATURE_SVE -D__FP_FAST_FMA -D__ELF__ -D__unix__ -D__linux__ -D__linux -Asystem(unix) -D__LIBC_6B -D__LP64__ -D_LP64 --K=omp -DSVE -DBLAS --K=noocl -D_REENTRANT -D__MT__ --zmode=64 --sys_include=/opt/FJSVstclanga/v1.1.0/bin/../include --sys_include=/opt/FJSVxos/devkit/aarch64/rfs/usr/include --K=opt diffusion_ker40.c -- -ncmdname=fccpx -nspopt=\"-c -Kfast,openmp -O3 -DSVE -DBLAS -Khpctag -Nfjomplib -Nfjprof -Nlst=t -Koptmsg=2 -S\" -zcfc=target_sve -O3 -x- -Komitfp,mfunc,eval,fp_relaxed,fz,fast_matmul,fp_contract,ilfunc,simd_packed_promotion,openmp,threadsafe -O3 -Khpctag -Nfjomplib -Nfjprof -Nlst=t -Koptmsg=2 -zsta=am -Klargepage -zsrc=diffusion_ker40.c diffusion_ker40.s $"
	.file	"diffusion_ker40.c"
	.ident	"$Compiler: Fujitsu C/C++ Compiler 4.2.0 (Jun 15 2020 13:54:08) diffusion_ker40.c allocate_ker40 $"
	.text
	.align	2
	.global	allocate_ker40
	.type	allocate_ker40, %function
allocate_ker40:
	.file 1 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/bits/stdio.h"
	.file 2 "/opt/FJSVstclanga/v1.1.0/bin/../include/stdarg.h"
	.file 3 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/bits/types/FILE.h"
	.file 4 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/bits/types/struct_FILE.h"
	.file 5 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/stdlib.h"
	.file 6 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/bits/byteswap.h"
	.file 7 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/bits/types.h"
	.file 8 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/bits/uintn-identity.h"
	.file 9 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/bits/stdlib-bsearch.h"
	.file 10 "/opt/FJSVstclanga/v1.1.0/bin/../include/stddef.h"
	.file 11 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/bits/stdlib-float.h"
	.file 12 "diffusion_ker40.c"
	.loc 12 24 0
..LDL1:
.LFB0:
	.cfi_startproc
	.loc 12 26 0
..LDL2:
/*     26 */	sxtw	x1, w1
/*     26 */	sxtw	x3, w3
/*     26 */	sbfiz	x2, x2, 3, 32
/*     26 */	mul	x3, x1, x3
/*     26 */	mov	x1, 64
/*     26 */	mul	x2, x2, x3
/*     26 */	b	posix_memalign
..D1.pchi:
	.cfi_endproc
.LFE0:
	.size	allocate_ker40, .-allocate_ker40
	.ident	"$Compiler: Fujitsu C/C++ Compiler 4.2.0 (Jun 15 2020 13:54:08) diffusion_ker40.c init_ker40 $"
	.text
	.align	2
	.global	init_ker40
	.type	init_ker40, %function
init_ker40:
	.loc 12 30 0
..LDL3:
.LFB1:
	.cfi_startproc
/*    ??? */	str	x30, [sp, -16]!	//  (*)
	.cfi_def_cfa_offset 16
	.cfi_offset 30, -16
/*     63 */	mov	w4, w1
/*    ??? */	sub	sp, sp, 176
	.cfi_def_cfa_offset 192
/*     30 */	add	x1, sp, 80
/*     30 */	stp	d5, d4, [x1, 24]
/*     30 */	stp	d3, d2, [x1, 40]
/*     30 */	stp	d1, d0, [x1, 56]
/*     30 */	stp	w3, w2, [x1, 72]
/*     30 */	str	w4, [x1, 80]
/*     30 */	str	x0, [x1, 88]
	.loc 12 37 0
..LDL4:
/*     37 */	fneg	d2, d7
/*     37 */	ldr	d1, [x1, 64]
/*     37 */	ptrue	p1.d, ALL
/*     37 */	ptrue	p0.d, VL1
/*     37 */	adrp	x0, __fj_dexp_const_tbl_
/*     37 */	fdup	z0.d, 1.000000e+00
/*     37 */	add	x0, x0, :lo12:__fj_dexp_const_tbl_
/*     37 */	ld1rd	{z21.d}, p1/z, [x0, 8]	//  "__fj_dexp_const_tbl_"
/*     37 */	ld1rd	{z16.d}, p1/z, [x0, 16]	//  "__fj_dexp_const_tbl_"
/*     37 */	fmul	d23, d2, d6
/*     37 */	ld1rd	{z20.d}, p1/z, [x0, 88]	//  "__fj_dexp_const_tbl_"
/*     37 */	ld1rd	{z7.d}, p1/z, [x0, 24]	//  "__fj_dexp_const_tbl_"
/*     37 */	ld1rd	{z5.d}, p1/z, [x0, 32]	//  "__fj_dexp_const_tbl_"
/*     37 */	ld1rd	{z2.d}, p1/z, [x0, 96]	//  "__fj_dexp_const_tbl_"
/*     37 */	ld1rd	{z19.d}, p1/z, [x0, 104]	//  "__fj_dexp_const_tbl_"
/*     37 */	ld1rd	{z4.d}, p1/z, [x0, 112]	//  "__fj_dexp_const_tbl_"
/*     37 */	ld1rd	{z3.d}, p1/z, [x0, 120]	//  "__fj_dexp_const_tbl_"
/*     37 */	ld1rd	{z22.d}, p1/z, [x0]	//  "__fj_dexp_const_tbl_"
/*     37 */	ld1rd	{z18.d}, p1/z, [x0, 128]	//  "__fj_dexp_const_tbl_"
/*     37 */	fmul	d6, d23, d1
/*     37 */	fmul	d1, d1, d6
/*     37 */	dup	z6.d, z1.d[0]
/*     37 */	fcmgt	p1.d, p0/z, z21.d, z6.d
/*     37 */	fcmge	p2.d, p0/z, z6.d, z22.d
/*     37 */	fmov	z6.d, p1/m, 0.000000e+00
/*     37 */	movprfx	z17.d, p0/z, z6.d
/*     37 */	fmad	z17.d, p0/m, z16.d, z20.d
/*     37 */	fsub	z1.d, z17.d, z20.d
/*     37 */	fexpa	z17.d, z17.d
/*     37 */	fmla	z6.d, p0/m, z1.d, z7.d
/*     37 */	fmla	z6.d, p0/m, z1.d, z5.d
/*     37 */	mov	z1.d, z2.d
/*     37 */	movprfx	z1.d, p0/m, z2.d
/*     37 */	fmad	z1.d, p0/m, z6.d, z19.d
/*     37 */	fmad	z1.d, p0/m, z6.d, z4.d
/*     37 */	fmad	z1.d, p0/m, z6.d, z3.d
/*     37 */	fmad	z1.d, p0/m, z6.d, z0.d
/*     37 */	fmad	z6.d, p0/m, z1.d, z0.d
/*     37 */	fmul	z1.d, z6.d, z17.d
/*     37 */	fmov	z1.d, p1/m, 0.000000e+00
/*     37 */	sel	z1.d, p2, z18.d, z1.d
/*     37 */	str	d1, [x1, 16]
	.loc 12 38 0
..LDL5:
/*     38 */	ldr	d1, [x1, 56]
/*     38 */	fmul	d6, d23, d1
/*     38 */	fmul	d1, d1, d6
/*     38 */	dup	z6.d, z1.d[0]
/*     38 */	fcmgt	p1.d, p0/z, z21.d, z6.d
/*     38 */	fcmge	p2.d, p0/z, z6.d, z22.d
/*     38 */	fmov	z6.d, p1/m, 0.000000e+00
/*     38 */	movprfx	z17.d, p0/z, z6.d
/*     38 */	fmad	z17.d, p0/m, z16.d, z20.d
/*     38 */	fsub	z1.d, z17.d, z20.d
/*     38 */	fexpa	z17.d, z17.d
/*     38 */	fmla	z6.d, p0/m, z1.d, z7.d
/*     38 */	fmla	z6.d, p0/m, z1.d, z5.d
/*     38 */	mov	z1.d, z2.d
/*     38 */	movprfx	z1.d, p0/m, z2.d
/*     38 */	fmad	z1.d, p0/m, z6.d, z19.d
/*     38 */	fmad	z1.d, p0/m, z6.d, z4.d
/*     38 */	fmad	z1.d, p0/m, z6.d, z3.d
/*     38 */	fmad	z1.d, p0/m, z6.d, z0.d
/*     38 */	fmad	z6.d, p0/m, z1.d, z0.d
/*     38 */	fmul	z1.d, z6.d, z17.d
/*     38 */	fmov	z1.d, p1/m, 0.000000e+00
/*     38 */	sel	z1.d, p2, z18.d, z1.d
/*     38 */	str	d1, [x1, 8]
	.loc 12 39 0
..LDL6:
/*     39 */	ldr	d1, [x1, 48]
/*     39 */	fmul	d6, d23, d1
/*     39 */	fmul	d1, d1, d6
/*     39 */	dup	z1.d, z1.d[0]
/*     39 */	fcmgt	p2.d, p0/z, z21.d, z1.d
/*     39 */	fcmge	p1.d, p0/z, z1.d, z22.d
/*     39 */	fmov	z1.d, p2/m, 0.000000e+00
/*     39 */	fmad	z16.d, p0/m, z1.d, z20.d
/*     39 */	fsub	z6.d, z16.d, z20.d
/*     39 */	fexpa	z16.d, z16.d
/*     39 */	fmla	z1.d, p0/m, z6.d, z7.d
/*     39 */	fmla	z1.d, p0/m, z6.d, z5.d
/*     39 */	fmad	z2.d, p0/m, z1.d, z19.d
/*     39 */	fmad	z2.d, p0/m, z1.d, z4.d
/*     39 */	fmad	z2.d, p0/m, z1.d, z3.d
/*     39 */	fmad	z2.d, p0/m, z1.d, z0.d
/*     39 */	fmad	z1.d, p0/m, z2.d, z0.d
/*     39 */	fmul	z0.d, z1.d, z16.d
/*     39 */	fmov	z0.d, p2/m, 0.000000e+00
/*     39 */	sel	z0.d, p1, z18.d, z0.d
/*     39 */	str	d0, [x1]
	.loc 12 40 0 is_stmt 0
..LDL7:
/*     40 */	mov	x2, 0
/*     40 */	adrp	x0, init_ker40._OMP_1
/*     40 */	add	x0, x0, :lo12:init_ker40._OMP_1
/*     40 */	bl	__mpc_opar
	.loc 12 64 0 is_stmt 1
..LDL8:
/*    ??? */	add	sp, sp, 176
	.cfi_def_cfa_offset 16
/*    ??? */	ldr	x30, [sp], 16	//  (*)
	.cfi_restore 30
	.cfi_def_cfa_offset 0
/*     64 */	ret	
..D2.pchi:
	.cfi_endproc
.LFE1:
	.size	init_ker40, .-init_ker40
	.ident	"$Compiler: Fujitsu C/C++ Compiler 4.2.0 (Jun 15 2020 13:54:08) diffusion_ker40.c init_ker40._OMP_1 $"
	.text
	.align	2
	.type	init_ker40._OMP_1, %function
init_ker40._OMP_1:
	.loc 12 40 0
..LDL9:
.LFB2:
	.cfi_startproc
/*    ??? */	addvl	sp, sp, -11
/*    ??? */	stp	x29, x30, [sp]	//  (*)
/*     28 */	add	x29, sp, 0
	.cfi_escape 0xf,0xb,0x92,0x1d,0x0,0x11,0xd8,0x0,0x92,0x2e,0x0,0x1e,0x22
	.cfi_escape 0x10,0x1d,0x8,0x11,0xa8,0x7f,0x92,0x2e,0x0,0x1e,0x22
	.cfi_escape 0x10,0x1e,0xb,0x11,0x8,0x22,0x11,0xa8,0x7f,0x92,0x2e,0x0,0x1e,0x22
/*     28 */	sub	sp, sp, 160
/*    ??? */	stp	x19, x20, [x29, -16]	//  (*)
	.cfi_escape 0x10,0x13,0xb,0x11,0x70,0x22,0x11,0xa8,0x7f,0x92,0x2e,0x0,0x1e,0x22
	.cfi_escape 0x10,0x14,0xb,0x11,0x78,0x22,0x11,0xa8,0x7f,0x92,0x2e,0x0,0x1e,0x22
/*    ??? */	stp	d8, d9, [x29, -40]	//  (*)
	.cfi_escape 0x10,0x68,0xb,0x11,0x58,0x22,0x11,0xa8,0x7f,0x92,0x2e,0x0,0x1e,0x22
	.cfi_escape 0x10,0x69,0xb,0x11,0x60,0x22,0x11,0xa8,0x7f,0x92,0x2e,0x0,0x1e,0x22
/*     28 */	add	x19, sp, 0
/*     28 */	mov	x20, x0
/*    ??? */	stp	d10, d11, [x29, -56]	//  (*)
	.cfi_escape 0x10,0x6a,0xb,0x11,0x48,0x22,0x11,0xa8,0x7f,0x92,0x2e,0x0,0x1e,0x22
	.cfi_escape 0x10,0x6b,0xb,0x11,0x50,0x22,0x11,0xa8,0x7f,0x92,0x2e,0x0,0x1e,0x22
/*    ??? */	stp	d12, d13, [x29, -72]	//  (*)
	.cfi_escape 0x10,0x6c,0xc,0x11,0xb8,0x7f,0x22,0x11,0xa8,0x7f,0x92,0x2e,0x0,0x1e,0x22
	.cfi_escape 0x10,0x6d,0xb,0x11,0x40,0x22,0x11,0xa8,0x7f,0x92,0x2e,0x0,0x1e,0x22
/*    ??? */	stp	d14, d15, [x29, -88]	//  (*)
	.cfi_escape 0x10,0x6e,0xc,0x11,0xa8,0x7f,0x22,0x11,0xa8,0x7f,0x92,0x2e,0x0,0x1e,0x22
	.cfi_escape 0x10,0x6f,0xc,0x11,0xb0,0x7f,0x22,0x11,0xa8,0x7f,0x92,0x2e,0x0,0x1e,0x22
/*    ??? */	str	x21, [x29, -24]	//  (*)
	.cfi_escape 0x10,0x15,0xb,0x11,0x68,0x22,0x11,0xa8,0x7f,0x92,0x2e,0x0,0x1e,0x22
/*     28 */	and	sp, x19, -64
/*     28 */	str	x1, [x19, 48]
/*     28 */	str	x2, [x19, 40]
/*     28 */	str	x3, [x19, 32]
/*     28 */	str	x4, [x19, 24]
	.loc 12 42 0
..LDL10:
/*     42 */	bl	omp_get_thread_num
/*     42 */	mov	w21, w0
	.loc 12 43 0
..LDL11:
/*     43 */	bl	omp_get_num_threads
	.loc 12 44 0
..LDL12:
/*     44 */	mov	x6, 715784192
	.loc 12 47 0
..LDL13:
/*     47 */	sub	w5, w0, 1
	.loc 12 46 0
..LDL14:
/*     46 */	ldr	w3, [x20, 76]	//  "ny"
	.loc 12 47 0
..LDL15:
/*     47 */	ldr	w2, [x20, 72]	//  "nz"
	.loc 12 44 0
..LDL16:
/*     44 */	movk	x6, 43691, lsl #0
	.loc 12 47 0
..LDL17:
/*     47 */	sxtw	x1, w5
	.loc 12 44 0
..LDL18:
/*     44 */	sxtw	x0, w21
	.loc 12 47 0
..LDL19:
/*     47 */	mul	x1, x1, x6
	.loc 12 44 0
..LDL20:
/*     44 */	mul	x0, x0, x6
	.loc 12 46 0
..LDL21:
/*     46 */	sub	w4, w3, 1
/*     46 */	sxtw	x7, w4
	.loc 12 47 0
..LDL22:
/*     47 */	lsr	x1, x1, 32
	.loc 12 44 0
..LDL23:
/*     44 */	lsr	x0, x0, 32
	.loc 12 47 0
..LDL24:
/*     47 */	asr	w8, w1, 1
	.loc 12 44 0
..LDL25:
/*     44 */	asr	w1, w0, 1
	.loc 12 47 0
..LDL26:
/*     47 */	sub	w8, w8, w5, asr #31
	.loc 12 46 0
..LDL27:
/*     46 */	mul	x0, x7, x6
	.loc 12 44 0
..LDL28:
/*     44 */	sub	w5, w1, w21, asr #31
	.loc 12 47 0
..LDL29:
/*     47 */	add	w1, w8, 1
/*     47 */	sdiv	w7, w2, w1
	.loc 12 45 0
..LDL30:
/*     45 */	add	w1, w5, w5
	.loc 12 48 0
..LDL31:
/*     48 */	add	w6, w5, 1
	.loc 12 45 0
..LDL32:
/*     45 */	add	w1, w1, w5
/*     45 */	sub	w1, w21, w1, lsl #2
	.loc 12 46 0
..LDL33:
/*     46 */	lsr	x0, x0, 32
/*     46 */	asr	w0, w0, 1
/*     46 */	sub	w0, w0, w4, asr #31
/*     46 */	add	w0, w0, 1
	.loc 12 48 0
..LDL34:
/*     48 */	mul	w4, w6, w7
/*     48 */	mul	w16, w7, w5
/*     48 */	cmp	w2, w4
/*     48 */	csel	w4, w2, w4, le
/*     48 */	cmp	w16, w4
/*     48 */	bge	.L259
	.loc 12 49 0 is_stmt 0
..LDL35:
/*     49 */	add	w5, w1, 1
	.loc 12 55 0
..LDL36:
/*     55 */	ptrue	p0.d, ALL
/*     55 */	ldr	d3, [x20, 32]	//  "dy"
	.loc 12 49 0
..LDL37:
/*     49 */	mul	w15, w0, w1
/*     49 */	mul	w5, w5, w0
	.loc 12 55 0
..LDL38:
/*     55 */	ldr	d2, [x20, 56]	//  "ky"
/*     55 */	ldr	d1, [x20, 24]	//  "dz"
	.loc 12 50 0
..LDL39:
/*     50 */	ldr	w2, [x20, 80]	//  "nx"
	.loc 12 55 0
..LDL40:
/*     55 */	ldr	d4, [x20, 48]	//  "kz"
	.loc 12 52 0
..LDL41:
/*     52 */	ld1rd	{z0.d}, p0/z, [x20, 40]	//  "dx"
	.loc 12 55 0
..LDL42:
/*     55 */	adrp	x0, __fj_dsin_const_tbl_
/*     55 */	add	x8, x0, :lo12:__fj_dsin_const_tbl_
	.loc 12 49 0
..LDL43:
/*     49 */	cmp	w3, w5
/*     49 */	csel	w13, w3, w5, le
	.loc 12 50 0
..LDL44:
/*     50 */	cmp	w2, 0
	.loc 12 59 0
..LDL45:
/*     59 */	sxtw	x12, w2
	.loc 12 50 0
..LDL46:
/*     50 */	csel	w0, w2, wzr, ge
/*     50 */	asr	w1, w0, 2
/*     50 */	add	w1, w0, w1, lsr #29
/*     50 */	asr	w11, w1, 3
/*     50 */	lsl	w1, w11, 3
	.loc 12 52 0
..LDL47:
/*    ??? */	str	z0, [x29, 3, mul vl]	//  (*)
	.loc 12 50 0
..LDL48:
/*     50 */	sub	w5, w0, w1
	.loc 12 55 0
..LDL49:
/*     55 */	ld1rd	{z0.d}, p0/z, [x20, 64]	//  "kx"
	.loc 12 50 0
..LDL50:
/*     50 */	sbfiz	x3, x11, 3, 32
	.loc 12 55 0
..LDL51:
/*     55 */	whilelt	p1.d, wzr, w5
/*    ??? */	str	z0, [x29, 4, mul vl]	//  (*)
/*     55 */	ld1rd	{z0.d}, p0/z, [x20, 16]	//  "ax"
/*    ??? */	str	z0, [x29, 2, mul vl]	//  (*)
/*     55 */	fmul	d0, d3, d2
/*    ??? */	str	d0, [x19, 8]	//  (*)
/*     55 */	fmul	d0, d1, d4
	.loc 12 52 0
..LDL52:
/*     52 */	dup	z1.s, w1
	.loc 12 55 0
..LDL53:
/*    ??? */	str	d0, [x19]	//  (*)
	.loc 12 52 0
..LDL54:
/*     52 */	index	z0.d, 0, 1
/*     52 */	add	z0.s, z0.s, z1.s
/*    ??? */	ldr	z1, [x29, 3, mul vl]	//  (*)
/*     52 */	scvtf	z0.d, p1/m, z0.s
/*     52 */	fadd	z0.d, p1/m, z0.d, #0.5
/*     52 */	fmul	z1.d, z0.d, z1.d
	.loc 12 55 0
..LDL55:
/*    ??? */	ldr	z0, [x29, 4, mul vl]	//  (*)
/*     55 */	fmul	z0.d, z1.d, z0.d
/*    ??? */	str	z0, [x29, 1, mul vl]	//  (*)
.L243:					// :entr
	.loc 12 49 0 is_stmt 1
..LDL56:
/*     49 */	cmp	w15, w13
/*     49 */	bge	.L257
	.loc 12 51 0 is_stmt 0
..LDL57:
/*     51 */	ldr	w0, [x20, 76]	//  "ny"
	.loc 12 53 0
..LDL58:
/*     53 */	scvtf	d0, w15
	.loc 12 51 0
..LDL59:
/*     51 */	ldr	w6, [x20, 80]	//  "nx"
	.loc 12 50 0
..LDL60:
/*     50 */	mov	w9, w15
	.loc 12 51 0
..LDL61:
/*     51 */	madd	w0, w0, w16, w15
/*     51 */	mul	w0, w0, w6
	.loc 12 53 0
..LDL62:
/*    ??? */	str	d0, [x19, 16]	//  (*)
	.loc 12 59 0
..LDL63:
/*     59 */	sxtw	x14, w0
	.loc 12 50 0
..LDL64:
/*     50 */	add	x10, x14, x3
.L246:					// :entr
	.loc 12 50 0 is_stmt 1
..LDL65:
/*     50 */	cmp	w6, 0
/*     50 */	ble	.L255
	.loc 12 54 0 is_stmt 0
..LDL66:
/*     54 */	scvtf	d5, w16
	.loc 12 55 0
..LDL67:
/*     55 */	fmov	d0, 5.000000e-01
/*    ??? */	ldr	d1, [x19, 16]	//  (*)
/*     55 */	ptrue	p2.d, ALL
/*     55 */	ptrue	p3.d, VL1
/*     55 */	ldr	d16, [x20, 8]	//  "ay"
/*     55 */	ldr	d17, [x20]	//  "az"
/*     55 */	ld1rd	{z2.d}, p2/z, [x8, 16]	//  "__fj_dsin_const_tbl_"
/*     55 */	ld1rd	{z18.d}, p2/z, [x8, 48]	//  "__fj_dsin_const_tbl_"
/*     55 */	ld1rd	{z6.d}, p2/z, [x8, 24]	//  "__fj_dsin_const_tbl_"
/*     55 */	ld1rd	{z20.d}, p2/z, [x8]	//  "__fj_dsin_const_tbl_"
/*     55 */	ld1rd	{z3.d}, p2/z, [x8, 32]	//  "__fj_dsin_const_tbl_"
/*     55 */	ld1rd	{z7.d}, p2/z, [x8, 56]	//  "__fj_dsin_const_tbl_"
/*     55 */	fadd	d4, d1, d0
/*    ??? */	ldr	d0, [x19, 8]	//  (*)
/*     55 */	ld1rd	{z1.d}, p2/z, [x8, 40]	//  "__fj_dsin_const_tbl_"
/*     55 */	fmul	d0, d4, d0
/*     55 */	fmov	d4, 5.000000e-01
/*     55 */	fadd	d5, d5, d4
/*     55 */	dup	z4.d, z0.d[0]
/*    ??? */	ldr	d0, [x19]	//  (*)
/*     55 */	movprfx	z19.d, p3/z, z2.d
/*     55 */	fmla	z19.d, p3/m, z4.d, z18.d
/*     55 */	facgt	p4.d, p3/z, z4.d, z20.d
/*     55 */	fmul	d0, d5, d0
/*     55 */	fsub	z5.d, z19.d, z2.d
/*     55 */	dup	z0.d, z0.d[0]
/*     55 */	fmad	z18.d, p3/m, z0.d, z2.d
/*     55 */	fmls	z4.d, p3/m, z5.d, z6.d
/*     55 */	facgt	p2.d, p3/z, z0.d, z20.d
/*     55 */	fsub	z2.d, z18.d, z2.d
/*     55 */	fmls	z4.d, p3/m, z5.d, z3.d
/*     55 */	fmls	z0.d, p3/m, z2.d, z6.d
/*     55 */	fmls	z4.d, p3/m, z5.d, z1.d
/*     55 */	fmls	z0.d, p3/m, z2.d, z3.d
/*     55 */	ftsmul	z3.d, z4.d, z19.d
/*     55 */	ftssel	z4.d, z4.d, z19.d
/*     55 */	fmls	z0.d, p3/m, z2.d, z1.d
/*     55 */	fmov	z1.d, 0.000000e+00
/*     55 */	ftmad	z1.d, z1.d, z3.d, 7
/*     55 */	ftsmul	z2.d, z0.d, z18.d
/*     55 */	ftssel	z5.d, z0.d, z18.d
/*     55 */	fmov	z0.d, 0.000000e+00
/*     55 */	ftmad	z1.d, z1.d, z3.d, 6
/*     55 */	ftmad	z0.d, z0.d, z2.d, 7
/*     55 */	ftmad	z1.d, z1.d, z3.d, 5
/*     55 */	ftmad	z0.d, z0.d, z2.d, 6
/*     55 */	ftmad	z1.d, z1.d, z3.d, 4
/*     55 */	ftmad	z0.d, z0.d, z2.d, 5
/*     55 */	ftmad	z1.d, z1.d, z3.d, 3
/*     55 */	ftmad	z0.d, z0.d, z2.d, 4
/*     55 */	ftmad	z1.d, z1.d, z3.d, 2
/*     55 */	ftmad	z0.d, z0.d, z2.d, 3
/*     55 */	ftmad	z1.d, z1.d, z3.d, 1
/*     55 */	ftmad	z0.d, z0.d, z2.d, 2
/*     55 */	ftmad	z1.d, z1.d, z3.d, 0
/*     55 */	ftmad	z0.d, z0.d, z2.d, 1
/*     55 */	fmul	z1.d, z1.d, z4.d
/*     55 */	ftmad	z0.d, z0.d, z2.d, 0
/*     55 */	fmov	d2, 1.000000e+00
/*     55 */	sel	z1.d, p4, z7.d, z1.d
/*     55 */	fmul	z0.d, z0.d, z5.d
/*     55 */	fmsub	d1, d16, d1, d2
/*     55 */	sel	z2.d, p2, z7.d, z0.d
/*     55 */	fmov	d0, 1.000000e+00
/*     55 */	fmsub	d2, d17, d2, d0
/*     55 */	fmov	d0, 1.250000e-01
/*     55 */	fmul	d0, d2, d0
/*     55 */	fmul	d0, d1, d0
/*     55 */	dup	z15.d, z0.d[0]
	.loc 12 50 0
..LDL68:
/*     50 */	cbz	w11, .L252
/*     50 */	mov	w7, 0
	.loc 12 55 0
..LDL69:
/*     55 */	ld1rd	{z0.d}, p0/z, [x8]	//  "__fj_dsin_const_tbl_"
/*     55 */	ld1rd	{z14.d}, p0/z, [x8, 16]	//  "__fj_dsin_const_tbl_"
/*     55 */	mov	w2, w11
/*     55 */	mov	x1, x14
	.loc 12 50 0
..LDL70:
/*     50 */	cmp	w2, 16
	.loc 12 55 0
..LDL71:
/*    ??? */	str	z0, [x29, 5, mul vl]	//  (*)
/*     55 */	ld1rd	{z0.d}, p0/z, [x8, 24]	//  "__fj_dsin_const_tbl_"
/*    ??? */	str	z0, [x29, 7, mul vl]	//  (*)
/*     55 */	ld1rd	{z0.d}, p0/z, [x8, 32]	//  "__fj_dsin_const_tbl_"
/*    ??? */	str	z0, [x29, 8, mul vl]	//  (*)
/*     55 */	ld1rd	{z0.d}, p0/z, [x8, 40]	//  "__fj_dsin_const_tbl_"
/*    ??? */	str	z0, [x29, 9, mul vl]	//  (*)
/*     55 */	ld1rd	{z0.d}, p0/z, [x8, 48]	//  "__fj_dsin_const_tbl_"
/*    ??? */	str	z0, [x29, 6, mul vl]	//  (*)
/*     55 */	ld1rd	{z0.d}, p0/z, [x8, 56]	//  "__fj_dsin_const_tbl_"
/*    ??? */	str	z0, [x29, 10, mul vl]	//  (*)
	.loc 12 50 0
..LDL72:
/*     50 */	blt	.L309
	.loc 12 55 0
..LDL73:
/*     55 */	mov	x0, x1
	.loc 12 52 0
..LDL74:
/*     52 */	index	z0.d, 0, 1
/*     52 */	add	w1, w7, 8
	.loc 12 55 0
..LDL75:
/*    ??? */	ldr	z16, [x29, 6, mul vl]	//  (*)
/*     55 */	mov	w17, w7
	.loc 12 52 0
..LDL76:
/*     52 */	dup	z1.s, w7
	.loc 12 55 0
..LDL77:
/*    ??? */	ldr	z28, [x29, 7, mul vl]	//  (*)
/*     55 */	sub	w2, w2, 6
	.loc 12 52 0
..LDL78:
/*     52 */	dup	z2.s, w1
/*     52 */	add	w1, w7, 16
	.loc 12 55 0
..LDL79:
/*    ??? */	ldr	z7, [x29, 9, mul vl]	//  (*)
/*     55 */	sub	w2, w2, 4
/*    ??? */	ldr	z25, [x29, 6, mul vl]	//  (*)
/*     55 */	fmov	z26.d, 0.000000e+00
/*    ??? */	ldr	z6, [x29, 6, mul vl]	//  (*)
/*     55 */	fmov	z10.d, 0.000000e+00
/*     55 */	fmov	z9.d, 0.000000e+00
	.loc 12 52 0
..LDL80:
/*     52 */	add	z0.s, z1.s, z0.s
/*    ??? */	ldr	z1, [x29, 3, mul vl]	//  (*)
/*     52 */	scvtf	z0.d, p0/m, z0.s
/*     52 */	fadd	z0.d, p0/m, z0.d, #0.5
/*     52 */	fmul	z1.d, z0.d, z1.d
/*     52 */	index	z0.d, 0, 1
/*     52 */	add	z0.s, z2.s, z0.s
	.loc 12 55 0
..LDL81:
/*    ??? */	ldr	z2, [x29, 4, mul vl]	//  (*)
	.loc 12 52 0
..LDL82:
/*     52 */	scvtf	z0.d, p0/m, z0.s
	.loc 12 55 0
..LDL83:
/*     55 */	fmul	z20.d, z1.d, z2.d
	.loc 12 52 0
..LDL84:
/*    ??? */	ldr	z1, [x29, 3, mul vl]	//  (*)
/*     52 */	dup	z2.s, w1
/*     52 */	add	w1, w7, 24
/*     52 */	add	w7, w7, 32
/*     52 */	fadd	z0.d, p0/m, z0.d, #0.5
/*     52 */	dup	z27.s, w1
/*     52 */	add	w1, w17, 40
/*     52 */	dup	z5.s, w7
	.loc 12 55 0
..LDL85:
/*     55 */	fmad	z16.d, p0/m, z20.d, z14.d
	.loc 12 52 0
..LDL86:
/*     52 */	fmul	z1.d, z0.d, z1.d
/*     52 */	index	z0.d, 0, 1
	.loc 12 55 0
..LDL87:
/*     55 */	fsub	z17.d, z16.d, z14.d
/*     55 */	mov	z23.d, z16.d
	.loc 12 52 0
..LDL88:
/*     52 */	add	z0.s, z2.s, z0.s
	.loc 12 55 0
..LDL89:
/*     55 */	fmsb	z28.d, p0/m, z17.d, z20.d
	.loc 12 52 0
..LDL90:
/*     52 */	scvtf	z18.d, p0/m, z0.s
	.loc 12 55 0
..LDL91:
/*    ??? */	ldr	z0, [x29, 4, mul vl]	//  (*)
	.loc 12 52 0
..LDL92:
/*     52 */	fadd	z18.d, p0/m, z18.d, #0.5
	.loc 12 55 0
..LDL93:
/*     55 */	fmul	z21.d, z1.d, z0.d
/*    ??? */	ldr	z0, [x29, 8, mul vl]	//  (*)
	.loc 12 52 0
..LDL94:
/*    ??? */	ldr	z1, [x29, 3, mul vl]	//  (*)
	.loc 12 55 0
..LDL95:
/*     55 */	fmad	z25.d, p0/m, z21.d, z14.d
/*     55 */	fmsb	z0.d, p0/m, z17.d, z28.d
	.loc 12 52 0
..LDL96:
/*     52 */	fmul	z3.d, z18.d, z1.d
/*     52 */	index	z1.d, 0, 1
	.loc 12 55 0
..LDL97:
/*     55 */	fmov	z28.d, 0.000000e+00
/*     55 */	fsub	z30.d, z25.d, z14.d
/*     55 */	fmsb	z7.d, p0/m, z17.d, z0.d
/*    ??? */	ldr	z0, [x29, 4, mul vl]	//  (*)
/*     55 */	mov	z17.d, z25.d
	.loc 12 52 0
..LDL98:
/*     52 */	add	z1.s, z27.s, z1.s
/*     52 */	scvtf	z2.d, p0/m, z1.s
	.loc 12 55 0
..LDL99:
/*     55 */	ftsmul	z29.d, z7.d, z23.d
/*    ??? */	ldr	z1, [x29, 6, mul vl]	//  (*)
/*     55 */	ftssel	z23.d, z7.d, z23.d
/*    ??? */	ldr	z7, [x29, 6, mul vl]	//  (*)
/*     55 */	fmul	z19.d, z3.d, z0.d
/*    ??? */	ldr	z3, [x29, 7, mul vl]	//  (*)
/*    ??? */	ldr	z0, [x29, 8, mul vl]	//  (*)
	.loc 12 52 0
..LDL100:
/*     52 */	fadd	z2.d, p0/m, z2.d, #0.5
	.loc 12 55 0
..LDL101:
/*     55 */	ftmad	z28.d, z28.d, z29.d, 7
/*     55 */	fmad	z1.d, p0/m, z19.d, z14.d
/*     55 */	fmsb	z3.d, p0/m, z30.d, z21.d
/*     55 */	ftmad	z28.d, z28.d, z29.d, 6
/*     55 */	fmls	z3.d, p0/m, z30.d, z0.d
	.loc 12 52 0
..LDL102:
/*    ??? */	ldr	z0, [x29, 3, mul vl]	//  (*)
	.loc 12 55 0
..LDL103:
/*     55 */	ftmad	z28.d, z28.d, z29.d, 5
	.loc 12 52 0
..LDL104:
/*     52 */	fmul	z4.d, z2.d, z0.d
/*     52 */	index	z0.d, 0, 1
	.loc 12 55 0
..LDL105:
/*     55 */	fsub	z2.d, z1.d, z14.d
/*     55 */	ftmad	z28.d, z28.d, z29.d, 4
/*     55 */	ftmad	z28.d, z28.d, z29.d, 3
	.loc 12 52 0
..LDL106:
/*     52 */	add	z0.s, z5.s, z0.s
/*     52 */	scvtf	z5.d, p0/m, z0.s
	.loc 12 55 0
..LDL107:
/*    ??? */	ldr	z0, [x29, 9, mul vl]	//  (*)
/*     55 */	ftmad	z28.d, z28.d, z29.d, 2
	.loc 12 52 0
..LDL108:
/*     52 */	fadd	z5.d, p0/m, z5.d, #0.5
	.loc 12 55 0
..LDL109:
/*     55 */	fmsb	z30.d, p0/m, z0.d, z3.d
/*    ??? */	ldr	z0, [x29, 4, mul vl]	//  (*)
	.loc 12 52 0
..LDL110:
/*     52 */	dup	z3.s, w1
/*     52 */	add	w1, w17, 48
/*     52 */	dup	z22.s, w1
	.loc 12 55 0
..LDL111:
/*     55 */	ftmad	z28.d, z28.d, z29.d, 1
	.loc 12 52 0
..LDL112:
/*     52 */	add	w7, w1, 8
	.loc 12 55 0
..LDL113:
/*     55 */	ftsmul	z16.d, z30.d, z17.d
/*     55 */	ftssel	z17.d, z30.d, z17.d
/*    ??? */	ldr	z30, [x29, 8, mul vl]	//  (*)
/*     55 */	fmul	z13.d, z4.d, z0.d
/*    ??? */	ldr	z4, [x29, 7, mul vl]	//  (*)
/*     55 */	ftmad	z28.d, z28.d, z29.d, 0
/*    ??? */	ldr	z0, [x29, 8, mul vl]	//  (*)
/*    ??? */	ldr	z29, [x29, 8, mul vl]	//  (*)
/*     55 */	ftmad	z26.d, z26.d, z16.d, 7
/*     55 */	fmad	z6.d, p0/m, z13.d, z14.d
/*     55 */	fmsb	z4.d, p0/m, z2.d, z19.d
/*     55 */	ftmad	z26.d, z26.d, z16.d, 6
/*     55 */	fmls	z4.d, p0/m, z2.d, z0.d
	.loc 12 52 0
..LDL114:
/*    ??? */	ldr	z0, [x29, 3, mul vl]	//  (*)
	.loc 12 55 0
..LDL115:
/*     55 */	ftmad	z26.d, z26.d, z16.d, 5
	.loc 12 52 0
..LDL116:
/*     52 */	fmul	z18.d, z5.d, z0.d
/*     52 */	index	z0.d, 0, 1
	.loc 12 55 0
..LDL117:
/*     55 */	ftmad	z26.d, z26.d, z16.d, 4
/*     55 */	ftmad	z26.d, z26.d, z16.d, 3
	.loc 12 52 0
..LDL118:
/*     52 */	add	z3.s, z3.s, z0.s
	.loc 12 55 0
..LDL119:
/*     55 */	fsub	z0.d, z6.d, z14.d
	.loc 12 52 0
..LDL120:
/*     52 */	scvtf	z5.d, p0/m, z3.s
	.loc 12 55 0
..LDL121:
/*    ??? */	ldr	z3, [x29, 9, mul vl]	//  (*)
/*     55 */	ftmad	z26.d, z26.d, z16.d, 2
	.loc 12 52 0
..LDL122:
/*     52 */	fadd	z5.d, p0/m, z5.d, #0.5
	.loc 12 55 0
..LDL123:
/*     55 */	fmsb	z2.d, p0/m, z3.d, z4.d
/*    ??? */	ldr	z3, [x29, 4, mul vl]	//  (*)
/*    ??? */	ldr	z4, [x29, 6, mul vl]	//  (*)
/*     55 */	ftmad	z26.d, z26.d, z16.d, 1
/*     55 */	ftsmul	z27.d, z2.d, z1.d
/*     55 */	fmul	z24.d, z18.d, z3.d
/*    ??? */	ldr	z18, [x29, 7, mul vl]	//  (*)
/*     55 */	ftmad	z26.d, z26.d, z16.d, 0
/*    ??? */	ldr	z3, [x29, 8, mul vl]	//  (*)
/*    ??? */	ldr	z16, [x29, 6, mul vl]	//  (*)
/*     55 */	fmad	z4.d, p0/m, z24.d, z14.d
/*     55 */	fmul	z26.d, z26.d, z17.d
	.loc 12 52 0
..LDL124:
/*     52 */	index	z17.d, 0, 1
	.loc 12 55 0
..LDL125:
/*     55 */	fmsb	z18.d, p0/m, z0.d, z13.d
/*     55 */	fmls	z18.d, p0/m, z0.d, z3.d
	.loc 12 52 0
..LDL126:
/*    ??? */	ldr	z3, [x29, 3, mul vl]	//  (*)
/*     52 */	fmul	z31.d, z5.d, z3.d
/*     52 */	index	z5.d, 0, 1
	.loc 12 55 0
..LDL127:
/*     55 */	fmov	z3.d, 0.000000e+00
/*     55 */	ftmad	z3.d, z3.d, z27.d, 7
	.loc 12 52 0
..LDL128:
/*     52 */	add	z22.s, z22.s, z5.s
	.loc 12 55 0
..LDL129:
/*     55 */	fsub	z5.d, z4.d, z14.d
/*     55 */	ftmad	z3.d, z3.d, z27.d, 6
	.loc 12 52 0
..LDL130:
/*     52 */	scvtf	z25.d, p0/m, z22.s
	.loc 12 55 0
..LDL131:
/*    ??? */	ldr	z22, [x29, 9, mul vl]	//  (*)
/*     55 */	ftmad	z3.d, z3.d, z27.d, 5
	.loc 12 52 0
..LDL132:
/*     52 */	fadd	z25.d, p0/m, z25.d, #0.5
	.loc 12 55 0
..LDL133:
/*     55 */	fmsb	z0.d, p0/m, z22.d, z18.d
/*    ??? */	ldr	z18, [x29, 4, mul vl]	//  (*)
/*     55 */	ftmad	z3.d, z3.d, z27.d, 4
/*     55 */	ftsmul	z12.d, z0.d, z6.d
/*     55 */	fmul	z22.d, z31.d, z18.d
/*    ??? */	ldr	z18, [x29, 7, mul vl]	//  (*)
	.loc 12 52 0
..LDL134:
/*     52 */	dup	z31.s, w7
/*     52 */	add	w7, w1, 16
/*     52 */	add	w1, w1, 24
	.loc 12 55 0
..LDL135:
/*     55 */	ftmad	z3.d, z3.d, z27.d, 3
/*     55 */	ftmad	z10.d, z10.d, z12.d, 7
/*     55 */	fmad	z7.d, p0/m, z22.d, z14.d
/*     55 */	fmsb	z18.d, p0/m, z5.d, z24.d
/*     55 */	ftmad	z3.d, z3.d, z27.d, 2
/*     55 */	ftmad	z10.d, z10.d, z12.d, 6
/*     55 */	fmls	z18.d, p0/m, z5.d, z29.d
	.loc 12 52 0
..LDL136:
/*    ??? */	ldr	z29, [x29, 3, mul vl]	//  (*)
	.loc 12 55 0
..LDL137:
/*     55 */	ftmad	z3.d, z3.d, z27.d, 1
/*     55 */	ftmad	z10.d, z10.d, z12.d, 5
/*     55 */	ftmad	z3.d, z3.d, z27.d, 0
	.loc 12 52 0
..LDL138:
/*     52 */	dup	z27.s, w1
/*     52 */	fmul	z25.d, z25.d, z29.d
	.loc 12 55 0
..LDL139:
/*     55 */	fmul	z29.d, z28.d, z23.d
	.loc 12 52 0
..LDL140:
/*     52 */	index	z23.d, 0, 1
	.loc 12 55 0
..LDL141:
/*     55 */	ftmad	z10.d, z10.d, z12.d, 4
/*     55 */	ftmad	z10.d, z10.d, z12.d, 3
	.loc 12 52 0
..LDL142:
/*     52 */	add	z28.s, z31.s, z23.s
	.loc 12 55 0
..LDL143:
/*     55 */	fsub	z23.d, z7.d, z14.d
/*    ??? */	ldr	z31, [x29, 5, mul vl]	//  (*)
	.loc 12 52 0
..LDL144:
/*     52 */	scvtf	z28.d, p0/m, z28.s
	.loc 12 55 0
..LDL145:
/*     55 */	facgt	p2.d, p0/z, z20.d, z31.d
/*    ??? */	ldr	z20, [x29, 9, mul vl]	//  (*)
	.loc 12 52 0
..LDL146:
/*     52 */	fadd	z28.d, p0/m, z28.d, #0.5
	.loc 12 55 0
..LDL147:
/*     55 */	fmsb	z5.d, p0/m, z20.d, z18.d
/*    ??? */	ldr	z18, [x29, 4, mul vl]	//  (*)
/*     55 */	ftsmul	z11.d, z5.d, z4.d
/*     55 */	fmul	z20.d, z25.d, z18.d
/*    ??? */	ldr	z18, [x29, 10, mul vl]	//  (*)
/*    ??? */	ldr	z25, [x29, 7, mul vl]	//  (*)
/*     55 */	ftmad	z9.d, z9.d, z11.d, 7
/*     55 */	fmad	z16.d, p0/m, z20.d, z14.d
/*     55 */	sel	z18.d, p2, z18.d, z29.d
/*    ??? */	ldr	z29, [x29, 2, mul vl]	//  (*)
/*     55 */	fmsb	z25.d, p0/m, z23.d, z22.d
/*     55 */	facgt	p2.d, p0/z, z21.d, z31.d
/*    ??? */	ldr	z21, [x29, 9, mul vl]	//  (*)
/*     55 */	ftmad	z9.d, z9.d, z11.d, 6
/*     55 */	fmls	z25.d, p0/m, z23.d, z30.d
	.loc 12 52 0
..LDL148:
/*    ??? */	ldr	z30, [x29, 3, mul vl]	//  (*)
	.loc 12 55 0
..LDL149:
/*     55 */	fmul	z29.d, z18.d, z29.d
	.loc 12 52 0
..LDL150:
/*     52 */	dup	z18.s, w7
	.loc 12 59 0
..LDL151:
/*     59 */	ldr	x7, [x20, 88]	//  "buff1"
	.loc 12 55 0
..LDL152:
/*     55 */	ftmad	z9.d, z9.d, z11.d, 5
/*     55 */	fmsb	z23.d, p0/m, z21.d, z25.d
/*    ??? */	ldr	z21, [x29, 4, mul vl]	//  (*)
/*     55 */	fsubr	z29.d, p0/m, z29.d, #1.0
	.loc 12 52 0
..LDL153:
/*     52 */	add	z18.s, z18.s, z17.s
/*     52 */	fmul	z28.d, z28.d, z30.d
	.loc 12 55 0
..LDL154:
/*     55 */	fsub	z17.d, z16.d, z14.d
	.loc 12 52 0
..LDL155:
/*     52 */	scvtf	z18.d, p0/m, z18.s
	.loc 12 55 0
..LDL156:
/*     55 */	ftsmul	z8.d, z23.d, z7.d
/*     55 */	fmul	z25.d, z29.d, z15.d
/*    ??? */	ldr	z29, [x29, 2, mul vl]	//  (*)
/*     55 */	fmul	z21.d, z28.d, z21.d
/*    ??? */	ldr	z28, [x29, 10, mul vl]	//  (*)
	.loc 12 52 0
..LDL157:
/*     52 */	fadd	z18.d, p0/m, z18.d, #0.5
	.loc 12 59 0
..LDL158:
/*     59 */	st1d	{z25.d}, p0, [x7, x0, lsl #3]	//  (*)
	.loc 12 55 0
..LDL159:
/*    ??? */	ldr	z25, [x29, 6, mul vl]	//  (*)
/*     55 */	sel	z26.d, p2, z28.d, z26.d
/*    ??? */	ldr	z28, [x29, 7, mul vl]	//  (*)
/*     55 */	fmul	z26.d, z26.d, z29.d
/*     55 */	fmad	z25.d, p0/m, z21.d, z14.d
/*     55 */	fmsb	z28.d, p0/m, z17.d, z20.d
	.p2align 5
.L250:					// :entr:term:swpl
/*     55 */	ftssel	z2.d, z2.d, z1.d
/* #00001 */	ldr	z1, [x29, 8, mul vl]	//  (*)
/*     55 */	sub	w2, w2, 6
/*     55 */	fmov	z30.d, 0.000000e+00
/*     55 */	cmp	w2, 6
/*     55 */	fmls	z28.d, p0/m, z17.d, z1.d
/*     55 */	ftmad	z10.d, z10.d, z12.d, 2
/*     55 */	movprfx	z1.d, p0/m, z26.d
/*     55 */	fsubr	z1.d, p0/m, z1.d, #1.0
	.loc 12 52 0
..LDL160:
/* #00001 */	ldr	z26, [x29, 3, mul vl]	//  (*)
/*     52 */	fmul	z26.d, z18.d, z26.d
	.loc 12 55 0
..LDL161:
/*     55 */	ftmad	z30.d, z30.d, z8.d, 7
/*     55 */	fmul	z18.d, z3.d, z2.d
	.loc 12 52 0
..LDL162:
/*     52 */	index	z2.d, 0, 1
/*     52 */	add	z3.s, z27.s, z2.s
	.loc 12 55 0
..LDL163:
/* #00001 */	ldr	z27, [x29, 5, mul vl]	//  (*)
/*     55 */	fsub	z2.d, z25.d, z14.d
/*     55 */	ftmad	z9.d, z9.d, z11.d, 4
	.loc 12 52 0
..LDL164:
/*     52 */	scvtf	z3.d, p0/m, z3.s
	.loc 12 55 0
..LDL165:
/*     55 */	facgt	p2.d, p0/z, z19.d, z27.d
/* #00001 */	ldr	z19, [x29, 9, mul vl]	//  (*)
/*     55 */	fmls	z28.d, p0/m, z17.d, z19.d
/* #00001 */	ldr	z17, [x29, 4, mul vl]	//  (*)
/*     55 */	ftmad	z10.d, z10.d, z12.d, 1
/*     55 */	fmul	z1.d, z1.d, z15.d
/*     55 */	fmul	z19.d, z26.d, z17.d
/* #00001 */	ldr	z17, [x29, 10, mul vl]	//  (*)
/*     55 */	ftmad	z30.d, z30.d, z8.d, 6
/* #00001 */	ldr	z26, [x29, 7, mul vl]	//  (*)
/*     55 */	sel	z17.d, p2, z17.d, z18.d
/* #00001 */	ldr	z18, [x29, 2, mul vl]	//  (*)
	.loc 12 59 0
..LDL166:
/*     59 */	ldr	x17, [x20, 88]	//  "buff1"
	.loc 12 55 0
..LDL167:
/*     55 */	fmsb	z26.d, p0/m, z2.d, z21.d
/*     55 */	ftmad	z9.d, z9.d, z11.d, 3
/*     55 */	fmul	z17.d, z17.d, z18.d
	.loc 12 52 0
..LDL168:
/*     52 */	fadd	z3.d, p0/m, z3.d, #0.5
	.loc 12 55 0
..LDL169:
/*     55 */	add	x7, x0, 8
/*     55 */	ftsmul	z29.d, z28.d, z16.d
/*     55 */	ftmad	z10.d, z10.d, z12.d, 0
	.loc 12 59 0
..LDL170:
/*     59 */	st1d	{z1.d}, p0, [x17, x7, lsl #3]	//  (*)
	.loc 12 52 0
..LDL171:
/*     52 */	add	w7, w1, 8
	.loc 12 55 0
..LDL172:
/* #00001 */	ldr	z1, [x29, 6, mul vl]	//  (*)
	.loc 12 52 0
..LDL173:
/*     52 */	dup	z18.s, w7
	.loc 12 55 0
..LDL174:
/*     55 */	fmad	z1.d, p0/m, z19.d, z14.d
/*     55 */	ftmad	z30.d, z30.d, z8.d, 5
/*     55 */	ftssel	z0.d, z0.d, z6.d
/* #00001 */	ldr	z6, [x29, 8, mul vl]	//  (*)
/*     55 */	fmls	z26.d, p0/m, z2.d, z6.d
/*     55 */	ftmad	z9.d, z9.d, z11.d, 2
/*     55 */	movprfx	z6.d, p0/m, z17.d
/*     55 */	fsubr	z6.d, p0/m, z6.d, #1.0
	.loc 12 52 0
..LDL175:
/* #00001 */	ldr	z17, [x29, 3, mul vl]	//  (*)
/*     52 */	fmul	z27.d, z3.d, z17.d
	.loc 12 55 0
..LDL176:
/*     55 */	fmov	z17.d, 0.000000e+00
/*     55 */	ftmad	z17.d, z17.d, z29.d, 7
/*     55 */	fmul	z31.d, z10.d, z0.d
	.loc 12 52 0
..LDL177:
/*     52 */	index	z0.d, 0, 1
/*     52 */	add	z0.s, z18.s, z0.s
	.loc 12 55 0
..LDL178:
/* #00001 */	ldr	z18, [x29, 5, mul vl]	//  (*)
/*     55 */	fsub	z3.d, z1.d, z14.d
/*     55 */	ftmad	z30.d, z30.d, z8.d, 4
	.loc 12 52 0
..LDL179:
/*     52 */	scvtf	z0.d, p0/m, z0.s
	.loc 12 55 0
..LDL180:
/*     55 */	facgt	p2.d, p0/z, z13.d, z18.d
/* #00001 */	ldr	z18, [x29, 9, mul vl]	//  (*)
/*     55 */	fmls	z26.d, p0/m, z2.d, z18.d
/* #00001 */	ldr	z2, [x29, 4, mul vl]	//  (*)
/*     55 */	ftmad	z9.d, z9.d, z11.d, 1
/*     55 */	fmul	z6.d, z6.d, z15.d
/*     55 */	fmul	z13.d, z27.d, z2.d
/* #00001 */	ldr	z2, [x29, 10, mul vl]	//  (*)
/*     55 */	ftmad	z17.d, z17.d, z29.d, 6
/* #00001 */	ldr	z27, [x29, 2, mul vl]	//  (*)
/*     55 */	sel	z18.d, p2, z2.d, z31.d
/* #00001 */	ldr	z2, [x29, 7, mul vl]	//  (*)
	.loc 12 59 0
..LDL181:
/*     59 */	ldr	x7, [x20, 88]	//  "buff1"
	.loc 12 55 0
..LDL182:
/*     55 */	fmsb	z2.d, p0/m, z3.d, z19.d
/*     55 */	ftmad	z30.d, z30.d, z8.d, 3
/*     55 */	fmul	z18.d, z18.d, z27.d
	.loc 12 52 0
..LDL183:
/*     52 */	fadd	z0.d, p0/m, z0.d, #0.5
	.loc 12 55 0
..LDL184:
/*     55 */	add	x17, x0, 16
/*     55 */	ftsmul	z27.d, z26.d, z25.d
/*     55 */	ftmad	z9.d, z9.d, z11.d, 0
	.loc 12 59 0
..LDL185:
/*     59 */	st1d	{z6.d}, p0, [x7, x17, lsl #3]	//  (*)
	.loc 12 52 0
..LDL186:
/*     52 */	add	w7, w1, 16
	.loc 12 55 0
..LDL187:
/* #00001 */	ldr	z6, [x29, 6, mul vl]	//  (*)
	.loc 12 52 0
..LDL188:
/*     52 */	dup	z10.s, w7
	.loc 12 55 0
..LDL189:
/*     55 */	fmad	z6.d, p0/m, z13.d, z14.d
/*     55 */	ftmad	z17.d, z17.d, z29.d, 5
/*     55 */	ftssel	z5.d, z5.d, z4.d
/* #00001 */	ldr	z4, [x29, 8, mul vl]	//  (*)
/*     55 */	fmls	z2.d, p0/m, z3.d, z4.d
/*     55 */	ftmad	z30.d, z30.d, z8.d, 2
/*     55 */	movprfx	z4.d, p0/m, z18.d
/*     55 */	fsubr	z4.d, p0/m, z4.d, #1.0
	.loc 12 52 0
..LDL190:
/* #00001 */	ldr	z18, [x29, 3, mul vl]	//  (*)
/*     52 */	fmul	z11.d, z0.d, z18.d
/*     52 */	index	z0.d, 0, 1
	.loc 12 55 0
..LDL191:
/*     55 */	fmov	z18.d, 0.000000e+00
/*     55 */	ftmad	z18.d, z18.d, z27.d, 7
/*     55 */	fmul	z31.d, z9.d, z5.d
/* #00001 */	ldr	z9, [x29, 5, mul vl]	//  (*)
	.loc 12 52 0
..LDL192:
/*     52 */	add	z5.s, z10.s, z0.s
	.loc 12 55 0
..LDL193:
/*     55 */	fsub	z0.d, z6.d, z14.d
/*     55 */	ftmad	z17.d, z17.d, z29.d, 4
	.loc 12 52 0
..LDL194:
/*     52 */	scvtf	z5.d, p0/m, z5.s
	.loc 12 55 0
..LDL195:
/*     55 */	facgt	p2.d, p0/z, z24.d, z9.d
/* #00001 */	ldr	z24, [x29, 9, mul vl]	//  (*)
/* #00001 */	ldr	z9, [x29, 7, mul vl]	//  (*)
/*     55 */	fmls	z2.d, p0/m, z3.d, z24.d
/* #00001 */	ldr	z3, [x29, 4, mul vl]	//  (*)
/*     55 */	ftmad	z30.d, z30.d, z8.d, 1
/*     55 */	fmul	z4.d, z4.d, z15.d
/*     55 */	fmul	z24.d, z11.d, z3.d
/* #00001 */	ldr	z3, [x29, 10, mul vl]	//  (*)
/*     55 */	ftmad	z18.d, z18.d, z27.d, 6
/*     55 */	sel	z3.d, p2, z3.d, z31.d
/* #00001 */	ldr	z31, [x29, 2, mul vl]	//  (*)
	.loc 12 59 0
..LDL196:
/*     59 */	ldr	x17, [x20, 88]	//  "buff1"
	.loc 12 55 0
..LDL197:
/*     55 */	fmsb	z9.d, p0/m, z0.d, z13.d
/*     55 */	ftmad	z17.d, z17.d, z29.d, 3
/*     55 */	fmul	z3.d, z3.d, z31.d
	.loc 12 52 0
..LDL198:
/*     52 */	fadd	z5.d, p0/m, z5.d, #0.5
	.loc 12 55 0
..LDL199:
/*     55 */	add	x7, x0, 24
/*     55 */	ftsmul	z31.d, z2.d, z1.d
/*     55 */	ftmad	z30.d, z30.d, z8.d, 0
	.loc 12 59 0
..LDL200:
/*     59 */	st1d	{z4.d}, p0, [x17, x7, lsl #3]	//  (*)
	.loc 12 52 0
..LDL201:
/*     52 */	add	w7, w1, 24
	.loc 12 55 0
..LDL202:
/* #00001 */	ldr	z4, [x29, 6, mul vl]	//  (*)
	.loc 12 52 0
..LDL203:
/*     52 */	dup	z8.s, w7
	.loc 12 55 0
..LDL204:
/*     55 */	fmad	z4.d, p0/m, z24.d, z14.d
/*     55 */	ftmad	z18.d, z18.d, z27.d, 5
/*     55 */	ftssel	z23.d, z23.d, z7.d
/* #00001 */	ldr	z7, [x29, 8, mul vl]	//  (*)
/*     55 */	fmls	z9.d, p0/m, z0.d, z7.d
/*     55 */	ftmad	z17.d, z17.d, z29.d, 2
/*     55 */	movprfx	z7.d, p0/m, z3.d
/*     55 */	fsubr	z7.d, p0/m, z7.d, #1.0
	.loc 12 52 0
..LDL205:
/* #00001 */	ldr	z3, [x29, 3, mul vl]	//  (*)
/*     52 */	fmul	z10.d, z5.d, z3.d
/*     52 */	index	z5.d, 0, 1
	.loc 12 55 0
..LDL206:
/*     55 */	fmov	z3.d, 0.000000e+00
/*     55 */	ftmad	z3.d, z3.d, z31.d, 7
/*     55 */	fmul	z30.d, z30.d, z23.d
	.loc 12 52 0
..LDL207:
/*     52 */	add	z23.s, z8.s, z5.s
	.loc 12 55 0
..LDL208:
/* #00001 */	ldr	z8, [x29, 5, mul vl]	//  (*)
/*     55 */	fsub	z5.d, z4.d, z14.d
/*     55 */	ftmad	z18.d, z18.d, z27.d, 4
	.loc 12 52 0
..LDL209:
/*     52 */	scvtf	z23.d, p0/m, z23.s
	.loc 12 55 0
..LDL210:
/*     55 */	facgt	p2.d, p0/z, z22.d, z8.d
/* #00001 */	ldr	z22, [x29, 9, mul vl]	//  (*)
/* #00001 */	ldr	z8, [x29, 10, mul vl]	//  (*)
/*     55 */	fmsb	z0.d, p0/m, z22.d, z9.d
/* #00001 */	ldr	z22, [x29, 4, mul vl]	//  (*)
/*     55 */	ftmad	z17.d, z17.d, z29.d, 1
/* #00001 */	ldr	z9, [x29, 2, mul vl]	//  (*)
/*     55 */	fmul	z7.d, z7.d, z15.d
/*     55 */	fmul	z22.d, z10.d, z22.d
/*     55 */	ftmad	z3.d, z3.d, z31.d, 6
/*     55 */	sel	z8.d, p2, z8.d, z30.d
/* #00001 */	ldr	z30, [x29, 7, mul vl]	//  (*)
	.loc 12 59 0
..LDL211:
/*     59 */	ldr	x17, [x20, 88]	//  "buff1"
	.loc 12 55 0
..LDL212:
/*     55 */	fmov	z10.d, 0.000000e+00
/*     55 */	fmsb	z30.d, p0/m, z5.d, z24.d
/*     55 */	ftmad	z18.d, z18.d, z27.d, 3
/*     55 */	fmul	z8.d, z8.d, z9.d
	.loc 12 52 0
..LDL213:
/*     52 */	fadd	z23.d, p0/m, z23.d, #0.5
	.loc 12 55 0
..LDL214:
/*     55 */	add	x7, x0, 32
/*     55 */	ftsmul	z12.d, z0.d, z6.d
/*     55 */	ftmad	z17.d, z17.d, z29.d, 0
	.loc 12 59 0
..LDL215:
/*     59 */	st1d	{z7.d}, p0, [x17, x7, lsl #3]	//  (*)
	.loc 12 52 0
..LDL216:
/*     52 */	add	w7, w1, 32
	.loc 12 55 0
..LDL217:
/* #00001 */	ldr	z7, [x29, 6, mul vl]	//  (*)
	.loc 12 52 0
..LDL218:
/*     52 */	dup	z29.s, w7
	.loc 12 55 0
..LDL219:
/*     55 */	fmad	z7.d, p0/m, z22.d, z14.d
/*     55 */	ftmad	z3.d, z3.d, z31.d, 5
/*     55 */	ftssel	z9.d, z28.d, z16.d
/* #00001 */	ldr	z16, [x29, 8, mul vl]	//  (*)
	.loc 12 52 0
..LDL220:
/* #00001 */	ldr	z28, [x29, 3, mul vl]	//  (*)
	.loc 12 55 0
..LDL221:
/*     55 */	fmls	z30.d, p0/m, z5.d, z16.d
/*     55 */	ftmad	z18.d, z18.d, z27.d, 2
/*     55 */	movprfx	z16.d, p0/m, z8.d
/*     55 */	fsubr	z16.d, p0/m, z16.d, #1.0
	.loc 12 52 0
..LDL222:
/*     52 */	fmul	z28.d, z23.d, z28.d
	.loc 12 55 0
..LDL223:
/*     55 */	ftmad	z10.d, z10.d, z12.d, 7
/*     55 */	fmul	z8.d, z17.d, z9.d
	.loc 12 52 0
..LDL224:
/*     52 */	index	z17.d, 0, 1
	.loc 12 55 0
..LDL225:
/*     55 */	fmov	z9.d, 0.000000e+00
	.loc 12 52 0
..LDL226:
/*     52 */	add	z17.s, z29.s, z17.s
	.loc 12 55 0
..LDL227:
/* #00001 */	ldr	z29, [x29, 5, mul vl]	//  (*)
/*     55 */	fsub	z23.d, z7.d, z14.d
/*     55 */	ftmad	z3.d, z3.d, z31.d, 4
	.loc 12 52 0
..LDL228:
/*     52 */	scvtf	z17.d, p0/m, z17.s
	.loc 12 55 0
..LDL229:
/*     55 */	facgt	p2.d, p0/z, z20.d, z29.d
/* #00001 */	ldr	z20, [x29, 9, mul vl]	//  (*)
/*     55 */	fmsb	z5.d, p0/m, z20.d, z30.d
/* #00001 */	ldr	z20, [x29, 4, mul vl]	//  (*)
/*     55 */	ftmad	z18.d, z18.d, z27.d, 1
/* #00001 */	ldr	z30, [x29, 2, mul vl]	//  (*)
/*     55 */	fmul	z16.d, z16.d, z15.d
/*     55 */	fmul	z20.d, z28.d, z20.d
/* #00001 */	ldr	z28, [x29, 10, mul vl]	//  (*)
/*     55 */	ftmad	z10.d, z10.d, z12.d, 6
/*     55 */	sel	z29.d, p2, z28.d, z8.d
/* #00001 */	ldr	z28, [x29, 7, mul vl]	//  (*)
	.loc 12 59 0
..LDL230:
/*     59 */	ldr	x7, [x20, 88]	//  "buff1"
	.loc 12 55 0
..LDL231:
/*     55 */	fmsb	z28.d, p0/m, z23.d, z22.d
/*     55 */	ftmad	z3.d, z3.d, z31.d, 3
/*     55 */	fmul	z29.d, z29.d, z30.d
	.loc 12 52 0
..LDL232:
/*     52 */	fadd	z17.d, p0/m, z17.d, #0.5
	.loc 12 55 0
..LDL233:
/*     55 */	add	x17, x0, 40
/*     55 */	ftsmul	z11.d, z5.d, z4.d
/*     55 */	ftmad	z18.d, z18.d, z27.d, 0
	.loc 12 59 0
..LDL234:
/*     59 */	st1d	{z16.d}, p0, [x7, x17, lsl #3]	//  (*)
	.loc 12 52 0
..LDL235:
/*     52 */	add	w7, w1, 40
	.loc 12 55 0
..LDL236:
/* #00001 */	ldr	z16, [x29, 6, mul vl]	//  (*)
	.loc 12 52 0
..LDL237:
/*     52 */	dup	z27.s, w7
	.loc 12 55 0
..LDL238:
/*     55 */	fmad	z16.d, p0/m, z20.d, z14.d
/*     55 */	ftmad	z10.d, z10.d, z12.d, 5
/*     55 */	ftssel	z30.d, z26.d, z25.d
/* #00001 */	ldr	z25, [x29, 8, mul vl]	//  (*)
	.loc 12 52 0
..LDL239:
/* #00001 */	ldr	z26, [x29, 3, mul vl]	//  (*)
	.loc 12 55 0
..LDL240:
/*     55 */	fmls	z28.d, p0/m, z23.d, z25.d
/*     55 */	ftmad	z3.d, z3.d, z31.d, 2
/*     55 */	movprfx	z25.d, p0/m, z29.d
/*     55 */	fsubr	z25.d, p0/m, z25.d, #1.0
	.loc 12 52 0
..LDL241:
/*     52 */	fmul	z26.d, z17.d, z26.d
/*     52 */	index	z17.d, 0, 1
	.loc 12 55 0
..LDL242:
/*     55 */	ftmad	z9.d, z9.d, z11.d, 7
/*     55 */	fmul	z29.d, z18.d, z30.d
	.loc 12 52 0
..LDL243:
/*     52 */	add	z18.s, z27.s, z17.s
	.loc 12 55 0
..LDL244:
/* #00001 */	ldr	z27, [x29, 5, mul vl]	//  (*)
/*     55 */	fsub	z17.d, z16.d, z14.d
/*     55 */	ftmad	z10.d, z10.d, z12.d, 4
	.loc 12 52 0
..LDL245:
/*     52 */	scvtf	z18.d, p0/m, z18.s
	.loc 12 55 0
..LDL246:
/*     55 */	facgt	p2.d, p0/z, z21.d, z27.d
/* #00001 */	ldr	z21, [x29, 9, mul vl]	//  (*)
/* #00001 */	ldr	z27, [x29, 2, mul vl]	//  (*)
/*     55 */	fmsb	z23.d, p0/m, z21.d, z28.d
/* #00001 */	ldr	z21, [x29, 4, mul vl]	//  (*)
/*     55 */	ftmad	z3.d, z3.d, z31.d, 1
/* #00001 */	ldr	z28, [x29, 7, mul vl]	//  (*)
/*     55 */	fmul	z25.d, z25.d, z15.d
/*     55 */	fmul	z21.d, z26.d, z21.d
/* #00001 */	ldr	z26, [x29, 10, mul vl]	//  (*)
/*     55 */	ftmad	z9.d, z9.d, z11.d, 6
/*     55 */	sel	z26.d, p2, z26.d, z29.d
	.loc 12 59 0
..LDL247:
/*     59 */	ldr	x7, [x20, 88]	//  "buff1"
	.loc 12 55 0
..LDL248:
/*     55 */	fmsb	z28.d, p0/m, z17.d, z20.d
/*     55 */	ftmad	z10.d, z10.d, z12.d, 3
/*     55 */	fmul	z26.d, z26.d, z27.d
	.loc 12 52 0
..LDL249:
/*     52 */	fadd	z18.d, p0/m, z18.d, #0.5
	.loc 12 55 0
..LDL250:
/*     55 */	add	x0, x0, 48
/*     55 */	ftsmul	z8.d, z23.d, z7.d
/*     55 */	ftmad	z3.d, z3.d, z31.d, 0
	.loc 12 59 0
..LDL251:
/*     59 */	st1d	{z25.d}, p0, [x7, x0, lsl #3]	//  (*)
	.loc 12 52 0
..LDL252:
/*     52 */	add	w1, w1, 48
	.loc 12 55 0
..LDL253:
/* #00001 */	ldr	z25, [x29, 6, mul vl]	//  (*)
	.loc 12 52 0
..LDL254:
/*     52 */	dup	z27.s, w1
	.loc 12 55 0
..LDL255:
/*     55 */	fmad	z25.d, p0/m, z21.d, z14.d
/*     55 */	ftmad	z9.d, z9.d, z11.d, 5
/*     55 */	bge	.L250
/*     55 */	ftssel	z2.d, z2.d, z1.d
/*    ??? */	ldr	z1, [x29, 8, mul vl]	//  (*)
/*     55 */	ftmad	z10.d, z10.d, z12.d, 2
/*     55 */	add	x17, x0, 8
/*     55 */	fsubr	z26.d, p0/m, z26.d, #1.0
/*    ??? */	ldr	z29, [x29, 5, mul vl]	//  (*)
/*     55 */	ftssel	z6.d, z0.d, z6.d
/*     55 */	ftmad	z9.d, z9.d, z11.d, 4
/*    ??? */	ldr	z30, [x29, 8, mul vl]	//  (*)
/*     55 */	ftssel	z7.d, z23.d, z7.d
	.loc 12 59 0
..LDL256:
/*     59 */	ldr	x7, [x20, 88]	//  "buff1"
	.loc 12 55 0
..LDL257:
/*     55 */	fmul	z2.d, z3.d, z2.d
	.loc 12 52 0
..LDL258:
/*     52 */	index	z3.d, 0, 1
	.loc 12 55 0
..LDL259:
/*     55 */	ftmad	z10.d, z10.d, z12.d, 1
/*     55 */	fmls	z28.d, p0/m, z17.d, z1.d
	.loc 12 52 0
..LDL260:
/*    ??? */	ldr	z1, [x29, 3, mul vl]	//  (*)
	.loc 12 55 0
..LDL261:
/*     55 */	ftmad	z9.d, z9.d, z11.d, 3
/*     55 */	facgt	p2.d, p0/z, z19.d, z29.d
/*    ??? */	ldr	z19, [x29, 9, mul vl]	//  (*)
	.loc 12 52 0
..LDL262:
/*     52 */	add	z27.s, z27.s, z3.s
	.loc 12 55 0
..LDL263:
/*     55 */	fsub	z3.d, z25.d, z14.d
/*     55 */	ftmad	z10.d, z10.d, z12.d, 0
/*     55 */	ftmad	z9.d, z9.d, z11.d, 2
	.loc 12 52 0
..LDL264:
/*     52 */	scvtf	z27.d, p0/m, z27.s
/*     52 */	fmul	z18.d, z18.d, z1.d
	.loc 12 55 0
..LDL265:
/*     55 */	fmov	z1.d, 0.000000e+00
/*     55 */	fmls	z28.d, p0/m, z17.d, z19.d
/*    ??? */	ldr	z19, [x29, 4, mul vl]	//  (*)
/*     55 */	fmul	z17.d, z26.d, z15.d
/*    ??? */	ldr	z26, [x29, 2, mul vl]	//  (*)
/*     55 */	ftmad	z1.d, z1.d, z8.d, 7
/*     55 */	fmul	z6.d, z10.d, z6.d
/*     55 */	fmov	z10.d, 0.000000e+00
/*     55 */	ftmad	z9.d, z9.d, z11.d, 1
	.loc 12 52 0
..LDL266:
/*     52 */	fadd	z27.d, p0/m, z27.d, #0.5
	.loc 12 59 0
..LDL267:
/*     59 */	st1d	{z17.d}, p0, [x7, x17, lsl #3]	//  (*)
	.loc 12 52 0
..LDL268:
/*     52 */	add	w7, w1, 8
	.loc 12 55 0
..LDL269:
/*     55 */	add	x1, x0, 16
/*     55 */	fmul	z19.d, z18.d, z19.d
/*    ??? */	ldr	z18, [x29, 10, mul vl]	//  (*)
/*    ??? */	ldr	z17, [x29, 6, mul vl]	//  (*)
/*     55 */	ftmad	z1.d, z1.d, z8.d, 6
/*     55 */	ftmad	z9.d, z9.d, z11.d, 0
	.loc 12 59 0
..LDL270:
/*     59 */	ldr	x17, [x20, 88]	//  "buff1"
	.loc 12 55 0
..LDL271:
/*     55 */	sel	z18.d, p2, z18.d, z2.d
/*     55 */	movprfx	z29, z1
/*     55 */	ftmad	z29.d, z29.d, z8.d, 5
	.loc 12 52 0
..LDL272:
/*    ??? */	ldr	z1, [x29, 3, mul vl]	//  (*)
	.loc 12 55 0
..LDL273:
/*    ??? */	ldr	z2, [x29, 7, mul vl]	//  (*)
/*     55 */	fmad	z17.d, p0/m, z19.d, z14.d
/*     55 */	fmul	z26.d, z18.d, z26.d
/*     55 */	ftsmul	z18.d, z28.d, z16.d
/*     55 */	ftmad	z29.d, z29.d, z8.d, 4
	.loc 12 52 0
..LDL274:
/*     52 */	fmul	z1.d, z27.d, z1.d
	.loc 12 55 0
..LDL275:
/*    ??? */	ldr	z27, [x29, 5, mul vl]	//  (*)
/*     55 */	fmsb	z2.d, p0/m, z3.d, z21.d
/*     55 */	movprfx	z0.d, p0/m, z26.d
/*     55 */	fsubr	z0.d, p0/m, z0.d, #1.0
/*     55 */	fmov	z26.d, 0.000000e+00
/*     55 */	ftmad	z26.d, z26.d, z18.d, 7
/*     55 */	ftmad	z29.d, z29.d, z8.d, 3
/*     55 */	fmsb	z30.d, p0/m, z3.d, z2.d
/*     55 */	fsub	z2.d, z17.d, z14.d
/*     55 */	facgt	p2.d, p0/z, z13.d, z27.d
/*    ??? */	ldr	z27, [x29, 9, mul vl]	//  (*)
/*     55 */	fmul	z0.d, z0.d, z15.d
/*     55 */	ftmad	z26.d, z26.d, z18.d, 6
/*     55 */	ftmad	z29.d, z29.d, z8.d, 2
	.loc 12 59 0
..LDL276:
/*     59 */	st1d	{z0.d}, p0, [x17, x1, lsl #3]	//  (*)
	.loc 12 55 0
..LDL277:
/*     55 */	ftssel	z0.d, z5.d, z4.d
/*     55 */	add	x1, x0, 24
/*     55 */	fmov	z5.d, 0.000000e+00
/*    ??? */	ldr	z4, [x29, 5, mul vl]	//  (*)
/*     55 */	fmls	z30.d, p0/m, z3.d, z27.d
/*    ??? */	ldr	z3, [x29, 4, mul vl]	//  (*)
/*    ??? */	ldr	z27, [x29, 7, mul vl]	//  (*)
/*     55 */	ftmad	z26.d, z26.d, z18.d, 5
/*     55 */	ftmad	z29.d, z29.d, z8.d, 1
	.loc 12 59 0
..LDL278:
/*     59 */	ldr	x17, [x20, 88]	//  "buff1"
	.loc 12 55 0
..LDL279:
/*     55 */	fmul	z13.d, z1.d, z3.d
/*    ??? */	ldr	z1, [x29, 10, mul vl]	//  (*)
/*     55 */	fmsb	z27.d, p0/m, z2.d, z19.d
/*    ??? */	ldr	z3, [x29, 2, mul vl]	//  (*)
/*     55 */	ftmad	z29.d, z29.d, z8.d, 0
/*     55 */	ftmad	z26.d, z26.d, z18.d, 4
/*     55 */	sel	z1.d, p2, z1.d, z6.d
/*    ??? */	ldr	z6, [x29, 6, mul vl]	//  (*)
/*     55 */	ftmad	z26.d, z26.d, z18.d, 3
/*     55 */	facgt	p2.d, p0/z, z24.d, z4.d
/*    ??? */	ldr	z4, [x29, 9, mul vl]	//  (*)
/*    ??? */	ldr	z24, [x29, 7, mul vl]	//  (*)
/*     55 */	fmul	z31.d, z1.d, z3.d
/*     55 */	ftsmul	z3.d, z30.d, z25.d
/*    ??? */	ldr	z1, [x29, 8, mul vl]	//  (*)
/*     55 */	ftmad	z26.d, z26.d, z18.d, 2
/*     55 */	fmad	z6.d, p0/m, z13.d, z14.d
/*     55 */	ftmad	z5.d, z5.d, z3.d, 7
/*     55 */	fmsb	z1.d, p0/m, z2.d, z27.d
/*     55 */	movprfx	z27.d, p0/m, z31.d
/*     55 */	fsubr	z27.d, p0/m, z27.d, #1.0
/*     55 */	fmul	z31.d, z9.d, z0.d
/*     55 */	ftmad	z26.d, z26.d, z18.d, 1
/*     55 */	fsub	z0.d, z6.d, z14.d
/*     55 */	ftmad	z5.d, z5.d, z3.d, 6
/*     55 */	fmsb	z2.d, p0/m, z4.d, z1.d
/*    ??? */	ldr	z1, [x29, 10, mul vl]	//  (*)
/*     55 */	fmul	z4.d, z27.d, z15.d
/*     55 */	ftmad	z26.d, z26.d, z18.d, 0
/*    ??? */	ldr	z18, [x29, 10, mul vl]	//  (*)
/*     55 */	fmsb	z24.d, p0/m, z0.d, z13.d
/*     55 */	ftmad	z5.d, z5.d, z3.d, 5
	.loc 12 59 0
..LDL280:
/*     59 */	st1d	{z4.d}, p0, [x17, x1, lsl #3]	//  (*)
	.loc 12 55 0
..LDL281:
/*     55 */	add	x1, x0, 32
/*     55 */	sel	z27.d, p2, z1.d, z31.d
/*     55 */	mov	z1.d, z17.d
/*    ??? */	ldr	z17, [x29, 2, mul vl]	//  (*)
/*    ??? */	ldr	z4, [x29, 8, mul vl]	//  (*)
/*     55 */	ftmad	z5.d, z5.d, z3.d, 4
	.loc 12 59 0
..LDL282:
/*     59 */	ldr	x17, [x20, 88]	//  "buff1"
	.loc 12 55 0
..LDL283:
/*     55 */	fmul	z27.d, z27.d, z17.d
/*     55 */	ftsmul	z17.d, z2.d, z1.d
/*     55 */	fmls	z24.d, p0/m, z0.d, z4.d
/*     55 */	ftssel	z1.d, z2.d, z1.d
/*     55 */	fmov	z4.d, 0.000000e+00
/*     55 */	ftmad	z5.d, z5.d, z3.d, 3
/*     55 */	movprfx	z23.d, p0/m, z27.d
/*     55 */	fsubr	z23.d, p0/m, z23.d, #1.0
/*     55 */	fmul	z27.d, z29.d, z7.d
/*    ??? */	ldr	z7, [x29, 5, mul vl]	//  (*)
/*     55 */	ftmad	z4.d, z4.d, z17.d, 7
/*     55 */	ftmad	z5.d, z5.d, z3.d, 2
/*     55 */	ftmad	z4.d, z4.d, z17.d, 6
/*     55 */	facgt	p2.d, p0/z, z22.d, z7.d
/*    ??? */	ldr	z7, [x29, 9, mul vl]	//  (*)
/*    ??? */	ldr	z22, [x29, 10, mul vl]	//  (*)
/*     55 */	ftmad	z5.d, z5.d, z3.d, 1
/*     55 */	ftmad	z4.d, z4.d, z17.d, 5
/*     55 */	ftmad	z5.d, z5.d, z3.d, 0
/*     55 */	ftssel	z3.d, z30.d, z25.d
/*     55 */	fmsb	z0.d, p0/m, z7.d, z24.d
/*     55 */	fmul	z7.d, z23.d, z15.d
/*    ??? */	ldr	z23, [x29, 2, mul vl]	//  (*)
/*     55 */	sel	z22.d, p2, z22.d, z27.d
/*     55 */	ftmad	z4.d, z4.d, z17.d, 4
/*     55 */	fmul	z3.d, z5.d, z3.d
	.loc 12 59 0
..LDL284:
/*     59 */	st1d	{z7.d}, p0, [x17, x1, lsl #3]	//  (*)
	.loc 12 55 0
..LDL285:
/*     55 */	add	x17, x0, 40
/*     55 */	ftsmul	z12.d, z0.d, z6.d
/*     55 */	add	x0, x0, 48
/*     55 */	ftssel	z0.d, z0.d, z6.d
/*     55 */	fmul	z22.d, z22.d, z23.d
/*    ??? */	ldr	z23, [x29, 5, mul vl]	//  (*)
	.loc 12 59 0
..LDL286:
/*     59 */	ldr	x1, [x20, 88]	//  "buff1"
	.loc 12 55 0
..LDL287:
/*     55 */	ftmad	z4.d, z4.d, z17.d, 3
/*     55 */	ftmad	z10.d, z10.d, z12.d, 7
/*     55 */	fsubr	z22.d, p0/m, z22.d, #1.0
/*     55 */	facgt	p2.d, p0/z, z20.d, z23.d
/*     55 */	ftssel	z20.d, z28.d, z16.d
/*     55 */	facgt	p3.d, p0/z, z21.d, z23.d
/*     55 */	ftmad	z4.d, z4.d, z17.d, 2
/*     55 */	fmul	z7.d, z26.d, z20.d
/*     55 */	ftmad	z10.d, z10.d, z12.d, 6
/*     55 */	fmul	z16.d, z22.d, z15.d
/*     55 */	ftmad	z4.d, z4.d, z17.d, 1
/*     55 */	sel	z18.d, p2, z18.d, z7.d
/*    ??? */	ldr	z7, [x29, 2, mul vl]	//  (*)
/*     55 */	ftmad	z10.d, z10.d, z12.d, 5
/*     55 */	facgt	p2.d, p0/z, z19.d, z23.d
	.loc 12 59 0
..LDL288:
/*     59 */	st1d	{z16.d}, p0, [x1, x17, lsl #3]	//  (*)
/*     59 */	ldr	x1, [x20, 88]	//  "buff1"
	.loc 12 55 0
..LDL289:
/*     55 */	ftmad	z10.d, z10.d, z12.d, 4
/*     55 */	fmul	z7.d, z18.d, z7.d
/*     55 */	ftmad	z10.d, z10.d, z12.d, 3
/*     55 */	fsubr	z7.d, p0/m, z7.d, #1.0
/*     55 */	ftmad	z10.d, z10.d, z12.d, 2
/*     55 */	fmul	z5.d, z7.d, z15.d
/*    ??? */	ldr	z7, [x29, 10, mul vl]	//  (*)
/*     55 */	ftmad	z10.d, z10.d, z12.d, 1
	.loc 12 59 0
..LDL290:
/*     59 */	st1d	{z5.d}, p0, [x1, x0, lsl #3]	//  (*)
	.loc 12 55 0
..LDL291:
/*     55 */	add	x1, x0, 8
/*     55 */	sel	z3.d, p3, z7.d, z3.d
/*    ??? */	ldr	z7, [x29, 2, mul vl]	//  (*)
	.loc 12 59 0
..LDL292:
/*     59 */	ldr	x17, [x20, 88]	//  "buff1"
	.loc 12 55 0
..LDL293:
/*     55 */	ftmad	z10.d, z10.d, z12.d, 0
/*     55 */	fmul	z26.d, z3.d, z7.d
/*     55 */	movprfx	z3, z4
/*     55 */	ftmad	z3.d, z3.d, z17.d, 0
/*     55 */	fmul	z0.d, z10.d, z0.d
/*     55 */	movprfx	z2.d, p0/m, z26.d
/*     55 */	fsubr	z2.d, p0/m, z2.d, #1.0
/*     55 */	fmul	z3.d, z3.d, z1.d
/*     55 */	fmul	z1.d, z2.d, z15.d
/*    ??? */	ldr	z2, [x29, 10, mul vl]	//  (*)
	.loc 12 59 0
..LDL294:
/*     59 */	st1d	{z1.d}, p0, [x17, x1, lsl #3]	//  (*)
	.loc 12 55 0
..LDL295:
/*     55 */	add	x1, x0, 16
/*     55 */	sel	z2.d, p2, z2.d, z3.d
/*     55 */	facgt	p2.d, p0/z, z13.d, z23.d
/*     55 */	fmul	z1.d, z2.d, z7.d
/*    ??? */	ldr	z2, [x29, 10, mul vl]	//  (*)
	.loc 12 59 0
..LDL296:
/*     59 */	ldr	x17, [x20, 88]	//  "buff1"
	.loc 12 55 0
..LDL297:
/*     55 */	fsubr	z1.d, p0/m, z1.d, #1.0
/*     55 */	sel	z2.d, p2, z2.d, z0.d
/*     55 */	fmul	z0.d, z2.d, z7.d
/*     55 */	fmul	z1.d, z1.d, z15.d
/*     55 */	fsubr	z0.d, p0/m, z0.d, #1.0
	.loc 12 59 0
..LDL298:
/*     59 */	st1d	{z1.d}, p0, [x17, x1, lsl #3]	//  (*)
	.loc 12 55 0
..LDL299:
/*     55 */	add	x1, x0, 24
/*     55 */	fmul	z0.d, z0.d, z15.d
	.loc 12 59 0
..LDL300:
/*     59 */	ldr	x17, [x20, 88]	//  "buff1"
/*     59 */	st1d	{z0.d}, p0, [x17, x1, lsl #3]	//  (*)
	.loc 12 55 0
..LDL301:
/*     55 */	add	x1, x0, 32
/*     55 */	cbz	w2, .L306
.L309:
	.p2align 5
.L312:					// :entr:term:mod:swpl
	.loc 12 52 0 is_stmt 1
..LDL302:
/*     52 */	index	z0.d, 0, 1
	.loc 12 55 0
..LDL303:
/* #00002 */	ldr	z2, [x29, 7, mul vl]	//  (*)
/*     55 */	subs	w2, w2, 1
	.loc 12 52 0
..LDL304:
/*     52 */	dup	z1.s, w7
	.loc 12 59 0
..LDL305:
/*     59 */	ldr	x0, [x20, 88]	//  "buff1"
	.loc 12 52 0 is_stmt 0
..LDL306:
/*     52 */	add	w7, w7, 8
	.loc 12 52 0 is_stmt 1
..LDL307:
/*     52 */	add	z0.s, z1.s, z0.s
/* #00002 */	ldr	z1, [x29, 3, mul vl]	//  (*)
/*     52 */	scvtf	z0.d, p0/m, z0.s
/*     52 */	fadd	z0.d, p0/m, z0.d, #0.5
/*     52 */	fmul	z1.d, z0.d, z1.d
	.loc 12 55 0
..LDL308:
/* #00002 */	ldr	z0, [x29, 4, mul vl]	//  (*)
/*     55 */	fmul	z1.d, z1.d, z0.d
/* #00002 */	ldr	z0, [x29, 5, mul vl]	//  (*)
/*     55 */	facgt	p2.d, p0/z, z1.d, z0.d
/* #00002 */	ldr	z0, [x29, 6, mul vl]	//  (*)
/*     55 */	fmad	z0.d, p0/m, z1.d, z14.d
/*     55 */	fsub	z3.d, z0.d, z14.d
/*     55 */	fmls	z1.d, p0/m, z3.d, z2.d
/* #00002 */	ldr	z2, [x29, 8, mul vl]	//  (*)
/*     55 */	fmls	z1.d, p0/m, z3.d, z2.d
/* #00002 */	ldr	z2, [x29, 9, mul vl]	//  (*)
/*     55 */	fmls	z1.d, p0/m, z3.d, z2.d
/*     55 */	ftsmul	z2.d, z1.d, z0.d
/*     55 */	ftssel	z1.d, z1.d, z0.d
/*     55 */	fmov	z0.d, 0.000000e+00
/*     55 */	ftmad	z0.d, z0.d, z2.d, 7
/*     55 */	ftmad	z0.d, z0.d, z2.d, 6
/*     55 */	ftmad	z0.d, z0.d, z2.d, 5
/*     55 */	ftmad	z0.d, z0.d, z2.d, 4
/*     55 */	ftmad	z0.d, z0.d, z2.d, 3
/*     55 */	ftmad	z0.d, z0.d, z2.d, 2
/*     55 */	ftmad	z0.d, z0.d, z2.d, 1
/*     55 */	ftmad	z0.d, z0.d, z2.d, 0
/*     55 */	fmul	z1.d, z0.d, z1.d
/* #00002 */	ldr	z0, [x29, 10, mul vl]	//  (*)
/*     55 */	sel	z1.d, p2, z0.d, z1.d
/* #00002 */	ldr	z0, [x29, 2, mul vl]	//  (*)
/*     55 */	fmul	z0.d, z1.d, z0.d
/*     55 */	fsubr	z0.d, p0/m, z0.d, #1.0
/*     55 */	fmul	z0.d, z0.d, z15.d
	.loc 12 59 0
..LDL309:
/*     59 */	st1d	{z0.d}, p0, [x0, x1, lsl #3]	//  (*)
	.loc 12 55 0 is_stmt 0
..LDL310:
/*     55 */	add	x1, x1, 8
/*     55 */	bne	.L312
.L306:
.L252:
	.loc 12 50 0 is_stmt 1
..LDL311:
/*     50 */	cbz	w5, .L255
	.loc 12 55 0
..LDL312:
/*     55 */	ld1rd	{z7.d}, p0/z, [x8, 16]	//  "__fj_dsin_const_tbl_"
/*     55 */	ld1rd	{z1.d}, p0/z, [x8, 48]	//  "__fj_dsin_const_tbl_"
/*    ??? */	ldr	z5, [x29, 1, mul vl]	//  (*)
/*     55 */	ld1rd	{z4.d}, p0/z, [x8, 24]	//  "__fj_dsin_const_tbl_"
/*     55 */	ld1rd	{z2.d}, p0/z, [x8]	//  "__fj_dsin_const_tbl_"
/*     55 */	ld1rd	{z3.d}, p0/z, [x8, 32]	//  "__fj_dsin_const_tbl_"
/*     55 */	ld1rd	{z0.d}, p0/z, [x8, 40]	//  "__fj_dsin_const_tbl_"
/*     55 */	ld1rd	{z6.d}, p0/z, [x8, 56]	//  "__fj_dsin_const_tbl_"
	.loc 12 59 0
..LDL313:
/*     59 */	ldr	x0, [x20, 88]	//  "buff1"
	.loc 12 55 0
..LDL314:
/*     55 */	fmad	z5.d, p1/m, z1.d, z7.d
/*    ??? */	ldr	z1, [x29, 1, mul vl]	//  (*)
/*     55 */	facgt	p2.d, p1/z, z1.d, z2.d
/*     55 */	fsub	z2.d, z5.d, z7.d
/*     55 */	fmls	z1.d, p1/m, z2.d, z4.d
/*     55 */	fmls	z1.d, p1/m, z2.d, z3.d
/*     55 */	fmls	z1.d, p1/m, z2.d, z0.d
/*     55 */	fmov	z0.d, 0.000000e+00
/*     55 */	ftsmul	z2.d, z1.d, z5.d
/*     55 */	ftssel	z1.d, z1.d, z5.d
/*     55 */	ftmad	z0.d, z0.d, z2.d, 7
/*     55 */	ftmad	z0.d, z0.d, z2.d, 6
/*     55 */	ftmad	z0.d, z0.d, z2.d, 5
/*     55 */	ftmad	z0.d, z0.d, z2.d, 4
/*     55 */	ftmad	z0.d, z0.d, z2.d, 3
/*     55 */	ftmad	z0.d, z0.d, z2.d, 2
/*     55 */	ftmad	z0.d, z0.d, z2.d, 1
/*     55 */	ftmad	z0.d, z0.d, z2.d, 0
/*     55 */	fmul	z0.d, z0.d, z1.d
/*     55 */	sel	z1.d, p2, z6.d, z0.d
/*    ??? */	ldr	z0, [x29, 2, mul vl]	//  (*)
/*     55 */	fmul	z0.d, z1.d, z0.d
/*     55 */	fsubr	z0.d, p1/m, z0.d, #1.0
/*     55 */	fmul	z0.d, z0.d, z15.d
	.loc 12 59 0
..LDL315:
/*     59 */	st1d	{z0.d}, p1, [x0, x10, lsl #3]	//  (*)
.L255:					// :term
	.loc 12 61 0
..LDL316:
/*     61 */	add	w9, w9, 1
/*     61 */	add	x14, x14, x12
	.loc 12 55 0 is_stmt 0
..LDL317:
/*     55 */	fmov	d1, 1.000000e+00
	.loc 12 61 0
..LDL318:
/*    ??? */	ldr	d0, [x19, 16]	//  (*)
/*     61 */	add	x10, x10, x12
/*     61 */	cmp	w9, w13
/*     61 */	fadd	d0, d0, d1
/*    ??? */	str	d0, [x19, 16]	//  (*)
/*     61 */	blt	.L246
.L257:					// :term
	.loc 12 62 0 is_stmt 1
..LDL319:
/*     62 */	add	w16, w16, 1
/*     62 */	cmp	w16, w4
/*     62 */	blt	.L243
.L259:
/*     63 */	add	x0, x19, 32
/*     63 */	ldr	x0, [x0]
/*     63 */	bl	__mpc_obar
	.loc 12 63 0
..LDL320:
/*    ??? */	ldp	x19, x20, [x29, -16]	//  (*)
	.cfi_restore 19
	.cfi_restore 20
/*    ??? */	ldr	x21, [x29, -24]	//  (*)
	.cfi_restore 21
/*    ??? */	ldp	d8, d9, [x29, -40]	//  (*)
	.cfi_restore 72
	.cfi_restore 73
/*    ??? */	ldp	d10, d11, [x29, -56]	//  (*)
	.cfi_restore 74
	.cfi_restore 75
/*    ??? */	ldp	d12, d13, [x29, -72]	//  (*)
	.cfi_restore 76
	.cfi_restore 77
/*    ??? */	ldp	d14, d15, [x29, -88]	//  (*)
	.cfi_restore 78
	.cfi_restore 79
/*    ??? */	add	sp, x29, 0
/*    ??? */	ldp	x29, x30, [sp]	//  (*)
	.cfi_restore 29
	.cfi_restore 30
/*    ??? */	addvl	sp, sp, 11
	.cfi_def_cfa_offset 0
/*     63 */	ret	
..D3.pchi:
	.cfi_endproc
.LFE2:
	.size	init_ker40._OMP_1, .-init_ker40._OMP_1
	.ident	"$Compiler: Fujitsu C/C++ Compiler 4.2.0 (Jun 15 2020 13:54:08) diffusion_ker40.c diffusion_ker40 $"
	.text
	.align	2
	.global	diffusion_ker40
	.type	diffusion_ker40, %function
diffusion_ker40:
	.loc 12 67 0
..LDL321:
.LFB3:
	.cfi_startproc
/*    ??? */	str	x30, [sp, -16]!	//  (*)
	.cfi_def_cfa_offset 16
	.cfi_offset 30, -16
/*   2563 */	mov	x8, x1
/*    ??? */	sub	sp, sp, 304
	.cfi_def_cfa_offset 320
/*     67 */	add	x1, sp, 144
/*   2563 */	ldr	x9, [sp, 320]	//  (*)
/*     67 */	stp	x7, x6, [x1, 8]
/*     67 */	str	x5, [x1, 24]
/*     67 */	str	d7, [x1, 32]
/*     67 */	str	x9, [x1]
/*     67 */	stp	d6, d5, [x1, 40]
/*     67 */	stp	d4, d3, [x1, 56]
/*     67 */	stp	d2, d1, [x1, 72]
/*     67 */	str	d0, [x1, 88]
/*     67 */	stp	w4, w3, [x1, 96]
/*     67 */	str	w2, [x1, 104]
/*     67 */	stp	x8, x0, [x1, 112]
	.loc 12 72 0 is_stmt 0
..LDL322:
/*     72 */	mov	x2, 0
/*     72 */	adrp	x0, diffusion_ker40._OMP_2
/*     72 */	add	x0, x0, :lo12:diffusion_ker40._OMP_2
/*     72 */	bl	__mpc_opar
	.loc 12 2564 0 is_stmt 1
..LDL323:
/*    ??? */	add	sp, sp, 304
	.cfi_def_cfa_offset 16
/*    ??? */	ldr	x30, [sp], 16	//  (*)
	.cfi_restore 30
	.cfi_def_cfa_offset 0
/*   2564 */	ret	
..D4.pchi:
	.cfi_endproc
.LFE3:
	.size	diffusion_ker40, .-diffusion_ker40
	.ident	"$Compiler: Fujitsu C/C++ Compiler 4.2.0 (Jun 15 2020 13:54:08) diffusion_ker40.c diffusion_ker40._OMP_2 $"
	.text
	.align	2
	.type	diffusion_ker40._OMP_2, %function
diffusion_ker40._OMP_2:
	.loc 12 72 0
..LDL324:
.LFB4:
	.cfi_startproc
/*    ??? */	addvl	sp, sp, -8
/*    ??? */	stp	x29, x30, [sp]	//  (*)
/*     64 */	add	x29, sp, 0
	.cfi_escape 0xf,0xb,0x92,0x1d,0x0,0x11,0xc0,0x0,0x92,0x2e,0x0,0x1e,0x22
	.cfi_escape 0x10,0x1d,0x7,0x11,0x40,0x92,0x2e,0x0,0x1e,0x22
	.cfi_escape 0x10,0x1e,0xa,0x11,0x8,0x22,0x11,0x40,0x92,0x2e,0x0,0x1e,0x22
/*     64 */	sub	sp, sp, 464
/*    ??? */	stp	x19, x20, [x29, -16]	//  (*)
	.cfi_escape 0x10,0x13,0xa,0x11,0x70,0x22,0x11,0x40,0x92,0x2e,0x0,0x1e,0x22
	.cfi_escape 0x10,0x14,0xa,0x11,0x78,0x22,0x11,0x40,0x92,0x2e,0x0,0x1e,0x22
/*    ??? */	stp	d8, d9, [x29, -96]	//  (*)
	.cfi_escape 0x10,0x68,0xb,0x11,0xa0,0x7f,0x22,0x11,0x40,0x92,0x2e,0x0,0x1e,0x22
	.cfi_escape 0x10,0x69,0xb,0x11,0xa8,0x7f,0x22,0x11,0x40,0x92,0x2e,0x0,0x1e,0x22
/*     64 */	add	x19, sp, 0
/*    ??? */	stp	x21, x22, [x29, -32]	//  (*)
	.cfi_escape 0x10,0x15,0xa,0x11,0x60,0x22,0x11,0x40,0x92,0x2e,0x0,0x1e,0x22
	.cfi_escape 0x10,0x16,0xa,0x11,0x68,0x22,0x11,0x40,0x92,0x2e,0x0,0x1e,0x22
/*    ??? */	stp	d10, d11, [x29, -112]	//  (*)
	.cfi_escape 0x10,0x6a,0xb,0x11,0x90,0x7f,0x22,0x11,0x40,0x92,0x2e,0x0,0x1e,0x22
	.cfi_escape 0x10,0x6b,0xb,0x11,0x98,0x7f,0x22,0x11,0x40,0x92,0x2e,0x0,0x1e,0x22
/*    ??? */	stp	x23, x24, [x29, -48]	//  (*)
	.cfi_escape 0x10,0x17,0xa,0x11,0x50,0x22,0x11,0x40,0x92,0x2e,0x0,0x1e,0x22
	.cfi_escape 0x10,0x18,0xa,0x11,0x58,0x22,0x11,0x40,0x92,0x2e,0x0,0x1e,0x22
/*    ??? */	stp	x25, x26, [x29, -64]	//  (*)
	.cfi_escape 0x10,0x19,0xa,0x11,0x40,0x22,0x11,0x40,0x92,0x2e,0x0,0x1e,0x22
	.cfi_escape 0x10,0x1a,0xa,0x11,0x48,0x22,0x11,0x40,0x92,0x2e,0x0,0x1e,0x22
/*    ??? */	stp	x27, x28, [x29, -80]	//  (*)
	.cfi_escape 0x10,0x1b,0xb,0x11,0xb0,0x7f,0x22,0x11,0x40,0x92,0x2e,0x0,0x1e,0x22
	.cfi_escape 0x10,0x1c,0xb,0x11,0xb8,0x7f,0x22,0x11,0x40,0x92,0x2e,0x0,0x1e,0x22
/*     64 */	and	sp, x19, -64
/*    ??? */	str	x0, [x19, 176]	//  (*)
/*     64 */	str	x1, [x19, 208]
/*     64 */	str	x2, [x19, 200]
/*     64 */	str	x3, [x19, 192]
/*     64 */	str	x4, [x19, 184]
	.loc 12 74 0
..LDL325:
/*     74 */	eor	v10.8b, v10.8b, v10.8b
	.loc 12 75 0
..LDL326:
/*     75 */	mov	w0, 0
/*    ??? */	str	w0, [x19, 164]	//  (*)
	.loc 12 76 0
..LDL327:
/*    ??? */	ldr	x0, [x19, 176]	//  (*)
/*     76 */	ldr	x20, [x0, 120]	//  "f1"
	.loc 12 77 0
..LDL328:
/*     77 */	ldr	x0, [x0, 112]	//  "f2"
/*    ??? */	str	x0, [x19, 16]	//  (*)
/*     80 */	bl	omp_get_thread_num
/*     80 */	mov	w21, w0
	.loc 12 81 0
..LDL329:
/*     81 */	bl	omp_get_num_threads
	.loc 12 90 0
..LDL330:
/*     90 */	sub	w4, w0, 1
/*    ??? */	ldr	x0, [x19, 176]	//  (*)
	.loc 12 107 0
..LDL331:
/*    107 */	ptrue	p0.d, ALL
/*    ??? */	ldr	x7, [x19, 176]	//  (*)
	.loc 12 87 0
..LDL332:
/*     87 */	mov	x2, 715784192
	.loc 12 90 0
..LDL333:
/*     90 */	sxtw	x1, w4
	.loc 12 167 0
..LDL334:
	.loc 12 2555 0 is_stmt 0
..LDL335:
/*   2555 */	fmov	d8, 5.000000e-01
/*   2555 */	adrp	x8, .LCP1
	.loc 12 87 0 is_stmt 1
..LDL336:
/*     87 */	movk	x2, 43691, lsl #0
	.loc 12 167 0
..LDL337:
	.loc 12 2555 0 is_stmt 0
..LDL338:
/*   2555 */	ldr	d9, [x8, :lo12:.LCP1]	//  1.000000e-01
	.loc 12 90 0 is_stmt 1
..LDL339:
/*     90 */	mul	x1, x1, x2
/*     90 */	ldr	w5, [x0, 96]	//  "nz"
	.loc 12 107 0
..LDL340:
/*    107 */	ld1rd	{z0.d}, p0/z, [x7, 40]	//  "cc"
	.loc 12 89 0
..LDL341:
/*     89 */	ldr	w6, [x0, 100]	//  "ny"
	.loc 12 87 0
..LDL342:
/*     87 */	sxtw	x0, w21
/*     87 */	mul	x0, x0, x2
	.loc 12 90 0
..LDL343:
/*     90 */	lsr	x1, x1, 32
/*     90 */	asr	w1, w1, 1
	.loc 12 89 0
..LDL344:
/*     89 */	sub	w3, w6, 1
	.loc 12 90 0
..LDL345:
/*     90 */	sub	w4, w1, w4, asr #31
	.loc 12 87 0
..LDL346:
/*     87 */	lsr	x0, x0, 32
/*     87 */	asr	w0, w0, 1
/*     87 */	sub	w1, w0, w21, asr #31
	.loc 12 90 0
..LDL347:
/*     90 */	add	w0, w4, 1
	.loc 12 107 0
..LDL348:
/*    ??? */	str	z0, [x29, 1, mul vl]	//  (*)
	.loc 12 90 0
..LDL349:
/*     90 */	sdiv	w4, w5, w0
	.loc 12 108 0
..LDL350:
/*    108 */	ld1rd	{z0.d}, p0/z, [x7, 80]	//  "cw"
	.loc 12 88 0
..LDL351:
/*     88 */	add	w0, w1, w1
/*     88 */	add	w0, w0, w1
	.loc 12 108 0
..LDL352:
/*    ??? */	str	z0, [x29, 2, mul vl]	//  (*)
	.loc 12 109 0
..LDL353:
/*    109 */	ld1rd	{z0.d}, p0/z, [x7, 88]	//  "ce"
/*    ??? */	str	z0, [x29, 3, mul vl]	//  (*)
	.loc 12 110 0
..LDL354:
/*    110 */	ld1rd	{z0.d}, p0/z, [x7, 72]	//  "cn"
/*    ??? */	str	z0, [x29, 4, mul vl]	//  (*)
	.loc 12 111 0
..LDL355:
/*    111 */	ld1rd	{z0.d}, p0/z, [x7, 64]	//  "cs"
/*    ??? */	str	z0, [x29, 5, mul vl]	//  (*)
	.loc 12 112 0
..LDL356:
/*    112 */	ld1rd	{z0.d}, p0/z, [x7, 48]	//  "cb"
	.loc 12 89 0
..LDL357:
/*     89 */	sxtw	x7, w3
/*     89 */	mul	x8, x7, x2
	.loc 12 88 0
..LDL358:
/*     88 */	sub	w7, w21, w0, lsl #2
	.loc 12 92 0
..LDL359:
/*     92 */	add	w2, w1, 1
	.loc 12 91 0
..LDL360:
/*     91 */	mul	w1, w4, w1
	.loc 12 92 0
..LDL361:
/*     92 */	mul	w2, w4, w2
	.loc 12 89 0
..LDL362:
/*     89 */	lsr	x0, x8, 32
/*     89 */	asr	w8, w0, 1
	.loc 12 94 0
..LDL363:
/*     94 */	add	w0, w7, 1
	.loc 12 89 0
..LDL364:
/*     89 */	sub	w3, w8, w3, asr #31
	.loc 12 92 0
..LDL365:
/*     92 */	cmp	w5, w2
	.loc 12 112 0
..LDL366:
/*    ??? */	str	z0, [x29, 6, mul vl]	//  (*)
	.loc 12 89 0
..LDL367:
/*     89 */	add	w3, w3, 1
	.loc 12 94 0
..LDL368:
/*     94 */	mul	w0, w3, w0
	.loc 12 93 0
..LDL369:
/*     93 */	mul	w3, w3, w7
	.loc 12 113 0
..LDL370:
/*    ??? */	ldr	x9, [x19, 176]	//  (*)
/*    113 */	ld1rd	{z0.d}, p0/z, [x9, 56]	//  "ct"
/*    ??? */	str	z0, [x29, 7, mul vl]	//  (*)
	.loc 12 91 0
..LDL371:
/*    ??? */	str	w1, [x19, 56]	//  (*)
	.loc 12 92 0
..LDL372:
/*     92 */	csel	w1, w5, w2, le
	.loc 12 94 0
..LDL373:
/*     94 */	cmp	w6, w0
	.loc 12 93 0
..LDL374:
/*    ??? */	str	w3, [x19, 80]	//  (*)
	.loc 12 94 0
..LDL375:
/*     94 */	csel	w0, w6, w0, le
	.loc 12 92 0
..LDL376:
/*    ??? */	str	w1, [x19, 48]	//  (*)
	.loc 12 94 0
..LDL377:
/*    ??? */	str	w0, [x19, 72]	//  (*)
	.loc 12 167 0
..LDL378:
	.loc 12 171 0 is_stmt 0
..LDL379:
/*    171 */	sub	w0, w0, w3
/*    ??? */	str	w0, [x19, 76]	//  (*)
/*    ??? */	ldr	w1, [x19, 56]	//  (*)
/*    ??? */	ldr	w2, [x19, 48]	//  (*)
/*    168 */	sub	w1, w2, w1
/*    ??? */	str	w1, [x19, 52]	//  (*)
.L263:					// :entr
	.loc 12 167 0
..LDL380:
/*    ??? */	ldr	x0, [x19, 16]	//  (*)
/*    ??? */	str	x20, [x19, 16]	//  (*)
/*    ??? */	str	x0, [x19, 168]	//  (*)
	.loc 12 168 0 is_stmt 1
..LDL381:
/*    ??? */	ldr	w0, [x19, 56]	//  (*)
/*    ??? */	ldr	w1, [x19, 48]	//  (*)
/*    168 */	cmp	w0, w1
/*    168 */	bge	.L278
	.loc 12 169 0 is_stmt 0
..LDL382:
/*    ??? */	ldr	x3, [x19, 176]	//  (*)
	.loc 12 426 0
..LDL383:
/*    ??? */	ldr	w0, [x19, 56]	//  (*)
/*    ??? */	str	w0, [x19, 160]	//  (*)
	.loc 12 169 0
..LDL384:
/*    169 */	ldr	w2, [x3, 104]	//  "nx"
	.loc 12 426 0
..LDL385:
/*    ??? */	ldr	w0, [x19, 52]	//  (*)
	.loc 12 169 0
..LDL386:
/*    169 */	ldr	w1, [x3, 100]	//  "ny"
	.loc 12 426 0
..LDL387:
/*    ??? */	str	w0, [x19, 156]	//  (*)
	.loc 12 169 0
..LDL388:
/*    169 */	neg	w4, w2
/*    ??? */	stp	w1, w2, [x19, 84]	//  (*)
	.loc 12 170 0
..LDL389:
/*    170 */	ldr	w0, [x3, 96]	//  "nz"
	.loc 12 646 0
..LDL390:
/*    646 */	sub	w3, w2, 65
	.loc 12 169 0
..LDL391:
/*    ??? */	str	w4, [x19, 96]	//  (*)
/*    169 */	mul	w4, w1, w4
	.loc 12 170 0
..LDL392:
/*    170 */	mul	w1, w1, w2
/*    170 */	sub	w0, w0, 1
/*    ??? */	str	w0, [x19, 64]	//  (*)
	.loc 12 426 0
..LDL393:
/*    ??? */	ldr	w0, [x19, 84]	//  (*)
	.loc 12 169 0
..LDL394:
/*    ??? */	str	w4, [x19, 68]	//  (*)
	.loc 12 646 0
..LDL395:
/*    646 */	asr	w4, w3, 5
	.loc 12 170 0
..LDL396:
/*    ??? */	str	w1, [x19, 60]	//  (*)
	.loc 12 646 0
..LDL397:
/*    646 */	add	w1, w3, w4, lsr #26
	.loc 12 426 0
..LDL398:
/*    426 */	sub	w0, w0, 1
/*    ??? */	str	w0, [x19, 92]	//  (*)
	.loc 12 646 0
..LDL399:
/*    646 */	sub	w0, w2, 64
/*    ??? */	str	w0, [x19, 100]	//  (*)
/*    646 */	asr	w0, w1, 6
/*    ??? */	str	w0, [x19, 116]	//  (*)
.L266:					// :entr
	.loc 12 169 0 is_stmt 1
..LDL400:
/*    ??? */	ldr	w0, [x19, 160]	//  (*)
/*    ??? */	ldr	w1, [x19, 68]	//  (*)
/*    169 */	cmp	w0, 0
/*    169 */	csel	w1, w1, wzr, ne
/*    ??? */	str	w1, [x19, 152]	//  (*)
	.loc 12 170 0
..LDL401:
/*    ??? */	ldr	w1, [x19, 64]	//  (*)
/*    170 */	cmp	w1, w0
/*    ??? */	ldr	w0, [x19, 60]	//  (*)
/*    170 */	csel	w0, w0, wzr, ne
/*    ??? */	str	w0, [x19, 148]	//  (*)
	.loc 12 171 0
..LDL402:
/*    ??? */	ldr	w1, [x19, 80]	//  (*)
/*    ??? */	ldr	w0, [x19, 72]	//  (*)
/*    171 */	cmp	w1, w0
/*    171 */	bge	.L276
	.loc 12 429 0 is_stmt 0
..LDL403:
/*    ??? */	ldr	w0, [x19, 80]	//  (*)
/*    ??? */	str	w0, [x19, 144]	//  (*)
/*    ??? */	ldr	w0, [x19, 76]	//  (*)
/*    ??? */	str	w0, [x19, 140]	//  (*)
.L269:					// :entr
	.loc 12 425 0 is_stmt 1
..LDL404:
/*    ??? */	ldr	w0, [x19, 144]	//  (*)
/*    ??? */	ldr	w1, [x19, 96]	//  (*)
	.loc 12 445 0
..LDL405:
	.loc 12 107 0 is_stmt 0
..LDL406:
/*    107 */	ptrue	p0.d, ALL
	.loc 12 429 0 is_stmt 1
..LDL407:
/*    ??? */	ldr	w3, [x19, 160]	//  (*)
	.loc 12 425 0
..LDL408:
/*    425 */	cmp	w0, 0
/*    425 */	csel	w1, wzr, w1, eq
/*    ??? */	str	w1, [x19, 12]	//  (*)
	.loc 12 426 0
..LDL409:
/*    ??? */	ldr	w1, [x19, 92]	//  (*)
/*    426 */	cmp	w1, w0
/*    ??? */	ldr	w1, [x19, 88]	//  (*)
/*    426 */	csel	w2, w1, wzr, ne
/*    ??? */	str	w2, [x19, 8]	//  (*)
	.loc 12 429 0
..LDL410:
/*    ??? */	ldr	w2, [x19, 84]	//  (*)
/*    429 */	madd	w0, w2, w3, w0
/*    429 */	mul	w17, w0, w1
	.loc 12 449 0
..LDL411:
/*    ??? */	ldr	x0, [x19, 16]	//  (*)
	.loc 12 440 0
..LDL412:
/*    440 */	add	w13, w17, 32
	.loc 12 439 0
..LDL413:
/*    439 */	add	w14, w17, 24
	.loc 12 449 0
..LDL414:
/*    449 */	sxtw	x1, w13
	.loc 12 448 0
..LDL415:
/*    448 */	sxtw	x6, w14
/*    448 */	mov	x2, x0
	.loc 12 449 0
..LDL416:
/*    449 */	add	x9, x0, x1, lsl #3
	.loc 12 437 0
..LDL417:
/*    437 */	add	w16, w17, 8
	.loc 12 448 0
..LDL418:
/*    ??? */	str	x6, [x19, 104]	//  (*)
	.loc 12 446 0
..LDL419:
/*    446 */	sxtw	x0, w16
	.loc 12 430 0
..LDL420:
/*    430 */	sxtw	x7, w17
	.loc 12 446 0
..LDL421:
/*    446 */	add	x20, x2, x0, lsl #3
	.loc 12 430 0
..LDL422:
/*    430 */	add	x5, x7, 2
/*    430 */	ldr	d22, [x2, x7, lsl #3]
	.loc 12 441 0
..LDL423:
/*    441 */	add	w12, w17, 40
	.loc 12 430 0
..LDL424:
/*    430 */	add	x8, x7, 4
/*    430 */	ldr	d16, [x2, x5, lsl #3]
	.loc 12 445 0
..LDL425:
/*    445 */	add	x22, x2, x7, lsl #3
	.loc 12 430 0
..LDL426:
/*    430 */	ldr	d6, [x2, x8, lsl #3]
	.loc 12 450 0
..LDL427:
/*    450 */	sxtw	x5, w12
	.loc 12 446 0
..LDL428:
/*    446 */	ld1d	{z3.d}, p0/z, [x20, 0, mul vl]
	.loc 12 430 0
..LDL429:
/*    430 */	add	x21, x7, 1
/*    430 */	add	x24, x7, 3
	.loc 12 450 0
..LDL430:
/*    450 */	add	x8, x2, x5, lsl #3
	.loc 12 454 0
..LDL431:
/*    ??? */	ldr	z0, [x29, 1, mul vl]	//  (*)
	.loc 12 430 0
..LDL432:
/*    430 */	add	x3, x7, 6
/*    430 */	add	x23, x7, 5
/*    430 */	ldr	d17, [x2, x21, lsl #3]
	.loc 12 445 0
..LDL433:
/*    445 */	ld1d	{z18.d}, p0/z, [x22, 0, mul vl]
	.loc 12 438 0
..LDL434:
/*    438 */	add	w15, w17, 16
	.loc 12 430 0
..LDL435:
/*    430 */	ldr	d7, [x2, x24, lsl #3]
/*    430 */	ldr	d5, [x2, x23, lsl #3]
	.loc 12 474 0
..LDL436:
	.loc 12 167 0 is_stmt 0
..LDL437:
/*    167 */	add	x23, x19, 288
	.loc 12 447 0 is_stmt 1
..LDL438:
/*    447 */	sxtw	x4, w15
	.loc 12 430 0
..LDL439:
/*    430 */	ldr	d4, [x2, x3, lsl #3]
	.loc 12 451 0
..LDL440:
/*    ??? */	ldr	x3, [x19, 16]	//  (*)
	.loc 12 448 0
..LDL441:
/*    448 */	add	x10, x2, x6, lsl #3
	.loc 12 442 0
..LDL442:
/*    442 */	add	w11, w17, 48
	.loc 12 443 0
..LDL443:
/*    443 */	add	w18, w17, 56
	.loc 12 447 0
..LDL444:
/*    447 */	add	x30, x2, x4, lsl #3
	.loc 12 451 0
..LDL445:
/*    451 */	sxtw	x2, w11
	.loc 12 448 0
..LDL446:
/*    448 */	ld1d	{z1.d}, p0/z, [x10, 0, mul vl]
	.loc 12 467 0
..LDL447:
/*    467 */	add	w22, w17, 15
	.loc 12 451 0
..LDL448:
/*    451 */	add	x20, x3, x2, lsl #3
	.loc 12 447 0
..LDL449:
/*    447 */	ld1d	{z2.d}, p0/z, [x30, 0, mul vl]
	.loc 12 452 0
..LDL450:
/*    ??? */	ldr	x30, [x19, 16]	//  (*)
/*    452 */	sxtw	x3, w18
	.loc 12 476 0
..LDL451:
/*    476 */	sxtw	x22, w22
	.loc 12 455 0
..LDL452:
/*    455 */	fmul	z21.d, z3.d, z0.d
	.loc 12 450 0
..LDL453:
/*    450 */	ld1d	{z3.d}, p0/z, [x8, 0, mul vl]
	.loc 12 471 0
..LDL454:
/*    471 */	add	w10, w17, 47
	.loc 12 472 0
..LDL455:
/*    472 */	add	w21, w17, 55
	.loc 12 454 0
..LDL456:
/*    454 */	fmul	z23.d, z18.d, z0.d
	.loc 12 449 0
..LDL457:
/*    449 */	ld1d	{z18.d}, p0/z, [x9, 0, mul vl]
	.loc 12 430 0
..LDL458:
	.loc 12 167 0 is_stmt 0
..LDL459:
/*    167 */	add	x9, x19, 288
	.loc 12 452 0 is_stmt 1
..LDL460:
/*    452 */	add	x30, x30, x3, lsl #3
	.loc 12 430 0
..LDL461:
/*    430 */	fmov	x8, d22
	.loc 12 457 0
..LDL462:
/*    457 */	fmul	z19.d, z1.d, z0.d
	.loc 12 452 0
..LDL463:
/*    452 */	ld1d	{z1.d}, p0/z, [x30, 0, mul vl]
	.loc 12 470 0
..LDL464:
/*    470 */	add	w30, w17, 39
	.loc 12 456 0
..LDL465:
/*    456 */	fmul	z20.d, z2.d, z0.d
	.loc 12 451 0
..LDL466:
/*    451 */	ld1d	{z2.d}, p0/z, [x20, 0, mul vl]
	.loc 12 469 0
..LDL467:
/*    469 */	add	w20, w17, 31
	.loc 12 458 0
..LDL468:
/*    458 */	fmul	z18.d, z18.d, z0.d
	.loc 12 430 0
..LDL469:
/*    430 */	str	x8, [x9]	//  "fcm1_arr"
	.loc 12 167 0 is_stmt 0
..LDL470:
/*    167 */	add	x9, x19, 288
	.loc 12 430 0
..LDL471:
/*    430 */	str	x8, [x9, 8]	//  "fcm1_arr"
	.loc 12 167 0
..LDL472:
/*    167 */	add	x8, x19, 288
	.loc 12 468 0 is_stmt 1
..LDL473:
/*    468 */	add	w9, w17, 23
	.loc 12 430 0
..LDL474:
/*    430 */	str	d17, [x8, 16]	//  "fcm1_arr"
	.loc 12 167 0 is_stmt 0
..LDL475:
/*    167 */	add	x8, x19, 288
	.loc 12 459 0 is_stmt 1
..LDL476:
/*    459 */	fmul	z17.d, z3.d, z0.d
	.loc 12 477 0
..LDL477:
/*    477 */	sxtw	x9, w9
	.loc 12 430 0
..LDL478:
/*    430 */	str	d16, [x8, 24]	//  "fcm1_arr"
	.loc 12 167 0 is_stmt 0
..LDL479:
/*    167 */	add	x8, x19, 288
	.loc 12 430 0
..LDL480:
/*    430 */	str	d7, [x8, 32]	//  "fcm1_arr"
	.loc 12 167 0
..LDL481:
/*    167 */	add	x8, x19, 288
	.loc 12 430 0
..LDL482:
/*    430 */	str	d6, [x8, 40]	//  "fcm1_arr"
	.loc 12 167 0
..LDL483:
/*    167 */	add	x8, x19, 288
	.loc 12 461 0 is_stmt 1
..LDL484:
/*    461 */	fmul	z1.d, z1.d, z0.d
	.loc 12 430 0
..LDL485:
/*    430 */	str	d5, [x8, 48]	//  "fcm1_arr"
	.loc 12 167 0 is_stmt 0
..LDL486:
/*    167 */	add	x8, x19, 288
	.loc 12 430 0
..LDL487:
/*    430 */	str	d4, [x8, 56]	//  "fcm1_arr"
	.loc 12 460 0 is_stmt 1
..LDL488:
/*    460 */	fmul	z5.d, z2.d, z0.d
	.loc 12 466 0
..LDL489:
/*    466 */	add	w8, w17, 7
	.loc 12 475 0
..LDL490:
/*    475 */	sxtw	x8, w8
	.loc 12 483 0
..LDL491:
/*    ??? */	ldr	z0, [x29, 2, mul vl]	//  (*)
	.loc 12 474 0
..LDL492:
/*    474 */	ld1d	{z24.d}, p0/z, [x23, 0, mul vl]	//  "fcm1_arr"
	.loc 12 475 0
..LDL493:
/*    ??? */	ldr	x23, [x19, 16]	//  (*)
/*    475 */	add	x8, x23, x8, lsl #3
/*    475 */	ld1d	{z22.d}, p0/z, [x8, 0, mul vl]
	.loc 12 476 0
..LDL494:
/*    476 */	add	x8, x23, x22, lsl #3
	.loc 12 478 0
..LDL495:
/*    478 */	sxtw	x22, w20
	.loc 12 477 0
..LDL496:
/*    477 */	add	x9, x23, x9, lsl #3
	.loc 12 476 0
..LDL497:
/*    476 */	ld1d	{z16.d}, p0/z, [x8, 0, mul vl]
	.loc 12 479 0
..LDL498:
/*    479 */	sxtw	x8, w30
	.loc 12 498 0
..LDL499:
/*    498 */	add	w30, w17, 33
	.loc 12 477 0
..LDL500:
/*    477 */	ld1d	{z7.d}, p0/z, [x9, 0, mul vl]
	.loc 12 479 0
..LDL501:
/*    479 */	add	x8, x23, x8, lsl #3
	.loc 12 483 0
..LDL502:
/*    483 */	fmad	z24.d, p0/m, z0.d, z23.d
	.loc 12 496 0
..LDL503:
/*    496 */	add	w9, w17, 17
	.loc 12 479 0
..LDL504:
/*    479 */	ld1d	{z4.d}, p0/z, [x8, 0, mul vl]
	.loc 12 480 0
..LDL505:
/*    480 */	sxtw	x8, w10
	.loc 12 494 0
..LDL506:
/*    494 */	add	w10, w17, 1
	.loc 12 503 0
..LDL507:
/*    503 */	sxtw	x24, w10
/*    ??? */	ldr	x10, [x19, 16]	//  (*)
	.loc 12 480 0
..LDL508:
/*    480 */	add	x8, x23, x8, lsl #3
	.loc 12 478 0
..LDL509:
/*    478 */	add	x20, x23, x22, lsl #3
	.loc 12 480 0
..LDL510:
/*    480 */	ld1d	{z3.d}, p0/z, [x8, 0, mul vl]
	.loc 12 481 0
..LDL511:
/*    481 */	sxtw	x8, w21
	.loc 12 484 0
..LDL512:
/*    484 */	fmad	z22.d, p0/m, z0.d, z21.d
	.loc 12 497 0
..LDL513:
/*    497 */	add	w22, w17, 25
	.loc 12 478 0
..LDL514:
/*    478 */	ld1d	{z6.d}, p0/z, [x20, 0, mul vl]
	.loc 12 499 0
..LDL515:
/*    499 */	add	w21, w17, 41
	.loc 12 503 0
..LDL516:
/*    503 */	add	x10, x10, x24, lsl #3
	.loc 12 485 0
..LDL517:
/*    485 */	fmad	z16.d, p0/m, z0.d, z20.d
	.loc 12 501 0
..LDL518:
/*    501 */	add	w20, w17, 57
	.loc 12 481 0
..LDL519:
/*    481 */	add	x8, x23, x8, lsl #3
	.loc 12 495 0
..LDL520:
/*    495 */	add	w23, w17, 9
	.loc 12 503 0
..LDL521:
/*    503 */	ld1d	{z23.d}, p0/z, [x10, 0, mul vl]
	.loc 12 504 0
..LDL522:
/*    ??? */	ldr	x10, [x19, 16]	//  (*)
	.loc 12 486 0
..LDL523:
/*    486 */	fmad	z7.d, p0/m, z0.d, z19.d
	.loc 12 504 0
..LDL524:
/*    504 */	sxtw	x23, w23
	.loc 12 481 0
..LDL525:
/*    481 */	ld1d	{z2.d}, p0/z, [x8, 0, mul vl]
	.loc 12 500 0
..LDL526:
/*    500 */	add	w8, w17, 49
	.loc 12 488 0
..LDL527:
/*    488 */	fmad	z4.d, p0/m, z0.d, z17.d
	.loc 12 509 0
..LDL528:
/*    509 */	sxtw	x8, w8
	.loc 12 504 0
..LDL529:
/*    504 */	add	x10, x10, x23, lsl #3
	.loc 12 489 0
..LDL530:
/*    489 */	fmad	z3.d, p0/m, z0.d, z5.d
	.loc 12 504 0
..LDL531:
/*    504 */	ld1d	{z21.d}, p0/z, [x10, 0, mul vl]
	.loc 12 505 0
..LDL532:
/*    505 */	sxtw	x10, w9
/*    ??? */	ldr	x9, [x19, 16]	//  (*)
	.loc 12 487 0
..LDL533:
/*    487 */	fmad	z6.d, p0/m, z0.d, z18.d
	.loc 12 505 0
..LDL534:
/*    505 */	add	x9, x9, x10, lsl #3
	.loc 12 506 0
..LDL535:
/*    ??? */	ldr	x10, [x19, 16]	//  (*)
	.loc 12 490 0
..LDL536:
/*    490 */	fmad	z0.d, p0/m, z2.d, z1.d
	.loc 12 512 0
..LDL537:
/*    ??? */	ldr	z1, [x29, 3, mul vl]	//  (*)
	.loc 12 505 0
..LDL538:
/*    505 */	ld1d	{z20.d}, p0/z, [x9, 0, mul vl]
	.loc 12 506 0
..LDL539:
/*    506 */	sxtw	x9, w22
/*    506 */	add	x9, x10, x9, lsl #3
/*    506 */	ld1d	{z19.d}, p0/z, [x9, 0, mul vl]
	.loc 12 507 0
..LDL540:
/*    507 */	sxtw	x9, w30
	.loc 12 509 0
..LDL541:
/*    509 */	add	x8, x10, x8, lsl #3
	.loc 12 507 0
..LDL542:
/*    507 */	add	x9, x10, x9, lsl #3
	.loc 12 509 0
..LDL543:
/*    509 */	ld1d	{z5.d}, p0/z, [x8, 0, mul vl]
	.loc 12 510 0
..LDL544:
/*    510 */	sxtw	x8, w20
	.loc 12 507 0
..LDL545:
/*    507 */	ld1d	{z18.d}, p0/z, [x9, 0, mul vl]
	.loc 12 508 0
..LDL546:
/*    508 */	sxtw	x9, w21
	.loc 12 510 0
..LDL547:
/*    510 */	add	x8, x10, x8, lsl #3
	.loc 12 523 0
..LDL548:
/*    ??? */	ldr	w21, [x19, 12]	//  (*)
	.loc 12 512 0
..LDL549:
/*    512 */	fmad	z23.d, p0/m, z1.d, z24.d
	.loc 12 513 0
..LDL550:
/*    513 */	fmad	z21.d, p0/m, z1.d, z22.d
	.loc 12 508 0
..LDL551:
/*    508 */	add	x9, x10, x9, lsl #3
	.loc 12 510 0
..LDL552:
/*    510 */	ld1d	{z2.d}, p0/z, [x8, 0, mul vl]
	.loc 12 514 0
..LDL553:
/*    514 */	fmad	z20.d, p0/m, z1.d, z16.d
	.loc 12 508 0
..LDL554:
/*    508 */	ld1d	{z17.d}, p0/z, [x9, 0, mul vl]
	.loc 12 523 0
..LDL555:
/*    523 */	add	w20, w17, w21
	.loc 12 524 0
..LDL556:
/*    524 */	add	w9, w16, w21
	.loc 12 525 0
..LDL557:
/*    525 */	add	w22, w15, w21
	.loc 12 526 0
..LDL558:
/*    526 */	add	w8, w14, w21
	.loc 12 515 0
..LDL559:
/*    515 */	fmad	z19.d, p0/m, z1.d, z7.d
	.loc 12 527 0
..LDL560:
/*    527 */	add	w30, w13, w21
	.loc 12 528 0
..LDL561:
/*    528 */	add	w10, w12, w21
	.loc 12 529 0
..LDL562:
/*    529 */	add	w23, w11, w21
	.loc 12 530 0
..LDL563:
/*    530 */	add	w24, w18, w21
	.loc 12 518 0
..LDL564:
/*    518 */	fmad	z5.d, p0/m, z1.d, z3.d
	.loc 12 532 0
..LDL565:
/*    532 */	sxtw	x21, w20
/*    ??? */	ldr	x20, [x19, 16]	//  (*)
	.loc 12 535 0
..LDL566:
/*    535 */	sxtw	x8, w8
	.loc 12 516 0
..LDL567:
/*    516 */	fmad	z18.d, p0/m, z1.d, z6.d
	.loc 12 532 0
..LDL568:
/*    532 */	add	x20, x20, x21, lsl #3
	.loc 12 517 0
..LDL569:
/*    517 */	fmad	z17.d, p0/m, z1.d, z4.d
	.loc 12 519 0
..LDL570:
/*    519 */	fmad	z1.d, p0/m, z2.d, z0.d
	.loc 12 541 0
..LDL571:
/*    ??? */	ldr	z0, [x29, 4, mul vl]	//  (*)
	.loc 12 532 0
..LDL572:
/*    532 */	ld1d	{z24.d}, p0/z, [x20, 0, mul vl]
	.loc 12 533 0
..LDL573:
/*    533 */	sxtw	x20, w9
/*    ??? */	ldr	x9, [x19, 16]	//  (*)
/*    533 */	add	x9, x9, x20, lsl #3
	.loc 12 534 0
..LDL574:
/*    534 */	sxtw	x20, w22
	.loc 12 533 0
..LDL575:
/*    533 */	ld1d	{z22.d}, p0/z, [x9, 0, mul vl]
	.loc 12 534 0
..LDL576:
/*    ??? */	ldr	x9, [x19, 16]	//  (*)
	.loc 12 541 0
..LDL577:
/*    541 */	fmad	z24.d, p0/m, z0.d, z23.d
	.loc 12 534 0
..LDL578:
/*    534 */	add	x9, x9, x20, lsl #3
	.loc 12 552 0
..LDL579:
/*    ??? */	ldr	w20, [x19, 8]	//  (*)
	.loc 12 534 0
..LDL580:
/*    534 */	ld1d	{z16.d}, p0/z, [x9, 0, mul vl]
	.loc 12 535 0
..LDL581:
/*    ??? */	ldr	x9, [x19, 16]	//  (*)
	.loc 12 555 0
..LDL582:
/*    555 */	add	w22, w14, w20
	.loc 12 557 0
..LDL583:
/*    557 */	add	w21, w12, w20
	.loc 12 542 0
..LDL584:
/*    542 */	fmad	z22.d, p0/m, z0.d, z21.d
	.loc 12 535 0
..LDL585:
/*    535 */	add	x8, x9, x8, lsl #3
/*    535 */	ld1d	{z7.d}, p0/z, [x8, 0, mul vl]
	.loc 12 536 0
..LDL586:
/*    536 */	sxtw	x8, w30
	.loc 12 556 0
..LDL587:
/*    556 */	add	w30, w13, w20
	.loc 12 536 0
..LDL588:
/*    536 */	add	x8, x9, x8, lsl #3
/*    536 */	ld1d	{z6.d}, p0/z, [x8, 0, mul vl]
	.loc 12 537 0
..LDL589:
/*    537 */	sxtw	x8, w10
	.loc 12 543 0
..LDL590:
/*    543 */	fmad	z16.d, p0/m, z0.d, z20.d
	.loc 12 552 0
..LDL591:
/*    552 */	add	w10, w17, w20
	.loc 12 537 0
..LDL592:
/*    537 */	add	x8, x9, x8, lsl #3
/*    537 */	ld1d	{z4.d}, p0/z, [x8, 0, mul vl]
	.loc 12 538 0
..LDL593:
/*    538 */	sxtw	x8, w23
	.loc 12 553 0
..LDL594:
/*    553 */	add	w23, w16, w20
	.loc 12 562 0
..LDL595:
/*    562 */	sxtw	x23, w23
	.loc 12 538 0
..LDL596:
/*    538 */	add	x8, x9, x8, lsl #3
	.loc 12 544 0
..LDL597:
/*    544 */	fmad	z7.d, p0/m, z0.d, z19.d
	.loc 12 538 0
..LDL598:
/*    538 */	ld1d	{z3.d}, p0/z, [x8, 0, mul vl]
	.loc 12 539 0
..LDL599:
/*    539 */	sxtw	x8, w24
	.loc 12 561 0
..LDL600:
/*    561 */	sxtw	x24, w10
/*    ??? */	ldr	x10, [x19, 16]	//  (*)
	.loc 12 539 0
..LDL601:
/*    539 */	add	x8, x9, x8, lsl #3
	.loc 12 554 0
..LDL602:
/*    554 */	add	w9, w15, w20
	.loc 12 545 0
..LDL603:
/*    545 */	fmad	z6.d, p0/m, z0.d, z18.d
	.loc 12 539 0
..LDL604:
/*    539 */	ld1d	{z2.d}, p0/z, [x8, 0, mul vl]
	.loc 12 558 0
..LDL605:
/*    558 */	add	w8, w11, w20
	.loc 12 559 0
..LDL606:
/*    559 */	add	w20, w18, w20
	.loc 12 561 0
..LDL607:
/*    561 */	add	x10, x10, x24, lsl #3
	.loc 12 567 0
..LDL608:
/*    567 */	sxtw	x8, w8
	.loc 12 546 0
..LDL609:
/*    546 */	fmad	z4.d, p0/m, z0.d, z17.d
	.loc 12 561 0
..LDL610:
/*    561 */	ld1d	{z23.d}, p0/z, [x10, 0, mul vl]
	.loc 12 562 0
..LDL611:
/*    ??? */	ldr	x10, [x19, 16]	//  (*)
	.loc 12 547 0
..LDL612:
/*    547 */	fmad	z3.d, p0/m, z0.d, z5.d
	.loc 12 570 0
..LDL613:
/*    ??? */	ldr	z5, [x29, 5, mul vl]	//  (*)
	.loc 12 562 0
..LDL614:
/*    562 */	add	x10, x10, x23, lsl #3
/*    562 */	ld1d	{z21.d}, p0/z, [x10, 0, mul vl]
	.loc 12 563 0
..LDL615:
/*    563 */	sxtw	x10, w9
/*    ??? */	ldr	x9, [x19, 16]	//  (*)
	.loc 12 548 0
..LDL616:
/*    548 */	fmad	z0.d, p0/m, z2.d, z1.d
	.loc 12 563 0
..LDL617:
/*    563 */	add	x9, x9, x10, lsl #3
	.loc 12 564 0
..LDL618:
/*    ??? */	ldr	x10, [x19, 16]	//  (*)
	.loc 12 563 0
..LDL619:
/*    563 */	ld1d	{z20.d}, p0/z, [x9, 0, mul vl]
	.loc 12 564 0
..LDL620:
/*    564 */	sxtw	x9, w22
	.loc 12 570 0
..LDL621:
/*    570 */	fmad	z23.d, p0/m, z5.d, z24.d
	.loc 12 564 0
..LDL622:
/*    564 */	add	x9, x10, x9, lsl #3
	.loc 12 571 0
..LDL623:
/*    571 */	fmad	z21.d, p0/m, z5.d, z22.d
	.loc 12 564 0
..LDL624:
/*    564 */	ld1d	{z19.d}, p0/z, [x9, 0, mul vl]
	.loc 12 565 0
..LDL625:
/*    565 */	sxtw	x9, w30
	.loc 12 567 0
..LDL626:
/*    567 */	add	x8, x10, x8, lsl #3
	.loc 12 565 0
..LDL627:
/*    565 */	add	x9, x10, x9, lsl #3
	.loc 12 567 0
..LDL628:
/*    567 */	ld1d	{z1.d}, p0/z, [x8, 0, mul vl]
	.loc 12 568 0
..LDL629:
/*    568 */	sxtw	x8, w20
	.loc 12 565 0
..LDL630:
/*    565 */	ld1d	{z18.d}, p0/z, [x9, 0, mul vl]
	.loc 12 566 0
..LDL631:
/*    566 */	sxtw	x9, w21
	.loc 12 568 0
..LDL632:
/*    568 */	add	x8, x10, x8, lsl #3
	.loc 12 581 0
..LDL633:
/*    ??? */	ldr	w21, [x19, 152]	//  (*)
	.loc 12 566 0
..LDL634:
/*    566 */	add	x9, x10, x9, lsl #3
	.loc 12 568 0
..LDL635:
/*    568 */	ld1d	{z2.d}, p0/z, [x8, 0, mul vl]
	.loc 12 572 0
..LDL636:
/*    572 */	fmad	z20.d, p0/m, z5.d, z16.d
	.loc 12 566 0
..LDL637:
/*    566 */	ld1d	{z17.d}, p0/z, [x9, 0, mul vl]
	.loc 12 581 0
..LDL638:
/*    581 */	add	w20, w17, w21
	.loc 12 582 0
..LDL639:
/*    582 */	add	w8, w16, w21
	.loc 12 583 0
..LDL640:
/*    583 */	add	w9, w15, w21
	.loc 12 584 0
..LDL641:
/*    584 */	add	w30, w14, w21
	.loc 12 573 0
..LDL642:
/*    573 */	fmad	z19.d, p0/m, z5.d, z7.d
	.loc 12 585 0
..LDL643:
/*    585 */	add	w22, w13, w21
	.loc 12 586 0
..LDL644:
/*    586 */	add	w10, w12, w21
	.loc 12 587 0
..LDL645:
/*    587 */	add	w23, w11, w21
	.loc 12 588 0
..LDL646:
/*    588 */	add	w24, w18, w21
	.loc 12 590 0
..LDL647:
/*    ??? */	ldr	x21, [x19, 16]	//  (*)
	.loc 12 576 0
..LDL648:
/*    576 */	fmad	z1.d, p0/m, z5.d, z3.d
	.loc 12 590 0
..LDL649:
/*    590 */	sxtw	x20, w20
	.loc 12 591 0
..LDL650:
/*    591 */	sxtw	x8, w8
	.loc 12 574 0
..LDL651:
/*    574 */	fmad	z18.d, p0/m, z5.d, z6.d
	.loc 12 591 0
..LDL652:
/*    591 */	add	x8, x21, x8, lsl #3
	.loc 12 575 0
..LDL653:
/*    575 */	fmad	z17.d, p0/m, z5.d, z4.d
	.loc 12 577 0
..LDL654:
/*    577 */	fmad	z5.d, p0/m, z2.d, z0.d
	.loc 12 591 0
..LDL655:
/*    591 */	ld1d	{z7.d}, p0/z, [x8, 0, mul vl]
	.loc 12 592 0
..LDL656:
/*    592 */	sxtw	x8, w9
	.loc 12 597 0
..LDL657:
/*    597 */	sxtw	x9, w24
	.loc 12 599 0
..LDL658:
/*    ??? */	ldr	z0, [x29, 6, mul vl]	//  (*)
	.loc 12 592 0
..LDL659:
/*    592 */	add	x8, x21, x8, lsl #3
	.loc 12 590 0
..LDL660:
/*    590 */	add	x20, x21, x20, lsl #3
	.loc 12 592 0
..LDL661:
/*    592 */	ld1d	{z6.d}, p0/z, [x8, 0, mul vl]
	.loc 12 593 0
..LDL662:
/*    593 */	sxtw	x8, w30
	.loc 12 612 0
..LDL663:
/*    ??? */	ldr	w30, [x19, 148]	//  (*)
	.loc 12 590 0
..LDL664:
/*    590 */	ld1d	{z22.d}, p0/z, [x20, 0, mul vl]
	.loc 12 593 0
..LDL665:
/*    593 */	add	x8, x21, x8, lsl #3
/*    593 */	ld1d	{z3.d}, p0/z, [x8, 0, mul vl]
	.loc 12 594 0
..LDL666:
/*    594 */	sxtw	x8, w22
	.loc 12 623 0
..LDL667:
/*    ??? */	ldr	x22, [x19, 16]	//  (*)
	.loc 12 594 0
..LDL668:
/*    594 */	add	x8, x21, x8, lsl #3
/*    594 */	ld1d	{z2.d}, p0/z, [x8, 0, mul vl]
	.loc 12 595 0
..LDL669:
/*    595 */	sxtw	x8, w10
	.loc 12 600 0
..LDL670:
/*    600 */	fmad	z7.d, p0/m, z0.d, z21.d
	.loc 12 610 0
..LDL671:
/*    ??? */	ldr	w10, [x19, 148]	//  (*)
	.loc 12 595 0
..LDL672:
/*    595 */	add	x8, x21, x8, lsl #3
	.loc 12 601 0
..LDL673:
/*    601 */	fmad	z6.d, p0/m, z0.d, z20.d
	.loc 12 595 0
..LDL674:
/*    595 */	ld1d	{z16.d}, p0/z, [x8, 0, mul vl]
	.loc 12 596 0
..LDL675:
/*    596 */	sxtw	x8, w23
	.loc 12 599 0
..LDL676:
/*    599 */	fmad	z22.d, p0/m, z0.d, z23.d
	.loc 12 596 0
..LDL677:
/*    596 */	add	x8, x21, x8, lsl #3
	.loc 12 612 0
..LDL678:
/*    612 */	add	w21, w15, w30
	.loc 12 619 0
..LDL679:
/*    ??? */	ldr	x30, [x19, 16]	//  (*)
	.loc 12 602 0
..LDL680:
/*    602 */	fmad	z3.d, p0/m, z0.d, z19.d
	.loc 12 596 0
..LDL681:
/*    596 */	ld1d	{z4.d}, p0/z, [x8, 0, mul vl]
	.loc 12 610 0
..LDL682:
/*    610 */	add	w8, w17, w10
	.loc 12 611 0
..LDL683:
/*    611 */	add	w10, w16, w10
	.loc 12 619 0
..LDL684:
/*    619 */	sxtw	x8, w8
	.loc 12 620 0
..LDL685:
/*    620 */	sxtw	x10, w10
	.loc 12 603 0
..LDL686:
/*    603 */	fmad	z2.d, p0/m, z0.d, z18.d
	.loc 12 619 0
..LDL687:
/*    619 */	add	x20, x30, x8, lsl #3
	.loc 12 613 0
..LDL688:
/*    ??? */	ldr	w8, [x19, 148]	//  (*)
	.loc 12 619 0
..LDL689:
/*    619 */	ld1d	{z21.d}, p0/z, [x20, 0, mul vl]
	.loc 12 620 0
..LDL690:
/*    ??? */	ldr	x20, [x19, 16]	//  (*)
	.loc 12 604 0
..LDL691:
/*    604 */	fmad	z16.d, p0/m, z0.d, z17.d
	.loc 12 613 0
..LDL692:
/*    613 */	add	w30, w14, w8
	.loc 12 621 0
..LDL693:
/*    621 */	sxtw	x8, w21
	.loc 12 616 0
..LDL694:
/*    ??? */	ldr	w21, [x19, 148]	//  (*)
	.loc 12 622 0
..LDL695:
/*    622 */	sxtw	x30, w30
	.loc 12 620 0
..LDL696:
/*    620 */	add	x20, x20, x10, lsl #3
	.loc 12 614 0
..LDL697:
/*    ??? */	ldr	w10, [x19, 148]	//  (*)
	.loc 12 605 0
..LDL698:
/*    605 */	fmad	z4.d, p0/m, z0.d, z1.d
	.loc 12 628 0
..LDL699:
/*    ??? */	ldr	z1, [x29, 7, mul vl]	//  (*)
	.loc 12 620 0
..LDL700:
/*    620 */	ld1d	{z20.d}, p0/z, [x20, 0, mul vl]
	.loc 12 621 0
..LDL701:
/*    ??? */	ldr	x20, [x19, 16]	//  (*)
	.loc 12 614 0
..LDL702:
/*    614 */	add	w10, w13, w10
	.loc 12 623 0
..LDL703:
/*    623 */	sxtw	x10, w10
	.loc 12 621 0
..LDL704:
/*    621 */	add	x20, x20, x8, lsl #3
	.loc 12 615 0
..LDL705:
/*    ??? */	ldr	w8, [x19, 148]	//  (*)
	.loc 12 621 0
..LDL706:
/*    621 */	ld1d	{z19.d}, p0/z, [x20, 0, mul vl]
	.loc 12 622 0
..LDL707:
/*    ??? */	ldr	x20, [x19, 16]	//  (*)
	.loc 12 623 0
..LDL708:
/*    623 */	add	x22, x22, x10, lsl #3
/*    623 */	ld1d	{z17.d}, p0/z, [x22, 0, mul vl]
	.loc 12 615 0
..LDL709:
/*    615 */	add	w8, w12, w8
	.loc 12 628 0
..LDL710:
/*    628 */	fmad	z21.d, p0/m, z1.d, z22.d
	.loc 12 624 0
..LDL711:
/*    624 */	sxtw	x8, w8
	.loc 12 629 0
..LDL712:
/*    629 */	fmad	z20.d, p0/m, z1.d, z7.d
	.loc 12 622 0
..LDL713:
/*    622 */	add	x30, x20, x30, lsl #3
	.loc 12 616 0
..LDL714:
/*    616 */	add	w20, w11, w21
	.loc 12 617 0
..LDL715:
/*    617 */	add	w21, w18, w21
	.loc 12 625 0
..LDL716:
/*    625 */	sxtw	x20, w20
	.loc 12 626 0
..LDL717:
/*    626 */	sxtw	x10, w21
	.loc 12 638 0
..LDL718:
/*    ??? */	ldr	x21, [x19, 168]	//  (*)
	.loc 12 622 0
..LDL719:
/*    622 */	ld1d	{z18.d}, p0/z, [x30, 0, mul vl]
	.loc 12 637 0
..LDL720:
/*    ??? */	ldr	x30, [x19, 168]	//  (*)
	.loc 12 630 0
..LDL721:
/*    630 */	fmad	z19.d, p0/m, z1.d, z6.d
	.loc 12 638 0
..LDL722:
/*    638 */	add	x22, x21, x0, lsl #3
	.loc 12 624 0
..LDL723:
/*    ??? */	ldr	x21, [x19, 16]	//  (*)
	.loc 12 632 0
..LDL724:
/*    632 */	fmad	z17.d, p0/m, z1.d, z2.d
	.loc 12 637 0
..LDL725:
/*    637 */	add	x30, x30, x7, lsl #3
/*    637 */	st1d	{z21.d}, p0, [x30, 0, mul vl]
	.loc 12 624 0
..LDL726:
/*    624 */	add	x8, x21, x8, lsl #3
	.loc 12 638 0
..LDL727:
/*    638 */	st1d	{z20.d}, p0, [x22, 0, mul vl]
	.loc 12 624 0
..LDL728:
/*    624 */	ld1d	{z7.d}, p0/z, [x8, 0, mul vl]
	.loc 12 639 0
..LDL729:
/*    ??? */	ldr	x8, [x19, 168]	//  (*)
	.loc 12 631 0
..LDL730:
/*    631 */	fmad	z18.d, p0/m, z1.d, z3.d
	.loc 12 639 0
..LDL731:
/*    639 */	add	x21, x8, x4, lsl #3
	.loc 12 597 0
..LDL732:
/*    ??? */	ldr	x8, [x19, 16]	//  (*)
	.loc 12 639 0
..LDL733:
/*    639 */	st1d	{z19.d}, p0, [x21, 0, mul vl]
	.loc 12 597 0
..LDL734:
/*    597 */	add	x8, x8, x9, lsl #3
	.loc 12 640 0
..LDL735:
/*    ??? */	ldr	x9, [x19, 168]	//  (*)
	.loc 12 633 0
..LDL736:
/*    633 */	fmad	z7.d, p0/m, z1.d, z16.d
	.loc 12 597 0
..LDL737:
/*    597 */	ld1d	{z6.d}, p0/z, [x8, 0, mul vl]
	.loc 12 625 0
..LDL738:
/*    ??? */	ldr	x8, [x19, 16]	//  (*)
	.loc 12 640 0
..LDL739:
/*    640 */	add	x6, x9, x6, lsl #3
	.loc 12 625 0
..LDL740:
/*    625 */	add	x8, x8, x20, lsl #3
	.loc 12 640 0
..LDL741:
/*    640 */	st1d	{z18.d}, p0, [x6, 0, mul vl]
	.loc 12 625 0
..LDL742:
/*    625 */	ld1d	{z3.d}, p0/z, [x8, 0, mul vl]
	.loc 12 626 0
..LDL743:
/*    ??? */	ldr	x8, [x19, 16]	//  (*)
	.loc 12 641 0
..LDL744:
/*    641 */	add	x9, x9, x1, lsl #3
	.loc 12 642 0
..LDL745:
/*    ??? */	ldr	x6, [x19, 168]	//  (*)
	.loc 12 641 0
..LDL746:
/*    641 */	st1d	{z17.d}, p0, [x9, 0, mul vl]
	.loc 12 644 0
..LDL747:
/*    ??? */	ldr	x9, [x19, 168]	//  (*)
	.loc 12 606 0
..LDL748:
/*    606 */	fmad	z0.d, p0/m, z6.d, z5.d
	.loc 12 626 0
..LDL749:
/*    626 */	add	x8, x8, x10, lsl #3
/*    626 */	ld1d	{z2.d}, p0/z, [x8, 0, mul vl]
	.loc 12 642 0
..LDL750:
/*    642 */	add	x6, x6, x5, lsl #3
	.loc 12 643 0
..LDL751:
/*    ??? */	ldr	x8, [x19, 168]	//  (*)
	.loc 12 644 0
..LDL752:
/*    644 */	add	x9, x9, x3, lsl #3
	.loc 12 642 0
..LDL753:
/*    642 */	st1d	{z7.d}, p0, [x6, 0, mul vl]
	.loc 12 634 0
..LDL754:
/*    634 */	fmad	z3.d, p0/m, z1.d, z4.d
	.loc 12 643 0
..LDL755:
/*    643 */	add	x8, x8, x2, lsl #3
	.loc 12 635 0
..LDL756:
/*    635 */	fmad	z1.d, p0/m, z2.d, z0.d
	.loc 12 643 0
..LDL757:
/*    643 */	st1d	{z3.d}, p0, [x8, 0, mul vl]
	.loc 12 644 0
..LDL758:
/*    644 */	st1d	{z1.d}, p0, [x9, 0, mul vl]
	.loc 12 646 0
..LDL759:
/*    ??? */	ldr	w6, [x19, 100]	//  (*)
/*    646 */	cmp	w6, 64
/*    646 */	ble	.L274
/*    646 */	mov	w6, 64
	.loc 12 945 0 is_stmt 0
..LDL760:
/*    ??? */	ldr	w30, [x19, 116]	//  (*)
/*    ??? */	ldr	x8, [x19, 168]	//  (*)
/*    ??? */	ldp	w10, w9, [x19, 8]	//  (*)
/*    ??? */	str	x8, [x19, 120]	//  (*)
/*    ??? */	ldr	w8, [x19, 148]	//  (*)
/*    ??? */	ldr	z18, [x29, 7, mul vl]	//  (*)
/*    ??? */	ldr	z17, [x29, 6, mul vl]	//  (*)
/*    ??? */	str	w8, [x19, 132]	//  (*)
/*    ??? */	ldr	w8, [x19, 152]	//  (*)
/*    ??? */	ldr	z16, [x29, 5, mul vl]	//  (*)
/*    ??? */	str	w8, [x19, 136]	//  (*)
/*    ??? */	ldr	x8, [x19, 16]	//  (*)
/*    ??? */	ldr	x20, [x19, 104]	//  (*)
/*    ??? */	ldr	z7, [x29, 4, mul vl]	//  (*)
/*    ??? */	ldr	z6, [x29, 3, mul vl]	//  (*)
/*    ??? */	ldr	z5, [x29, 2, mul vl]	//  (*)
/*    ??? */	ldr	z4, [x29, 1, mul vl]	//  (*)
	.p2align 5
.L272:					// :entr:term
	.loc 12 931 0 is_stmt 1
..LDL761:
/*    931 */	add	x4, x4, 64
	.loc 12 930 0
..LDL762:
/*    930 */	add	x0, x0, 64
	.loc 12 938 0
..LDL763:
	.loc 12 107 0 is_stmt 0
..LDL764:
/*    107 */	ptrue	p0.d, ALL
	.loc 12 932 0 is_stmt 1
..LDL765:
/*    932 */	add	x20, x20, 64
	.loc 12 933 0
..LDL766:
/*    933 */	add	x1, x1, 64
	.loc 12 934 0
..LDL767:
/*    934 */	add	x5, x5, 64
	.loc 12 935 0
..LDL768:
/*    935 */	add	x2, x2, 64
	.loc 12 929 0
..LDL769:
/*    929 */	add	w17, w17, 64
	.loc 12 939 0
..LDL770:
/*    939 */	ld1d	{z21.d}, p0/z, [x8, x0, lsl #3]
	.loc 12 940 0
..LDL771:
/*    940 */	ld1d	{z20.d}, p0/z, [x8, x4, lsl #3]
	.loc 12 958 0
..LDL772:
/*    958 */	asr	w21, w6, 20
	.loc 12 934 0
..LDL773:
/*    934 */	add	w12, w12, 64
	.loc 12 941 0
..LDL774:
/*    941 */	ld1d	{z19.d}, p0/z, [x8, x20, lsl #3]
	.loc 12 942 0
..LDL775:
/*    942 */	ld1d	{z3.d}, p0/z, [x8, x1, lsl #3]
	.loc 12 958 0
..LDL776:
/*    958 */	sub	w24, w21, 1
	.loc 12 929 0
..LDL777:
/*    929 */	add	x7, x7, 64
	.loc 12 943 0
..LDL778:
/*    943 */	ld1d	{z2.d}, p0/z, [x8, x5, lsl #3]
	.loc 12 944 0
..LDL779:
/*    944 */	ld1d	{z1.d}, p0/z, [x8, x2, lsl #3]
	.loc 12 959 0
..LDL780:
/*    959 */	add	w22, w24, w17
	.loc 12 938 0
..LDL781:
/*    938 */	ld1d	{z22.d}, p0/z, [x8, x7, lsl #3]
	.loc 12 964 0
..LDL782:
/*    964 */	add	w21, w24, w12
	.loc 12 968 0
..LDL783:
/*    968 */	sxtw	x22, w22
	.loc 12 936 0
..LDL784:
/*    936 */	add	x3, x3, 64
	.loc 12 968 0
..LDL785:
/*    968 */	ld1d	{z29.d}, p0/z, [x8, x22, lsl #3]
	.loc 12 973 0
..LDL786:
/*    973 */	sxtw	x21, w21
	.loc 12 933 0
..LDL787:
/*    933 */	add	w13, w13, 64
	.loc 12 935 0
..LDL788:
/*    935 */	add	w11, w11, 64
	.loc 12 945 0
..LDL789:
/*    945 */	ld1d	{z0.d}, p0/z, [x8, x3, lsl #3]
	.loc 12 973 0
..LDL790:
/*    973 */	ld1d	{z26.d}, p0/z, [x8, x21, lsl #3]
	.loc 12 963 0
..LDL791:
/*    963 */	add	w23, w24, w13
	.loc 12 965 0
..LDL792:
/*    965 */	add	w22, w24, w11
	.loc 12 972 0
..LDL793:
/*    972 */	sxtw	x23, w23
	.loc 12 974 0
..LDL794:
/*    974 */	sxtw	x21, w22
	.loc 12 932 0
..LDL795:
/*    932 */	add	w14, w14, 64
	.loc 12 931 0
..LDL796:
/*    931 */	add	w15, w15, 64
	.loc 12 974 0
..LDL797:
/*    974 */	ld1d	{z24.d}, p0/z, [x8, x21, lsl #3]
	.loc 12 930 0
..LDL798:
/*    930 */	add	w16, w16, 64
	.loc 12 936 0
..LDL799:
/*    936 */	add	w18, w18, 64
	.loc 12 960 0
..LDL800:
/*    960 */	add	w27, w24, w16
	.loc 12 961 0
..LDL801:
/*    961 */	add	w26, w24, w15
	.loc 12 948 0
..LDL802:
/*    948 */	fmul	z28.d, z21.d, z4.d
	.loc 12 949 0
..LDL803:
/*    949 */	fmul	z21.d, z20.d, z4.d
	.loc 12 962 0
..LDL804:
/*    962 */	add	w25, w24, w14
	.loc 12 966 0
..LDL805:
/*    966 */	add	w24, w24, w18
	.loc 12 950 0
..LDL806:
/*    950 */	fmul	z20.d, z19.d, z4.d
	.loc 12 951 0
..LDL807:
/*    951 */	fmul	z19.d, z3.d, z4.d
	.loc 12 971 0
..LDL808:
/*    971 */	sxtw	x25, w25
	.loc 12 975 0
..LDL809:
/*    975 */	sxtw	x21, w24
	.loc 12 952 0
..LDL810:
/*    952 */	fmul	z3.d, z2.d, z4.d
	.loc 12 953 0
..LDL811:
/*    953 */	fmul	z2.d, z1.d, z4.d
	.loc 12 969 0
..LDL812:
/*    969 */	sxtw	x27, w27
	.loc 12 971 0
..LDL813:
/*    971 */	ld1d	{z23.d}, p0/z, [x8, x25, lsl #3]
	.loc 12 947 0
..LDL814:
/*    947 */	fmul	z30.d, z22.d, z4.d
	.loc 12 970 0
..LDL815:
/*    970 */	sxtw	x26, w26
	.loc 12 972 0
..LDL816:
/*    972 */	ld1d	{z22.d}, p0/z, [x8, x23, lsl #3]
	.loc 12 1145 0
..LDL817:
/*   1145 */	subs	w30, w30, 1
	.loc 12 969 0
..LDL818:
/*    969 */	ld1d	{z27.d}, p0/z, [x8, x27, lsl #3]
	.loc 12 970 0
..LDL819:
/*    970 */	ld1d	{z25.d}, p0/z, [x8, x26, lsl #3]
	.loc 12 954 0
..LDL820:
/*    954 */	fmul	z1.d, z0.d, z4.d
	.loc 12 975 0
..LDL821:
/*    975 */	ld1d	{z0.d}, p0/z, [x8, x21, lsl #3]
	.loc 12 988 0
..LDL822:
/*    988 */	asr	w21, w6, 21
/*    988 */	add	w23, w21, 1
	.loc 12 990 0
..LDL823:
/*    990 */	add	w22, w23, w16
	.loc 12 991 0
..LDL824:
/*    991 */	add	w27, w23, w15
	.loc 12 992 0
..LDL825:
/*    992 */	add	w24, w23, w14
	.loc 12 999 0
..LDL826:
/*    999 */	sxtw	x22, w22
	.loc 12 989 0
..LDL827:
/*    989 */	add	w25, w23, w17
	.loc 12 994 0
..LDL828:
/*    994 */	add	w21, w23, w12
	.loc 12 998 0
..LDL829:
/*    998 */	sxtw	x25, w25
	.loc 12 1003 0
..LDL830:
/*   1003 */	sxtw	x21, w21
	.loc 12 982 0
..LDL831:
/*    982 */	fmad	z26.d, p0/m, z5.d, z3.d
	.loc 12 983 0
..LDL832:
/*    983 */	fmad	z24.d, p0/m, z5.d, z2.d
	.loc 12 993 0
..LDL833:
/*    993 */	add	w26, w23, w13
	.loc 12 977 0
..LDL834:
/*    977 */	fmad	z29.d, p0/m, z5.d, z30.d
	.loc 12 980 0
..LDL835:
/*    980 */	fmad	z23.d, p0/m, z5.d, z20.d
	.loc 12 981 0
..LDL836:
/*    981 */	fmad	z22.d, p0/m, z5.d, z19.d
	.loc 12 999 0
..LDL837:
/*    999 */	ld1d	{z19.d}, p0/z, [x8, x22, lsl #3]
	.loc 12 1000 0
..LDL838:
/*   1000 */	sxtw	x22, w27
	.loc 12 978 0
..LDL839:
/*    978 */	fmad	z27.d, p0/m, z5.d, z28.d
	.loc 12 979 0
..LDL840:
/*    979 */	fmad	z25.d, p0/m, z5.d, z21.d
	.loc 12 1000 0
..LDL841:
/*   1000 */	ld1d	{z20.d}, p0/z, [x8, x22, lsl #3]
	.loc 12 1001 0
..LDL842:
/*   1001 */	sxtw	x22, w24
	.loc 12 984 0
..LDL843:
/*    984 */	fmad	z0.d, p0/m, z5.d, z1.d
	.loc 12 1001 0
..LDL844:
/*   1001 */	ld1d	{z11.d}, p0/z, [x8, x22, lsl #3]
	.loc 12 1002 0
..LDL845:
/*   1002 */	sxtw	x22, w26
	.loc 12 998 0
..LDL846:
/*    998 */	ld1d	{z1.d}, p0/z, [x8, x25, lsl #3]
	.loc 12 995 0
..LDL847:
/*    995 */	add	w25, w23, w11
	.loc 12 996 0
..LDL848:
/*    996 */	add	w23, w23, w18
	.loc 12 1003 0
..LDL849:
/*   1003 */	ld1d	{z21.d}, p0/z, [x8, x21, lsl #3]
	.loc 12 1002 0
..LDL850:
/*   1002 */	ld1d	{z28.d}, p0/z, [x8, x22, lsl #3]
	.loc 12 1004 0
..LDL851:
/*   1004 */	sxtw	x21, w25
	.loc 12 1018 0
..LDL852:
/*   1018 */	add	w26, w9, w6, asr #22
	.loc 12 1004 0
..LDL853:
/*   1004 */	ld1d	{z3.d}, p0/z, [x8, x21, lsl #3]
	.loc 12 1005 0
..LDL854:
/*   1005 */	sxtw	x21, w23
/*   1005 */	ld1d	{z2.d}, p0/z, [x8, x21, lsl #3]
	.loc 12 1019 0
..LDL855:
/*   1019 */	add	w27, w26, w17
	.loc 12 1020 0
..LDL856:
/*   1020 */	add	w25, w26, w16
	.loc 12 1028 0
..LDL857:
/*   1028 */	sxtw	x27, w27
	.loc 12 1029 0
..LDL858:
/*   1029 */	sxtw	x25, w25
	.loc 12 1021 0
..LDL859:
/*   1021 */	add	w21, w26, w15
	.loc 12 1022 0
..LDL860:
/*   1022 */	add	w28, w26, w14
	.loc 12 1024 0
..LDL861:
/*   1024 */	add	w22, w26, w12
	.loc 12 1025 0
..LDL862:
/*   1025 */	add	w24, w26, w11
	.loc 12 1023 0
..LDL863:
/*   1023 */	add	w23, w26, w13
	.loc 12 1034 0
..LDL864:
/*   1034 */	sxtw	x24, w24
	.loc 12 1008 0
..LDL865:
/*   1008 */	fmad	z19.d, p0/m, z6.d, z27.d
	.loc 12 1026 0
..LDL866:
/*   1026 */	add	w26, w26, w18
	.loc 12 1032 0
..LDL867:
/*   1032 */	sxtw	x23, w23
	.loc 12 1009 0
..LDL868:
/*   1009 */	fmad	z20.d, p0/m, z6.d, z25.d
	.loc 12 1035 0
..LDL869:
/*   1035 */	sxtw	x26, w26
	.loc 12 1032 0
..LDL870:
/*   1032 */	ld1d	{z30.d}, p0/z, [x8, x23, lsl #3]
	.loc 12 1010 0
..LDL871:
/*   1010 */	fmad	z11.d, p0/m, z6.d, z23.d
	.loc 12 1034 0
..LDL872:
/*   1034 */	ld1d	{z23.d}, p0/z, [x8, x24, lsl #3]
	.loc 12 1007 0
..LDL873:
/*   1007 */	fmad	z1.d, p0/m, z6.d, z29.d
	.loc 12 1012 0
..LDL874:
/*   1012 */	fmla	z26.d, p0/m, z21.d, z6.d
	.loc 12 1029 0
..LDL875:
/*   1029 */	ld1d	{z21.d}, p0/z, [x8, x25, lsl #3]
	.loc 12 1033 0
..LDL876:
/*   1033 */	sxtw	x25, w22
	.loc 12 1011 0
..LDL877:
/*   1011 */	fmad	z28.d, p0/m, z6.d, z22.d
	.loc 12 1028 0
..LDL878:
/*   1028 */	ld1d	{z22.d}, p0/z, [x8, x27, lsl #3]
	.loc 12 1030 0
..LDL879:
/*   1030 */	sxtw	x27, w21
	.loc 12 1031 0
..LDL880:
/*   1031 */	sxtw	x21, w28
	.loc 12 1013 0
..LDL881:
/*   1013 */	fmla	z24.d, p0/m, z3.d, z6.d
	.loc 12 1031 0
..LDL882:
/*   1031 */	ld1d	{z31.d}, p0/z, [x8, x21, lsl #3]
	.loc 12 1048 0
..LDL883:
/*   1048 */	add	w22, w10, w6, asr #23
	.loc 12 1035 0
..LDL884:
/*   1035 */	ld1d	{z3.d}, p0/z, [x8, x26, lsl #3]
	.loc 12 1014 0
..LDL885:
/*   1014 */	fmad	z2.d, p0/m, z6.d, z0.d
	.loc 12 1033 0
..LDL886:
/*   1033 */	ld1d	{z25.d}, p0/z, [x8, x25, lsl #3]
	.loc 12 1030 0
..LDL887:
/*   1030 */	ld1d	{z27.d}, p0/z, [x8, x27, lsl #3]
	.loc 12 1049 0
..LDL888:
/*   1049 */	add	w21, w22, w17
	.loc 12 1050 0
..LDL889:
/*   1050 */	add	w24, w22, w16
	.loc 12 1058 0
..LDL890:
/*   1058 */	sxtw	x21, w21
	.loc 12 1059 0
..LDL891:
/*   1059 */	sxtw	x24, w24
	.loc 12 1051 0
..LDL892:
/*   1051 */	add	w28, w22, w15
	.loc 12 1053 0
..LDL893:
/*   1053 */	add	w26, w22, w13
	.loc 12 1058 0
..LDL894:
/*   1058 */	ld1d	{z0.d}, p0/z, [x8, x21, lsl #3]
	.loc 12 1055 0
..LDL895:
/*   1055 */	add	w25, w22, w11
	.loc 12 1060 0
..LDL896:
/*   1060 */	sxtw	x21, w28
	.loc 12 1052 0
..LDL897:
/*   1052 */	add	w27, w22, w14
	.loc 12 1054 0
..LDL898:
/*   1054 */	add	w23, w22, w12
	.loc 12 1056 0
..LDL899:
/*   1056 */	add	w22, w22, w18
	.loc 12 1061 0
..LDL900:
/*   1061 */	sxtw	x27, w27
	.loc 12 1063 0
..LDL901:
/*   1063 */	sxtw	x23, w23
	.loc 12 1065 0
..LDL902:
/*   1065 */	sxtw	x22, w22
	.loc 12 1041 0
..LDL903:
/*   1041 */	fmad	z30.d, p0/m, z7.d, z28.d
	.loc 12 1038 0
..LDL904:
/*   1038 */	fmad	z21.d, p0/m, z7.d, z19.d
	.loc 12 1060 0
..LDL905:
/*   1060 */	ld1d	{z19.d}, p0/z, [x8, x21, lsl #3]
	.loc 12 1064 0
..LDL906:
/*   1064 */	sxtw	x21, w25
	.loc 12 1037 0
..LDL907:
/*   1037 */	fmla	z1.d, p0/m, z22.d, z7.d
	.loc 12 1059 0
..LDL908:
/*   1059 */	ld1d	{z22.d}, p0/z, [x8, x24, lsl #3]
	.loc 12 1062 0
..LDL909:
/*   1062 */	sxtw	x24, w26
/*   1062 */	ld1d	{z29.d}, p0/z, [x8, x24, lsl #3]
	.loc 12 1064 0
..LDL910:
/*   1064 */	ld1d	{z28.d}, p0/z, [x8, x21, lsl #3]
	.loc 12 1043 0
..LDL911:
/*   1043 */	fmad	z23.d, p0/m, z7.d, z24.d
	.loc 12 1044 0
..LDL912:
/*   1044 */	fmad	z3.d, p0/m, z7.d, z2.d
	.loc 12 1078 0
..LDL913:
/* #00003 */	ldr	w24, [x19, 136]	//  (*)
	.loc 12 1040 0
..LDL914:
/*   1040 */	fmad	z31.d, p0/m, z7.d, z11.d
	.loc 12 1042 0
..LDL915:
/*   1042 */	fmad	z25.d, p0/m, z7.d, z26.d
	.loc 12 1065 0
..LDL916:
/*   1065 */	ld1d	{z24.d}, p0/z, [x8, x22, lsl #3]
	.loc 12 1039 0
..LDL917:
/*   1039 */	fmla	z20.d, p0/m, z27.d, z7.d
	.loc 12 1061 0
..LDL918:
/*   1061 */	ld1d	{z27.d}, p0/z, [x8, x27, lsl #3]
	.loc 12 1063 0
..LDL919:
/*   1063 */	ld1d	{z11.d}, p0/z, [x8, x23, lsl #3]
	.loc 12 1078 0
..LDL920:
/*   1078 */	add	w26, w24, w6, asr #24
	.loc 12 1079 0
..LDL921:
/*   1079 */	add	w21, w26, w17
	.loc 12 1080 0
..LDL922:
/*   1080 */	add	w28, w26, w16
	.loc 12 1067 0
..LDL923:
/*   1067 */	fmla	z1.d, p0/m, z0.d, z16.d
	.loc 12 1081 0
..LDL924:
/*   1081 */	add	w22, w26, w15
	.loc 12 1088 0
..LDL925:
/*   1088 */	sxtw	x21, w21
/*   1088 */	ld1d	{z2.d}, p0/z, [x8, x21, lsl #3]
	.loc 12 1089 0
..LDL926:
/*   1089 */	sxtw	x21, w28
	.loc 12 1090 0
..LDL927:
/*   1090 */	sxtw	x22, w22
	.loc 12 1068 0
..LDL928:
/*   1068 */	fmad	z22.d, p0/m, z16.d, z21.d
	.loc 12 1082 0
..LDL929:
/*   1082 */	add	w24, w26, w14
	.loc 12 1089 0
..LDL930:
/*   1089 */	ld1d	{z0.d}, p0/z, [x8, x21, lsl #3]
	.loc 12 1090 0
..LDL931:
/*   1090 */	ld1d	{z21.d}, p0/z, [x8, x22, lsl #3]
	.loc 12 1069 0
..LDL932:
/*   1069 */	fmla	z20.d, p0/m, z19.d, z16.d
	.loc 12 1083 0
..LDL933:
/*   1083 */	add	w27, w26, w13
	.loc 12 1091 0
..LDL934:
/*   1091 */	sxtw	x21, w24
	.loc 12 1071 0
..LDL935:
/*   1071 */	fmla	z30.d, p0/m, z29.d, z16.d
	.loc 12 1074 0
..LDL936:
/*   1074 */	fmla	z3.d, p0/m, z24.d, z16.d
	.loc 12 1084 0
..LDL937:
/*   1084 */	add	w25, w26, w12
	.loc 12 1092 0
..LDL938:
/*   1092 */	sxtw	x24, w27
	.loc 12 1091 0
..LDL939:
/*   1091 */	ld1d	{z19.d}, p0/z, [x8, x21, lsl #3]
	.loc 12 1070 0
..LDL940:
/*   1070 */	fmla	z31.d, p0/m, z27.d, z16.d
	.loc 12 1093 0
..LDL941:
/*   1093 */	sxtw	x22, w25
	.loc 12 1108 0
..LDL942:
/* #00003 */	ldr	w21, [x19, 132]	//  (*)
	.loc 12 1092 0
..LDL943:
/*   1092 */	ld1d	{z27.d}, p0/z, [x8, x24, lsl #3]
	.loc 12 1072 0
..LDL944:
/*   1072 */	fmla	z25.d, p0/m, z11.d, z16.d
	.loc 12 1085 0
..LDL945:
/*   1085 */	add	w23, w26, w11
	.loc 12 1086 0
..LDL946:
/*   1086 */	add	w26, w26, w18
	.loc 12 1093 0
..LDL947:
/*   1093 */	ld1d	{z26.d}, p0/z, [x8, x22, lsl #3]
	.loc 12 1073 0
..LDL948:
/*   1073 */	fmla	z23.d, p0/m, z28.d, z16.d
	.loc 12 1094 0
..LDL949:
/*   1094 */	sxtw	x23, w23
	.loc 12 1095 0
..LDL950:
/*   1095 */	sxtw	x25, w26
	.loc 12 1094 0
..LDL951:
/*   1094 */	ld1d	{z29.d}, p0/z, [x8, x23, lsl #3]
	.loc 12 1095 0
..LDL952:
/*   1095 */	ld1d	{z28.d}, p0/z, [x8, x25, lsl #3]
	.loc 12 1108 0
..LDL953:
/*   1108 */	add	w21, w21, w6, asr #25
	.loc 12 1145 0
..LDL954:
/*   1145 */	add	w6, w6, 64
	.loc 12 1097 0
..LDL955:
/*   1097 */	fmla	z1.d, p0/m, z2.d, z17.d
	.loc 12 1109 0
..LDL956:
/*   1109 */	add	w22, w21, w17
	.loc 12 1098 0
..LDL957:
/*   1098 */	fmad	z0.d, p0/m, z17.d, z22.d
	.loc 12 1099 0
..LDL958:
/*   1099 */	fmla	z20.d, p0/m, z21.d, z17.d
	.loc 12 1118 0
..LDL959:
/*   1118 */	sxtw	x22, w22
	.loc 12 1100 0
..LDL960:
/*   1100 */	fmad	z19.d, p0/m, z17.d, z31.d
	.loc 12 1101 0
..LDL961:
/*   1101 */	fmad	z27.d, p0/m, z17.d, z30.d
	.loc 12 1118 0
..LDL962:
/*   1118 */	ld1d	{z30.d}, p0/z, [x8, x22, lsl #3]
	.loc 12 1110 0
..LDL963:
/*   1110 */	add	w22, w21, w16
	.loc 12 1102 0
..LDL964:
/*   1102 */	fmad	z26.d, p0/m, z17.d, z25.d
	.loc 12 1119 0
..LDL965:
/*   1119 */	sxtw	x22, w22
/*   1119 */	ld1d	{z25.d}, p0/z, [x8, x22, lsl #3]
	.loc 12 1111 0
..LDL966:
/*   1111 */	add	w22, w21, w15
	.loc 12 1103 0
..LDL967:
/*   1103 */	fmad	z29.d, p0/m, z17.d, z23.d
	.loc 12 1104 0
..LDL968:
/*   1104 */	fmad	z28.d, p0/m, z17.d, z3.d
	.loc 12 1120 0
..LDL969:
/*   1120 */	sxtw	x22, w22
/*   1120 */	ld1d	{z24.d}, p0/z, [x8, x22, lsl #3]
	.loc 12 1112 0
..LDL970:
/*   1112 */	add	w22, w21, w14
	.loc 12 1121 0
..LDL971:
/*   1121 */	sxtw	x22, w22
/*   1121 */	ld1d	{z23.d}, p0/z, [x8, x22, lsl #3]
	.loc 12 1113 0
..LDL972:
/*   1113 */	add	w22, w21, w13
	.loc 12 1122 0
..LDL973:
/*   1122 */	sxtw	x22, w22
/*   1122 */	ld1d	{z22.d}, p0/z, [x8, x22, lsl #3]
	.loc 12 1114 0
..LDL974:
/*   1114 */	add	w22, w21, w12
	.loc 12 1123 0
..LDL975:
/*   1123 */	sxtw	x22, w22
/*   1123 */	ld1d	{z21.d}, p0/z, [x8, x22, lsl #3]
	.loc 12 1115 0
..LDL976:
/*   1115 */	add	w22, w21, w11
	.loc 12 1116 0
..LDL977:
/*   1116 */	add	w21, w21, w18
	.loc 12 1124 0
..LDL978:
/*   1124 */	sxtw	x22, w22
	.loc 12 1125 0
..LDL979:
/*   1125 */	sxtw	x21, w21
	.loc 12 1127 0
..LDL980:
/*   1127 */	fmad	z30.d, p0/m, z18.d, z1.d
	.loc 12 1125 0
..LDL981:
/*   1125 */	ld1d	{z2.d}, p0/z, [x8, x21, lsl #3]
	.loc 12 1136 0
..LDL982:
/* #00003 */	ldr	x21, [x19, 120]	//  (*)
	.loc 12 1124 0
..LDL983:
/*   1124 */	ld1d	{z3.d}, p0/z, [x8, x22, lsl #3]
	.loc 12 1128 0
..LDL984:
/*   1128 */	fmad	z25.d, p0/m, z18.d, z0.d
	.loc 12 1129 0
..LDL985:
/*   1129 */	fmad	z24.d, p0/m, z18.d, z20.d
	.loc 12 1130 0
..LDL986:
/*   1130 */	fmad	z23.d, p0/m, z18.d, z19.d
	.loc 12 1131 0
..LDL987:
/*   1131 */	fmad	z22.d, p0/m, z18.d, z27.d
	.loc 12 1136 0
..LDL988:
/*   1136 */	st1d	{z30.d}, p0, [x21, x7, lsl #3]
	.loc 12 1132 0
..LDL989:
/*   1132 */	fmad	z21.d, p0/m, z18.d, z26.d
	.loc 12 1137 0
..LDL990:
/*   1137 */	st1d	{z25.d}, p0, [x21, x0, lsl #3]
	.loc 12 1134 0
..LDL991:
/*   1134 */	fmad	z2.d, p0/m, z18.d, z28.d
	.loc 12 1133 0
..LDL992:
/*   1133 */	fmad	z3.d, p0/m, z18.d, z29.d
	.loc 12 1138 0
..LDL993:
/*   1138 */	st1d	{z24.d}, p0, [x21, x4, lsl #3]
	.loc 12 1139 0
..LDL994:
/*   1139 */	st1d	{z23.d}, p0, [x21, x20, lsl #3]
	.loc 12 1140 0
..LDL995:
/*   1140 */	st1d	{z22.d}, p0, [x21, x1, lsl #3]
	.loc 12 1141 0
..LDL996:
/*   1141 */	st1d	{z21.d}, p0, [x21, x5, lsl #3]
	.loc 12 1142 0
..LDL997:
/*   1142 */	st1d	{z3.d}, p0, [x21, x2, lsl #3]
	.loc 12 1143 0
..LDL998:
/*   1143 */	st1d	{z2.d}, p0, [x21, x3, lsl #3]
	.loc 12 1145 0
..LDL999:
/*   1145 */	bne	.L272
.L274:					// :term
	.loc 12 1150 0
..LDL1000:
/*   1150 */	add	w2, w17, 64
	.loc 12 1159 0
..LDL1001:
/*    ??? */	ldr	x5, [x19, 16]	//  (*)
	.loc 12 107 0 is_stmt 0
..LDL1002:
/*    107 */	ptrue	p0.d, ALL
	.loc 12 1151 0 is_stmt 1
..LDL1003:
/*   1151 */	add	w8, w16, 64
	.loc 12 1159 0
..LDL1004:
/*   1159 */	sxtw	x4, w2
	.loc 12 1152 0
..LDL1005:
/*   1152 */	add	w3, w15, 64
	.loc 12 1153 0
..LDL1006:
/*   1153 */	add	w0, w14, 64
	.loc 12 1154 0
..LDL1007:
/*   1154 */	add	w1, w13, 64
	.loc 12 1159 0
..LDL1008:
/*    ??? */	str	x4, [x19, 40]	//  (*)
	.loc 12 1162 0
..LDL1009:
/*   1162 */	sxtw	x28, w0
	.loc 12 1163 0
..LDL1010:
/*   1163 */	sxtw	x7, w1
	.loc 12 1159 0
..LDL1011:
/*   1159 */	add	x4, x5, x4, lsl #3
	.loc 12 1155 0
..LDL1012:
/*   1155 */	add	w6, w12, 64
	.loc 12 1156 0
..LDL1013:
/*   1156 */	add	w9, w11, 64
	.loc 12 1159 0
..LDL1014:
/*   1159 */	ld1d	{z16.d}, p0/z, [x4, 0, mul vl]
	.loc 12 1160 0
..LDL1015:
/*   1160 */	sxtw	x4, w8
	.loc 12 1179 0
..LDL1016:
/*   1179 */	add	w25, w17, 63
	.loc 12 1160 0
..LDL1017:
/*    ??? */	str	x4, [x19, 32]	//  (*)
/*   1160 */	add	x4, x5, x4, lsl #3
	.loc 12 1188 0
..LDL1018:
/*   1188 */	sxtw	x26, w25
	.loc 12 1157 0
..LDL1019:
/*   1157 */	add	w10, w18, 64
	.loc 12 1160 0
..LDL1020:
/*   1160 */	ld1d	{z6.d}, p0/z, [x4, 0, mul vl]
	.loc 12 1161 0
..LDL1021:
/*   1161 */	sxtw	x4, w3
	.loc 12 1166 0
..LDL1022:
/*   1166 */	sxtw	x27, w10
	.loc 12 1161 0
..LDL1023:
/*    ??? */	str	x4, [x19, 24]	//  (*)
/*   1161 */	add	x4, x5, x4, lsl #3
	.loc 12 1180 0
..LDL1024:
/*   1180 */	add	w24, w16, 63
	.loc 12 1181 0
..LDL1025:
/*   1181 */	add	w23, w15, 63
	.loc 12 1182 0
..LDL1026:
/*   1182 */	add	w22, w14, 63
	.loc 12 1161 0
..LDL1027:
/*   1161 */	ld1d	{z7.d}, p0/z, [x4, 0, mul vl]
	.loc 12 1162 0
..LDL1028:
/*   1162 */	add	x4, x5, x28, lsl #3
	.loc 12 1183 0
..LDL1029:
/*   1183 */	add	w21, w13, 63
	.loc 12 1184 0
..LDL1030:
/*   1184 */	add	w20, w12, 63
	.loc 12 1189 0
..LDL1031:
/*   1189 */	sxtw	x24, w24
	.loc 12 1162 0
..LDL1032:
/*   1162 */	ld1d	{z4.d}, p0/z, [x4, 0, mul vl]
	.loc 12 1163 0
..LDL1033:
/*   1163 */	add	x4, x5, x7, lsl #3
	.loc 12 1164 0
..LDL1034:
/*   1164 */	sxtw	x5, w6
	.loc 12 1190 0
..LDL1035:
/*   1190 */	sxtw	x23, w23
	.loc 12 1163 0
..LDL1036:
/*   1163 */	ld1d	{z5.d}, p0/z, [x4, 0, mul vl]
	.loc 12 1164 0
..LDL1037:
/*    ??? */	ldr	x4, [x19, 16]	//  (*)
	.loc 12 1191 0
..LDL1038:
/*   1191 */	sxtw	x22, w22
	.loc 12 1192 0
..LDL1039:
/*   1192 */	sxtw	x21, w21
	.loc 12 1165 0
..LDL1040:
/*    ??? */	ldr	x30, [x19, 16]	//  (*)
	.loc 12 1168 0
..LDL1041:
/*    ??? */	ldr	z2, [x29, 1, mul vl]	//  (*)
	.loc 12 1193 0
..LDL1042:
/*   1193 */	sxtw	x20, w20
	.loc 12 1188 0
..LDL1043:
/*    ??? */	ldr	x25, [x19, 16]	//  (*)
	.loc 12 1186 0
..LDL1044:
/*   1186 */	add	w18, w18, 63
	.loc 12 1211 0
..LDL1045:
/*   1211 */	add	w14, w14, 65
	.loc 12 1164 0
..LDL1046:
/*   1164 */	add	x4, x4, x5, lsl #3
	.loc 12 1195 0
..LDL1047:
/*   1195 */	sxtw	x18, w18
	.loc 12 1213 0
..LDL1048:
/*   1213 */	add	w12, w12, 65
	.loc 12 1164 0
..LDL1049:
/*   1164 */	ld1d	{z1.d}, p0/z, [x4, 0, mul vl]
	.loc 12 1165 0
..LDL1050:
/*   1165 */	sxtw	x4, w9
	.loc 12 1188 0
..LDL1051:
/*   1188 */	add	x25, x25, x26, lsl #3
	.loc 12 1165 0
..LDL1052:
/*   1165 */	add	x30, x30, x4, lsl #3
	.loc 12 1188 0
..LDL1053:
/*   1188 */	ld1d	{z21.d}, p0/z, [x25, 0, mul vl]
	.loc 12 1189 0
..LDL1054:
/*    ??? */	ldr	x25, [x19, 16]	//  (*)
	.loc 12 1165 0
..LDL1055:
/*   1165 */	ld1d	{z0.d}, p0/z, [x30, 0, mul vl]
	.loc 12 1166 0
..LDL1056:
/*    ??? */	ldr	x30, [x19, 16]	//  (*)
	.loc 12 1168 0
..LDL1057:
/*   1168 */	fmul	z24.d, z16.d, z2.d
	.loc 12 1169 0
..LDL1058:
/*   1169 */	fmul	z23.d, z6.d, z2.d
	.loc 12 1170 0
..LDL1059:
/*   1170 */	fmul	z22.d, z7.d, z2.d
	.loc 12 1171 0
..LDL1060:
/*   1171 */	fmul	z20.d, z4.d, z2.d
	.loc 12 1172 0
..LDL1061:
/*   1172 */	fmul	z19.d, z5.d, z2.d
	.loc 12 1166 0
..LDL1062:
/*   1166 */	add	x30, x30, x27, lsl #3
/*   1166 */	ld1d	{z3.d}, p0/z, [x30, 0, mul vl]
	.loc 12 1173 0
..LDL1063:
/*   1173 */	fmul	z17.d, z1.d, z2.d
	.loc 12 1185 0
..LDL1064:
/*   1185 */	add	w30, w11, 63
	.loc 12 1189 0
..LDL1065:
/*   1189 */	add	x24, x25, x24, lsl #3
	.loc 12 1194 0
..LDL1066:
/*   1194 */	sxtw	x30, w30
	.loc 12 1189 0
..LDL1067:
/*   1189 */	ld1d	{z18.d}, p0/z, [x24, 0, mul vl]
	.loc 12 1190 0
..LDL1068:
/*   1190 */	add	x23, x25, x23, lsl #3
	.loc 12 1174 0
..LDL1069:
/*   1174 */	fmul	z16.d, z0.d, z2.d
	.loc 12 1197 0
..LDL1070:
/*    ??? */	ldr	z0, [x29, 2, mul vl]	//  (*)
	.loc 12 1190 0
..LDL1071:
/*   1190 */	ld1d	{z7.d}, p0/z, [x23, 0, mul vl]
	.loc 12 1191 0
..LDL1072:
/*   1191 */	add	x22, x25, x22, lsl #3
/*   1191 */	ld1d	{z6.d}, p0/z, [x22, 0, mul vl]
	.loc 12 1192 0
..LDL1073:
/*   1192 */	add	x21, x25, x21, lsl #3
/*   1192 */	ld1d	{z5.d}, p0/z, [x21, 0, mul vl]
	.loc 12 1193 0
..LDL1074:
/*   1193 */	add	x20, x25, x20, lsl #3
	.loc 12 1175 0
..LDL1075:
/*   1175 */	fmul	z1.d, z3.d, z2.d
	.loc 12 1193 0
..LDL1076:
/*   1193 */	ld1d	{z4.d}, p0/z, [x20, 0, mul vl]
	.loc 12 1194 0
..LDL1077:
/*   1194 */	add	x30, x25, x30, lsl #3
/*   1194 */	ld1d	{z3.d}, p0/z, [x30, 0, mul vl]
	.loc 12 1195 0
..LDL1078:
/*   1195 */	add	x18, x25, x18, lsl #3
	.loc 12 1217 0
..LDL1079:
/*   1217 */	add	x30, x27, 3
	.loc 12 1195 0
..LDL1080:
/*   1195 */	ld1d	{z2.d}, p0/z, [x18, 0, mul vl]
	.loc 12 1197 0
..LDL1081:
/*   1197 */	fmad	z21.d, p0/m, z0.d, z24.d
	.loc 12 1198 0
..LDL1082:
/*   1198 */	fmad	z18.d, p0/m, z0.d, z23.d
	.loc 12 1208 0
..LDL1083:
/*   1208 */	add	w18, w17, 65
	.loc 12 1199 0
..LDL1084:
/*   1199 */	fmad	z7.d, p0/m, z0.d, z22.d
	.loc 12 1209 0
..LDL1085:
/*   1209 */	add	w17, w16, 65
	.loc 12 1210 0
..LDL1086:
/*   1210 */	add	w16, w15, 65
	.loc 12 1212 0
..LDL1087:
/*   1212 */	add	w15, w13, 65
	.loc 12 1214 0
..LDL1088:
/*   1214 */	add	w13, w11, 65
	.loc 12 1200 0
..LDL1089:
/*   1200 */	fmad	z6.d, p0/m, z0.d, z20.d
	.loc 12 1217 0
..LDL1090:
/*   1217 */	add	x11, x27, 1
	.loc 12 1201 0
..LDL1091:
/*   1201 */	fmad	z5.d, p0/m, z0.d, z19.d
	.loc 12 1202 0
..LDL1092:
/*   1202 */	fmad	z4.d, p0/m, z0.d, z17.d
	.loc 12 1203 0
..LDL1093:
/*   1203 */	fmad	z3.d, p0/m, z0.d, z16.d
	.loc 12 1204 0
..LDL1094:
/*   1204 */	fmad	z0.d, p0/m, z2.d, z1.d
	.loc 12 1217 0
..LDL1095:
/*   1217 */	ldr	d1, [x25, x11, lsl #3]
	.loc 12 167 0 is_stmt 0
..LDL1096:
/*    167 */	add	x11, x19, 288
	.loc 12 1217 0
..LDL1097:
/*   1217 */	str	d1, [x11, -64]	//  "fcp1_arr"
/*   1217 */	add	x11, x27, 2
/*   1217 */	ldr	d1, [x25, x11, lsl #3]
	.loc 12 167 0
..LDL1098:
/*    167 */	add	x11, x19, 288
	.loc 12 1217 0
..LDL1099:
/*   1217 */	str	d1, [x11, -56]	//  "fcp1_arr"
	.loc 12 167 0
..LDL1100:
/*    167 */	add	x11, x19, 288
	.loc 12 1217 0
..LDL1101:
/*   1217 */	ldr	d1, [x25, x30, lsl #3]
/*   1217 */	add	x30, x27, 6
/*   1217 */	str	d1, [x11, -48]	//  "fcp1_arr"
/*   1217 */	add	x11, x27, 4
/*   1217 */	ldr	d1, [x25, x11, lsl #3]
	.loc 12 167 0
..LDL1102:
/*    167 */	add	x11, x19, 288
	.loc 12 1217 0
..LDL1103:
/*   1217 */	str	d1, [x11, -40]	//  "fcp1_arr"
/*   1217 */	add	x11, x27, 5
/*   1217 */	ldr	d1, [x25, x11, lsl #3]
	.loc 12 167 0
..LDL1104:
/*    167 */	add	x11, x19, 288
	.loc 12 1217 0
..LDL1105:
/*   1217 */	str	d1, [x11, -32]	//  "fcp1_arr"
	.loc 12 167 0
..LDL1106:
/*    167 */	add	x11, x19, 288
	.loc 12 1217 0
..LDL1107:
/*   1217 */	ldr	d1, [x25, x30, lsl #3]
	.loc 12 167 0
..LDL1108:
/*    167 */	add	x30, x19, 288
	.loc 12 1217 0
..LDL1109:
/*   1217 */	str	d1, [x11, -24]	//  "fcp1_arr"
/*   1217 */	add	x11, x27, 7
/*   1217 */	ldr	d1, [x25, x11, lsl #3]
/*   1217 */	fmov	x11, d1
	.loc 12 1227 0 is_stmt 1
..LDL1110:
/*    ??? */	ldr	z1, [x29, 3, mul vl]	//  (*)
	.loc 12 1217 0
..LDL1111:
/*   1217 */	str	x11, [x30, -16]	//  "fcp1_arr"
	.loc 12 167 0 is_stmt 0
..LDL1112:
/*    167 */	add	x30, x19, 288
	.loc 12 1217 0
..LDL1113:
/*   1217 */	str	x11, [x30, -8]	//  "fcp1_arr"
	.loc 12 1218 0 is_stmt 1
..LDL1114:
/*   1218 */	sxtw	x11, w18
/*   1218 */	add	x11, x25, x11, lsl #3
/*   1218 */	ld1d	{z24.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1219 0
..LDL1115:
/*   1219 */	sxtw	x11, w17
/*   1219 */	add	x11, x25, x11, lsl #3
/*   1219 */	ld1d	{z23.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1220 0
..LDL1116:
/*   1220 */	sxtw	x11, w16
/*   1220 */	add	x11, x25, x11, lsl #3
/*   1220 */	ld1d	{z22.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1221 0
..LDL1117:
/*   1221 */	sxtw	x11, w14
/*   1221 */	add	x11, x25, x11, lsl #3
	.loc 12 1227 0
..LDL1118:
/*   1227 */	fmad	z24.d, p0/m, z1.d, z21.d
	.loc 12 1221 0
..LDL1119:
/*   1221 */	ld1d	{z20.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1222 0
..LDL1120:
/*   1222 */	sxtw	x11, w15
/*   1222 */	add	x11, x25, x11, lsl #3
	.loc 12 1228 0
..LDL1121:
/*   1228 */	fmad	z23.d, p0/m, z1.d, z18.d
	.loc 12 1222 0
..LDL1122:
/*   1222 */	ld1d	{z19.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1223 0
..LDL1123:
/*   1223 */	sxtw	x11, w12
	.loc 12 1238 0
..LDL1124:
/*    ??? */	ldr	w12, [x19, 12]	//  (*)
	.loc 12 1223 0
..LDL1125:
/*   1223 */	add	x11, x25, x11, lsl #3
	.loc 12 1229 0
..LDL1126:
/*   1229 */	fmad	z22.d, p0/m, z1.d, z7.d
	.loc 12 1223 0
..LDL1127:
/*   1223 */	ld1d	{z17.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1224 0
..LDL1128:
/*   1224 */	sxtw	x11, w13
	.loc 12 1239 0
..LDL1129:
/*   1239 */	add	w18, w8, w12
	.loc 12 1224 0
..LDL1130:
/*   1224 */	add	x11, x25, x11, lsl #3
	.loc 12 1240 0
..LDL1131:
/*   1240 */	add	w17, w3, w12
	.loc 12 1230 0
..LDL1132:
/*   1230 */	fmad	z20.d, p0/m, z1.d, z6.d
	.loc 12 1241 0
..LDL1133:
/*   1241 */	add	w16, w0, w12
	.loc 12 1242 0
..LDL1134:
/*   1242 */	add	w15, w1, w12
	.loc 12 1224 0
..LDL1135:
/*   1224 */	ld1d	{z16.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1225 0
..LDL1136:
	.loc 12 167 0 is_stmt 0
..LDL1137:
/*    167 */	add	x11, x19, 288
	.loc 12 1243 0 is_stmt 1
..LDL1138:
/*   1243 */	add	w14, w6, w12
	.loc 12 1225 0
..LDL1139:
/*   1225 */	ld1d	{z2.d}, p0/z, [x11, -1, mul vl]	//  "fcp1_arr"
	.loc 12 1238 0
..LDL1140:
/*   1238 */	add	w11, w2, w12
	.loc 12 1244 0
..LDL1141:
/*   1244 */	add	w13, w9, w12
	.loc 12 1245 0
..LDL1142:
/*   1245 */	add	w12, w10, w12
	.loc 12 1247 0
..LDL1143:
/*   1247 */	sxtw	x11, w11
	.loc 12 1231 0
..LDL1144:
/*   1231 */	fmad	z19.d, p0/m, z1.d, z5.d
	.loc 12 1247 0
..LDL1145:
/*   1247 */	add	x11, x25, x11, lsl #3
/*   1247 */	ld1d	{z21.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1248 0
..LDL1146:
/*   1248 */	sxtw	x11, w18
	.loc 12 1232 0
..LDL1147:
/*   1232 */	fmad	z17.d, p0/m, z1.d, z4.d
	.loc 12 1248 0
..LDL1148:
/*   1248 */	add	x11, x25, x11, lsl #3
/*   1248 */	ld1d	{z18.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1249 0
..LDL1149:
/*   1249 */	sxtw	x11, w17
	.loc 12 1233 0
..LDL1150:
/*   1233 */	fmad	z16.d, p0/m, z1.d, z3.d
	.loc 12 1234 0
..LDL1151:
/*   1234 */	fmad	z1.d, p0/m, z2.d, z0.d
	.loc 12 1249 0
..LDL1152:
/*   1249 */	add	x11, x25, x11, lsl #3
	.loc 12 1256 0
..LDL1153:
/*    ??? */	ldr	z0, [x29, 4, mul vl]	//  (*)
	.loc 12 1249 0
..LDL1154:
/*   1249 */	ld1d	{z7.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1250 0
..LDL1155:
/*   1250 */	sxtw	x11, w16
/*   1250 */	add	x11, x25, x11, lsl #3
/*   1250 */	ld1d	{z6.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1251 0
..LDL1156:
/*   1251 */	sxtw	x11, w15
/*   1251 */	add	x11, x25, x11, lsl #3
/*   1251 */	ld1d	{z5.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1252 0
..LDL1157:
/*   1252 */	sxtw	x11, w14
	.loc 12 1256 0
..LDL1158:
/*   1256 */	fmad	z21.d, p0/m, z0.d, z24.d
	.loc 12 1257 0
..LDL1159:
/*   1257 */	fmad	z18.d, p0/m, z0.d, z23.d
	.loc 12 1252 0
..LDL1160:
/*   1252 */	add	x11, x25, x11, lsl #3
	.loc 12 1258 0
..LDL1161:
/*   1258 */	fmad	z7.d, p0/m, z0.d, z22.d
	.loc 12 1252 0
..LDL1162:
/*   1252 */	ld1d	{z4.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1253 0
..LDL1163:
/*   1253 */	sxtw	x11, w13
/*   1253 */	add	x11, x25, x11, lsl #3
	.loc 12 1259 0
..LDL1164:
/*   1259 */	fmad	z6.d, p0/m, z0.d, z20.d
	.loc 12 1253 0
..LDL1165:
/*   1253 */	ld1d	{z3.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1254 0
..LDL1166:
/*   1254 */	sxtw	x11, w12
	.loc 12 1267 0
..LDL1167:
/*    ??? */	ldr	w12, [x19, 8]	//  (*)
	.loc 12 1254 0
..LDL1168:
/*   1254 */	add	x11, x25, x11, lsl #3
	.loc 12 1260 0
..LDL1169:
/*   1260 */	fmad	z5.d, p0/m, z0.d, z19.d
	.loc 12 1254 0
..LDL1170:
/*   1254 */	ld1d	{z2.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1267 0
..LDL1171:
/*   1267 */	add	w11, w2, w12
	.loc 12 1268 0
..LDL1172:
/*   1268 */	add	w18, w8, w12
	.loc 12 1269 0
..LDL1173:
/*   1269 */	add	w17, w3, w12
	.loc 12 1276 0
..LDL1174:
/*   1276 */	sxtw	x11, w11
	.loc 12 1261 0
..LDL1175:
/*   1261 */	fmad	z4.d, p0/m, z0.d, z17.d
	.loc 12 1270 0
..LDL1176:
/*   1270 */	add	w16, w0, w12
	.loc 12 1271 0
..LDL1177:
/*   1271 */	add	w14, w1, w12
	.loc 12 1276 0
..LDL1178:
/*   1276 */	add	x11, x25, x11, lsl #3
	.loc 12 1272 0
..LDL1179:
/*   1272 */	add	w15, w6, w12
	.loc 12 1273 0
..LDL1180:
/*   1273 */	add	w13, w9, w12
	.loc 12 1274 0
..LDL1181:
/*   1274 */	add	w12, w10, w12
	.loc 12 1276 0
..LDL1182:
/*   1276 */	ld1d	{z24.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1277 0
..LDL1183:
/*   1277 */	sxtw	x11, w18
	.loc 12 1262 0
..LDL1184:
/*   1262 */	fmad	z3.d, p0/m, z0.d, z16.d
	.loc 12 1277 0
..LDL1185:
/*   1277 */	add	x11, x25, x11, lsl #3
/*   1277 */	ld1d	{z23.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1278 0
..LDL1186:
/*   1278 */	sxtw	x11, w17
	.loc 12 1263 0
..LDL1187:
/*   1263 */	fmad	z0.d, p0/m, z2.d, z1.d
	.loc 12 1285 0
..LDL1188:
/*    ??? */	ldr	z1, [x29, 5, mul vl]	//  (*)
	.loc 12 1278 0
..LDL1189:
/*   1278 */	add	x11, x25, x11, lsl #3
/*   1278 */	ld1d	{z22.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1279 0
..LDL1190:
/*   1279 */	sxtw	x11, w16
/*   1279 */	add	x11, x25, x11, lsl #3
/*   1279 */	ld1d	{z20.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1280 0
..LDL1191:
/*   1280 */	sxtw	x11, w14
/*   1280 */	add	x11, x25, x11, lsl #3
/*   1280 */	ld1d	{z19.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1281 0
..LDL1192:
/*   1281 */	sxtw	x11, w15
	.loc 12 1285 0
..LDL1193:
/*   1285 */	fmad	z24.d, p0/m, z1.d, z21.d
	.loc 12 1286 0
..LDL1194:
/*   1286 */	fmad	z23.d, p0/m, z1.d, z18.d
	.loc 12 1281 0
..LDL1195:
/*   1281 */	add	x11, x25, x11, lsl #3
	.loc 12 1287 0
..LDL1196:
/*   1287 */	fmad	z22.d, p0/m, z1.d, z7.d
	.loc 12 1281 0
..LDL1197:
/*   1281 */	ld1d	{z17.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1282 0
..LDL1198:
/*   1282 */	sxtw	x11, w13
/*   1282 */	add	x11, x25, x11, lsl #3
	.loc 12 1288 0
..LDL1199:
/*   1288 */	fmad	z20.d, p0/m, z1.d, z6.d
	.loc 12 1282 0
..LDL1200:
/*   1282 */	ld1d	{z16.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1283 0
..LDL1201:
/*   1283 */	sxtw	x11, w12
	.loc 12 1296 0
..LDL1202:
/*    ??? */	ldr	w12, [x19, 152]	//  (*)
	.loc 12 1283 0
..LDL1203:
/*   1283 */	add	x11, x25, x11, lsl #3
	.loc 12 1289 0
..LDL1204:
/*   1289 */	fmad	z19.d, p0/m, z1.d, z5.d
	.loc 12 1283 0
..LDL1205:
/*   1283 */	ld1d	{z2.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1296 0
..LDL1206:
/*   1296 */	add	w11, w2, w12
	.loc 12 1297 0
..LDL1207:
/*   1297 */	add	w18, w8, w12
	.loc 12 1298 0
..LDL1208:
/*   1298 */	add	w17, w3, w12
	.loc 12 1305 0
..LDL1209:
/*   1305 */	sxtw	x11, w11
	.loc 12 1290 0
..LDL1210:
/*   1290 */	fmad	z17.d, p0/m, z1.d, z4.d
	.loc 12 1299 0
..LDL1211:
/*   1299 */	add	w16, w0, w12
	.loc 12 1300 0
..LDL1212:
/*   1300 */	add	w15, w1, w12
	.loc 12 1305 0
..LDL1213:
/*   1305 */	add	x11, x25, x11, lsl #3
	.loc 12 1301 0
..LDL1214:
/*   1301 */	add	w14, w6, w12
	.loc 12 1302 0
..LDL1215:
/*   1302 */	add	w13, w9, w12
	.loc 12 1303 0
..LDL1216:
/*   1303 */	add	w12, w10, w12
	.loc 12 1305 0
..LDL1217:
/*   1305 */	ld1d	{z21.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1306 0
..LDL1218:
/*   1306 */	sxtw	x11, w18
	.loc 12 1291 0
..LDL1219:
/*   1291 */	fmad	z16.d, p0/m, z1.d, z3.d
	.loc 12 1306 0
..LDL1220:
/*   1306 */	add	x11, x25, x11, lsl #3
/*   1306 */	ld1d	{z5.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1307 0
..LDL1221:
/*   1307 */	sxtw	x11, w17
	.loc 12 1292 0
..LDL1222:
/*   1292 */	fmad	z1.d, p0/m, z2.d, z0.d
	.loc 12 1314 0
..LDL1223:
/*    ??? */	ldr	z0, [x29, 6, mul vl]	//  (*)
	.loc 12 1307 0
..LDL1224:
/*   1307 */	add	x11, x25, x11, lsl #3
/*   1307 */	ld1d	{z18.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1308 0
..LDL1225:
/*   1308 */	sxtw	x11, w16
/*   1308 */	add	x11, x25, x11, lsl #3
/*   1308 */	ld1d	{z3.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1309 0
..LDL1226:
/*   1309 */	sxtw	x11, w15
/*   1309 */	add	x11, x25, x11, lsl #3
/*   1309 */	ld1d	{z2.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1310 0
..LDL1227:
/*   1310 */	sxtw	x11, w14
	.loc 12 1314 0
..LDL1228:
/*   1314 */	fmad	z21.d, p0/m, z0.d, z24.d
	.loc 12 1315 0
..LDL1229:
/*   1315 */	fmad	z5.d, p0/m, z0.d, z23.d
	.loc 12 1310 0
..LDL1230:
/*   1310 */	add	x11, x25, x11, lsl #3
	.loc 12 1316 0
..LDL1231:
/*   1316 */	fmad	z18.d, p0/m, z0.d, z22.d
	.loc 12 1310 0
..LDL1232:
/*   1310 */	ld1d	{z6.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1311 0
..LDL1233:
/*   1311 */	sxtw	x11, w13
/*   1311 */	add	x11, x25, x11, lsl #3
	.loc 12 1317 0
..LDL1234:
/*   1317 */	fmad	z3.d, p0/m, z0.d, z20.d
	.loc 12 1311 0
..LDL1235:
/*   1311 */	ld1d	{z4.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1312 0
..LDL1236:
/*   1312 */	sxtw	x11, w12
/*   1312 */	add	x11, x25, x11, lsl #3
	.loc 12 1318 0
..LDL1237:
/*   1318 */	fmad	z2.d, p0/m, z0.d, z19.d
	.loc 12 1312 0
..LDL1238:
/*   1312 */	ld1d	{z7.d}, p0/z, [x11, 0, mul vl]
	.loc 12 1325 0
..LDL1239:
/*    ??? */	ldr	w11, [x19, 148]	//  (*)
	.loc 12 1319 0
..LDL1240:
/*   1319 */	fmad	z6.d, p0/m, z0.d, z17.d
	.loc 12 1325 0
..LDL1241:
/*   1325 */	add	w2, w2, w11
	.loc 12 1326 0
..LDL1242:
/*   1326 */	add	w11, w8, w11
	.loc 12 1334 0
..LDL1243:
/*   1334 */	sxtw	x8, w2
	.loc 12 1335 0
..LDL1244:
/*   1335 */	sxtw	x2, w11
	.loc 12 1327 0
..LDL1245:
/*    ??? */	ldr	w11, [x19, 148]	//  (*)
	.loc 12 1320 0
..LDL1246:
/*   1320 */	fmad	z4.d, p0/m, z0.d, z16.d
	.loc 12 1334 0
..LDL1247:
/*   1334 */	add	x8, x25, x8, lsl #3
/*   1334 */	ld1d	{z20.d}, p0/z, [x8, 0, mul vl]
	.loc 12 1335 0
..LDL1248:
/*   1335 */	add	x8, x25, x2, lsl #3
	.loc 12 1329 0
..LDL1249:
/*    ??? */	ldr	w2, [x19, 148]	//  (*)
	.loc 12 1321 0
..LDL1250:
/*   1321 */	fmad	z0.d, p0/m, z7.d, z1.d
	.loc 12 1327 0
..LDL1251:
/*   1327 */	add	w11, w3, w11
	.loc 12 1328 0
..LDL1252:
/*    ??? */	ldr	w3, [x19, 148]	//  (*)
	.loc 12 1335 0
..LDL1253:
/*   1335 */	ld1d	{z19.d}, p0/z, [x8, 0, mul vl]
	.loc 12 1329 0
..LDL1254:
/*   1329 */	add	w2, w1, w2
	.loc 12 1328 0
..LDL1255:
/*   1328 */	add	w3, w0, w3
	.loc 12 1336 0
..LDL1256:
/*   1336 */	sxtw	x0, w11
	.loc 12 1337 0
..LDL1257:
/*   1337 */	sxtw	x1, w3
	.loc 12 1336 0
..LDL1258:
/*   1336 */	add	x8, x25, x0, lsl #3
	.loc 12 1330 0
..LDL1259:
/*    ??? */	ldr	w0, [x19, 148]	//  (*)
	.loc 12 1336 0
..LDL1260:
/*   1336 */	ld1d	{z17.d}, p0/z, [x8, 0, mul vl]
	.loc 12 1330 0
..LDL1261:
/*   1330 */	add	w3, w6, w0
	.loc 12 1337 0
..LDL1262:
/*   1337 */	add	x6, x25, x1, lsl #3
	.loc 12 1331 0
..LDL1263:
/*    ??? */	ldr	w1, [x19, 148]	//  (*)
	.loc 12 1338 0
..LDL1264:
/*   1338 */	sxtw	x0, w2
	.loc 12 1339 0
..LDL1265:
/*   1339 */	sxtw	x2, w3
	.loc 12 1352 0
..LDL1266:
/*    ??? */	ldr	x3, [x19, 40]	//  (*)
	.loc 12 1337 0
..LDL1267:
/*   1337 */	ld1d	{z16.d}, p0/z, [x6, 0, mul vl]
	.loc 12 1352 0
..LDL1268:
/*    ??? */	ldr	x6, [x19, 168]	//  (*)
	.loc 12 1338 0
..LDL1269:
/*   1338 */	add	x8, x25, x0, lsl #3
/*   1338 */	ld1d	{z7.d}, p0/z, [x8, 0, mul vl]
	.loc 12 1353 0
..LDL1270:
/*    ??? */	ldr	x8, [x19, 168]	//  (*)
	.loc 12 1339 0
..LDL1271:
/*   1339 */	add	x2, x25, x2, lsl #3
	.loc 12 1331 0
..LDL1272:
/*   1331 */	add	w1, w9, w1
	.loc 12 1352 0
..LDL1273:
/*   1352 */	add	x6, x6, x3, lsl #3
	.loc 12 1332 0
..LDL1274:
/*    ??? */	ldr	w3, [x19, 148]	//  (*)
	.loc 12 1340 0
..LDL1275:
/*   1340 */	sxtw	x1, w1
/*   1340 */	add	x1, x25, x1, lsl #3
	.loc 12 1332 0
..LDL1276:
/*   1332 */	add	w3, w10, w3
	.loc 12 1341 0
..LDL1277:
/*   1341 */	sxtw	x0, w3
	.loc 12 2545 0
..LDL1278:
/*    ??? */	ldr	w3, [x19, 144]	//  (*)
	.loc 12 1341 0
..LDL1279:
/*   1341 */	add	x0, x25, x0, lsl #3
	.loc 12 2545 0
..LDL1280:
/*   2545 */	add	w3, w3, 1
/*    ??? */	str	w3, [x19, 144]	//  (*)
	.loc 12 1353 0
..LDL1281:
/*    ??? */	ldr	x3, [x19, 32]	//  (*)
/*   1353 */	add	x3, x8, x3, lsl #3
	.loc 12 1343 0
..LDL1282:
/*    ??? */	ldr	z1, [x29, 7, mul vl]	//  (*)
	.loc 12 2545 0
..LDL1283:
/*    ??? */	ldr	w8, [x19, 140]	//  (*)
/*   2545 */	subs	w8, w8, 1
/*    ??? */	str	w8, [x19, 140]	//  (*)
	.loc 12 1354 0
..LDL1284:
/*    ??? */	ldr	x8, [x19, 168]	//  (*)
	.loc 12 1343 0
..LDL1285:
/*   1343 */	fmad	z20.d, p0/m, z1.d, z21.d
	.loc 12 1344 0
..LDL1286:
/*   1344 */	fmad	z19.d, p0/m, z1.d, z5.d
	.loc 12 1339 0
..LDL1287:
/*   1339 */	ld1d	{z5.d}, p0/z, [x2, 0, mul vl]
	.loc 12 1354 0
..LDL1288:
/*    ??? */	ldr	x2, [x19, 24]	//  (*)
	.loc 12 1345 0
..LDL1289:
/*   1345 */	fmad	z17.d, p0/m, z1.d, z18.d
	.loc 12 1346 0
..LDL1290:
/*   1346 */	fmad	z16.d, p0/m, z1.d, z3.d
	.loc 12 1340 0
..LDL1291:
/*   1340 */	ld1d	{z3.d}, p0/z, [x1, 0, mul vl]
	.loc 12 1355 0
..LDL1292:
/*   1355 */	add	x1, x8, x28, lsl #3
	.loc 12 1347 0
..LDL1293:
/*   1347 */	fmad	z7.d, p0/m, z1.d, z2.d
	.loc 12 1341 0
..LDL1294:
/*   1341 */	ld1d	{z2.d}, p0/z, [x0, 0, mul vl]
	.loc 12 1357 0
..LDL1295:
/*   1357 */	add	x0, x8, x5, lsl #3
	.loc 12 1354 0
..LDL1296:
/*   1354 */	add	x2, x8, x2, lsl #3
	.loc 12 1352 0
..LDL1297:
/*   1352 */	st1d	{z20.d}, p0, [x6, 0, mul vl]
	.loc 12 1353 0
..LDL1298:
/*   1353 */	st1d	{z19.d}, p0, [x3, 0, mul vl]
	.loc 12 1356 0
..LDL1299:
/*   1356 */	add	x3, x8, x7, lsl #3
	.loc 12 1354 0
..LDL1300:
/*   1354 */	st1d	{z17.d}, p0, [x2, 0, mul vl]
	.loc 12 1348 0
..LDL1301:
/*   1348 */	fmad	z5.d, p0/m, z1.d, z6.d
	.loc 12 1355 0
..LDL1302:
/*   1355 */	st1d	{z16.d}, p0, [x1, 0, mul vl]
	.loc 12 1358 0
..LDL1303:
/*   1358 */	add	x1, x8, x4, lsl #3
	.loc 12 1349 0
..LDL1304:
/*   1349 */	fmad	z3.d, p0/m, z1.d, z4.d
	.loc 12 1350 0
..LDL1305:
/*   1350 */	fmad	z1.d, p0/m, z2.d, z0.d
	.loc 12 1359 0
..LDL1306:
/*   1359 */	add	x2, x8, x27, lsl #3
	.loc 12 1356 0
..LDL1307:
/*   1356 */	st1d	{z7.d}, p0, [x3, 0, mul vl]
	.loc 12 1357 0
..LDL1308:
/*   1357 */	st1d	{z5.d}, p0, [x0, 0, mul vl]
	.loc 12 1358 0
..LDL1309:
/*   1358 */	st1d	{z3.d}, p0, [x1, 0, mul vl]
	.loc 12 1359 0
..LDL1310:
/*   1359 */	st1d	{z1.d}, p0, [x2, 0, mul vl]
	.loc 12 2545 0 is_stmt 0
..LDL1311:
/*   2545 */	bne	.L269
.L276:					// :term
	.loc 12 2546 0 is_stmt 1
..LDL1312:
/*    ??? */	ldr	w0, [x19, 160]	//  (*)
/*   2546 */	add	w0, w0, 1
/*    ??? */	str	w0, [x19, 160]	//  (*)
/*    ??? */	ldr	w0, [x19, 156]	//  (*)
/*   2546 */	subs	w0, w0, 1
/*    ??? */	str	w0, [x19, 156]	//  (*)
/*   2546 */	bne	.L266
.L278:					// :term
	.loc 12 2547 0 is_stmt 0
..LDL1313:
/*    167 */	add	x0, x19, 192
/*   2547 */	ldr	x0, [x0]
/*   2547 */	bl	__mpc_obar
	.loc 12 2551 0 is_stmt 1
..LDL1314:
/*    ??? */	ldr	x0, [x19, 176]	//  (*)
/*   2551 */	ldr	d0, [x0, 32]	//  "dt"
	.loc 12 2553 0
..LDL1315:
/*    ??? */	ldr	w0, [x19, 164]	//  (*)
/*   2553 */	add	w0, w0, 1
/*    ??? */	str	w0, [x19, 164]	//  (*)
	.loc 12 2551 0
..LDL1316:
/*   2551 */	fadd	d10, d10, d0
	.loc 12 2555 0
..LDL1317:
	.loc 12 2546 0 is_stmt 0
..LDL1318:
/*    ??? */	ldr	x20, [x19, 168]	//  (*)
/*   2555 */	fmadd	d0, d8, d0, d10
/*   2555 */	fcmpe	d0, d9
/*   2555 */	bmi	.L263
	.loc 12 2556 0 is_stmt 0
..LDL1319:
/*   2556 */	add	x0, x19, 200
/*   2556 */	ldr	x1, [x0]
/*   2556 */	cbnz	x1, .L281
	.loc 12 2558 0 is_stmt 1
..LDL1320:
/*    ??? */	ldr	x1, [x19, 176]	//  (*)
/*   2558 */	ldr	x2, [x1, 24]	//  "f1_ret"
/*    ??? */	ldr	x1, [x19, 168]	//  (*)
/*   2558 */	str	x1, [x2]	//  (*)
/*    ??? */	ldr	x1, [x19, 176]	//  (*)
/*   2558 */	ldr	x2, [x1, 16]	//  "f2_ret"
/*    ??? */	ldr	x1, [x19, 16]	//  (*)
/*   2558 */	str	x1, [x2]	//  (*)
	.loc 12 2559 0
..LDL1321:
/*    ??? */	ldr	x1, [x19, 176]	//  (*)
/*   2559 */	ldr	x1, [x1, 8]	//  "time_ret"
/*   2559 */	str	d10, [x1]	//  (*)
	.loc 12 2560 0
..LDL1322:
/*    ??? */	ldr	x1, [x19, 176]	//  (*)
/*   2560 */	ldr	x2, [x1]	//  "count_ret"
/*    ??? */	ldr	w1, [x19, 164]	//  (*)
/*   2560 */	str	w1, [x2]	//  (*)
.L281:
/*   2563 */	ldr	x0, [x0, -8]
/*   2563 */	bl	__mpc_obar
	.loc 12 2563 0
..LDL1323:
/*    ??? */	ldp	x19, x20, [x29, -16]	//  (*)
	.cfi_restore 19
	.cfi_restore 20
/*    ??? */	ldp	x21, x22, [x29, -32]	//  (*)
	.cfi_restore 21
	.cfi_restore 22
/*    ??? */	ldp	x23, x24, [x29, -48]	//  (*)
	.cfi_restore 23
	.cfi_restore 24
/*    ??? */	ldp	x25, x26, [x29, -64]	//  (*)
	.cfi_restore 25
	.cfi_restore 26
/*    ??? */	ldp	x27, x28, [x29, -80]	//  (*)
	.cfi_restore 27
	.cfi_restore 28
/*    ??? */	ldp	d8, d9, [x29, -96]	//  (*)
	.cfi_restore 72
	.cfi_restore 73
/*    ??? */	ldp	d10, d11, [x29, -112]	//  (*)
	.cfi_restore 74
	.cfi_restore 75
/*    ??? */	add	sp, x29, 0
/*    ??? */	ldp	x29, x30, [sp]	//  (*)
	.cfi_restore 29
	.cfi_restore 30
/*    ??? */	addvl	sp, sp, 8
	.cfi_def_cfa_offset 0
/*   2563 */	ret	
..D5.pchi:
	.cfi_endproc
.LFE4:
	.size	diffusion_ker40._OMP_2, .-diffusion_ker40._OMP_2
	.section	.rodata.cst8,"aM",@progbits,8
	.align	3
.LCP1:
	.word	0x9999999a,0x3fb99999
	.section	.rodata
	.align	7
__fj_dexp_const_tbl_:
	.word	0x89374bc7,0x40862e41
	.word	0xdd7abcd2,0xc086232b
	.word	0x652b82fe,0x3ff71547
	.word	0,0xbfe62e43
	.word	0xca86c39,0x3e205c61
	.word	0x624dd2f2,0x40734410
	.word	0x46f72a42,0xc0733a71
	.word	0x979a371,0x400a934f
	.word	0x50000000,0xbfd34413
	.word	0xde623e25,0xbe03ef3f
	.word	0xbbb55515,0x40026bb1
	.word	0xffc0,0x42d80000
	.word	0x7b85e9c4,0x3f811112
	.word	0x630a7716,0x3fa55557
	.word	0x55553e71,0x3fc55555
	.word	0xffffd08f,0x3fdfffff
	.word	0,0x7ff00000
	.section	.rodata
	.align	7
__fj_dsin_const_tbl_:
	.word	0x581d4000,0x43291508
	.word	0,0x43380000
	.word	0x1,0x43380000
	.word	0x50000000,0x3ff921fb
	.word	0x60000000,0x3e5110b4
	.word	0x33145c07,0x3c91a626
	.word	0x6dc9c882,0x3fe45f30
	.word	0,0x7ff80000
	.file 13 "/opt/FJSVstclanga/v1.1.0/bin/../include/arm_sve.h"
	.file 14 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/bits/stdint-intn.h"
	.file 15 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/stdio.h"
	.file 16 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/alloca.h"
	.file 17 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/bits/types/__fpos_t.h"
	.file 18 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/sys/select.h"
	.file 19 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/bits/types/struct_timeval.h"
	.file 20 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/bits/types/struct_timespec.h"
	.file 21 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/bits/types/__sigset_t.h"
	.file 22 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/bits/mathcalls-helper-functions.h"
	.file 23 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/bits/mathcalls.h"
	.file 24 "/opt/FJSVstclanga/v1.1.0/bin/../include/omp.h"
	.file 25 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/bits/stdint-uintn.h"
	.file 26 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/bits/types/__mbstate_t.h"
	.file 27 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/bits/types/__fpos64_t.h"
	.file 28 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/bits/types/__FILE.h"
	.file 29 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/bits/floatn.h"
	.file 30 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/bits/floatn-common.h"
	.file 31 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/sys/types.h"
	.file 32 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/bits/types/clock_t.h"
	.file 33 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/bits/types/clockid_t.h"
	.file 34 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/bits/types/time_t.h"
	.file 35 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/bits/types/timer_t.h"
	.file 36 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/bits/types/sigset_t.h"
	.file 37 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/bits/pthreadtypes-arch.h"
	.file 38 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/bits/thread-shared-types.h"
	.file 39 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/bits/pthreadtypes.h"
	.file 40 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/math.h"
	.file 41 "/opt/FJSVxos/devkit/aarch64/rfs/usr/include/stdint.h"
	.pushsection	.text
..text.e:
	.popsection
	.section	.debug_info
	.4byte	.LSEdebug_info-.LSBdebug_info	// Length of .debug_info section
.LSBdebug_info:
	.2byte	0x4	// Version of DWARF information
	.4byte	.Ldebug_abbrev	// Offset into .debug_abbrev section
	.byte	0x8	// Address size
	.uleb128	0x1	// DW_TAG_compile_unit (0xb)
	.ascii	"diffusion_ker40.c\0"	// DW_AT_name
	.4byte	.Ldebug_line	// DW_AT_stmt_list
	.8byte	..text.b	// DW_AT_low_pc
	.8byte	..text.e-..text.b	// DW_AT_high_pc
	.byte	0xc	// DW_AT_language
	.ascii	"/home/z30108/git_work/diffusion/FX\0"	// DW_AT_comp_dir
	.ascii	"ccpcompx: Fujitsu C/C++ Compiler 4.2.0 (Jun 15 2020 13:54:08)\0"	// DW_AT_producer
	.uleb128	0x2	// DW_TAG_subprogram (0x94)
	.ascii	"allocate_ker40\0"	// DW_AT_name
	.8byte	allocate_ker40	// DW_AT_low_pc
	.8byte	..D1.pchi-allocate_ker40	// DW_AT_high_pc
			// DW_AT_prototyped
	.byte	0xc	// DW_AT_decl_file
	.byte	0x18	// DW_AT_decl_line
			// DW_AT_external
	.uleb128	0x1	// DW_AT_frame_base
	.byte	0x9c	// DW_OP_call_frame_cfa
	.uleb128	0x3	// DW_TAG_subprogram (0xb8)
	.4byte	0x115	// DW_AT_sibling
	.ascii	"init_ker40\0"	// DW_AT_name
	.8byte	init_ker40	// DW_AT_low_pc
	.8byte	..D2.pchi-init_ker40	// DW_AT_high_pc
			// DW_AT_prototyped
	.byte	0xc	// DW_AT_decl_file
	.byte	0x1e	// DW_AT_decl_line
			// DW_AT_external
	.uleb128	0x1	// DW_AT_frame_base
	.byte	0x9c	// DW_OP_call_frame_cfa
	.uleb128	0x4	// DW_TAG_subprogram (0xdc)
	.ascii	"init_ker40._OMP_1\0"	// DW_AT_name
	.8byte	init_ker40._OMP_1	// DW_AT_low_pc
	.8byte	..D3.pchi-init_ker40._OMP_1	// DW_AT_high_pc
			// DW_AT_artificial
	.uleb128	0x1	// DW_AT_frame_base
	.byte	0x9c	// DW_OP_call_frame_cfa
	.uleb128	0x5	// DW_TAG_FJ_loop (0x101)
	.byte	0xc	// DW_AT_decl_file
	.byte	0x32	// DW_AT_FJ_loop_start_line
	.byte	0x3c	// DW_AT_FJ_loop_end_line
	.byte	0x3	// DW_AT_FJ_loop_nest_level
	.byte	0x5	// DW_AT_FJ_loop_type
	.uleb128	0x5	// DW_TAG_FJ_loop (0x107)
	.byte	0xc	// DW_AT_decl_file
	.byte	0x31	// DW_AT_FJ_loop_start_line
	.byte	0x3d	// DW_AT_FJ_loop_end_line
	.byte	0x2	// DW_AT_FJ_loop_nest_level
	.byte	0x5	// DW_AT_FJ_loop_type
	.uleb128	0x5	// DW_TAG_FJ_loop (0x10d)
	.byte	0xc	// DW_AT_decl_file
	.byte	0x30	// DW_AT_FJ_loop_start_line
	.byte	0x3e	// DW_AT_FJ_loop_end_line
	.byte	0x1	// DW_AT_FJ_loop_nest_level
	.byte	0x5	// DW_AT_FJ_loop_type
	.byte	0x0	// End of children (0xdc)
	.byte	0x0	// End of children (0xb8)
	.uleb128	0x6	// DW_TAG_subprogram (0x115)
	.ascii	"diffusion_ker40\0"	// DW_AT_name
	.8byte	diffusion_ker40	// DW_AT_low_pc
	.8byte	..D4.pchi-diffusion_ker40	// DW_AT_high_pc
			// DW_AT_prototyped
	.byte	0xc	// DW_AT_decl_file
	.byte	0x43	// DW_AT_decl_line
			// DW_AT_external
	.uleb128	0x1	// DW_AT_frame_base
	.byte	0x9c	// DW_OP_call_frame_cfa
	.uleb128	0x4	// DW_TAG_subprogram (0x13a)
	.ascii	"diffusion_ker40._OMP_2\0"	// DW_AT_name
	.8byte	diffusion_ker40._OMP_2	// DW_AT_low_pc
	.8byte	..D5.pchi-diffusion_ker40._OMP_2	// DW_AT_high_pc
			// DW_AT_artificial
	.uleb128	0x1	// DW_AT_frame_base
	.byte	0x9c	// DW_OP_call_frame_cfa
	.uleb128	0x7	// DW_TAG_FJ_loop (0x164)
	.byte	0xc	// DW_AT_decl_file
	.2byte	0x286	// DW_AT_FJ_loop_start_line
	.2byte	0x479	// DW_AT_FJ_loop_end_line
	.byte	0x4	// DW_AT_FJ_loop_nest_level
	.byte	0x5	// DW_AT_FJ_loop_type
	.uleb128	0x8	// DW_TAG_FJ_loop (0x16c)
	.byte	0xc	// DW_AT_decl_file
	.byte	0xab	// DW_AT_FJ_loop_start_line
	.2byte	0x9f1	// DW_AT_FJ_loop_end_line
	.byte	0x3	// DW_AT_FJ_loop_nest_level
	.byte	0x5	// DW_AT_FJ_loop_type
	.uleb128	0x8	// DW_TAG_FJ_loop (0x173)
	.byte	0xc	// DW_AT_decl_file
	.byte	0xa8	// DW_AT_FJ_loop_start_line
	.2byte	0x9f2	// DW_AT_FJ_loop_end_line
	.byte	0x2	// DW_AT_FJ_loop_nest_level
	.byte	0x5	// DW_AT_FJ_loop_type
	.uleb128	0x8	// DW_TAG_FJ_loop (0x17a)
	.byte	0xc	// DW_AT_decl_file
	.byte	0xa7	// DW_AT_FJ_loop_start_line
	.2byte	0x9fb	// DW_AT_FJ_loop_end_line
	.byte	0x1	// DW_AT_FJ_loop_nest_level
	.byte	0x2	// DW_AT_FJ_loop_type
	.byte	0x0	// End of children (0x13a)
	.byte	0x0	// End of children (0x115)
	.byte	0x0	// End of children (0xb)
.LSEdebug_info:
	.section	.debug_abbrev
.Ldebug_abbrev:
	.uleb128	0x1	// Abbreviation code
	.uleb128	0x11	// DW_TAG_compile_unit
	.byte	0x1	// DW_CHILDREN_yes
	.uleb128	0x3	// DW_AT_name
	.uleb128	0x8	// DW_FORM_string
	.uleb128	0x10	// DW_AT_stmt_list
	.uleb128	0x17	// DW_FORM_sec_offset
	.uleb128	0x11	// DW_AT_low_pc
	.uleb128	0x1	// DW_FORM_addr
	.uleb128	0x12	// DW_AT_high_pc
	.uleb128	0x7	// DW_FORM_data8
	.uleb128	0x13	// DW_AT_language
	.uleb128	0xb	// DW_FORM_data1
	.uleb128	0x1b	// DW_AT_comp_dir
	.uleb128	0x8	// DW_FORM_string
	.uleb128	0x25	// DW_AT_producer
	.uleb128	0x8	// DW_FORM_string
	.byte	0x0
	.byte	0x0
	.uleb128	0x2	// Abbreviation code
	.uleb128	0x2e	// DW_TAG_subprogram
	.byte	0x0	// DW_CHILDREN_no
	.uleb128	0x3	// DW_AT_name
	.uleb128	0x8	// DW_FORM_string
	.uleb128	0x11	// DW_AT_low_pc
	.uleb128	0x1	// DW_FORM_addr
	.uleb128	0x12	// DW_AT_high_pc
	.uleb128	0x7	// DW_FORM_data8
	.uleb128	0x27	// DW_AT_prototyped
	.uleb128	0x19	// DW_FORM_flag_present
	.uleb128	0x3a	// DW_AT_decl_file
	.uleb128	0xb	// DW_FORM_data1
	.uleb128	0x3b	// DW_AT_decl_line
	.uleb128	0xb	// DW_FORM_data1
	.uleb128	0x3f	// DW_AT_external
	.uleb128	0x19	// DW_FORM_flag_present
	.uleb128	0x40	// DW_AT_frame_base
	.uleb128	0x18	// DW_FORM_exprloc
	.byte	0x0
	.byte	0x0
	.uleb128	0x3	// Abbreviation code
	.uleb128	0x2e	// DW_TAG_subprogram
	.byte	0x1	// DW_CHILDREN_yes
	.uleb128	0x1	// DW_AT_sibling
	.uleb128	0x13	// DW_FORM_ref4
	.uleb128	0x3	// DW_AT_name
	.uleb128	0x8	// DW_FORM_string
	.uleb128	0x11	// DW_AT_low_pc
	.uleb128	0x1	// DW_FORM_addr
	.uleb128	0x12	// DW_AT_high_pc
	.uleb128	0x7	// DW_FORM_data8
	.uleb128	0x27	// DW_AT_prototyped
	.uleb128	0x19	// DW_FORM_flag_present
	.uleb128	0x3a	// DW_AT_decl_file
	.uleb128	0xb	// DW_FORM_data1
	.uleb128	0x3b	// DW_AT_decl_line
	.uleb128	0xb	// DW_FORM_data1
	.uleb128	0x3f	// DW_AT_external
	.uleb128	0x19	// DW_FORM_flag_present
	.uleb128	0x40	// DW_AT_frame_base
	.uleb128	0x18	// DW_FORM_exprloc
	.byte	0x0
	.byte	0x0
	.uleb128	0x4	// Abbreviation code
	.uleb128	0x2e	// DW_TAG_subprogram
	.byte	0x1	// DW_CHILDREN_yes
	.uleb128	0x3	// DW_AT_name
	.uleb128	0x8	// DW_FORM_string
	.uleb128	0x11	// DW_AT_low_pc
	.uleb128	0x1	// DW_FORM_addr
	.uleb128	0x12	// DW_AT_high_pc
	.uleb128	0x7	// DW_FORM_data8
	.uleb128	0x34	// DW_AT_artificial
	.uleb128	0x19	// DW_FORM_flag_present
	.uleb128	0x40	// DW_AT_frame_base
	.uleb128	0x18	// DW_FORM_exprloc
	.byte	0x0
	.byte	0x0
	.uleb128	0x5	// Abbreviation code
	.uleb128	0xf000	// DW_TAG_FJ_loop
	.byte	0x0	// DW_CHILDREN_no
	.uleb128	0x3a	// DW_AT_decl_file
	.uleb128	0xb	// DW_FORM_data1
	.uleb128	0x3300	// DW_AT_FJ_loop_start_line
	.uleb128	0xb	// DW_FORM_data1
	.uleb128	0x3301	// DW_AT_FJ_loop_end_line
	.uleb128	0xb	// DW_FORM_data1
	.uleb128	0x3302	// DW_AT_FJ_loop_nest_level
	.uleb128	0xb	// DW_FORM_data1
	.uleb128	0x3303	// DW_AT_FJ_loop_type
	.uleb128	0xb	// DW_FORM_data1
	.byte	0x0
	.byte	0x0
	.uleb128	0x6	// Abbreviation code
	.uleb128	0x2e	// DW_TAG_subprogram
	.byte	0x1	// DW_CHILDREN_yes
	.uleb128	0x3	// DW_AT_name
	.uleb128	0x8	// DW_FORM_string
	.uleb128	0x11	// DW_AT_low_pc
	.uleb128	0x1	// DW_FORM_addr
	.uleb128	0x12	// DW_AT_high_pc
	.uleb128	0x7	// DW_FORM_data8
	.uleb128	0x27	// DW_AT_prototyped
	.uleb128	0x19	// DW_FORM_flag_present
	.uleb128	0x3a	// DW_AT_decl_file
	.uleb128	0xb	// DW_FORM_data1
	.uleb128	0x3b	// DW_AT_decl_line
	.uleb128	0xb	// DW_FORM_data1
	.uleb128	0x3f	// DW_AT_external
	.uleb128	0x19	// DW_FORM_flag_present
	.uleb128	0x40	// DW_AT_frame_base
	.uleb128	0x18	// DW_FORM_exprloc
	.byte	0x0
	.byte	0x0
	.uleb128	0x7	// Abbreviation code
	.uleb128	0xf000	// DW_TAG_FJ_loop
	.byte	0x0	// DW_CHILDREN_no
	.uleb128	0x3a	// DW_AT_decl_file
	.uleb128	0xb	// DW_FORM_data1
	.uleb128	0x3300	// DW_AT_FJ_loop_start_line
	.uleb128	0x5	// DW_FORM_data2
	.uleb128	0x3301	// DW_AT_FJ_loop_end_line
	.uleb128	0x5	// DW_FORM_data2
	.uleb128	0x3302	// DW_AT_FJ_loop_nest_level
	.uleb128	0xb	// DW_FORM_data1
	.uleb128	0x3303	// DW_AT_FJ_loop_type
	.uleb128	0xb	// DW_FORM_data1
	.byte	0x0
	.byte	0x0
	.uleb128	0x8	// Abbreviation code
	.uleb128	0xf000	// DW_TAG_FJ_loop
	.byte	0x0	// DW_CHILDREN_no
	.uleb128	0x3a	// DW_AT_decl_file
	.uleb128	0xb	// DW_FORM_data1
	.uleb128	0x3300	// DW_AT_FJ_loop_start_line
	.uleb128	0xb	// DW_FORM_data1
	.uleb128	0x3301	// DW_AT_FJ_loop_end_line
	.uleb128	0x5	// DW_FORM_data2
	.uleb128	0x3302	// DW_AT_FJ_loop_nest_level
	.uleb128	0xb	// DW_FORM_data1
	.uleb128	0x3303	// DW_AT_FJ_loop_type
	.uleb128	0xb	// DW_FORM_data1
	.byte	0x0
	.byte	0x0
	.byte	0x0
	.section	.debug_line
.Ldebug_line:
	.section	.note.GNU-stack,"",%progbits
	.section	.fj.compile_info, "e"
	.ascii	"C::trad"
