#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>

typedef struct { int32_t t; int32_t _p; union { int64_t i64; double f64; void *p; }; } A;
#define kI 1
#define kF 3
#define kO 4
#define kD 7

extern void __cute_internal_fa2_enc_varlen_fwd_cuda_init(CUlibrary *);
extern void __cute_internal_fa2_enc_varlen_fwd_cuda_load(CUlibrary *);
extern int  __cute_internal_fa2_enc_varlen_fwd_cuda_load_to_device(CUlibrary *, int);
extern void __cute_internal_fa2_dec_varlen_fwd_cuda_init(CUlibrary *);
extern void __cute_internal_fa2_dec_varlen_fwd_cuda_load(CUlibrary *);
extern int  __cute_internal_fa2_dec_varlen_fwd_cuda_load_to_device(CUlibrary *, int);
extern void __cute_internal_fa2_enc_dense_tma_fwd_cuda_init(CUlibrary *);
extern void __cute_internal_fa2_enc_dense_tma_fwd_cuda_load(CUlibrary *);
extern int  __cute_internal_fa2_enc_dense_tma_fwd_cuda_load_to_device(CUlibrary *, int);
extern int32_t __tvm_ffi_fa2_enc_varlen_fwd(void *, const A *, int32_t, A *);
extern int32_t __tvm_ffi_fa2_dec_varlen_fwd(void *, const A *, int32_t, A *);
extern int32_t __tvm_ffi_fa2_enc_dense_tma_fwd(void *, const A *, int32_t, A *);
extern int cuda_dialect_init_library_once(void *st, void (*init)(CUlibrary *), int (*ld)(CUlibrary *, int), void *err);

extern int __cute_internal_fa2_enc_varlen_fwd_cutlass_fa2_enc_varlen_fwd_Tensorgmemoi64i641_Tensorgmemoi64i641_Tensorgmemoi64i641_Tensorgmemoi64i641____Tensorgmemo1_CUstream0x0(
    void *, int, int, int, int64_t, int64_t,
    void *, int, int, int, int64_t, int64_t,
    void *, int, int, int, int64_t, int64_t,
    void *, int, int, int, int64_t, int64_t,
    float, int, int,
    void *, int,
    cudaStream_t);
extern int __cute_internal_fa2_dec_varlen_fwd_cutlass_fa2_dec_varlen_fwd_Tensorgmemoi64i641_Tensorgmemoi64i641_Tensorgmemoi64i641_Tensorgmemoi64i641____Tensorgmemo1_CUstream0x0(
    void *, int, int, int, int64_t, int64_t,
    void *, int, int, int, int64_t, int64_t,
    void *, int, int, int, int64_t, int64_t,
    void *, int, int, int, int64_t, int64_t,
    float, int, int,
    void *, int,
    cudaStream_t);
extern int __cute_internal_fa2_enc_dense_tma_fwd_cutlass_fa2_enc_dense_tma_fwd_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641____CUstream0x0(
    void *, int, int, int, int, int64_t, int64_t, int64_t,
    void *, int, int, int, int, int64_t, int64_t, int64_t,
    void *, int, int, int, int, int64_t, int64_t, int64_t,
    void *, int, int, int, int, int64_t, int64_t, int64_t,
    float, int, int,
    cudaStream_t);

static DLTensor *mk_dl(void *d, int nd, int64_t *sh, int64_t *st) {
    DLTensor *t = calloc(1, sizeof(DLTensor));
    t->data = d; t->device = (DLDevice){kDLCUDA, 0};
    t->ndim = nd; t->dtype = (DLDataType){4, 16, 1};
    t->shape = sh; t->strides = st; return t;
}
static DLTensor *mk_dl_i32(void *d, int nd, int64_t *sh, int64_t *st) {
    DLTensor *t = calloc(1, sizeof(DLTensor));
    t->data = d; t->device = (DLDevice){kDLCUDA, 0};
    t->ndim = nd; t->dtype = (DLDataType){0, 32, 1};
    t->shape = sh; t->strides = st; return t;
}

static CUlibrary g_el, g_dl, g_tl; static int g_eo, g_do, g_to;
static int init_enc(void){if(g_eo)return 1;g_el=NULL;__cute_internal_fa2_enc_varlen_fwd_cuda_init(&g_el);if(!g_el)return 0;__cute_internal_fa2_enc_varlen_fwd_cuda_load(&g_el);__cute_internal_fa2_enc_varlen_fwd_cuda_load_to_device(&g_el,0);g_eo=1;return 1;}
static int init_dec(void){if(g_do)return 1;g_dl=NULL;__cute_internal_fa2_dec_varlen_fwd_cuda_init(&g_dl);if(!g_dl)return 0;__cute_internal_fa2_dec_varlen_fwd_cuda_load(&g_dl);__cute_internal_fa2_dec_varlen_fwd_cuda_load_to_device(&g_dl,0);g_do=1;return 1;}
static int init_tma(void){if(g_to)return 1;g_tl=NULL;__cute_internal_fa2_enc_dense_tma_fwd_cuda_init(&g_tl);if(!g_tl)return 0;__cute_internal_fa2_enc_dense_tma_fwd_cuda_load(&g_tl);__cute_internal_fa2_enc_dense_tma_fwd_cuda_load_to_device(&g_tl,0);g_to=1;return 1;}
static int use_tvm_ffi(void){const char *e=getenv("GLMASR_CUTEDSL_FA2_TVM_FFI");return e&&e[0]&&e[0]!='0';}
static int use_dec_tvm_ffi(void){const char *e=getenv("GLMASR_CUTEDSL_DEC_TVM_FFI");return e&&e[0]&&e[0]!='0';}

int flash_fwd_vllm_available(void){return init_enc();}
int flash_fwd_vllm_gqa_available(void){return init_dec();}
int flash_fwd_bridge_available(void){return init_enc();}
int flash_fwd_vllm_dense_tma_available(void){return init_tma();}

int flash_fwd_vllm_launch_varlen(void *q,void *k,void *v,void *o,int tl,int ml,int nh,int hd,int qs,int ks,int vs,int ns,int *cu,void *st){
    if(!init_enc()||hd!=64)return-1;(void)st;(void)ml;
    double ss=1.0/sqrt(64.0);
    if(!use_tvm_ffi()){
        return __cute_internal_fa2_enc_varlen_fwd_cutlass_fa2_enc_varlen_fwd_Tensorgmemoi64i641_Tensorgmemoi64i641_Tensorgmemoi64i641_Tensorgmemoi64i641____Tensorgmemo1_CUstream0x0(
            q,tl,nh,hd,qs,hd,k,tl,nh,hd,ks,hd,v,tl,nh,hd,vs,hd,o,tl,nh,hd,nh*hd,hd,(float)ss,ml,ns,cu,ns+1,(cudaStream_t)st);
    }
    int64_t qs_a[]={tl,nh,hd},qt_a[]={qs,hd,1},ks_a[]={tl,nh,hd},kt_a[]={ks,hd,1},vs_a[]={tl,nh,hd},vt_a[]={vs,hd,1},os_a[]={tl,nh,hd},ot_a[]={nh*hd,hd,1},cs[]={ns+1};
    DLTensor tQ={q,{kDLCUDA,0},3,{4,16,1},qs_a,qt_a,0};
    DLTensor tK={k,{kDLCUDA,0},3,{4,16,1},ks_a,kt_a,0};
    DLTensor tV={v,{kDLCUDA,0},3,{4,16,1},vs_a,vt_a,0};
    DLTensor tO={o,{kDLCUDA,0},3,{4,16,1},os_a,ot_a,0};
    DLTensor tC={cu,{kDLCUDA,0},1,{0,32,1},cs,(int64_t[]){1,1},0};
    A a[9]={{kD,0,{.p=&tQ}},{kD,0,{.p=&tK}},{kD,0,{.p=&tV}},{kD,0,{.p=&tO}},{kF,0,{.f64=ss}},{kI,0,{.i64=ml}},{kI,0,{.i64=ns}},{kD,0,{.p=&tC}},{kO,0}};
    A r={0}; return __tvm_ffi_fa2_enc_varlen_fwd(NULL,a,9,&r);
}

int flash_fwd_vllm_gqa_launch(void *q,void *k,void *v,void *o,int tl,int ml,int qh,int kvh,int hd,int qs,int ks,int vs,int ns,int *cu,int ca,void *st){
    if(!init_dec()||hd!=128)return-1;(void)ca;(void)st;(void)ml;
    double ss=1.0/sqrt(128.0);
    if(!use_dec_tvm_ffi()){
        return __cute_internal_fa2_dec_varlen_fwd_cutlass_fa2_dec_varlen_fwd_Tensorgmemoi64i641_Tensorgmemoi64i641_Tensorgmemoi64i641_Tensorgmemoi64i641____Tensorgmemo1_CUstream0x0(
            q,tl,qh,hd,qs,hd,k,tl,kvh,hd,ks,hd,v,tl,kvh,hd,vs,hd,o,tl,qh,hd,qh*hd,hd,(float)ss,ml,ns,cu,ns+1,(cudaStream_t)st);
    }
    int64_t qs_a[]={tl,qh,hd},qt_a[]={qs,hd,1},ks_a[]={tl,kvh,hd},kt_a[]={ks,hd,1},os_a[]={tl,qh,hd},ot_a[]={qh*hd,hd,1},cs[]={ns+1};
    DLTensor tQ={q,{kDLCUDA,0},3,{4,16,1},qs_a,qt_a,0};
    DLTensor tK={k,{kDLCUDA,0},3,{4,16,1},ks_a,kt_a,0};
    DLTensor tV={v,{kDLCUDA,0},3,{4,16,1},ks_a,kt_a,0};
    DLTensor tO={o,{kDLCUDA,0},3,{4,16,1},os_a,ot_a,0};
    DLTensor tC={cu,{kDLCUDA,0},1,{0,32,1},cs,(int64_t[]){1,1},0};
    A a[9]={{kD,0,{.p=&tQ}},{kD,0,{.p=&tK}},{kD,0,{.p=&tV}},{kD,0,{.p=&tO}},{kF,0,{.f64=ss}},{kI,0,{.i64=ml}},{kI,0,{.i64=ns}},{kD,0,{.p=&tC}},{kO,0}};
    A r={0};int32_t rc=__tvm_ffi_fa2_dec_varlen_fwd(NULL,a,9,&r);return rc;
}

int flash_fwd_vllm_launch(void *q,void *k,void *v,void *o,int sl,int nh,int hd,float sc,int *cu,int qs,int ks,int vs,void *st){
    if(!init_enc()||hd!=64)return-1;(void)sc;(void)st;
    double ss=1.0/sqrt(64.0);int cneed=sl+1,alloc=0;
    int *dcu=cu;if(!cu){cudaMalloc((void**)&dcu,cneed*4);alloc=1;int *hc=malloc(cneed*4);for(int i=0;i<=sl;i++)hc[i]=i;cudaMemcpy(dcu,hc,cneed*4,cudaMemcpyHostToDevice);free(hc);}
    int64_t qs_a[]={sl,nh,hd},qt_a[]={qs,hd,1},ks_a[]={sl,nh,hd},kt_a[]={ks,hd,1},vs_a[]={sl,nh,hd},vt_a[]={vs,hd,1},os_a[]={sl,nh,hd},ot_a[]={nh*hd,hd,1},cs[]={2};
    DLTensor tQ={q,{kDLCUDA,0},3,{4,16,1},qs_a,qt_a,0};
    DLTensor tK={k,{kDLCUDA,0},3,{4,16,1},ks_a,kt_a,0};
    DLTensor tV={v,{kDLCUDA,0},3,{4,16,1},vs_a,vt_a,0};
    DLTensor tO={o,{kDLCUDA,0},3,{4,16,1},os_a,ot_a,0};
    DLTensor tC={dcu,{kDLCUDA,0},1,{0,32,1},cs,(int64_t[]){1,1},0};
    if(!use_tvm_ffi()){
        int32_t rc=__cute_internal_fa2_enc_varlen_fwd_cutlass_fa2_enc_varlen_fwd_Tensorgmemoi64i641_Tensorgmemoi64i641_Tensorgmemoi64i641_Tensorgmemoi64i641____Tensorgmemo1_CUstream0x0(
            q,sl,nh,hd,qs,hd,k,sl,nh,hd,ks,hd,v,sl,nh,hd,vs,hd,o,sl,nh,hd,nh*hd,hd,(float)ss,sl,1,dcu,2,(cudaStream_t)st);
        if(alloc)cudaFree(dcu);return rc;
    }
    A a[9]={{kD,0,{.p=&tQ}},{kD,0,{.p=&tK}},{kD,0,{.p=&tV}},{kD,0,{.p=&tO}},{kF,0,{.f64=ss}},{kI,0,{.i64=sl}},{kI,0,{.i64=1}},{kD,0,{.p=&tC}},{kO,0}};
    A r={0};int32_t rc=__tvm_ffi_fa2_enc_varlen_fwd(NULL,a,9,&r);
    if(alloc)cudaFree(dcu);return rc;
}

int flash_fwd_vllm_launch_dense_tma(void *q,void *k,void *v,void *o,int bs,int sl,int nh,int hd,int qs,int ks,int vs,void *st){
    if(!init_tma()||bs<=0||sl<=0||nh<=0||hd!=64)return-1;
    double ss=1.0/sqrt(64.0);(void)st;
    int64_t qbs=(int64_t)sl*qs,kbs=(int64_t)sl*ks,vbs=(int64_t)sl*vs,obs=(int64_t)sl*nh*hd;
    if(!use_tvm_ffi()){
        return __cute_internal_fa2_enc_dense_tma_fwd_cutlass_fa2_enc_dense_tma_fwd_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641_Tensorgmemoi64i64i641____CUstream0x0(
            q,bs,sl,nh,hd,qbs,qs,hd,
            k,bs,sl,nh,hd,kbs,ks,hd,
            v,bs,sl,nh,hd,vbs,vs,hd,
            o,bs,sl,nh,hd,obs,nh*hd,hd,
            (float)ss,sl,bs,(cudaStream_t)st);
    }
    int64_t qsh[]={bs,sl,nh,hd},qst[]={qbs,qs,hd,1};
    int64_t ksh[]={bs,sl,nh,hd},kst[]={kbs,ks,hd,1};
    int64_t vsh[]={bs,sl,nh,hd},vst[]={vbs,vs,hd,1};
    int64_t osh[]={bs,sl,nh,hd},ost[]={obs,nh*hd,hd,1};
    DLTensor tQ={q,{kDLCUDA,0},4,{4,16,1},qsh,qst,0};
    DLTensor tK={k,{kDLCUDA,0},4,{4,16,1},ksh,kst,0};
    DLTensor tV={v,{kDLCUDA,0},4,{4,16,1},vsh,vst,0};
    DLTensor tO={o,{kDLCUDA,0},4,{4,16,1},osh,ost,0};
    A a[8]={{kD,0,{.p=&tQ}},{kD,0,{.p=&tK}},{kD,0,{.p=&tV}},{kD,0,{.p=&tO}},{kF,0,{.f64=ss}},{kI,0,{.i64=sl}},{kI,0,{.i64=bs}},{kO,0}};
    A r={0};return __tvm_ffi_fa2_enc_dense_tma_fwd(NULL,a,8,&r);
}
