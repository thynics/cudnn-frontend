// V4 rev1 probe R14 (+R6): warp-us additivity under mixed residency.
// One 512-thread block (= V4 CTA shape, 16 warps) on one SM. Roles per warp id
// per config. Each role runs a FIXED work quantum per rep; per-warp clock64
// deltas -> per-role mean time. Inflation(role) = mixed / solo.
//
// Roles:
//   EXP  : fused P/dS chain (FFMA scale -> MUFU ex2 -> FFMA dS -> bf16 pack),
//          ILP form (elements independent). Quantum = 2048 fused elems/warp-lane.
//   REDG : red.global.add.v4.f32, deep-unrolled x8, rows spread (real drain
//          shape: 2KB rows, distinct rows per warp). Quantum = 512 v4-reds/lane.
//   STS  : st.shared.v4 128b to bank-spread addresses (stmatrix LSU-port proxy;
//          NOTE: not true STSM, lower bound proxy). Quantum = 1024 st/lane.
//
// Configs (role by warp id, '-' = exit immediately):
//   0 solo_exp8    : w0-7 EXP
//   1 solo_exp12   : w0-11 EXP
//   2 solo_redg4   : w0-3 REDG
//   3 solo_redg8   : w0-7 REDG
//   4 solo_sts4    : w0-3 STS
//   5 mix_8e_4r    : w0-7 EXP, w8-11 REDG            (V4 instant: two math squads + drain)
//   6 mix_4e_4s_4r : w0-3 EXP, w4-7 STS, w8-11 REDG  (publish-phase mix)
//   7 mix_12e_4r   : w0-11 EXP, w12-15 REDG          (pool saturated + drain overlap)

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA_ERR %d %s\n",__LINE__,cudaGetErrorString(e)); return 1;} } while(0)

__device__ __forceinline__ float ex2_mufu(float t){
  float r; asm volatile("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(t)); return r;
}
__device__ __forceinline__ uint32_t bf16pack(float a, float b){
  uint32_t ua=__float_as_uint(a), ub=__float_as_uint(b);
  ua += 0x7fff + ((ua>>16)&1); ub += 0x7fff + ((ub>>16)&1);
  return (ua>>16) | (ub & 0xffff0000u);
}

__device__ float g_src[8192];

__device__ __forceinline__ long long run_exp(int lane, int iters){
  const float k=1.4426950408889634f, lse=1.5f, delta=0.25f, sc=0.044194174f;
  long long t0 = clock64();
  uint32_t sink=0;
  for (int it=0; it<iters; ++it){
    #pragma unroll 8
    for (int i=0; i<2048; i+=2){
      float x0=g_src[(lane+i)&8191], x1=g_src[(lane+i+1)&8191];
      float p0=ex2_mufu(fmaf(k,x0,-lse)), p1=ex2_mufu(fmaf(k,x1,-lse));
      float d0=fmaf(p0,x0-delta,0.f)*sc, d1=fmaf(p1,x1-delta,0.f)*sc;
      sink ^= bf16pack(d0,d1);
    }
  }
  if (sink==0xdeadbeefu) g_src[lane]=1.f;
  return clock64()-t0;
}

__device__ __forceinline__ long long run_redg(float* buf, int warp, int lane, int iters){
  long long t0 = clock64();
  for (int it=0; it<iters; ++it){
    #pragma unroll 8
    for (int r=0; r<512; ++r){
      float* p = buf + ((size_t)((warp*512+r) & 4095))*512 + (lane*4 & 511);
      asm volatile("red.global.add.v4.f32 [%0], {%1,%1,%1,%1};" :: "l"(p), "f"(1.0f) : "memory");
    }
  }
  return clock64()-t0;
}

extern __shared__ float smem[];
__device__ __forceinline__ long long run_sts(int warp, int lane, int iters){
  long long t0 = clock64();
  float4 v = make_float4(1.f,2.f,3.f,4.f);
  for (int it=0; it<iters; ++it){
    #pragma unroll 8
    for (int i=0; i<1024; ++i){
      int off = ((warp*1024+i*32+lane) & 2047) * 4;
      float4* p = reinterpret_cast<float4*>(smem + off);
      asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};"
                   :: "l"(__cvta_generic_to_shared(p)), "f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w));
    }
  }
  return clock64()-t0;
}

// role code per (config, warp): 0=exit 1=EXP 2=REDG 3=STS
__constant__ signed char ROLES[8][16] = {
  {1,1,1,1,1,1,1,1, 0,0,0,0, 0,0,0,0},
  {1,1,1,1,1,1,1,1, 1,1,1,1, 0,0,0,0},
  {2,2,2,2, 0,0,0,0, 0,0,0,0, 0,0,0,0},
  {2,2,2,2,2,2,2,2, 0,0,0,0, 0,0,0,0},
  {3,3,3,3, 0,0,0,0, 0,0,0,0, 0,0,0,0},
  {1,1,1,1,1,1,1,1, 2,2,2,2, 0,0,0,0},
  {1,1,1,1, 3,3,3,3, 2,2,2,2, 0,0,0,0},
  {1,1,1,1,1,1,1,1, 1,1,1,1, 2,2,2,2},
};

__global__ void mixed_kernel(int config, float* redbuf, long long* out, int iters){
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int role = ROLES[config][warp];
  long long cyc = 0;
  if (role==1) cyc = run_exp(lane, iters);
  else if (role==2) cyc = run_redg(redbuf, warp, lane, iters);
  else if (role==3) cyc = run_sts(warp, lane, iters);
  if (lane==0) out[warp] = cyc;
}

int main(){
  int khz=0; cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0);
  float ghz = khz>0 ? khz/1e6f : 1.86f;
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,0));
  printf("device=%s clock=%.2fGHz (single block on single SM)\n", prop.name, ghz);
  float* redbuf; CK(cudaMalloc(&redbuf, (size_t)4096*512*4)); CK(cudaMemset(redbuf,0,(size_t)4096*512*4));
  long long *out_d, out_h[16]; CK(cudaMalloc(&out_d, 16*8));
  const char* names[8]={"solo_exp8","solo_exp12","solo_redg4","solo_redg8","solo_sts4",
                        "mix_8e_4r","mix_4e_4s_4r","mix_12e_4r"};
  const int iters=32;
  double solo[4]={0,0,0,0}; // role -> ns per quantum solo (from configs 0..4)
  printf("%-13s %-28s %s\n","config","role: mean_ns_per_quantum","inflation_vs_solo");
  for (int cfg=0; cfg<8; ++cfg){
    CK(cudaMemset(out_d,0,16*8));
    for (int rep=0; rep<2; ++rep)
      mixed_kernel<<<1,512,48*1024>>>(cfg, redbuf, out_d, iters);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(out_h,out_d,16*8,cudaMemcpyDeviceToHost));
    double sum[4]={0,0,0,0}; int cnt[4]={0,0,0,0};
    signed char roles[16]; CK(cudaMemcpyFromSymbol(roles, ROLES, 16, cfg*16));
    for (int w=0; w<16; ++w){ int r=roles[w]; if(r){ sum[r]+= (double)out_h[w]/ghz/iters; cnt[r]++; } }
    char line[256]; int off=0;
    off+=snprintf(line+off,sizeof(line)-off,"%-13s ",names[cfg]);
    const char* rn[4]={"","EXP","REDG","STS"};
    for (int r=1;r<4;++r) if(cnt[r]){
      double ns = sum[r]/cnt[r];
      off+=snprintf(line+off,sizeof(line)-off,"%s=%.0fns ",rn[r],ns);
      if (cfg<5){ if(solo[r]==0) solo[r]=ns; }
      else if (solo[r]>0) off+=snprintf(line+off,sizeof(line)-off,"(x%.2f) ",ns/solo[r]);
    }
    printf("%s\n",line);
  }
  printf("判读: inflation ~1.0-1.2 => warp·µs 可加(乐观界成立); >1.5 => 保守界; EXP@mix_8e_4r 对 R6, REDG 对 R5 联动\n");
  printf("PROBE_DONE\n");
  return 0;
}
