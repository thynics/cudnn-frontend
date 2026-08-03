// V4 rev1 probe R5/R15: REDG channel rate with kernel-grade code shape.
// Part1 (mixed_residency) naive loop hit ~10ns/warp-instr SM rate; v12's real
// drain achieved ~3.9ns. Difference suspects: per-row table load + per-iter
// address math. This probe issues reds with register-resident precomputed
// bases and pointer-increment only, unroll 16, independent addresses.
//   A: per-SM rate vs warps {2,4,8} and unroll shape  -> true channel ns/instr
//   B: aggregate at 148 blocks, 8MB L2-resident working set (real dKV size)
//      -> is >=4.5TB/s reachable (R15: V4 needs ~4.3TB/s device-wide)

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA_ERR %d %s\n",__LINE__,cudaGetErrorString(e)); return 1;} } while(0)

__device__ __forceinline__ void redv4(float* p){
  asm volatile("red.global.add.v4.f32 [%0], {%1,%1,%1,%1};" :: "l"(p), "f"(1.0f) : "memory");
}

// each warp: ROWS rows x (512 floats/row -> lanes cover 128 v4-chunks... lane
// handles 4 chunks/row via 4 base pointers). All addresses precomputed, inner
// loop = pointer increment by constant row stride only.
__global__ void drain_kernel(float* buf, int rows_per_warp, int iters,
                             long long* cycles, int row_words, int wrap_rows){
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  size_t row0 = ((size_t)(blockIdx.x * (blockDim.x >> 5) + warp) * rows_per_warp) % wrap_rows;
  float* b0 = buf + row0 * row_words + lane * 4;
  float* b1 = b0 + 32 * 4;      // lane covers 4 x 16B chunks per 2KB row
  float* b2 = b0 + 64 * 4;
  float* b3 = b0 + 96 * 4;
  size_t stride = row_words;    // next row
  long long t0 = clock64();
  for (int it = 0; it < iters; ++it){
    float *p0=b0, *p1=b1, *p2=b2, *p3=b3;
    #pragma unroll 4
    for (int r = 0; r < rows_per_warp; ++r){
      redv4(p0); redv4(p1); redv4(p2); redv4(p3);   // 4 independent reds back-to-back
      p0 += stride; p1 += stride; p2 += stride; p3 += stride;
    }
  }
  long long t1 = clock64();
  if (lane == 0) cycles[blockIdx.x * 16 + warp] = t1 - t0;
}

int main(){
  int khz=0; cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0);
  float ghz = khz>0 ? khz/1e6f : 1.86f;
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,0));
  printf("device=%s SMs=%d clock=%.2fGHz\n", prop.name, prop.multiProcessorCount, ghz);
  const int ROW_WORDS = 512;                 // 2KB rows (real dKV row shape)
  const int WRAP = 4096;                     // 8MB working set: L2-resident like real dKV
  float* buf; CK(cudaMalloc(&buf, (size_t)WRAP * ROW_WORDS * 4));
  CK(cudaMemset(buf, 0, (size_t)WRAP * ROW_WORDS * 4));
  long long* cyc; CK(cudaMalloc(&cyc, 148*16*8));
  long long h[148*16];
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

  printf("--- A: per-SM channel rate (1 block, kernel-grade shape) ---\n");
  for (int warps : {2,4,8}){
    const int rows = 128, iters = 64;
    for (int rep=0; rep<2; ++rep) drain_kernel<<<1, warps*32>>>(buf, rows, iters, cyc, ROW_WORDS, WRAP);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h, cyc, 16*8, cudaMemcpyDeviceToHost));
    double mx=0; for (int w=0; w<warps; ++w) mx = h[w]>mx? (double)h[w]:mx;
    double ns = mx/ghz;
    long long instr_per_warp = (long long)rows*4*iters;      // warp-level red instrs
    double ns_warp_instr = ns/instr_per_warp;
    double sm_rate = ns/( (double)instr_per_warp*warps );    // ns per instr SM-wide
    double gbps = (double)warps*instr_per_warp*512.0/ns;     // 512B per warp-instr
    printf("warps=%d  ns/instr/warp=%.2f  SM-rate=%.2fns/instr  perSM_GBps=%.1f\n",
           warps, ns_warp_instr, sm_rate, gbps);
  }

  printf("--- B: aggregate 148 blocks, 8MB L2-resident set ---\n");
  for (int warps : {4,8}){
    const int rows = 128, iters = 32;
    for (int rep=0; rep<2; ++rep) drain_kernel<<<148, warps*32>>>(buf, rows, iters, cyc, ROW_WORDS, WRAP);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0));
    drain_kernel<<<148, warps*32>>>(buf, rows, iters, cyc, ROW_WORDS, WRAP);
    CK(cudaEventRecord(e1));
    CK(cudaDeviceSynchronize());
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    double bytes = 148.0*warps*(double)rows*4*iters*512.0;
    printf("warps=%d  total_ms=%.3f  aggregate_TBps=%.2f  (V4 需求 ~4.3TB/s)\n",
           warps, ms, bytes/(ms*1e-3)/1e12);
  }
  printf("PROBE_DONE\n");
  return 0;
}
