// Probe week: M1/M2 (math exp-chain latency tax) + D4/D2 (drain REDG anatomy).
// Plain CUDA C++ — compiled with nvcc inside the B200 worker container.
//
// Part 1 (M1/M2): per-warp exp->dS->bf16 chain, 3 scheduling variants:
//   serial  : elements chained by fake dependency (exposes MUFU latency = today's
//             DSL-emitted serial chain behaviour, upper bound)
//   ilp     : elements independent, compiler free to interleave (M1 strip-mining
//             upper bound)
//   poly    : same but exp via FFMA-only deg-6 polynomial (M2, MUFU-free)
//   Sweep warps/SM in {4, 8} (M5's parallelism axis). Report ns/elem and
//   projected us per (128h x 64kv) tile-half (8192 elems).
//
// Part 2 (D4/D2): red.global.add.v4.f32 issue-rate with the real drain address
//   pattern (rows of 512 f32, 2KB apart). Contention variants:
//   disjoint  : every block reduces its own private rows (zero conflicts)
//   same      : all blocks hammer the same 128 rows (max same-address queuing)
//   overlap50 : ~50% row overlap between neighbouring blocks (topk-realistic)
//   Report ns per v4-red and aggregate GB/s across 148 blocks.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA_ERR %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1;} } while(0)

__device__ __forceinline__ float ex2_mufu(float t){
  float r; asm volatile("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(t)); return r;
}
__device__ __forceinline__ float exp_poly6(float t){
  // 2^t: magic-const round-to-nearest + Taylor deg-6 on [-0.5, 0.5], pure FFMA.
  const float MAGIC = 12582912.0f; // 1.5 * 2^23
  float z = t + MAGIC;
  int   n = __float_as_int(z) & 0x007FFFFF; n -= 0x00400000;
  float f = t - (z - MAGIC);
  float p = 0.00015403530f;
  p = fmaf(p, f, 0.00133335581f);
  p = fmaf(p, f, 0.00961812911f);
  p = fmaf(p, f, 0.05550410866f);
  p = fmaf(p, f, 0.24022650696f);
  p = fmaf(p, f, 0.69314718056f);
  p = fmaf(p, f, 1.0f);
  int eb = ((__float_as_int(p) >> 23) & 0xFF) + n;
  if (eb <= 0) return 0.0f;
  return __int_as_float((__float_as_int(p) & 0x807FFFFF) | (eb << 23));
}
__device__ __forceinline__ uint32_t bf16pack(float a, float b){
  uint32_t ua=__float_as_uint(a), ub=__float_as_uint(b);
  ua += 0x7fff + ((ua>>16)&1); ub += 0x7fff + ((ub>>16)&1);
  return (ua>>16) | (ub & 0xffff0000u);
}

// mode: 0=serial(MUFU, fake dep), 1=ilp(MUFU), 2=poly(FFMA)
template<int MODE>
__global__ void math_chain_kernel(const float* __restrict__ s_in,
                                  const float* __restrict__ dp_in,
                                  uint32_t* __restrict__ out,
                                  int elems_per_thread, int iters,
                                  long long* __restrict__ cycles){
  const float k = 1.4426950408889634f, lse = 1.5f, delta = 0.25f, sc = 0.044194174f;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float carry = 0.0f;
  long long t0 = clock64();
  uint32_t sink = 0;
  for (int it = 0; it < iters; ++it) {
    #pragma unroll 8
    for (int i = 0; i < elems_per_thread; i += 2) {
      float x0 = s_in[(tid + i)   & 0xFFFF] + carry;
      float x1 = s_in[(tid + i+1) & 0xFFFF];
      float d0 = dp_in[(tid + i)  & 0xFFFF];
      float d1 = dp_in[(tid + i+1)& 0xFFFF];
      float p0, p1;
      if (MODE == 2) { p0 = exp_poly6(fmaf(k, x0, -lse)); p1 = exp_poly6(fmaf(k, x1, -lse)); }
      else           { p0 = ex2_mufu (fmaf(k, x0, -lse)); p1 = ex2_mufu (fmaf(k, x1, -lse)); }
      float ds0 = p0 * (d0 - delta) * sc;
      float ds1 = p1 * (d1 - delta) * sc;
      uint32_t packed = bf16pack(ds0, ds1);
      sink ^= packed;
      if (MODE == 0) carry = __int_as_float(packed & 1); // fake dep: serialize chain
    }
  }
  long long t1 = clock64();
  if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
  out[tid & 0xFFFF] = sink;
}

// v4 red probe. rows table gives each block its 128 target rows.
__global__ void redg_kernel(float* __restrict__ buf,
                            const int* __restrict__ rows,
                            int rows_per_block, int floats_per_row, int iters,
                            long long* __restrict__ cycles){
  int lane = threadIdx.x;                       // 256 threads = 8 warps ("reducers")
  const int* my_rows = rows + blockIdx.x * rows_per_block;
  long long t0 = clock64();
  for (int it = 0; it < iters; ++it) {
    for (int r = 0; r < rows_per_block; ++r) {
      float* base = buf + (size_t)my_rows[r] * floats_per_row;
      for (int c = lane * 4; c < floats_per_row; c += blockDim.x * 4) {
        float4 v = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        asm volatile("red.global.add.v4.f32 [%0], {%1,%2,%3,%4};"
                     :: "l"(base + c), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
                     : "memory");
      }
    }
  }
  long long t1 = clock64();
  if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
}

static float ghz = 1.86f;

static void report_math(const char* name, long long cyc, int elems, int iters, int warps){
  double ns = (double)cyc / ghz;
  double per_elem = ns / ((double)elems * iters);   // ns per element per thread(lane)
  // one P+dS tile pass per CTA ~= 8192 element-ops spread over warps*32 lanes
  printf("MATH %-8s warps=%d  ns/elem/lane=%.3f  proj_us_per_8192elems=%.3f\n",
         name, warps, per_elem, per_elem * (8192.0 / (warps * 32)) / 1e3);
}

int main(int argc, char** argv){
  int iters = 64;
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  int khz = 0;
  cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0);   // removed from struct in CUDA 13
  ghz = khz > 0 ? khz / 1e6f : 1.86f;
  printf("device=%s SMs=%d clock=%.2fGHz\n", prop.name, prop.multiProcessorCount, ghz);

  // ---------------- Part 1: math chain ----------------
  const int ELEMS = 2048;      // per thread per iter
  float *s_in, *dp_in; uint32_t* out; long long* cyc;
  CK(cudaMalloc(&s_in, 65536*4)); CK(cudaMalloc(&dp_in, 65536*4));
  CK(cudaMalloc(&out, 65536*4)); CK(cudaMalloc(&cyc, 1024*8));
  CK(cudaMemset(s_in, 0, 65536*4)); CK(cudaMemset(dp_in, 0, 65536*4));
  long long hcyc;
  for (int warps : {4, 8}) {
    int threads = warps * 32;
    for (int mode = 0; mode < 3; ++mode) {
      const char* nm = mode==0 ? "serial" : (mode==1 ? "ilp" : "poly");
      // warmup + timed (single block => single SM, worst-case no cross-block hiding)
      for (int rep = 0; rep < 2; ++rep) {
        if (mode==0) math_chain_kernel<0><<<1, threads>>>(s_in, dp_in, out, ELEMS, iters, cyc);
        if (mode==1) math_chain_kernel<1><<<1, threads>>>(s_in, dp_in, out, ELEMS, iters, cyc);
        if (mode==2) math_chain_kernel<2><<<1, threads>>>(s_in, dp_in, out, ELEMS, iters, cyc);
      }
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(&hcyc, cyc, 8, cudaMemcpyDeviceToHost));
      report_math(nm, hcyc, ELEMS, iters, warps);
    }
  }

  // ---------------- Part 2: REDG ----------------
  const int NBLOCKS = 148, RPB = 128, FPR = 512;   // 128 rows x 2KB per block/iter
  const int NROWS = 65536;
  float* buf; int* rows_d; long long* cyc2;
  CK(cudaMalloc(&buf, (size_t)NROWS * FPR * 4));
  CK(cudaMemset(buf, 0, (size_t)NROWS * FPR * 4));
  CK(cudaMalloc(&rows_d, NBLOCKS * RPB * 4));
  CK(cudaMalloc(&cyc2, NBLOCKS * 8));
  int* rows_h = new int[NBLOCKS * RPB];
  cudaEvent_t ev0, ev1; CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));
  const int riters = 16;
  // rev2: concurrency x hot-set sweep to separate solo issue floor / scaling /
  // L2-resident vs DRAM-resident RMW. modes: same(256KB hot), l2res(rows in
  // 4096-row=8MB window, mimics real dKV tensor), dram(spread over 128MB).
  int concs[4] = {1, 37, 74, 148};
  for (int mode = 0; mode < 3; ++mode) {
    const char* nm = mode==0 ? "same256K" : (mode==1 ? "l2res8MB" : "dram128MB");
    for (int ci = 0; ci < 4; ++ci) {
      int nb = concs[ci];
      for (int b = 0; b < nb; ++b)
        for (int r = 0; r < RPB; ++r) {
          if (mode == 0) rows_h[b*RPB + r] = r;
          if (mode == 1) rows_h[b*RPB + r] = (b*RPB + r) % 4096;
          if (mode == 2) rows_h[b*RPB + r] = (b*RPB + r) % NROWS;
        }
      CK(cudaMemcpy(rows_d, rows_h, nb*RPB*4, cudaMemcpyHostToDevice));
      redg_kernel<<<nb, 256>>>(buf, rows_d, RPB, FPR, 1, cyc2); // warmup
      CK(cudaDeviceSynchronize());
      CK(cudaEventRecord(ev0));
      redg_kernel<<<nb, 256>>>(buf, rows_d, RPB, FPR, riters, cyc2);
      CK(cudaEventRecord(ev1));
      CK(cudaDeviceSynchronize());
      float ms; CK(cudaEventElapsedTime(&ms, ev0, ev1));
      double per_block_us_per_iter = ms * 1e3 / riters;             // one drain-equivalent
      double bytes = (double)nb * RPB * FPR * 4 * riters;
      printf("REDG %-9s conc=%-3d  per_block_us_per_tile_drain=%.3f  aggr_GBps=%.1f\n",
             nm, nb, per_block_us_per_iter, bytes / (ms * 1e-3) / 1e9);
    }
  }
  printf("PROBE_DONE\n");
  return 0;
}
