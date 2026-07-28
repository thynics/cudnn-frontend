# DSA Sparse Attention Backward SM100 Baseline（当前 1-CTA 实现）

> 本文件只描述 cuDNN Frontend 当前已经存在的 SM100 sparse DSA backward
> baseline。这里的 “baseline” 始终指
> `CtaGroup.ONE + cluster=[1,1,1] + block_tile=64` 的现有实现。
>
> 即将实现的协作式 2-CTA 方案只在
> [`优化设计文档.md`](优化设计文档.md) 中定义。2-CTA 的
> 内部结构与同步协议都不属于 baseline，也不在本文展开。

## 1. 事实源、范围与验证状态

### 1.1 固定事实源

| 类型 | 路径 / 版本 |
|---|---|
| `refcode4agent` commit | `b18ddf05e7a324bc9cec9982908759f89bf533ac` |
| cuDNN Frontend commit | `0c93f09cb36bea27d17066288afe8dfa26ac2398` |
| baseline kernel | `refcode4agent/gemm/cudnn-frontend/python/cudnn/deepseek_sparse_attention/sparse_attention_backward/dsa_bwd_sm100.py` |
| launch / interface | `refcode4agent/gemm/cudnn-frontend/python/cudnn/deepseek_sparse_attention/sparse_attention_backward/_interface_sm100.py` |
| correctness harness | `refcode4agent/gemm/cudnn-frontend/test/python/fe_api/dsa/test_DSA_sparse_attention_backward.py` |
| numerical reference | `refcode4agent/gemm/cudnn-frontend/test/python/fe_api/dsa/dsa_reference.py` |
| benchmark harness | `refcode4agent/gemm/cudnn-frontend/benchmark/dsa/benchmark_dsa_sparse_attention_backward.py` |
| 已有 baseline 摘要 | `context/baseline/DSA GQA128 BWD Baseline.md` |

本文中的结构、shape、资源和时序描述均来自上述固定版本。没有源码或
harness 证据的性能判断不作为 baseline 事实。

### 1.2 本任务审计的 baseline shape

```text
batch                 = 1（interface 内部固定）
H_q                   = 128
H_kv                  = 1（KV tensor 没有 head 维，实际为 MQA）
GQA ratio             = 128
Q/K dimension         = 512
V/O dimension         = 512
block_tile            = 64
topk                  = runtime / compile-key 参数
operand storage       = BF16
Tensor Core accumulator = FP32
CUDA Core math        = FP32
```

`topk=128` 可作为两个 N64 tile 的分析例子，但不是 kernel 的固定常量。
interface 还包含 `Q/K dimension=576, V/O dimension=512` 的 RoPE-tail
分支；本文只描述本任务使用的 `512/512` 路径。

### 1.3 当前验证边界

- kernel 和公开 interface 已存在，不是待实现的伪代码。
- checked-in correctness 参数当前固定为 `num_heads=64, D=512,
  topk=512`，覆盖有/无 `topk_length` 两种形式。
- benchmark 默认也是 `num_heads=64`；虽然 CLI 可以传 `128`，仓内没有
  随本文固定的 GQA=128 correctness 或性能结果。
- 因此本文可以把 GQA=128 时的 grid/ownership 从代码形式化推导出来，
  但不能声称该目标 shape 已经通过实机 correctness 或 benchmark。
- 本文只把 BF16 作为已审计数据路径。interface 的 dtype 检查还接受
  FP16，但 baseline kernel 内部的 `element_dtype` 固定为 BF16；FP16
  是否为有效生产路径未在本文核实。

## 2. 外部接口与 kernel 序列

interface 接收 flat、batch-1 tensor：

```text
Q           [S_q, H_q, D_qk]       BF16
KV          [S_kv, D_qk]           BF16, K=V, 无 KV-head 维
O, dO       [S_q, H_q, D_v]        BF16
LSE         [S_q, H_q]             FP32
attn_sink   [H_q]                  FP32
topk_idxs   [S_q, max_topk]        INT32, global KV row index
topk_length [S_q]                  INT32, optional

dQ          [S_q, H_q, D_qk]       BF16
dKV         [S_kv, D_qk]           BF16
dSink       [H_q]                  FP32
```

一次 wrapper 调用依次启动四类工作：

```text
1. sum_OdO:
     计算 sum_OdO 和带 attention sink 的 scaled_LSE

2. bwd:
     主 sparse backward kernel，生成 dQ 和 FP32 dKV workspace

3. convert:
     FP32 dKV workspace -> 外部 dKV dtype

4. sum_dSink:
     D_qk == D_v 时独立归约 dSink
```

所以 `dSink` 不在主 backward kernel 内与五个 GEMM 融合；dKV 的
global accumulation 和最终 dtype conversion 也明确分成两步。

## 3. 数学数据流

对一个 query head 和它选择的 KV rows，记：

```text
S   = Q K^T
P   = softmax_with_sink(scale * S)
dP  = dO V^T

delta = -sum_d(O * dO)
dS    = scale * P * (dP + delta)

dV  = P^T dO
dK  = dS^T Q
dQ  = dS K
dKV = dV + dK                    # K=V，共享同一参数
```

主 kernel 用转置输出方向发射五个逻辑 GEMM：

```text
S      = Q   @ K^T
dP     = dO  @ V^T
dV^T   = dO^T @ P
dK^T   = Q^T  @ dS
dQ^T  += K^T  @ dS^T
```

`sum_OdO` 预处理同时计算：

```text
sum_OdO[h,q]  = -sum_d(O[q,h,d] * dO[q,h,d])
scaled_LSE    = -log2(exp(LSE) + exp(attn_sink))
```

Compute warps 在 FP32 中执行：

```text
P  = exp2(S * softmax_scale * log2(e) + scaled_LSE)
dS = softmax_scale * P * (dP + sum_OdO)
```

随后把 P/dS downcast 为 BF16 SMEM operand，供 dV/dK/dQ 的 Tensor Core
GEMM 使用。FP32 的 P register fragment 会继续参与 dS 计算，不因 BF16
store 而提前降精度。

## 4. CTA、grid 与数据 ownership

### 4.1 launch

主 backward launch 为：

```text
grid    = (S_q, ceil_div(H_q, 64), 1)
block   = (640, 1, 1)             # 20 warps
cluster = (1, 1, 1)
MMA     = tcgen05.CtaGroup.ONE
```

当 `H_q=128` 时，对每个 query token 恰好有两个互不协作的 CTA：

```text
CTA(head_block=0) owns Q heads [ 0, 64)
CTA(head_block=1) owns Q heads [64,128)
```

它们不是一个 2-CTA cluster，也没有共享 task state、DSMEM exchange 或
cluster barrier。

### 4.2 输入 ownership

每个 CTA 独占并常驻：

```text
Q      [H64,D512]
dO     [H64,D512]
LSE    [H64]
sumOdO [H64]
```

两 CTA 使用相同 token 的 `topk_idxs`，并分别 gather 完整：

```text
KV tile [N64,D512]
```

由于 `H_kv=1` 且 CTA 之间不协作，GQA=128 的两个 head CTA 会重复读取
同一组 KV rows。

### 4.3 输出 ownership

```text
dQ:
  每 CTA 独占 [H64,D512]
  跨全部 KV tiles 在本 CTA 的 TMEM 中累加
  最后直接写回，不需要 atomic

dKV:
  每 CTA 产生本 H64 对应的完整 [D512,N64] partial
  CTA 内先完成 dV+dK
  再 FP32 atomicAdd 到同一个 global workspace
```

因此 GQA=128 的两个 baseline CTA 会对同一 dKV element 分别贡献一个
H64 reduction partial。不同 query token和重复的 global topk index 也会
继续汇入同一 workspace。

## 5. GEMM tiling

`block_tile=64` 同时是 CTA 的 head tile 和 sparse KV tile：

| GEMM | CTA logical output | reduction | baseline MMA |
|---|---:|---:|---|
| `S` | `[H64,N64]` | D512 | CG1，沿 D 累加 |
| `dP` | `[H64,N64]` | D512 | CG1，沿 D 累加 |
| `dV^T` | `[D512,N64]` | H64 | 4 个 `[D128,N64]` panel |
| `dK^T` | `[D512,N64]` | H64 | 4 个 `[D128,N64]` panel |
| `dQ^T` | `[D512,H64]` | N64 | 4 个 `[D128,H64]` panel |

四个 D128 panel 是 D 维的 concat，不是同一输出元素的 split-K：

```text
panel 0 = D[  0:128]
panel 1 = D[128:256]
panel 2 = D[256:384]
panel 3 = D[384:512]
```

dV 以 overwrite 初始化对应 dKV panel；dK 对完全相同的 TMEM panel
使用 accumulate，从而在 CTA 内直接得到 `dV+dK`。

## 6. Warp specialization

一个 CTA 使用 20 个 warps：

| Warp | role |
|---|---|
| W0-W3 | `cp.async` gather/zero-fill sparse KV rows |
| W4-W7 | S/dP T2R、FP32 P/dS math、P/dS BF16 store、dQ epilogue |
| W8-W15 | dKV T2R、FP32 workspace atomicAdd |
| W16 | elected thread 发射全部 `CtaGroup.ONE` MMA |
| W17 | TMA load Q、dO、LSE、sum_OdO |
| W18-W19 | 不参与计算分支 |

register budget 在源码中固定为：

```text
KV load / MMA / QdO load / idle = 40 registers/thread
compute / reduce                = 128 registers/thread
```

Tensor Core accumulator 位于 TMEM，因而 W16 主要保存 descriptor、地址和
pipeline state；Compute/Reduce warps 获得较高 register budget。

## 7. 实际流水线

### 7.1 pipeline stage 数

baseline 没有三级通用 operand FIFO。源码中的 pipeline stage 数为：

| producer -> consumer | stage |
|---|---:|
| Q/dO load -> MMA | 1 |
| KV gather -> MMA | 1 |
| LSE load -> compute | 1 |
| sum_OdO load -> compute | 1 |
| S MMA -> compute | 1 |
| dP MMA -> compute | 1 |
| compute P -> MMA | 1 |
| compute dS -> MMA | 1 |
| dQ MMA -> epilogue | 1 |
| dKV MMA -> reduce | 2 |

Q/dO/stats 只在 CTA 开始时生产一次，直到所有 KV tile 消费结束才释放。
KV、P、dS 则按 tile 复用各自的单 stage buffer。dKV 的两个 pipeline
stage 用于让 reducer 和后续 MMA overlap。

### 7.2 每个 KV tile 的固定顺序

KV tiles 在实现中按 index 从后向前遍历。对每个 tile，W16 的关键顺序是：

```text
S
-> dP
-> wait(P)
-> dV panel 0/1
-> wait(dS)
-> accumulate dK panel 0/1
-> publish dKV panel 0/1
-> dQ panel 0/1/2/3
-> dV panel 2/3
-> accumulate dK panel 2/3
-> publish dKV panel 2/3
```

并行关系为：

```text
KV loader: gather next tile

Compute:
  wait(S)  -> T2R -> FP32 P  -> BF16 P publish
  wait(dP) -> T2R -> FP32 dS -> BF16 dS publish

Reduce:
  wait(dKV 0/1) -> T2R -> release TMEM -> atomicAdd from registers
  wait(dKV 2/3) -> T2R -> release TMEM -> atomicAdd from registers
```

把 dQ 放在两组 dKV panel 之间，使 reducer 能较早取得 panel 0/1，并让
其 global atomic 与后续 Tensor Core 工作重叠。

### 7.3 tail

`tile_count=ceil_div(topk,64)`。partial tile 或 non-compact 输入中的
无效 index 由 KV loader zero-fill；dKV reducer 对无效 row 不发
workspace update。当前 checked-in harness 覆盖有/无 `topk_length`，
但没有把零 active tile 固定为默认 correctness case，因此本文不为
`topk=0` 声明已验证行为。

## 8. SMEM 资源与生命周期

目标 D512 路径的逻辑 payload 为：

| storage | logical shape | BF16 payload |
|---|---:|---:|
| `sQ` | `[H64,D512]` | 64 KiB |
| `sdO` | `[H64,D512]` | 64 KiB |
| `sK` / `sV` | `[N64,D512]` | 64 KiB |
| `sP` | `[H64,N64]` | 8 KiB |
| `sdS` | `[H64,N64]` | 8 KiB |
| LSE + sum_OdO | `2 x [H64] FP32` | 0.5 KiB |

这是逻辑 payload，不包含 swizzle、alignment 和 mbarrier。源码用实际
CuTe `cosize` 构造 `SharedStorage`，并在编译期断言：

```text
SharedStorage.size_in_bytes() <= 227 * 1024
```

当前文档没有固定一次真实编译打印出的最终字节数，因此不能把逻辑
208.5 KiB 当成最终 `SharedStorage`。

主要复用关系：

- `K=V`：S 和 dP 读取同一个 `sK` allocation。
- `sQT`、`sdOT`、`sK_2`、`sdST` 都是现有 storage 的 transpose/layout
  view，不分配转置副本。
- P 在 FP32 registers 中生成；同一 fragment downcast/store 到 `sP`
  后，FP32 P 继续用于计算 dS。
- `sP` 和 `sdS` 在各自 consumer release 后供下一 KV tile 覆盖。
- 所有 KV tile 消费结束后，`sK` 已死亡；dQ epilogue 才把同一 allocation
  重新解释为 `sdQ` staging。两者不同时存活。

## 9. TMEM 资源与生命周期

kernel 固定申请 512 TMEM columns。D512 路径的实际 placement 为：

```text
columns [  0, 64), lanes [ 0,16): S
columns [  0, 64), lanes [16,32): dP
columns [ 64,128): dKV panel 0
columns [128,192): dKV panel 1 / panel 3 lifetime alias
columns [192,256): dQ panel 0
columns [256,320): dQ panel 1
columns [320,384): dQ panel 2
columns [384,448): dQ panel 3
columns [448,512): dKV panel 2
```

S 和 dP 使用相同 column index，但源码给 dP pointer 增加
`lane_id = 16 << 16`；两者位于不相交的 TMEM lane half，可以同时
pending，不是先后覆盖同一 accumulator。

形式化生命周期：

```text
S:
  MMA write -> compute T2R/fence -> 下一 tile 才能覆盖 S lane-half

dP:
  MMA write -> compute T2R/fence -> 下一 tile 才能覆盖 dP lane-half

dKV panel 0/2:
  各自占有独立 64-column region
  -> dV overwrite -> dK accumulate -> reducer consume

dKV panel 1/3:
  共用 columns [128,192)
  -> panel 1 的 reducer T2R/fence 完成后，MMA 才能覆盖为 panel 3
  -> panel 3 的 reducer T2R/fence 完成后，下一 tile 才能写 panel 1
  -> atomicAdd 只依赖 register fragment，不继续持有 TMEM

dQ panel 0..3:
  first active KV tile overwrite
  -> subsequent tiles accumulate
  -> loop 结束后 T2R/R2S/TMA store
  -> epilogue 完成后释放
```

源码用 named barrier 闭合 dKV panel 1/3 的 alias handoff：MMA 在覆盖
该 region 前等待 Reduce 完成 T2R；不是以 “MMA 已 issue” 代替
accumulator 安全。

## 10. 同步与正确性不变量

- 所有同步均在单 CTA 内；baseline 不存在 CTA-pair barrier 或 remote
  SMEM transaction。
- Q/dO full 必须先于首个 S/dP；最后一个 tile 完成前不得覆盖。
- 一个 KV stage 同时服务 S、dP 和 dQ，三者都完成后才允许 loader
  覆盖该 stage。
- P 必须在 dV 前 ready；dS 必须在 dK 和 dQ 前 ready。
- dV 对每个 dKV panel overwrite，dK 对同一 panel accumulate。
- dKV panel 只有在 Reduce T2R 和 async-TMEM-load fence 后才能复用；
  atomicAdd 可以在 TMEM release 后继续。
- dQ 第一个实际 tile 使用 overwrite，后续 tile accumulate；全部 tile
  完成后才执行 epilogue。
- `sK -> sdQ` alias 只能发生在 KV loop 完整结束之后。
- dQ 由 head CTA 独占，不使用 atomic；dKV 必须写 FP32 workspace，再由
  独立 convert kernel 产生外部 dtype。
- GQA=128 的两个 head CTA 彼此没有归约或去重；它们对同一 dKV workspace
  地址的 H64 partial 都必须保留。
- Tensor Core accumulator、softmax/dS、dKV workspace accumulation 和
  dSink reduction 使用 FP32。

## 11. Baseline 的已知结构性成本

这些是由 ownership 直接推出的结构事实，不是未经 profile 的性能结论：

- 每个 token 的两个 H64 CTA 会重复 gather 同一 N64 KV tile。
- 两个 CTA 分别产生完整 D512 的 H64 dKV partial，并对相同 workspace
  元素各做一次 atomic contribution。
- 20 个 warps 中 W18-W19 没有计算职责。
- Q/dO 常驻使每 CTA 只加载一次这两个 H64xD512 tile，但也长期占用
  128 KiB 逻辑 SMEM payload。
- K pipeline 和 P/dS pipeline 都是单 stage；实际 stall 来源和是否值得
  加深必须由 GQA=128 的 profile 证明。

不能仅凭上述结构断言 2-CTA 一定更快。GQA=128 的 Tensor Core busy、
KV cache 命中、atomic contention、prologue/epilogue 占比以及最终 latency
仍需在同一环境中实测。

## 12. 作为 2-CTA 实现依据时必须保持的 baseline 合同

2-CTA 方案可以改变 CTA ownership、MMA group、load 复用和内部 pipeline，
但必须保持以下外部语义：

- sparse `topk_idxs` 是 global KV row index，支持 optional
  `topk_length` 和 `-1` padding；
- `K=V`、`H_kv=1`；
- 输出仍为 dQ、dKV 和 dSink；
- dQ 是所有有效 sparse KV contribution 的和；
- dKV 必须汇总全部 query token、全部 Q heads 和重复 index 的贡献；
- attention sink 必须参与 normalization，并由独立或数学等价的路径产生
  dSink；
- BF16 storage 路径上，Tensor Core accumulator 和 CUDA Core math 保持
  FP32；
- correctness 与性能都必须相对本文件固定的 1-CTA sparse baseline
  测量，而不是相对 dense prototype。

目标 2-CTA 的内部 shape、资源分配、FIFO、exchange 和 barrier 合同以
[`优化设计文档.md`](优化设计文档.md) 为唯一设计入口。

## 13. 仍待补齐的 baseline 证据

在比较 baseline 与 2-CTA v0 的正式性能结论前，还需单独完成并记录：

- GQA=128、D512 的 checked-in correctness case；
- compact / non-compact、partial tile、重复 index 和零 active tile 的
  明确覆盖；
- GQA=128、D512、目标 Top-K 集合的同机 benchmark；
- 编译后真实 `SharedStorage.size_in_bytes()`、register 和 occupancy；
- NCU 中 Tensor Core busy、KV load、pipeline wait 和 FP32 atomic 指标；
- 主 bwd、sum_OdO、convert 和 sum_dSink 各自的 latency 占比。

在这些数据产生前，本文只作为当前实现的形式化结构说明，不提供性能收益
结论。
