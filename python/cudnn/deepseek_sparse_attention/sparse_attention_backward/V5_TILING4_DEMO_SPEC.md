# v5 —— tiling4 overlap 演示版建造规格（2026-08-04 用户裁决）

**使命（用户原话口径）**：不关注 drain 代价与 e2e 性能，只回答一个问题——**head-outer
tiling4 的 S(t)→M(t)→G(t) 三级子块流水能否真的 overlap 起来**。交付物 = 硬件 trace。
correctness 4/4 仍是硬门（数学等价，f32 换序合法）。

基座：`dsa_bwd_sm100_2cta_v5.py` = v32 rev5（commit 2a638ea，未旋转形态）逐字节克隆。
类名合同不变：`FlashAttentionDSABackwardSm100TwoCTAV2`。IKET 合同不变（canonical 三名 +
≤29 静态名，不加新名）。

## 结构变换（8 个区，其余全部冻结）

### Z1 常量层
- `SUB_TILES = 4`，子块宽 h32（簇 N=32，B N 半 h16/CTA，MMA N32 合法档）。
- TMEM 重排：dQ [0,256) 不变；S pp [256,288)=2 stage×16 列；dP pp [288,320)=2×16；
  dV [320,384)、dK [384,448) 不变形态（64 列块）；[448,512) 空闲。
  合计 448/512。子块累加器 [kv128×h32] 簇 = 16 列/CTA（M128 fold N/2）。

### Z2 gather：chase 重流 ×4
- 每 bundle 填 32 个 piece（4 pass × 8）：piece_global = t*8 + p，D-slice 索引 = p，
  同一批 kv 行每 pass 重 gather（pass≥1 时 L2 热）。2 槽环、信用协议、r1/r0 kdq
  握手位置全部不变（只有循环计数 8→32 与 D-slice 取 p%8…按实际代码形态改）。

### Z3 leader score：head-outer
- 每 bundle：for t in 0..3：acquire s_done/dp_done（2-stage 管线，节奏 4 commit/bundle，
  自然形成 t/t+1 在飞）→ 8 piece × 2 gemm（S_t、dP_t，K-outer within sub-tile，
  ACCUMULATE=False 于 piece 0）→ commit s/dp(t)。
- B fragment = 面板 h16 窗口（面板行 D 连续、h 步长 = D512×2B=1KB，h16 窗偏移
  16KB 对齐，swizzle 原子安全；面板本体不动）。
- **子块内旋转（overlap 的主结构）**：leader 序 = score(0); for t in 1..3:
  { score(t); grads(t-1) }; grads(3); G5r0+G5r1（bundle 级不变）; TAIL。
  即 math(t-1) 藏在 score(t) 的发射窗下，grads(t-1) 紧随其后。
  相位安全论证：score(t) 的 s/dp commit 在自己 8 piece 后立即发（子块级，无 zip 相位
  塌陷问题——教训 #16 只适用于桶级 commit 后移）。

### Z4 math：4 子块
- chunk 循环 2(h64)→4(h32)，T2R/逐元素/列 stats 逻辑不变（窗口减半）。
- 发布 per 子块：P slab 子像(t) [kv64×h32]（slab 总 16KB 不变，2 子像→4 子像）；
  dS slab 同；dq_b own 半的 h32 切片写入（同一 dq_b 像内按 t 偏移，v32 的 h64 own 半
  本就是 2×h32 box 写，改 4×…实际是切片粒度变化）。
- **pds 交棒改子块级**：pds 管线 num_stages 1→2，节奏 4/bundle（math 每子块
  producer_commit，leader grads(t) consumer_wait/release）。武装计数遵守教训 #10
  （簇管线 consumer ×atom_thr_size(2)）与 #11（裸门三件套）。
- dq_b 的 relay peer 推送保持 **bundle 级**（等 4 子块 own 半全落，8KB 一次推，
  v32 原样）——G5 路径零改动的前提。

### Z5 grads：t-major 子块级 + 部分和 drain（代价已获批）
- grads(t) = 4 D-round × (dV, dK)：每块 accumulate=False 起（子块内新鲜），
  B = 该子块的半宽 gen（Z6），A = P/dS slab 子像(t)。
- 块在子块内单链完工 → **立即按 v32 既有 per-block drain 机器 drain**（f32 原子，
  部分和跨子块在 GMEM 累加——与跨 token 累加同类同语义，正确性无虞）。
  drain 次数 8→32/bundle，dkv_done 节奏 ×4。**性能不设指标。**

### Z6 W17 供应：QDO gen 半宽 ×4
- gen 重切为 per-(t, r)：[D128×h32] 簇（8KB），32 gen/bundle（4t×4r×2 张量…按
  dO/Q 交替的实际编排改）。round ring 字节不变（2×16KB 各装 2 个半宽 gen 或
  维持 16KB gen 内含双 h32 box 由 grads 开窗——**取最小 diff 的形态，建造者自决**，
  供应成本明确不设指标）。kdq 轮不动。
- 延迟提交格/相位律照 v32 原样按新 gen 数重推（gens mod 2 == 0 律保持）。

### Z7 drain/reduce warps
- 循环计数 ×4，块坐标解码不变（同块形态）。WAIT_dK/REDUCE span payload 打包加 t。

### Z8 IKET
- S_ISSUE(i)/dP_ISSUE(i) payload = bundle*4 + t（名字不变，canonical 合同完好）；
  dVdK_ISSUE(i,r,p) payload = ((bundle*4+t)*4+r)*2+p；MAT_QDO/MAT_ACQ payload 随
  gen 数扩位。静态名净增 0（28≤29 维持）。

## 冻结区（一字不动）
G5 双波 + dq_b DSM relay + dq_done/TAIL、chase 环深、面板装载、stats 装载、
dQ epilogue、convert_canonical、全部 kdq 机器。

## 判读合同（trace 侧）
overlap 成立 ⇔ 稳态 bundle 内可见：S_ISSUE(4i+t+1) 与 math chunk(4i+t) 时间重叠、
grads(4i+t) 的 dVdK 发射与 math(4i+t+1) 重叠、REDUCE 泳道在子块间连续。
对照物 = v32 rev5 trace（S 整桶 → math 整桶 → grads 整桶的串行三段）。

## 建造纪律（v32 硬化协议原样）
心跳 `V5_BUILD_LOG.md` 每区一行；单编辑 ≤250 行；每区完成 py_compile；禁全量读；
DSL 教训 #1-#16 全表过（HANDOFF_20260803.md §1/§3 + V32_BUILD_LOG.md [13]/[14]）；
staged 修改整块重审；struct 断言上界式+回显。
