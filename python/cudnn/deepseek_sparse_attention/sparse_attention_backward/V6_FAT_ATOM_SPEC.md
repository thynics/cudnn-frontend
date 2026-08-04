# V6 肥原子方案（精简 spec，2026-08-04）

## 论题（一句话）

生产粒度与消费粒度解耦：score 面生产原子 h32→**h64**（摊薄 ~180ns/条的
enqueue 固定税，r11 定罪 v5 释放版 ~85% 被 176 条小原子的 TC 时间钉死），
消费侧（math T2R 切片、grads chunk-major、驱逐/drain 机器）保持 v5 已验证
形态。基座 = v5 终态（bd22106，correctness 4/4），文件 fork 为
dsa_bwd_sm100_2cta_v6.py，类名保持 FlashAttentionDSABackwardSm100TwoCTAV2
（harness 槽位契约）。

## 预算（先算死再动刀）

- enqueue/bundle：score 128→64（8 D64 pieces × 2 k_blocks × 2 半 × G1G2）
  + grads ~32 + evict 16 ≈ **112 vs v5 176**；预期 release 12-14µs/64kv 档
  （原子价曲线微基准并行验证中，若固定税假说破产则本方案中止）。
- TMEM：S 2-stage h64 = 2×32 + dP 2×32 = **128** | dkv 4×64 = 256 |
  dQ 2×16 = 32 → **416 ≤ 512 ✓**（断言带回显）。dkv 槽 mod-4 代数不动。
- SMEM：strip/K 驻留/P/dS 槽字节量全部不变（h 粒度变化不改任何缓冲尺寸，
  只改索引算术）——若实现中发现任何缓冲需要扩，STOP 汇报。

## 手术区（按依赖序）

1. **常量层**：SUB_TILES 4→2；SUB_TILE_BOX 16→32（若该常量语义为
   每 sub-tile 列宽的 TMEM 编码半宽——建造者核实语义后改，不确定即 STOP）；
   S/dP TMEM 槽宽 16→32 列，映射表与 TMEM_BUDGET 断言同步。
2. **score 发射环**（S_ISSUE/dP_ISSUE 所在 helper）：N=64 原子；
   B 窗索引 (sub_tile%2)*8+piece → sub_tile*8+piece（strip 2-stage 的
   stage=t%2 语义在 t∈{0,1} 下自然保形——每 h64 子块恰一 strip stage）；
   s/dp done 管线 4 commit/bundle → 2。
3. **math 消费环**：外层 stage 循环 4→2；stage 内 T2R/softmax/pds 按
   h32 切片×2 内层化（寄存器预算不变——消费粒度不动是本方案的核心纪律；
   pds→SMEM 存储的 chunk-major 目标布局不变）。
4. **leader 轮转**：score(0); {score(1); grads-batch(0)}; grads-batch(1); G5。
   grads 面若现状已按 h64 chunk 消费（P/dS 槽本就 [kv64×h64] chunk-major），
   则 G3/G4/G5 与驱逐机器**零触碰**——建造者核实后在日志记录判定结果。
5. **trace 合同**：span 名不变（29 上限不动）；S_ISSUE/dP_ISSUE payload
   b*4+t → b*2+t；MAT_ACQ gen 序号表随 strip 消费节奏核对（数目不变则不动）。

## 冻结区（IDENTICAL 要求，违者 STOP）

dQ 驱逐/offload 四变体、融合 drain、K 驻留+gather、kdq gen 机器、relay、
chase loader、dkv 槽算术、workspace 开凿。

## 纪律（v5 战役全套继承）

- 忠于方案；做不通 STOP 汇报，禁止静默降级/伪串行化；
- 每 30s 心跳追加 V6_BUILD_LOG.md（[z0] 起编号，格式沿 V5_BUILD_LOG）；
- 每完成一个手术区 py_compile；全部完成后总 diff 统计 + 协议账
  （done 管线 产=耗 核对）+ 自知风险清单。

## 验收门

1. py_compile；2. correctness 4/4（bit 级等价不可期待——S 累加序随原子
形状变化，但 f32 全程累加 + 单次终舍入的精度等级不变，容差门判）；
3. release e2e 对 v5 的 35.653ms 显著下降（预言带 20-25ms）；
4. trace：S_ISSUE 逐条价（预期 ~0.22µs×31→高 N 下条数减半）、leader 大停摆、
   TC 占用率重估。

## 修正案 #1（2026-08-04，用户批准 R-B）

[v6-STOP] 裁决：采纳 R-B。strip 重装箱 [h16×D512]→[h32×D256]（同 16,384B/箱、
同 4 gen/bundle，字节恒等式 s*2048=(s//4)*8192+(s%4)*2048，描述符 canonical），
**W17 strip gen 区解除冻结、列为手术区 1.5**（约 30 行重写：gen g=(t6=g//2,
dhalf=g%2)，gmem 窗 = 头 [t6*64+r*32,+32) × D[dhalf*256,+256)，h32-tile 索引
2*(g//2)+rank）。score-B = auto make_smem_layout_b(N64,(128,64,64),8 stage) 直接
绑定；B 窗索引合同以 R-B 为准（piece 8 宽末模），spec 原 16 宽字面式作废。
pass 内早释放 stage0（piece3 后）保 W17 预填重叠；strip 管线 4 产=4 耗。
手术序：zone1（常量+SUB_TILE_BOX 逐点+头解码乘数换钩）→ zone1.5（strip 重装箱）
→ zone2（score 环）→ zone3（math）→ zone4/5。zone1 的 [z5] 处置一并批准。
注：haifa 已弃用——建造者无需任何探针轮；测试统一由协调者经 ~/proxy 委托。
