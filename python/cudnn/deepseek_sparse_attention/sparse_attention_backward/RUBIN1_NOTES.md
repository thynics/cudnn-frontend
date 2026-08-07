# rubin_1 运行与验证备忘（dsa_bwd_sm100_2cta_rubin_1.py）

**日期**：2026-08-07 ｜ **基座**：`dsa_bwd_sm100_2cta_final.py`（vk 血统交付件，
SM100 9.441ms / 1.1368×）｜ **目标硬件**：Rubin（假定 328KB SMEM/SM，
单 CTA cap 记作 334,848B——落地时按真实工具链修正）

## 1. 三刀（对应 RUBIN_HANDOFF.md 的应收 B/C + loan 解耦）

| # | 手术 | 字节 | 打掉的账目 |
|---|---|---|---|
| 1 | **loan 退役**：dO_r0 h0/h1 改为环代 g2/g3（环 8→10 代/tile），kscore 回归单代（LOAD_K(t+1) 门 = dP(t) release） | 0 | kscore 串行链删一环（vk_6 实测该链 +0.94µs/tile 量级）；score_kv 生命期解耦（epi 门语义简化） |
| 2 | **环深 2→5**（相位合法集 {2,5,10}） | +49,152B | 供应墙：0.45µs/gen 信用平台 + dVdK gap Σ2.9（应收 C，−0.35~0.55ms 起） |
| 3 | **P/dS 双面**（tile 奇偶；pipe_pds 1→2 stage） | +24,576B（ds_xchg 退役出资 −4,096×2） | 锚点边：MATH_PDS_ACQ（ve_1 硅上实证 0.823→0.056；应收 B，−0.3~0.6ms） |

**面选择实现**：不做 2-tile 宏展开，改用 v7 判例（乒乓）——动态 if 包住两个
全静态臂，臂内只有纯拷贝/GEMM；PipelineState 一律在分支外推进。所有按面裸
mbar 的相位是算术式 `phase(t) = (t // PDS_FACES) % 2`，无状态翻转变量。

## 2. 双档 profile（import 时读 env）

| 开关 | 默认 | 构型 |
|---|---|---|
| `DSA_RUBIN1_B200_COMPAT` | `0` | **Rubin 档**：环 5、双面、cap 334,848B；SMEM ≈ 304,640B（≤327KB，余 ~30KB）。SM100 上无法 launch。 |
| `DSA_RUBIN1_B200_COMPAT=1` | — | **COMPAT 档**：环 2、单面、cap 232,448B；SMEM ≈ 226,816B。语义 = final − loan（dO_r0 走环），用于 B200 回归验证推广后的机器。 |

两个数字恰与 v2_1（v12 血统容量版，commit 1d9820e）的已验证账目重合，交叉校验。
类名保持 `FlashAttentionDSABackwardSm100TwoCTAV2`（harness 字面类名合同）。

## 3. 环代与槽位（10 代/tile）

```
gen:  g0   g1   g2      g3      g4      g5      g6      g7      g8      g9
内容: kdq0 kdq1 dO_r0h0 dO_r0h1 Q_r0h0  Q_r0h1  dO_r1h0 dO_r1h1 Q_r1h0  Q_r1h1
槽(R5): 0   1    2       3       4       0       1       2       3       4
槽(R2): 0   1    0       1       0       1       0       1       0       1
消费:  dQr0 dQr1 dVr0h0  dVr0h1  dKr0h0  dKr0h1  dVr1h0  dVr1h1  dKr1h0  dKr1h1
```
- kdq 恒占槽 0/1（两档同）；填充序 == 消费序（FIFO）。
- dO_r0 own 半区 bulk 源偏移 = loan 时代同款（stationary_do_raw +0 / +8192 元素），
  peer 半区 TMA = `t_dot_gmem[None, 0, {0,1}]` —— 字节恒等于被退役的 loan。
- W19 每槽一个具名相位变量（槽在 tile 内被踩次数不均，只有 per-slot 相位安全）。

## 4. 验证计划

1. **B200 COMPAT 回归（先行，走 ~/proxy 委托）**：`DSA_RUBIN1_B200_COMPAT=1`
   跑 smoke —— 正确性 4/4 硬门；release 预期 ≈ final + loan 退役税
   （~+0.4µs/tile ≈ +0.7ms 量级，回归门只看正确性不看性能）。
2. **零 GPU**：host 断言全绿（Rubin 档 storage assert 带实际值回显）+
   py_compile（已过）+ 本审计面板（三维对抗）。
3. **待 Rubin 工具链**：改 arch target 与真实 cap；跑 smoke + trace。
   **验收签名**：dVdK per-pass 间距 0.43→~0.15µs；MATH_PDS_ACQ→~0.06µs
   （ve_1 机制值）；RK_ACQ 头段 0.494→~0.09（kscore 单租户）；
   period 预期 ~5.0±0.3（Blackwell 系数口径；Rubin 核频/张量核另算）。
4. 环让位后立即成对复测 **M2 deg6 exp**（应收 A，代码在库 vk_4/v_w3_3，
   门 = W17 链 < 5.84µs）。

## 5. 已知未验证 / 防复议

- **Rubin ISA 前提**：tcgen05 CG2 语义、共享描述符 rank 同址规则、TMEM 512 列、
  DSM/cluster 语义均按 SM100 假设外推；描述符规则若变，死家族图谱须重判
  （RUBIN_HANDOFF §3）。arch 类属性仍为 100，待工具链落地改 target。
- FP8/f16 一律不碰（精度不降级铁律）；本版全部精度中性。
- 全常驻形态（256KB/CTA 面板→dkv-A 零拷贝视图）不在本版：需求 ~330-355KB
  贴线 328KB 不稳，且需要真实容量数字后再判（handoff §5 三选一的第一优先）。
- 与 v_s1（dO 切片流，SM100 无字节解耦线）正交：若 v_s1 落地为新 final，
  rubin_1 的 loan 退役部分与其重合，环深/双面手术直接叠其上（10 代相位同）。
