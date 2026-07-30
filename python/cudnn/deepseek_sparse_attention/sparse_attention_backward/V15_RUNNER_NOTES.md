# v15 runner 注记（随提交发布）

v15 = 性能候选（v12 基础 + 三主杠杆 + 一 rider，全部 env 可 bisect）。
产物目录协议照旧：`/home/longcheng/artifact/v15_run1(.partial→mv 原子发布)`，
失败写 `v15_run1.FAILED`。

## 杠杆与 bisect 开关（import 时读 env）

| 开关 | 默认 | 含义 |
|---|---|---|
| `DSA_V15_L2X` | 1 | L2-staged exchange：P/dS 经 8KB pds_stage → HBM 环 → W18 五发 G2S 回填；DSM 发送/count-128 握手/math 槽位停车全拆 |
| `DSA_V15_REGSWAP` | **1**（run1 G2 后从 2 改默认） | 0=v12 分布；1=变体 B（W16-19 统一 64，reduce 120）；2=变体 C（W16-19 统一 56，gather 40——已知 56 下 W17 的 ALLTMA 循环有 2 个 STL 轮转位点）。全部池精确 61,440 |
| `DSA_V15_DQ_MERGE` | 1 | dQ 两 round 并一协议块（kdq 信用本就成对提交，无格死锁） |
| `DSA_V15_ALLTMA` | 1 | own-half DSM bulk 退役，走纯 TMA 回退路径（run2 实测 DSM 腿慢 1.7×） |

## 【必读】harness 适配：workspace_pds（仅 L2X=1 需要）

v15 的 `__call__` 末尾新增**尾参**（带默认值，老调用点不传也能编译）：
`workspace_pds: Optional[cute.Tensor] = None`。

L2X=1 时必须分配并传入（缺失会在编译期以明确 assert 失败，不会静默出错）：

```python
ws_pds_shape = ImplClass._get_workspace_size_pds(total_S_q)   # (total_S_q, 65536)
workspace_pds = torch.zeros(*ws_pds_shape, dtype=torch.uint8, device=device)
workspace_pds_tensor = to_cute_tensor(workspace_pds)
# cute.compile(...) 与运行调用的参数表末尾各追加 workspace_pds(_tensor)
```

256MiB 分配、热集 ~4.9MB（74 并发 cluster × 64KB），无需 L2 carve-out。
不想改 harness 时：`DSA_V15_L2X=0` 可直接跑（release/correctness）。

## IKET / 比较器

- L2X=1 trace：**28 名**（23 range + 4 ROLE + provenance）。退役：MATH_PDS_ACQ、
  MATH_BAR1、ROUTE_P、ROUTE_dS；新增：W18_PDS(i)。MATH_STORE 语义变更：现为
  双波发布全程（stmatrix×2 + S2G×2 + gen_ready arrives）。
- `DSA_V15_L2X=0` 的 bisect build 是 v12 名字集（31 名）——**超 29 上限，
  只跑 release/correctness，不要跑 trace**。
- 比较器按 vm5probe run2 先例跳过/适配（span 语义图变更同前）。

## Stage-0 SASS 门（编译产物，B200 run 之前）

- G0: SMEM ≤ 232,448（struct assert 编译期兜底）。
- G1: USETMAXREG 按变体核对（默认 B: gather 48 / W16-19 64 / math 128 / reduce 120）。
- G2: W16/W17/W18 分支零新增 STL/LDL（nvdisasm 分支归因，vm5probe 先例）；
  记录 leader 区 spill 数 vs v12（regswap 的收益指纹）。
- G3: math 分支 stmatrix m8n8.x4.trans 仍在、零 STS.U16 回退（build assert 兜底）；
  每 tile 恰好 2 条 cp.async.bulk S2G + W18 5 条 G2S。
- G4（默认 B 生效）: reduce 120 的 drain spill ≤ 23（v11 实测残余基线）。

## 运行序列

1. correctness 4/4（两个 parity）；
2. release 三方对时 v15/v12/baseline → perf JSON；
3. trace（tiles 1-3 + 14-17 双窗口 raw 提取）；
4. ncu 伴随（未插桩 v15 + baseline，同 10 metric）。可证伪预测：
   v15 tensor 占空 28-33%（period 若如模型收缩）；L2 读命中 ≥93%；
   dram__bytes.sum ≤ v12 的 1.64GB +10%。
5. 若回归：按开关做 one-off 矩阵（先关 L2X，再 REGSWAP=0，再 DQ_MERGE=0，
   再 ALLTMA=0），每次单关一个。

## 判读指纹（每杠杆的"fired"证据）

- L2X：MATH_PDS_ACQ 消失、W18_PDS 出现（预期 ~0.6-0.9µs/tile）、
  publish 尾（store_end→首 dVdK）大幅缩短；
- REGSWAP：leader 区 SASS spill 下降 + dVdK_ISSUE 间隙缩短；
- DQ_MERGE：WAIT_dQ(r1) 提前到 r0 issue 之前；
- ALLTMA：MAT_QDO 内 own-half 腿的 p90 尾收敛到 TMA 腿水平。
