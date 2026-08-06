# K2（kdq 第二次稀疏 gather 退役）桌面审计（2026-08-06，基于 vk_2 代码精读）

结论先行：**外部复盘的 K2 按原文不可建**——其时序前提（"dQ-A own 半区 =
score_kv 双主序视图；现行释放序纸面可判"）被两条独立的代码事实证伪。
可建形态存在，但需要"dQ 反旋转"级的重设计，风险档从"中"升到"中高"，
EV 需扣除三笔新税。K1（vk_3）与 K4（M2）不受影响。

## 一、被证伪的前提：两条代码事实

**事实 1：kdq 镜像的消费相位晚一整窗。**
kdq(g) 的镜像在窗 g+1 内才填充/提交（W17 的 ROUTE_K 会合区，
[vk_2 L13928-13959]：持 2 个 ring 信用 → kdq_barrier 双握手 → gather 警组填
[N64×D128]×2 轮（[L11710-11800]，每行 256B D-四分片，两轮列偏移
256r+128·rank）→ 双 commit）。而 K(g) 的字节在窗 g 的 dP(g) 读完后就被
K(g+1) 重填覆盖（gather 软件流水刻意"先 score 后 kdq"，[L12793-12802] 注释
自证）。**别名/拷贝的源在目的地空出来之前一整窗就死了**——"同窗更早"不成立。

**事实 2：grads 窗口里 score_kv 躺的根本不是 K，是 loan。**
vc_2 的 dO_r0 loan 就是 kscore 管线的第二个 gen（[L12860, L12930]：
`pipe_kscore.producer_acquire → _fill_score_loan_do_r0_vc2 → commit`）。
score_kv 在 1-stage kscore 管线上按 K(t) / dO_r0 时分复用——dQ(t-1) 发射时
score_kv 持有的是 loan 镜像。双主序视图在当前调度下没有可指的字节。

## 二、可建形态：dQ 反旋转 + K-split 双 MMA

唯一闭合的结构（其余变体的死法见 §四）：

1. **dQ(t) 提前到窗 t 内发射**（dS(t) 发布 + DSM 落地后，dVdK(t-1) 之前或
   之后）；rotation 对 dQ 局部解开，对 S/dP 与 dVdK 不变。尾窗少一次跨窗
   依赖（dq_done 提前，epilogue 提早启动，附带小赚）。
2. **kscore 释放从 dP(t) 后移到 dQ(t) 读完后**（UMMA-tracked 加一个消费者）；
   loan 死区消失 → **loan 强制退役**，dO_r0×2 回 ring。
3. **K-split 双 MMA**：dQ 的 K-dim（n64）按 rank 半区拆两次 UMMA——
   own-n32：A = score_kv 的 MN-major 别名视图（合法性 = 探针
   `probe_score_kv_mn_major.py`，不闭合则退化为 gather 警组 S2S 拷贝进 ring
   槽，+2 gens）；peer-n32：A = DSM inbox（peer 在窗 t 中段推送，源活 ✓）。
   B = dSᵀ 的对应 n-半区——**ds_blocks/ds_image 本就按 rank 分块存在**
   （relay 直接送 ds_image+2048 即为证），B 侧零新建。
4. **dO_r0 的填充留在 gather 警组**（他们卸下了 GMEM 二次 gather，改做
   16KB panel S2S ~0.3µs，kdq_barrier 机器原样复用）——否则 dO_r0 火车挪给
   W17 会把省下的会合区全吃回去（W17 链 4.38×8/6≈5.8 + inbox ≈ 旧 period）。

## 三、修订后的账（对复盘 §5 的修正）

**收益侧**：W17 链 6.65 → ~4.7-5.0（会合区 2.27 → ~0.5-0.8：信用等待仍在
（K3 的靶），填充 1.5-2 → ~0.3）。period 期望 −1.3~−1.8µs/tile 上限。

**新税三笔（复盘未计）**：
1. loan 退役税 ~+0.4µs/tile（vh_1 实测）；
2. peer inbox 的信用尾巴：inbox 若走 ring（g8/g9，窗尾提交+消费）则 dQ 的
   操作数排进 2 槽信用链尾部，dQ 读完时刻推后；
3. **refill 挤压（最大的未闭合定量项）**：K(t+1) 重填起点从"dP(t) 后"推迟到
   "dQ(t) 读完后"（推迟 ~2-4µs），对 S/dP(t+1) 的旧口径余量只有 +1.20µs。
   新 period 下余量会变大，但这是耦合方程，纸面解不动——**必须按新等待图
   算保守界或小步探针**。

修订 EV：−0.25~−0.45ms（复盘报 −0.30~−0.55），风险中高。

## 四、已排除的低风险变体（死因备案）

| 变体 | 死因 |
|---|---|
| S2S 拷贝进 ring 槽（不动调度） | 源（K(t-1)）死于槽空出之前一整窗（事实 1） |
| 专用 kdq staging（不动调度） | 需 +16~32KB，SMEM 无此资金 |
| gather 直接产 ring gen（去会合） | vd_1 家族（双生产者拓扑等价 null/死锁史） |
| peer 部分和跨 CTA 归约 | 64KB f32 交换 >> 16KB bf16，负 EV |
| 保持调度只延长 kscore 生命期 | S/dP(t) 与 dQ(t-1) 需要不同 tile 的 K 在同一缓冲——矛盾，与释放时机无关 |

## 五、执行建议

1. **等 vk_3（K1）判决**——K1 与 K2 零耦合，先收尾段的钱；
2. **探针先行（零 GPU）**：`probe_score_kv_mn_major.py` 在容器跑一次，
   K2 的 own 半区走别名还是拷贝就定了（拷贝版少一个未知数，多 +2 gens）；
3. K2 重设计的等待图（三路径：稳态/首 tile/尾 tile + refill 保守界）值不值得
   做，建议放在 K1 落账、且 K4（M2 终裁）明朗之后再定——若 K4 放行，
   M2 是同量级收益（−0.15~0.30ms）且改动只有 math 一处，性价比先于 K2。
