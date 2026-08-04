# V7 dQ 驻留回归方案（精简 spec，2026-08-05）

## 论题（一句话）

v6r1b/r2 双 trace + baseline 对账定罪：周期被 reduce 车道 89% 的串行服务钉死，
其中 dQ offload（30µs/bundle 税后）是 baseline 完全不付的结构性开销（v5.2 驱逐
把 dQ 流量放大 16×：294KB/bundle RMW vs 驻留的 18KB 一次性写）。v7 = 驱逐路线
整体回滚：dQ 驻留 TMEM 至 token 块结束 + TMA epilogue 一次写出（v12 血统、
机器在 v5.2 中休眠未删），offload/驱逐机器整体退役，reduce 车道只剩纯 drain
（已实测与 baseline 原子服务同价 15.6µs/kv128）。基座 = v6 终态（a3dd79c，
correctness 4/4），fork 为 dsa_bwd_sm100_2cta_v7.py，类名保持 TwoCTAV2。

## TMEM 账（恰好闭合，断言带回显）

S 2×32=64 | dP 2×32=64 | dkv 2×64=**128**（环 4→2）| dQ 驻留 **256** = **512 整**。
dkv 槽代数 mod-4 → mod-2：slot = p%2（pair 不变性更简）；MMA_DONE_STAGES 4→2；
dkv_done mbar 8→4。dQ 驻留区 [256,512)，kdq MMA 直接以 ACCUMULATE=True 跨
bundle 累加（首 bundle False 起）。

## 手术区（依赖序）

1. **驱逐/offload 退役**：pipe_dq_evict、_issue_dq_block_v52、_offload_dq_block_v52、
   dq_b 双像、DQ_EVICT_* 常量、workspace dq 尾巴依赖（_carve_dq_acc 调用点回
   None 路径；接口分配扩容留着无害）。DQ_EPI span 名保留改回 TMA epilogue 语义
   （v12 口径），payload 沿旧表。
2. **dQ 驻留复活**：t_dq 静态 256 列 @ [256,512)；kdq gen 对（冻结机器不动）喂
   持久累加 MMA（每 bundle 16 块 → 4 条 kdq evict MMA 改 4 条驻留累加 MMA，
   发射点沿 G5 位置）；tile 末尾 _store_dq_from_tmem + dq epi TMA（休眠机器
   复活，参数表已在）；_zero_dq_v2 路径不动。
3. **dkv 环 4→2**：t_dkv_army 4 元组→2 元组视图、槽选择 p%2、初始信用 2、
   融合 drain 调用点/签名同步。活性论证：drain 对服务实测 3.9µs（offload 退役
   后预期 ≤3µs），grads 对发射需求 1.9µs/对——leader 会被环节流，但节流窗与
   S/math 全重叠，账面 pacer = drain 纯原子流（≈baseline 水位）。若实测
   drain>5µs/对成为新 pacer，STOP 汇报（环 3 需 dQ 让 64 列，为后备案）。
4. **协议账**：dkv_done 16 产=16 耗（环 2 相位）；round 20=20 不动（kdq gen
   双持双放沿旧）；strip/s/dp/pds/kres 全部 v6 恒等。
5. **trace**：span 名 29/29 不变；DQ_EPI 语义回 epilogue（每 tile 一次）；
   MAT_ACQ kdq 序号 20-23 不动。

## 冻结区（IDENTICAL）

融合 drain 本体、K 驻留+gather、kdq gen、relay、chase、strip（v6 形态）、
score 环（v6 形态）、math 消费环（v6 形态）。

## 纪律

v6 全套（心跳 ≤2min 进 V7_BUILD_LOG.md、逐区 py_compile、做不通 STOP、
禁伪串行化；测试由协调者 proxy 委托）。

## 验收门

1. py_compile；2. correctness 4/4（dQ 数值 = f32 TMEM 累加 + 单次终舍入，
精度等级与 v12 同类）；3. release e2e：判决带 **12-18ms**（reduce 车道卸下
30µs/bundle 后 pacer 回落）；4. trace 钩子：reduce 车道占用（89%→~50%）、
bundle 末大停摆（36µs→<8µs）、grads 块间 gap（环 2 节流形态）、S(1) 长度
（共驻衰减是否随 reduce 静默而消失——案 1 的自然实验判决）。
