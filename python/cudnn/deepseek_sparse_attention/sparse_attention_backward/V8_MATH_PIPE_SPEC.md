# V8-LE：math T2R 软件流水（精简 spec，2026-08-05）

## 论题
math 串行链（~3µs×4 slice/bundle）= 当前深层 pacer；NCU+SASS 取证：math 窗
83% long_sb，F2FP 打包 46k 采样全在等 T2R 数据链（T2R 百余周期延迟裸奔）。
L-E = math 内环软件流水：slice s 计算/打包期间预发 slice s+1 的 T2R——
用 ILP 藏 T2R 延迟，目标 math 每 slice 3.1→~2µs。

## 约束
1. 解冻区仅 math 消费内环（v7 文件 13900-14440 一带的 sub_tile×slice_j 双层
   + T2R 分区块）；pds 协议/publish 目标布局/头解码公式/寄存器外的一切不动。
2. **寄存器红线**：math 现 128 reg（v11 注记 watch item）。T2R 双缓冲 +16 f32
   寄存器必须靠内环复用抵销；编译后用 cuobjdump --dump-resource-usage 核
   STACK 与 math spill 计数不得高于现状（现状 156/84 LDL/STL——若超，STOP）。
   注：本轮允许你编译核查（compile-only，cute.compile 即可，不跑 GPU 测试）。
3. 依赖正确性：slice s+1 的 T2R 需 s+1 的 s_done/dp_done 已 wait——预取不得
   越过管线 wait（j==0 的真 wait 在 pass 边界；同 pass 内 j0→j1 预取天然合法，
   跨 pass 预取需先完成下一 pass 的 wait——两档方案：A 仅 pass 内流水（安全，
   藏一半）；B 跨 pass 流水（重排 wait 位置，藏全部；wait 提前的协议影响要论证）。
   先做 A 测收益，B 作为后续。
4. 心跳 [le-z0] 起；py_compile 逐步；不 commit；终报 [le-final] 带寄存器/
   spill 对比表。
