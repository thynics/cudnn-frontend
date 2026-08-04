# V9-LB：D128 K 向肥化（spec 草案，待用户裁决后定稿——涉最高危冻结区）

## 论题
leader 的每条 enqueue 在共驻活跃期恒付 ~100ns（物理载体未定罪但现象铁实），
score 面 64 条/bundle 是最大穿越集。D64→D128 piece 使 K 深翻倍：score 面
64→32 条、G1 原子 (128,64,64)→(128,64,128)（微基准锚定：单价仅 +~13ns 而
工作翻倍）。预估 release −3~−4µs/bundle ⇒ e2e 21.94→~18-19ms。

## 危险声明（为何需正规战役）
K 驻留 SMEM 布局（8×D64 piece 阶段）与 gather/chase 填充机器是**全文件最老的
冻结区**（v1 血统，历经 v5/v6/v7 全程零触碰）。D128 重装箱触及：
- score_a_layout（8 stage → 4×D128 stage）+ 字节恒等式重推；
- gather/chase 的 piece 填充窗与 LOAD_K TMA box（D64→D128 box 合法性）；
- kdq gen（K 数据的 G5 消费者）的 piece 索引（dq MMA 的 A 窗）；
- strip 侧 [h32×D256] 与 D128 piece 的对齐（天然 2:1，预判无扰）。
任何一处字节序错 = correctness 全灭。建造者必须先做 v6-R-B 级的字节恒等式
纸面推导，恒等式不成立即 STOP。

## 预算预检（纸面已过）
SMEM 字节量零变（重装箱）；TMEM 零变；发射账 score 64→32、grads/evict 不变
= 104/bundle；strip 协议 4=4 不变；MAT_ACQ 序号表 wide gen 16→8 需重编
（span 名不动）。

## 待办（定稿前）
1. 用户裁决 GO/NO-GO（高危区解冻授权）；
2. 建造者侦察轮：K 驻留布局的 D128 字节恒等式推导 + gather 填充窗代数，
   恒等式闭合才开修改权。
