# V7.1-LD：pds 发布存储 bank 冲突消除（精简 spec，2026-08-05）

## 论题
NCU（v7r1 全量计数器）：shared STORE 平均 7.3-way bank 冲突（11.3M 冲突 =
store wavefronts 的 18.3%），主要来自 math 的 P/dS 发布存储；NCU 估算修复
+8% 全核。math 串行链（~3µs×4 slice）现为 grads 相位深层 pacer——发布存储
在链上，冲突消除直接缩链。

## 约束（铁）
1. **slab 消费者側（grads/kdq 冻结机器）零触碰优先**：优先在**存储侧**解决
   ——重排 math 发布的 TV/tile 顺序（publish domain / J-mode store tiles /
   stmatrix 换挡位），使同 wavefront 内地址错 bank；slab 布局与消费者描述符
   不动。若存储侧无解（swizzle 数学上冲突不可避），STOP 汇报（改 slab 布局
   +消费者联动是 v7.2 级，需另批）。
2. 字节语义不变（同数据同址，只改写入顺序/原子挡位）——correctness 门终裁。
3. 逐区 py_compile；心跳进 V7_BUILD_LOG.md（[ld-z0] 起）；纪律 v7 全套。

## 侦察起点
- math 发布块：publish domain (16,2) 列箱、J-mode store tiles、v9_3 stmatrix
  （STSM）机器——先用 NCU source CSV 交叉定位哪条 store 指令族背 7.3-way
  （STSM vs STS vs F2FP 后的 STS.64/128），在日志记录证据行号再动刀。
- NCU 数据本地路径：/Users/longcheng/proxy/dsa-ncu-v7-1785872244/
  v7_full_source.csv（Address Space=Shared、Access Operation=Store 行的
  "L1 Conflicts Shared N-Way"列）。

## 验收
py_compile → 协调者 proxy 跑 correctness 4/4 + release（判决带：e2e 有可测
下降即胜，NCU 复测 store 冲突归零为终证）。
