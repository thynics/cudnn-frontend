# DSA all-in-one validation round

一条命令跑完一轮验证的全部固定环节，产物形状每轮一致。四轮 v15 STOP 的
往返延迟主要来自管线之外的手工衔接（stage-0 门每轮重写分析器、readout 每轮
重算、打包协议手工执行）——本目录把它们全部固化。

## 一条命令

```bash
DSA_STAGE0_CMD="<你的编译 helper 入口>" \
DSA_NCU_CMD="<可选 ncu 包装>" \
./benchmark/dsa/allinone/run_allinone.sh --impl v15 \
    --reference-capture /path/to/v12_capture \
    --gates benchmark/dsa/allinone/v15_gates.json
```

序列（任何一步 STOP 即写 `<artifact>.FAILED` 并保留 partial 供取证）：

1. **preflight**：impl 注册/无未提交改动/py_compile/记录 `DSA_V*` bisect 旗标；
2. **stage0**：`DSA_STAGE0_CMD <impl> <out>` 产出编译捕获
   （CUTE_DSL_KEEP=ptx,cubin,sass），`stage0_analyzer.py` 按
   `--gates` JSON 施加 G0-G4——**在占用 B200 服务前拦截**；
3. **pipeline**：仓库既有一键 `run_b200_pipeline.sh --impl vX`
   （correctness / release 双方 / 双 IKET / span 表——见
   `skills/validate-dsa-b200`，本脚本不替代它的任何职责）；
4. **ncu**（可选钩子）；
5. **readout**：`round_readout.py` 固化标准分析——逐 warp 原始层
   per-name 统计、双窗口 all-spans CSV（默认 1-3 与 14-17）、稳态指标
   （period/W17 链/dVdK 节拍/publish 尾）、**每杠杆 fired 指纹**
   （MATH_PDS_ACQ 消失、W18_PDS 出现、ROUTE_P/dS 消失、PROBE 流存在性）、
   perf 摘要；
6. **publish**：`.partial` → `MANIFEST.sha256` → 原子改名。

## 产物清单（每轮同构）

| 文件 | 来源 |
|---|---|
| `bisect_flags.txt` | preflight |
| `stage0_gate_report.md`（+ 失败时 `stage0_capture/`） | stage0 |
| `pipeline.log` + 管线自带表格/验证文件 | pipeline |
| ncu 报告（钩子提供） | ncu |
| `candidate_iket.decoded_results.json` | readout 输入固化 |
| `i{lo}_i{hi}_all_spans.csv` × N、`span_means_full_run.json`、`readout.md` | readout |
| `MANIFEST.sha256` | publish |

## 钩子约定（私有件不进仓库，只留接口）

- `DSA_STAGE0_CMD <impl> <capture-out-dir>`：编译捕获（私有 compile helper）。
  未设置且未给 `--stage0-capture` 时该步 SKIPPED（明示，不静默）。
- `DSA_PIPELINE_CMD`：默认仓库一键脚本；可换 harness 直入口。
- `DSA_NCU_CMD <impl> <out-dir>`：ncu 伴随。
- `DSA_DECODED_GLOB` / `DSA_RESULTS_ROOT`：decoded 结果定位。

## stage0_analyzer.py

源自 v15 rev3 STOP 轮 runner 的一次性分析器（角色窗口归因 / USETMAXREG /
LDL/STL 差分 / G2S 克隆归因逻辑原样保留），加了 `--expectations` 配置层：
JSON 键 = 模块常量名（见 `v15_gates.json`）。每个新 rev 只改 JSON 的
`TARGET_REVISION` / `TARGET_SOURCE_SHA256` 两行，不再重写分析器。

## 注意

- 本目录不试图复刻私有 harness 的任何职责；`skills/validate-dsa-b200`
  的约束（不得绕过一键管线跑正式验证）继续有效；
- readout 的窗口/稳态区间可参数化（`--windows`、`--steady-window`）；
- `DSA_V15_L2X=0` 的 bisect 组合按 `V15_RUNNER_NOTES.md`：仅
  release/correctness（31 IKET 名超 29 上限，trace 会失败）。
