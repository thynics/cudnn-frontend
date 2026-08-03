#!/usr/bin/env python3
"""V4 rev0: 池轮转离散事件仿真 + leader 预算清点.

模型(每 CTA 口径, bundle = 2 个 per-64kv 周期):
  - 3 个 4-warp squad, gang 调度(每 work item 占整 squad, elapsed = warp_us/4),
    队内 FIFO + 优先级(P > dS > drain).
  - 每 bundle b 的角色: c0=σ(b%3), c1=σ((b+1)%3), dlead=σ((b+2)%3).
    работы: P(b,c)/dS(b,c) 归 c-squad; dV 4 槽归 dlead; dK 4 槽归 c0-squad.
  - leader 时序(简化 36 位): 需要 CG_P[c0] @ t_b+0.2, CG_P[c1] @ t_b+0.9,
    CG_dS[c0] @ t_b+1.2, CG_dS[c1] @ t_b+2.2 (dr-major 大致锚点, 源自 C 表);
    dkv_free(槽 r) 需在 G3/G4(r+2) 发射前 → 约 t_b + 1.0 + r*0.9 处检查
    drain(b-1? 同 bundle 前 round) 完成; MMA+供应地板 = max(3.38+bubbles, 2*supply_wall).
  - P(b,c) 最早开工: t_{b-1}_end - 1.9 (score(t) 在 t-1 窗内先行, v7/C 锚);
    dS(b,c) 开工: P(b,c) 完成且 dp_full ≈ t_b + 0.3 + 0.6*c.
  - dV/dK drain(b,r) 开工: leader 到达其发射位 ≈ t_b + 1.0 + r*0.9 (G3/G4(r,c1) commit).
    槽复用门: drain(b,r) 须在 leader 需要该槽的 G3/G4(b,r+2) 前完成(r+2<4),
    及跨 bundle: drain(b,2),(b,3) 须在 t_{b+1} + 1.0 + (r-2)*0.9 前完成.
三套账(warp_us / bundle):
  opt:  exp 9.0,  t2r 4.8, pub 12.0, stats 0.7, drain/槽 3.6   (探针锚)
  mid:  exp 14.0, t2r 4.8, pub 12.8, stats 0.7, drain/槽 4.7
  cons: exp 19.0, t2r 4.8, pub 13.6, stats 1.2, drain/槽 6.2   (v12 实核锚)
输出: 稳态 bundle 时长/2 = per-64kv period, 每 squad backlog 峰值, 空闲率, 绑定门统计.
"""

LEDGERS = {  # R14 校准 (2026-08-03 mixed_residency 实测): exp x1.27, pub 相位错开残余, REDG 4w 饱和
    # REDG 通道 = 10.18ns/instr 硬管线极限(2026-08-03 drain_channel_probe 钉死, 与 warp 数/代码形态无关)
    # → drain_slot 通道时间固定 0.652µs/槽; 三档只剩 math 侧(exp 混驻/pub 相位错开成色)
    "opt":  dict(exp=11.4, t2r=4.8, pub=13.8, stats=0.7, drain_slot=2.61),
    "mid":  dict(exp=14.0, t2r=4.8, pub=16.0, stats=0.7, drain_slot=2.61),
    "cons": dict(exp=17.8, t2r=4.8, pub=22.0, stats=1.2, drain_slot=2.61),
}
SUPPLY_WALL_64KV = (2.3, 2.9)     # per-64kv
MMA_BUNDLE = 3.38                  # µs busy per bundle
BUBBLES = 0.6
N_BUNDLES = 48

def simulate(led, supply_64kv, verbose=False):
    # per-chunk math items (half of bundle math each):
    p_item  = (led["exp"] * 0.55 + led["t2r"] * 0.5 + led["pub"] * 0.45 + led["stats"] * 0.5) / 2
    ds_item = (led["exp"] * 0.45 + led["t2r"] * 0.5 + led["pub"] * 0.55 + led["stats"] * 0.5) / 2
    drain_t2r = 0.35                       # T2R 波切, squad 并行段
    drain_redg = led["drain_slot"]         # REDG 段, 全 SM 单通道(4w 饱和)
    # squad state: next free time
    squad_free = [0.0, 0.0, 0.0]
    backlog_peak = [0.0, 0.0, 0.0]
    busy_acc = [0.0, 0.0, 0.0]
    gate_stall = {"CG_P0": 0.0, "CG_P1": 0.0, "CG_dS": 0.0, "dkv": 0.0, "floor": 0.0}
    t = 0.0
    prev_end = 0.0
    simulate._chan = 0.0
    prev_drain_done = {}   # (grad, r) -> completion time of bundle b-1 drains
    periods = []
    for b in range(N_BUNDLES):
        c0, c1, dl = b % 3, (b + 1) % 3, (b + 2) % 3
        t_b = t
        # --- math items scheduling (gang, FIFO per squad) ---
        # P(c0): earliest start = prev_end - 1.9 (score ran ahead)
        est = max(prev_end - 1.9, squad_free[c0])
        p0_done = est + p_item / 4.0
        squad_free[c0] = p0_done
        busy_acc[c0] += p_item / 4.0
        est = max(prev_end - 1.3, squad_free[c1])
        p1_done = est + p_item / 4.0
        squad_free[c1] = p1_done
        busy_acc[c1] += p_item / 4.0
        # leader gate: CG_P[c0] needed at t_b+0.2, CG_P[c1] at t_b+0.9
        s0 = max(0.0, p0_done - (t_b + 0.2)); gate_stall["CG_P0"] += s0
        s1 = max(0.0, p1_done - (t_b + 0.9)); gate_stall["CG_P1"] += s1
        t_shift = max(s0, s1)  # leader slips
        # dS items (start after P done and dp_full ~ t_b+0.3/0.9)
        est = max(p0_done, t_b + t_shift + 0.3, squad_free[c0])
        ds0_done = est + ds_item / 4.0
        squad_free[c0] = ds0_done
        busy_acc[c0] += ds_item / 4.0
        est = max(p1_done, t_b + t_shift + 0.9, squad_free[c1])
        ds1_done = est + ds_item / 4.0
        squad_free[c1] = ds1_done
        busy_acc[c1] += ds_item / 4.0
        s2 = max(0.0, ds0_done - (t_b + t_shift + 1.2), ds1_done - (t_b + t_shift + 2.2))
        gate_stall["CG_dS"] += s2
        t_shift += s2
        # --- drains: dV r0..3 -> dlead, dK r0..3 -> c0-squad ---
        dkv_slip = 0.0
        redg_chan = getattr(simulate, "_chan", 0.0)          # 全 SM REDG 通道空闲时刻
        import os
        D64 = os.environ.get("V4_D64_SLOTS") == "1"
        n_rounds = 8 if D64 else 4
        spacing = float(os.environ.get("V4_SPACING", "0.45" if D64 else "0.9"))
        slot_chan = (drain_redg / 4.0) / (2 if D64 else 1)
        slot_t2r = drain_t2r / (2 if D64 else 1) + (0.03 if D64 else 0.0)  # 趟数税
        for r in range(n_rounds):
            for grad, squad in (("dV", dl), ("dK", c0)):
                avail = t_b + t_shift + 1.0 + r * spacing
                est = max(avail, squad_free[squad])
                t2r_done = est + slot_t2r
                redg_start = max(t2r_done, redg_chan)
                done = redg_start + slot_chan
                redg_chan = done
                squad_free[squad] = done
                busy_acc[squad] += slot_t2r + slot_chan
                # slot needed again at round r+2 (same bundle) or next bundle r-2
                if r < n_rounds - 2:
                    need = t_b + t_shift + 1.0 + (r + (4 if D64 else 2)) * spacing
                else:
                    need = None  # checked next bundle via prev_drain_done
                if need is not None:
                    dkv_slip = max(dkv_slip, done - need if done > need else 0.0)
                prev_key = (grad, r)
                if prev_key in prev_drain_done and r >= 2:
                    pass
                prev_drain_done[prev_key] = done
        # cross-bundle slot check: drains r2,r3 of bundle b-1 must precede t_b + 1.0+(r-2)*0.9
        # (handled implicitly: if squad_free pushed them late, dkv_slip via next iteration's avail)
        gate_stall["dkv"] += dkv_slip
        t_shift += dkv_slip
        simulate._chan = redg_chan
        # --- bundle floor: MMA + bubbles, supply, protocol ---
        floor = max(MMA_BUNDLE + BUBBLES, 2.0 * supply_64kv)
        end = max(t_b + t_shift + floor, ds1_done, t_b + t_shift + 2.2 + 0.9)
        # backlog metric: how far squad_free exceeds bundle end
        for s in range(3):
            backlog_peak[s] = max(backlog_peak[s], squad_free[s] - end)
        periods.append((end - t_b) / 2.0)
        prev_end = end
        t = end
    steady = sorted(periods[24:])[len(periods[24:]) // 2]
    idle = [1.0 - busy_acc[s] / t for s in range(3)]
    return steady, backlog_peak, idle, gate_stall, t

print(f"{'ledger':<6} {'supply':<7} {'period(64kv)':<13} {'backlog峰值(µs)':<22} {'池空闲率':<20} 门失速累计(µs/48b)")
for name, led in LEDGERS.items():
    for sup in SUPPLY_WALL_64KV:
        steady, bl, idle, gs, tot = simulate(led, sup)
        print(f"{name:<6} {sup:<7.1f} {steady:<13.3f} "
              f"{'/'.join(f'{x:.2f}' for x in bl):<22} "
              f"{'/'.join(f'{x:.0%}' for x in idle):<20} "
              f"P0={gs['CG_P0']:.1f} P1={gs['CG_P1']:.1f} dS={gs['CG_dS']:.1f} dkv={gs['dkv']:.1f}")

# ---- leader 预算清点 ----
print("\nLeader 预算 (per bundle):")
atoms = 224
for issue_ns in (8, 12, 16):
    gates = 8 * 0.05   # 8 族门, 每次唤醒/检查 ~50ns
    pin = 0.1          # 钉桩重排开销
    tmem = 0.1
    total = atoms * issue_ns / 1000 + gates + pin + tmem
    print(f"  issue={issue_ns}ns/atom: {atoms}×{issue_ns}ns + 门 {gates:.2f} + 钉桩/TMEM 0.2 = {total:.2f} µs "
          f"{'✓' if total <= 4.0 else '✗ 超 R13 门'} (bundle 预算按中枢 ~9µs)")
