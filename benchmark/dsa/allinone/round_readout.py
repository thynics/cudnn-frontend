#!/usr/bin/env python3
"""Standard per-round DSA trace/perf readout.

Freezes the analysis performed by hand on every validation round so the
artifacts come back pre-digested and identically shaped:

  * per-name span statistics over the full run (raw per-warp layer, no
    cross-warp envelope synthesis);
  * one all-spans CSV per requested tile window (schema matches the
    established i1_i3_all_spans.csv);
  * steady-state metrics (period from S_ISSUE deltas, W17 chain, dVdK
    cadence, publish tail);
  * per-lever "fired" fingerprints for the v15 flag set (each degrades
    to ABSENT when the corresponding spans are not in the trace);
  * optional perf digest when baseline/candidate JSONs are provided;
  * readout.md tying it together.

Input is the IKET decoder output (iket.decoded_results.json).  Pure
stdlib; no GPU, no harness dependencies.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--decoded", required=True, type=Path,
                   help="candidate iket.decoded_results.json")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--windows", default="1-3,14-17",
                   help="comma-separated tile windows, e.g. 1-3,14-17")
    p.add_argument("--steady-window", default="10-24",
                   help="tile window for steady metrics")
    p.add_argument("--perf-baseline", type=Path, default=None)
    p.add_argument("--perf-candidate", type=Path, default=None)
    p.add_argument("--perf-reference", type=Path, default=None,
                   help="optional third timing (e.g. v12) for 3-way")
    return p.parse_args()


def load_ranges(decoded: Path):
    doc = json.loads(decoded.read_text())
    launch = doc["launches"][0]
    strings = doc["stringTable"]
    locs = doc["locationTable"]
    t0 = min(r["startTs"] for r in launch["ranges"])
    rows = []
    for r in launch["ranges"]:
        li = r["warpLocIdxs"][0]
        loc = locs[li]
        ev = r.get("internalEvents") or []
        payload = ev[0]["payloadVal"] if ev else -1
        rows.append({
            "raw_name": strings[r["rangeNameIdx"]],
            "payload": payload,
            "cta_x": loc["ctaId"][0],
            "warp_id": loc["warpId"],
            "sm_id": loc["smId"],
            "start_us": (r["startTs"] - t0) / 1000.0,
            "end_us": (r["endTs"] - t0) / 1000.0,
        })
    rows.sort(key=lambda x: x["start_us"])
    return rows


def tile_of(row) -> int:
    """Best-effort tile index from the established payload encodings."""
    n, p = row["raw_name"], row["payload"]
    if p < 0:
        return -1
    if "(i,r,p)" in n:
        return p // 8
    if "(r)" in n and "(i" not in n and "(m" not in n:
        return -1  # per-launch names like DQ_EPI(r)
    if "(i,r)" in n or "(m,r)" in n:
        return p // 2
    if "(i,g)" in n:
        return p // 16
    if "(i,k)" in n:
        return p // 8
    return p  # plain (i) names


def in_window(row, lo: int, hi: int) -> bool:
    t = tile_of(row)
    return lo <= t <= hi


def write_window_csv(rows, lo, hi, out: Path):
    sel = [r for r in rows if in_window(r, lo, hi)]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "raw_name", "payload", "cta_x", "warp_id", "sm_id",
            "start_us", "end_us", "duration_us",
        ])
        w.writeheader()
        for r in sel:
            w.writerow({**{k: r[k] for k in w.fieldnames if k != "duration_us"},
                        "duration_us": round(r["end_us"] - r["start_us"], 3)})
    return len(sel)


def name_stats(rows):
    by = {}
    for r in rows:
        by.setdefault(r["raw_name"], []).append(r["end_us"] - r["start_us"])
    out = {}
    for n, ds in sorted(by.items()):
        ds.sort()
        out[n] = {
            "n": len(ds),
            "mean_us": round(statistics.mean(ds), 4),
            "median_us": round(statistics.median(ds), 4),
            "p90_us": round(ds[int(len(ds) * 0.9) - 1 if len(ds) > 1 else 0], 4),
            "max_us": round(ds[-1], 4),
        }
    return out


def series(rows, name, cta=None):
    sel = [r for r in rows if r["raw_name"] == name
           and (cta is None or r["cta_x"] == cta)]
    sel.sort(key=lambda r: (r["payload"], r["start_us"]))
    return sel


def steady_metrics(rows, lo, hi):
    m = {}
    s_issue = [r for r in series(rows, "S_ISSUE(i)", cta=0)
               if lo <= r["payload"] <= hi]
    if len(s_issue) >= 3:
        deltas = [b["start_us"] - a["start_us"]
                  for a, b in zip(s_issue, s_issue[1:])]
        m["period_us"] = round(statistics.mean(deltas), 3)
        m["period_min_max"] = [round(min(deltas), 3), round(max(deltas), 3)]
    for nm, key in [("ROUTE_K(i)", "route_k_mean_us"),
                    ("W18_PDS(i)", "w18_pds_mean_us"),
                    ("MATH_STORE(i)", "math_store_mean_us"),
                    ("MATH_PDS_ACQ(i)", "math_pds_acq_mean_us")]:
        sel = [r["end_us"] - r["start_us"] for r in series(rows, nm, cta=0)
               if lo <= r["payload"] <= hi]
        if sel:
            m[key] = round(statistics.mean(sel), 3)
    qdo = [r for r in series(rows, "MAT_QDO(m,r)", cta=0)
           if lo <= r["payload"] // 2 <= hi]
    if qdo:
        m["mat_qdo_mean_us"] = round(statistics.mean(
            [r["end_us"] - r["start_us"] for r in qdo]), 3)
    dv = [r for r in series(rows, "dVdK_ISSUE(i,r,p)", cta=0)
          if lo <= r["payload"] // 8 <= hi]
    if len(dv) >= 3:
        dv.sort(key=lambda r: r["start_us"])
        gaps = [b["start_us"] - a["end_us"] for a, b in zip(dv, dv[1:])
                if b["payload"] % 8 != 0]
        if gaps:
            m["dvdk_interpass_gap_mean_us"] = round(statistics.mean(gaps), 3)
        m["dvdk_issue_mean_us"] = round(statistics.mean(
            [r["end_us"] - r["start_us"] for r in dv]), 3)
    # publish tail: MATH_STORE(t) end -> first dVdK_ISSUE(t) start, per tile
    tails = []
    store = {r["payload"]: r for r in series(rows, "MATH_STORE(i)", cta=0)}
    for t in range(lo, hi + 1):
        dv_t = [r for r in dv if r["payload"] // 8 == t
                and r["payload"] % 8 == 0]
        if t in store and dv_t:
            tails.append(dv_t[0]["start_us"] - store[t]["end_us"])
    if tails:
        m["publish_tail_mean_us"] = round(statistics.mean(tails), 3)
    return m


def fingerprints(stats):
    fp = {}
    fp["L2X_math_pds_acq_absent"] = "MATH_PDS_ACQ(i)" not in stats
    fp["L2X_w18_pds_present"] = "W18_PDS(i)" in stats
    if "W18_PDS(i)" in stats:
        fp["L2X_w18_pds_mean_us"] = stats["W18_PDS(i)"]["mean_us"]
    fp["route_p_ds_absent"] = ("ROUTE_P(i)" not in stats
                               and "ROUTE_dS(i)" not in stats)
    for probe in ("PROBE_GEN(i,g)", "PROBE_DONE(i,k)"):
        fp[f"probe_{probe.split('(')[0]}"] = probe in stats
    return fp


def perf_digest(args):
    d = {}
    for tag, p in [("baseline", args.perf_baseline),
                   ("candidate", args.perf_candidate),
                   ("reference", args.perf_reference)]:
        if p is not None and p.exists():
            try:
                d[tag] = json.loads(p.read_text())
            except Exception as e:  # keep the round alive
                d[tag] = {"error": str(e)}
    return d


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_ranges(args.decoded)

    stats = name_stats(rows)
    (args.out_dir / "span_means_full_run.json").write_text(
        json.dumps(stats, indent=1))

    window_files = []
    for win in args.windows.split(","):
        lo, hi = (int(x) for x in win.split("-"))
        out = args.out_dir / f"i{lo}_i{hi}_all_spans.csv"
        n = write_window_csv(rows, lo, hi, out)
        window_files.append((out.name, n))

    lo, hi = (int(x) for x in args.steady_window.split("-"))
    steady = steady_metrics(rows, lo, hi)
    fp = fingerprints(stats)
    perf = perf_digest(args)

    md = ["# round readout (auto-generated)", ""]
    md.append(f"- decoded source: `{args.decoded}`")
    md.append(f"- total ranges: {len(rows)}; distinct names: {len(stats)}")
    md.append(f"- window CSVs: " + ", ".join(
        f"`{n}` ({c} rows)" for n, c in window_files))
    md.append("")
    md.append(f"## steady metrics (tiles {lo}-{hi})")
    md.append("")
    md.append("| metric | value |")
    md.append("|---|---|")
    for k, v in steady.items():
        md.append(f"| {k} | {v} |")
    md.append("")
    md.append("## lever fingerprints")
    md.append("")
    md.append("| fingerprint | value |")
    md.append("|---|---|")
    for k, v in fp.items():
        md.append(f"| {k} | {v} |")
    if perf:
        md.append("")
        md.append("## perf digest")
        md.append("")
        md.append("```json")
        md.append(json.dumps(perf, indent=1)[:4000])
        md.append("```")
    md.append("")
    md.append("## per-name means (full run)")
    md.append("")
    md.append("| name | n | mean | median | p90 | max |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for n, s in stats.items():
        md.append(f"| {n} | {s['n']} | {s['mean_us']} | {s['median_us']}"
                  f" | {s['p90_us']} | {s['max_us']} |")
    (args.out_dir / "readout.md").write_text("\n".join(md) + "\n")
    print(f"READOUT_OK {args.out_dir}/readout.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
