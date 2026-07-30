#!/usr/bin/env python3
"""DSA Stage-0 SASS gate analyzer (config-generalized).

Derived from the runner's v15 rev3 one-off analyzer; all revision-pinned
expectations (module-level UPPER_CASE constants) can now be overridden via
--expectations gates.json, whose keys are the constant names.  It consumes
a compile-only Stage-0 capture plus a matched reference capture and never
launches a workload or invokes a GPU tool.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable


TARGET_REVISION = "2960a6a3decea7b2c39259eaa651a1239a35ac38"
TARGET_SOURCE_SHA256 = (
    "0f4257f2b28dc4da40dd9b10aef5d4f933cff1b3650813fc3dd148659de10aeb"
)
REFERENCE_SOURCE_SHA256 = (
    "581e5e63d01991874569e7262100b3122951c1eb4838314f9330b2c71727e36b"
)
DYNAMIC_SMEM_EXPECTED = 231_424
DYNAMIC_SMEM_LIMIT = 232_448
STATIC_SHARED_EXPECTED = 1_024
REGSWAP_C_POOL = 61_440
COMPILER_EXPECTED = "4.6.1"
CODEGEN_KEEP_EXPECTED = "ptx,cubin,sass"
G2S_SITES_PER_SOURCE_BLOCK = 5
G2S_STATIC_CLONES_EXPECTED = 3
G2S_CLONE_GAP_MIN_BYTES = 0x200
# Config-hoisted expectations (override via --expectations JSON):
G1_EXPECTED_USETMAXREG = [
    ["DEALLOC", 40, 1],
    ["DEALLOC", 56, 1],
    ["TRY_ALLOC", 128, 2],
]
G2_ALLOWED_DELTAS = {"W16": [0, 0], "W17": [0, 0], "W18": [0, 0]}
MATH_STSM_EXPECTED = 8
MATH_STSM_WAVES_EXPECTED = [4, 4]
MATH_S2G_EXPECTED = 2
REFERENCE_ROLE_LOCAL_TOTALS = {"W16": 28, "W17": 8, "W18": 9}

INSTRUCTION_RE = re.compile(
    r"/\*([0-9a-fA-F]+)\*/\s+"
    r"(?:(?:@!?P(?:T|\d+)|@!?UP(?:T|\d+))\s+)?"
    r"([A-Z][A-Za-z0-9_.]*)"
)
LABEL_RE = re.compile(r"(\.L_x_\d+):")
BRANCH_TARGET_RE = re.compile(
    r"\bBRA(?:\.[A-Z0-9_.]+)?\s+"
    r"(?:!?UP\d+,\s+)?"
    r"`\((\.L_x_\d+)\)"
)
SPECIAL_ROLE_COMPARE_RE = re.compile(
    r"\bU?ISETP\.NE\.U32\.[A-Z0-9_.]+\b"
    r".*\b((?:U)?R\d+),\s*0x(10|11),\s*"
)
W18_COMPARE_RE = re.compile(
    r"\bU?ISETP\.NE\.U32\.[A-Z0-9_.]+\b"
    r".*\b((?:U)?R\d+),\s*0x12,\s*"
)
ROLE_BOUND_COMPARE_RE = re.compile(
    r"\bUISETP\.GT(?:\.U32)?\.AND\b"
    r".*\b(UR\d+),\s*0x([37]),\s*UPT"
)
USETMAXREG_RE = re.compile(
    r"\bUSETMAXREG\.(DEALLOC|TRY_ALLOC)\.CTAPOOL\b"
    r"(?:\s+UP\d+,)?\s+(0x[0-9a-fA-F]+|\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the v15 rev3 G0-G4 Stage-0 gates offline."
    )
    parser.add_argument(
        "--capture-root",
        required=True,
        type=Path,
        help="Extracted v15 Stage-0 capture directory.",
    )
    parser.add_argument(
        "--expectations",
        type=Path,
        default=None,
        help=(
            "JSON overriding module-level expectation constants "
            "(keys = constant names, e.g. TARGET_REVISION, "
            "DYNAMIC_SMEM_EXPECTED, REGSWAP_C_POOL)."
        ),
    )
    parser.add_argument(
        "--reference-root",
        required=True,
        type=Path,
        help="Matched v12 Stage-0 capture directory, or its main_kernel.sass.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New directory for JSON, Markdown, and SASS evidence windows.",
    )
    parser.add_argument(
        "--dynamic-smem-bytes",
        type=int,
        default=None,
        help=(
            "Optional observed dynamic SMEM override.  Without it, 231424 is "
            "accepted only when capture provenance matches the pinned v15 "
            "rev3 source hash."
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def find_manifest(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.is_file():
        return direct
    matches = sorted(
        path
        for path in root.rglob(name)
        if not any(
            part in {"analysis_fast", "analysis_v15", "stage0_analysis"}
            for part in path.parts
        )
    )
    return matches[0] if len(matches) == 1 else None


def candidate_sha_from_capture(root: Path) -> tuple[str | None, str | None]:
    image_path = find_manifest(root, "image_run_manifest.json")
    if image_path is not None:
        image = load_json(image_path)
        value = image.get("source_sha256", {}).get("selected_candidate")
        if value:
            return str(value), str(image.get("source_revision") or "")
    source_path = find_manifest(root, "source_manifest.json")
    if source_path is not None:
        source = load_json(source_path)
        value = source.get("active_candidate_sha256")
        if value:
            return str(value), str(source.get("source_revision") or "")
    return None, None


def codegen_metadata(root: Path) -> tuple[Path | None, dict[str, Any]]:
    if root.is_file():
        return None, {}
    conventional = (
        root / "logs" / "codegen" / "correctness" / "artifact_manifest.json"
    )
    if conventional.is_file():
        return conventional, load_json(conventional)
    matches = sorted(
        path
        for path in root.rglob("artifact_manifest.json")
        if "codegen" in path.parts
        and not any(
            part in {"analysis_fast", "analysis_v15", "stage0_analysis"}
            for part in path.parts
        )
    )
    if len(matches) == 1:
        return matches[0], load_json(matches[0])
    return None, {}


def find_codegen_file(root: Path, suffix: str) -> Path:
    conventional = root / "logs" / "codegen" / "correctness"
    if conventional.is_dir():
        matches = sorted(conventional.glob(f"*{suffix}"))
    else:
        matches = sorted(
            path
            for path in root.rglob(f"*{suffix}")
            if not any(
                part in {
                    "analysis_fast",
                    "analysis_v15",
                    "stage0",
                    "stage0_analysis",
                }
                for part in path.parts
            )
        )
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one raw {suffix} below {root}, found {matches}"
        )
    return matches[0]


def find_reference_sass(path: Path) -> Path:
    if path.is_file():
        return path
    preferred = (
        path / "analysis_fast" / "main_kernel.sass",
        path / "stage0" / "main_kernel.sass",
        path / "main_kernel.sass",
    )
    for candidate in preferred:
        if candidate.is_file():
            return candidate
    return find_codegen_file(path, ".sass")


def find_resource_usage(root: Path) -> Path:
    preferred = (
        root / "resource_usage.txt",
        root / "stage0" / "resource_usage.txt",
    )
    for candidate in preferred:
        if candidate.is_file():
            return candidate
    matches = sorted(root.rglob("resource_usage.txt"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one resource_usage.txt below {root}, "
            f"found {matches}"
        )
    return matches[0]


def extract_main_kernel(raw_sass: Path) -> list[str]:
    lines = raw_sass.read_text(errors="replace").splitlines()
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if line.startswith(
                "//--------------------- .text.kernel_cutlass_kernel_"
            )
        ),
        None,
    )
    if start is None:
        if any("USETMAXREG." in line for line in lines):
            return lines
        raise RuntimeError(f"main DSA kernel was not found in {raw_sass}")
    end = next(
        (
            index
            for index, line in enumerate(lines[start + 1 :], start + 1)
            if line.startswith("//--------------------- .text.")
        ),
        len(lines),
    )
    return lines[start:end]


def parse_instruction(line: str) -> tuple[int, str] | None:
    match = INSTRUCTION_RE.search(line)
    if match is None:
        return None
    return int(match.group(1), 16), match.group(2)


def instructions_in(
    lines: list[str], first_line: int = 0, end_line: int | None = None
) -> list[tuple[int, int, str, str]]:
    if end_line is None:
        end_line = len(lines)
    return [
        (line_index, address, opcode, lines[line_index])
        for line_index in range(first_line, end_line)
        if (parsed := parse_instruction(lines[line_index])) is not None
        for address, opcode in (parsed,)
    ]


def opcode_counter(
    instructions: Iterable[tuple[int, int, str, str]]
) -> Counter[str]:
    return Counter(opcode for _, _, opcode, _ in instructions)


def prefix_count(counter: Counter[str], prefix: str) -> int:
    return sum(
        count
        for opcode, count in counter.items()
        if opcode == prefix or opcode.startswith(prefix + ".")
    )


def branch_target(lines: list[str], compare_line: int) -> str:
    for line in lines[compare_line + 1 : compare_line + 7]:
        match = BRANCH_TARGET_RE.search(line)
        if match is not None:
            return match.group(1)
    raise RuntimeError(
        f"no branch target found after compare on SASS line {compare_line + 1}"
    )


def labels(lines: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for index, line in enumerate(lines):
        match = LABEL_RE.match(line)
        if match is not None:
            result[match.group(1)] = index
    return result


def label_addresses(lines: list[str]) -> dict[str, int]:
    """Map nvdisasm labels to the first instruction address in each block."""

    result: dict[str, int] = {}
    for name, line_index in labels(lines).items():
        next_instruction = next(
            (
                parse_instruction(lines[index])
                for index in range(line_index + 1, len(lines))
                if parse_instruction(lines[index]) is not None
            ),
            None,
        )
        if next_instruction is None:
            raise RuntimeError(f"label {name} has no following instruction")
        result[name] = next_instruction[0]
    return result


def branch_edges(
    lines: list[str],
    instructions: Iterable[tuple[int, int, str, str]],
) -> list[dict[str, Any]]:
    """Return resolved BRA edges for a SASS instruction slice."""

    label_to_address = label_addresses(lines)
    result: list[dict[str, Any]] = []
    for _, address, opcode, line in instructions:
        if opcode != "BRA" and not opcode.startswith("BRA."):
            continue
        match = BRANCH_TARGET_RE.search(line)
        if match is None or match.group(1) not in label_to_address:
            continue
        target_label = match.group(1)
        result.append(
            {
                "source": address,
                "target": label_to_address[target_label],
                "target_label": target_label,
                "line": line.strip(),
            }
        )
    return result


def special_role_windows(lines: list[str]) -> dict[str, tuple[int, int]]:
    """Locate the top-level W16/W17/W18 source-order branches.

    The compiler lowers the role chain into three unique scalar comparisons
    against warp ids 0x10, 0x11, and 0x12.  The first two branch targets are
    the next role, and the third target is the common tail.
    """

    label_lines = labels(lines)
    candidates = [
        (index, match.group(1), int(match.group(2), 16))
        for index, line in enumerate(lines)
        if (match := SPECIAL_ROLE_COMPARE_RE.search(line)) is not None
    ]
    chains: list[tuple[int, str, int]] = []
    for start16, register, immediate in candidates:
        if immediate != 0x10:
            continue
        try:
            w17_label = branch_target(lines, start16)
            start17 = label_lines[w17_label]
        except (KeyError, RuntimeError):
            continue
        next_compare = next(
            (
                (index, match.group(1))
                for index in range(start17, min(start17 + 12, len(lines)))
                if (match := SPECIAL_ROLE_COMPARE_RE.search(lines[index]))
                is not None
                and int(match.group(2), 16) == 0x11
            ),
            None,
        )
        if next_compare is not None and next_compare[1] == register:
            chains.append((start16, register, next_compare[0]))
    if len(chains) != 1:
        raise RuntimeError(
            "expected one chained warp-id 16->17 role dispatch, found "
            f"{chains}"
        )
    start16, warp_register, compare17 = chains[0]
    w17_label = branch_target(lines, start16)
    start17 = label_lines[w17_label]
    w18_label = branch_target(lines, compare17)
    start18 = label_lines[w18_label]
    w18_compare = next(
        (
            (index, match.group(1))
            for index in range(start18, min(start18 + 16, len(lines)))
            if (match := W18_COMPARE_RE.search(lines[index])) is not None
        ),
        None,
    )
    if w18_compare is None:
        raise RuntimeError("W18 dispatch compare was not found")
    w18_register = w18_compare[1]
    register_bridge = any(
        re.search(
            rf"\b{re.escape(w18_register)}\b.*\b{re.escape(warp_register)}\b",
            lines[index],
        )
        for index in range(start18, w18_compare[0])
    )
    if w18_register != warp_register and not register_bridge:
        raise RuntimeError(
            "W18 dispatch has no validated bridge from the uniform warp-id "
            f"register {warp_register} to {w18_register}"
        )
    common_label = branch_target(lines, w18_compare[0])
    common = label_lines[common_label]
    if not start16 < start17 < start18 < common:
        raise RuntimeError(
            "special-role SASS windows are not in source order: "
            f"{start16}, {start17}, {start18}, {common}"
        )
    return {
        "W16": (start16, start17),
        "W17": (start17, start18),
        "W18": (start18, common),
    }


def math_window(lines: list[str]) -> tuple[int, int]:
    """Locate W4-W7 from the two top-level uniform warp-id comparisons."""

    label_lines = labels(lines)
    chains: list[tuple[int, str, int, int]] = []
    for index, line in enumerate(lines):
        match = ROLE_BOUND_COMPARE_RE.search(line)
        if match is None or int(match.group(2), 16) != 0x3:
            continue
        register = match.group(1)
        try:
            start = label_lines[branch_target(lines, index)]
        except (KeyError, RuntimeError):
            continue
        reduce_cmp = next(
            (
                candidate
                for candidate in range(start, min(start + 8, len(lines)))
                if (next_match := ROLE_BOUND_COMPARE_RE.search(lines[candidate]))
                is not None
                and next_match.group(1) == register
                and int(next_match.group(2), 16) == 0x7
            ),
            None,
        )
        if reduce_cmp is not None:
            reduce_label = branch_target(lines, reduce_cmp)
            reduce_start = label_lines[reduce_label]
            semantic_counts = opcode_counter(
                instructions_in(lines, start, reduce_start)
            )
            # The register-allocation prelude repeats the same 3/7 bounds.
            # The true math role is the only chain whose body contains the
            # stmatrix publish lowering.
            if prefix_count(semantic_counts, "STSM") > 0:
                chains.append((index, register, start, reduce_cmp))
    if len(chains) != 1:
        raise RuntimeError(
            "expected one chained gather->math->reduce dispatch, found "
            f"{chains}"
        )
    _, _, math_start, reduce_cmp = chains[0]
    reduce_start = label_lines[branch_target(lines, reduce_cmp)]
    return math_start, reduce_start


def window_summary(
    lines: list[str], first: int, end: int
) -> dict[str, Any]:
    instructions = instructions_in(lines, first, end)
    counts = opcode_counter(instructions)
    local_variants = {
        opcode: count
        for opcode, count in sorted(counts.items())
        if opcode == "LDL"
        or opcode.startswith("LDL.")
        or opcode == "STL"
        or opcode.startswith("STL.")
    }
    addresses = [address for _, address, _, _ in instructions]
    return {
        "start": hex(min(addresses)),
        "end_inclusive": hex(max(addresses)),
        "LDL": prefix_count(counts, "LDL"),
        "STL": prefix_count(counts, "STL"),
        "local_total": (
            prefix_count(counts, "LDL") + prefix_count(counts, "STL")
        ),
        "local_variants": local_variants,
        "local_sites": [
            {
                "address": hex(address),
                "opcode": opcode,
                "instruction": line.strip(),
            }
            for _, address, opcode, line in instructions
            if opcode == "LDL"
            or opcode.startswith("LDL.")
            or opcode == "STL"
            or opcode.startswith("STL.")
        ],
        "MOV.SPILL": prefix_count(counts, "MOV.SPILL"),
        "STSM": prefix_count(counts, "STSM"),
        "STS.U16": prefix_count(counts, "STS.U16"),
        "UBLKCP.G.S": prefix_count(counts, "UBLKCP.G.S"),
        "UBLKCP.S.G": prefix_count(counts, "UBLKCP.S.G"),
    }


def write_window(
    path: Path, lines: list[str], first: int, end: int
) -> None:
    path.write_text("\n".join(lines[first:end]) + "\n")


def parse_resources(path: Path) -> dict[str, int]:
    text = path.read_text(errors="replace")
    match = re.search(
        r"Function kernel_cutlass_kernel_[^\n]+:\n"
        r"\s+REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)"
        r"\s+LOCAL:(\d+)",
        text,
    )
    if match is None:
        raise RuntimeError(f"main-kernel resources were not found in {path}")
    reg, stack, shared, local = map(int, match.groups())
    return {
        "REG": reg,
        "STACK": stack,
        "STATIC_SHARED": shared,
        "LOCAL": local,
    }


def parse_usetmaxreg(
    instructions: list[tuple[int, int, str, str]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for _, address, opcode, line in instructions:
        if not opcode.startswith("USETMAXREG."):
            continue
        match = USETMAXREG_RE.search(line)
        if match is None:
            raise RuntimeError(f"could not parse USETMAXREG at {hex(address)}")
        result.append(
            {
                "address": hex(address),
                "kind": match.group(1),
                "value": int(match.group(2), 0),
                "opcode": opcode,
            }
        )
    return result


def source_sha_for_reference(root: Path) -> str | None:
    if root.is_file():
        return None
    value, _ = candidate_sha_from_capture(root)
    return value


def status(pass_condition: bool) -> str:
    return "PASS" if pass_condition else "FAIL"


def first_and_last_address(
    instructions: list[tuple[int, int, str, str]]
) -> tuple[str, str]:
    return hex(instructions[0][1]), hex(instructions[-1][1])


def group_g2s_lowering_clones(
    instructions: list[tuple[int, int, str, str]],
) -> list[list[tuple[int, int, str, str]]]:
    groups: list[list[tuple[int, int, str, str]]] = []
    for instruction in instructions:
        if (
            not groups
            or instruction[1] - groups[-1][-1][1]
            > G2S_CLONE_GAP_MIN_BYTES
        ):
            groups.append([instruction])
        else:
            groups[-1].append(instruction)
    return groups


def prove_g2s_per_tile_cfg(
    lines: list[str],
    w18_instructions: list[tuple[int, int, str, str]],
    groups: list[list[tuple[int, int, str, str]]],
) -> dict[str, Any]:
    """Prove the three static G2S groups are per-iteration lowering clones.

    CUTLASS 4.6.1 lowers the source loop as a two-tile unrolled main body
    followed by an optional scalar tail.  Thus the first two five-site
    groups may both execute in one loop trip, but they belong to different
    logical tiles; the third group is the mutually exclusive odd tail.
    The three resolved CFG anchors below distinguish that lowering from a
    single tile issuing fifteen G2S operations.
    """

    proof: dict[str, Any] = {
        "status": "FAIL",
        "interpretation": (
            "unproven: expected a two-tile unrolled main body plus an "
            "optional scalar tail"
        ),
        "main_loop_backedge": None,
        "tail_entry": None,
        "tail_skip": None,
    }
    if len(groups) != G2S_STATIC_CLONES_EXPECTED or any(not group for group in groups):
        return proof

    group1, group2, group3 = groups
    group1_start = group1[0][1]
    group2_end = group2[-1][1]
    group3_start = group3[0][1]
    group3_end = group3[-1][1]
    edges = branch_edges(lines, w18_instructions)

    # The backedge lies after both five-site groups and returns before the
    # first group.  Counter updates in the same block advance the unrolled
    # tile state by two.
    main_loop_backedges = [
        edge
        for edge in edges
        if group2_end < edge["source"] < group3_start
        and edge["target"] < group1_start
    ]
    main_loop_backedge = (
        max(main_loop_backedges, key=lambda edge: edge["source"])
        if main_loop_backedges
        else None
    )

    # Entry can bypass the two-group main loop and land in the tail prelude.
    tail_entries = [
        edge
        for edge in edges
        if edge["source"] < group1_start
        and group2_end < edge["target"] < group3_start
    ]
    tail_entry = (
        max(tail_entries, key=lambda edge: edge["source"])
        if tail_entries
        else None
    )

    # After the unrolled loop, the parity branch can skip the tail group.
    tail_skips = [
        edge
        for edge in edges
        if group2_end < edge["source"] < group3_start
        and edge["target"] > group3_end
    ]
    tail_skip = (
        max(tail_skips, key=lambda edge: edge["source"])
        if tail_skips
        else None
    )

    counter_updates: list[dict[str, Any]] = []
    if main_loop_backedge is not None:
        counter_updates = [
            {
                "address": hex(address),
                "instruction": line.strip(),
            }
            for _, address, opcode, line in w18_instructions
            if group2_end < address < main_loop_backedge["source"]
            and (
                re.search(r"\bVIADD\b.*,\s*0x2\s*;", line) is not None
                or re.search(r"\bIADD3\b.*,\s*0x1,\s*RZ\s*;", line)
                is not None
            )
        ]
    has_step_two = any(
        re.search(r"\bVIADD\b.*,\s*0x2\s*;", item["instruction"])
        is not None
        for item in counter_updates
    )

    cfg_pass = (
        main_loop_backedge is not None
        and tail_entry is not None
        and tail_skip is not None
        and has_step_two
    )

    def report_edge(edge: dict[str, Any] | None) -> dict[str, Any] | None:
        if edge is None:
            return None
        return {
            "source": hex(edge["source"]),
            "target": hex(edge["target"]),
            "target_label": edge["target_label"],
            "instruction": edge["line"],
        }

    proof.update(
        {
            "status": status(cfg_pass),
            "interpretation": (
                "two five-site groups are distinct logical tiles in the "
                "step-2 unrolled main loop; the third five-site group is "
                "the parity-selected scalar tail, so each logical tile "
                "issues five G2S operations"
                if cfg_pass
                else proof["interpretation"]
            ),
            "main_loop_backedge": report_edge(main_loop_backedge),
            "tail_entry": report_edge(tail_entry),
            "tail_skip": report_edge(tail_skip),
            "counter_updates_before_backedge": counter_updates,
            "per_logical_tile_sites": G2S_SITES_PER_SOURCE_BLOCK,
            "kernel_wide_mutual_exclusivity": False,
            "per_logical_tile_clone_exclusivity": cfg_pass,
        }
    )
    return proof


def main() -> int:
    args = parse_args()
    if args.expectations is not None:
        overrides = json.loads(args.expectations.read_text())
        for key, value in overrides.items():
            if key.isupper() and key in globals() and not key.endswith("_RE"):
                globals()[key] = value
            else:
                raise SystemExit(
                    f"unknown expectation constant: {key}"
                )
    capture_root = args.capture_root.resolve()
    reference_root = args.reference_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    raw_sass = find_codegen_file(capture_root, ".sass")
    resource_path = find_resource_usage(capture_root)
    candidate_sha, source_revision = candidate_sha_from_capture(capture_root)
    main_lines = extract_main_kernel(raw_sass)
    main_instructions = instructions_in(main_lines)
    (output_dir / "main_kernel.sass").write_text(
        "\n".join(main_lines) + "\n"
    )

    reference_sass = find_reference_sass(reference_root)
    reference_lines = extract_main_kernel(reference_sass)
    reference_sha = source_sha_for_reference(reference_root)
    candidate_codegen_path, candidate_codegen = codegen_metadata(capture_root)
    reference_codegen_path, reference_codegen = codegen_metadata(reference_root)

    resources = parse_resources(resource_path)
    provenance_matches = candidate_sha == TARGET_SOURCE_SHA256
    if args.dynamic_smem_bytes is not None:
        dynamic_smem = args.dynamic_smem_bytes
        dynamic_smem_basis = "explicit analyzer argument"
        dynamic_smem_verified = True
    elif provenance_matches:
        dynamic_smem = DYNAMIC_SMEM_EXPECTED
        dynamic_smem_basis = (
            "pinned 2960a6a SharedStorageV2.size_in_bytes specialization; "
            "successful compile exercised its <=MAX_SMEM_BYTES assert"
        )
        dynamic_smem_verified = True
    else:
        dynamic_smem = None
        dynamic_smem_basis = (
            "unverified: no explicit value and candidate hash is not the "
            "pinned 2960a6a source"
        )
        dynamic_smem_verified = False
    g0_pass = (
        dynamic_smem_verified
        and dynamic_smem is not None
        and dynamic_smem <= DYNAMIC_SMEM_LIMIT
        and resources["STATIC_SHARED"] == STATIC_SHARED_EXPECTED
    )

    usetmaxreg = parse_usetmaxreg(main_instructions)
    observed_reg_contract = Counter(
        (item["kind"], item["value"]) for item in usetmaxreg
    )
    expected_reg_contract = Counter(
        {
            (kind, value): count
            for kind, value, count in G1_EXPECTED_USETMAXREG
        }
    )
    g1_pass = observed_reg_contract == expected_reg_contract

    candidate_roles = special_role_windows(main_lines)
    reference_roles = special_role_windows(reference_lines)
    candidate_role_data = {
        role: window_summary(main_lines, *window)
        for role, window in candidate_roles.items()
    }
    reference_role_data = {
        role: window_summary(reference_lines, *window)
        for role, window in reference_roles.items()
    }
    role_deltas: dict[str, dict[str, int]] = {}
    g2_pass = True
    for role in ("W16", "W17", "W18"):
        candidate = candidate_role_data[role]
        reference = reference_role_data[role]
        role_deltas[role] = {
            key: candidate[key] - reference[key]
            for key in ("LDL", "STL", "local_total")
        }
        allowed_ldl, allowed_stl = G2_ALLOWED_DELTAS.get(role, [0, 0])
        if (role_deltas[role]["LDL"] > allowed_ldl
                or role_deltas[role]["STL"] > allowed_stl):
            g2_pass = False
    leader_fingerprint = (
        "FIRED"
        if candidate_role_data["W16"]["local_total"]
        < reference_role_data["W16"]["local_total"]
        else "NOT_FIRED"
    )

    math_first, math_end = math_window(main_lines)
    math_instructions = instructions_in(main_lines, math_first, math_end)
    math_counts = opcode_counter(math_instructions)
    s2g_instructions = [
        item
        for item in math_instructions
        if item[2] == "UBLKCP.G.S" or item[2].startswith("UBLKCP.G.S.")
    ]
    all_g2s = [
        item
        for item in main_instructions
        if item[2] == "UBLKCP.S.G" or item[2].startswith("UBLKCP.S.G.")
    ]
    w18_first, w18_end = candidate_roles["W18"]
    w18_instructions = instructions_in(main_lines, w18_first, w18_end)
    w18_counts = opcode_counter(w18_instructions)
    w18_g2s_instructions = [
        item
        for item in w18_instructions
        if item[2] == "UBLKCP.S.G" or item[2].startswith("UBLKCP.S.G.")
    ]
    g2s_clone_groups = group_g2s_lowering_clones(w18_g2s_instructions)
    g2s_clone_group_report = [
        {
            "clone": index + 1,
            "count": len(group),
            "start": hex(group[0][1]),
            "end": hex(group[-1][1]),
            "sites": [hex(item[1]) for item in group],
        }
        for index, group in enumerate(g2s_clone_groups)
    ]
    all_s2g_count = prefix_count(opcode_counter(main_instructions), "UBLKCP.G.S")
    all_g2s_count = prefix_count(opcode_counter(main_instructions), "UBLKCP.S.G")

    waves: list[dict[str, Any]] = []
    previous_position = 0
    if len(s2g_instructions) == MATH_S2G_EXPECTED:
        for index, s2g in enumerate(s2g_instructions):
            position = math_instructions.index(s2g)
            segment = math_instructions[previous_position : position + 1]
            segment_counts = opcode_counter(segment)
            stsm_positions = [
                segment_index
                for segment_index, item in enumerate(segment)
                if item[2] == "STSM" or item[2].startswith("STSM.")
            ]
            if stsm_positions:
                segment = segment[stsm_positions[0] :]
                segment_counts = opcode_counter(segment)
            waves.append(
                {
                    "wave": index + 1,
                    "start": hex(segment[0][1]),
                    "s2g": hex(s2g[1]),
                    "STSM": prefix_count(segment_counts, "STSM"),
                    "STS.U16": prefix_count(segment_counts, "STS.U16"),
                    "UBLKCP.G.S": prefix_count(
                        segment_counts, "UBLKCP.G.S"
                    ),
                }
            )
            previous_position = position + 1

    math_stsm = prefix_count(math_counts, "STSM")
    math_sts_u16 = prefix_count(math_counts, "STS.U16")
    w18_g2s = prefix_count(w18_counts, "UBLKCP.S.G")
    wave_contract = (
        len(waves) == len(MATH_STSM_WAVES_EXPECTED)
        and [wave["STSM"] for wave in waves] == MATH_STSM_WAVES_EXPECTED
        and all(wave["STS.U16"] == 0 for wave in waves)
        and all(wave["UBLKCP.G.S"] == 1 for wave in waves)
    )
    g3_pass = (
        all_s2g_count == MATH_S2G_EXPECTED
        and len(s2g_instructions) == MATH_S2G_EXPECTED
        and all_g2s_count
        == G2S_SITES_PER_SOURCE_BLOCK * G2S_STATIC_CLONES_EXPECTED
        and len(all_g2s)
        == G2S_SITES_PER_SOURCE_BLOCK * G2S_STATIC_CLONES_EXPECTED
        and w18_g2s
        == G2S_SITES_PER_SOURCE_BLOCK * G2S_STATIC_CLONES_EXPECTED
        and len(g2s_clone_groups) == G2S_STATIC_CLONES_EXPECTED
        and all(
            len(group) == G2S_SITES_PER_SOURCE_BLOCK
            for group in g2s_clone_groups
        )
        and math_stsm == MATH_STSM_EXPECTED
        and wave_contract
    )

    reference_contract_ok = (
        reference_sha in {None, REFERENCE_SOURCE_SHA256}
        and all(
            reference_role_data[role]["local_total"] == expected
            for role, expected in REFERENCE_ROLE_LOCAL_TOTALS.items()
        )
        and all(reference_role_data[role]["STL"] == 0 for role in reference_roles)
    )
    codegen_contract_ok = (
        str(candidate_codegen.get("cutlass_version")) == COMPILER_EXPECTED
        and str(reference_codegen.get("cutlass_version")) == COMPILER_EXPECTED
        and str(candidate_codegen.get("keep")) == CODEGEN_KEEP_EXPECTED
        and str(reference_codegen.get("keep")) == CODEGEN_KEEP_EXPECTED
    )
    p0_pass = (
        provenance_matches
        and reference_contract_ok
        and codegen_contract_ok
        and bool(main_instructions)
        and bool(reference_lines)
    )

    gate_statuses = {
        "P0_provenance": status(p0_pass),
        "G0_smem": status(g0_pass),
        "G1_regswap_c": status(g1_pass),
        "G2_special_role_spills": status(g2_pass),
        "G3_publish_and_l2x_bulk": status(g3_pass),
        "G4_variant_b_reduce": "N/A",
    }
    overall_pass = all(
        value == "PASS"
        for key, value in gate_statuses.items()
        if key != "G4_variant_b_reduce"
    )

    for role, window in candidate_roles.items():
        write_window(
            output_dir / f"{role.lower()}_candidate.sass",
            main_lines,
            *window,
        )
    for role, window in reference_roles.items():
        write_window(
            output_dir / f"{role.lower()}_v12_reference.sass",
            reference_lines,
            *window,
        )
    write_window(
        output_dir / "math_candidate.sass",
        main_lines,
        math_first,
        math_end,
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "analyzer_contract": "v15 rev3 2960a6a one-off",
        "analyzer_path": str(Path(__file__).resolve()),
        "analyzer_sha256": sha256(Path(__file__).resolve()),
        "overall_status": "PASS" if overall_pass else "FAIL",
        "stop_downstream": not overall_pass,
        "validation_authorized": overall_pass,
        "gate_statuses": gate_statuses,
        "provenance": {
            "target_revision_expected": TARGET_REVISION,
            "source_revision_observed": source_revision,
            "candidate_sha256_expected": TARGET_SOURCE_SHA256,
            "candidate_sha256_observed": candidate_sha,
            "candidate_sass": str(raw_sass),
            "candidate_sass_sha256": sha256(raw_sass),
            "reference_source_sha256_expected": REFERENCE_SOURCE_SHA256,
            "reference_source_sha256_observed": reference_sha,
            "reference_sass": str(reference_sass),
            "reference_sass_sha256": sha256(reference_sass),
            "resource_usage": str(resource_path),
            "resource_usage_sha256": sha256(resource_path),
            "candidate_codegen_manifest": (
                str(candidate_codegen_path)
                if candidate_codegen_path is not None
                else None
            ),
            "candidate_cutlass_version": candidate_codegen.get(
                "cutlass_version"
            ),
            "candidate_codegen_keep": candidate_codegen.get("keep"),
            "reference_codegen_manifest": (
                str(reference_codegen_path)
                if reference_codegen_path is not None
                else None
            ),
            "reference_cutlass_version": reference_codegen.get(
                "cutlass_version"
            ),
            "reference_codegen_keep": reference_codegen.get("keep"),
        },
        "gates": {
            "P0_provenance": {
                "status": gate_statuses["P0_provenance"],
                "candidate_source_match": provenance_matches,
                "reference_contract_match": reference_contract_ok,
                "codegen_contract_match": codegen_contract_ok,
                "compiler_expected": COMPILER_EXPECTED,
                "keep_expected": CODEGEN_KEEP_EXPECTED,
            },
            "G0_smem": {
                "status": gate_statuses["G0_smem"],
                "dynamic_smem_bytes": dynamic_smem,
                "dynamic_smem_limit": DYNAMIC_SMEM_LIMIT,
                "headroom_bytes": (
                    DYNAMIC_SMEM_LIMIT - dynamic_smem
                    if dynamic_smem is not None
                    else None
                ),
                "basis": dynamic_smem_basis,
                "static_shared_bytes": resources["STATIC_SHARED"],
                "note": (
                    "cuobjdump SHARED is static allocation only; it is not "
                    "the dynamic SharedStorageV2 launch size."
                ),
                "resources": resources,
            },
            "G1_regswap_c": {
                "status": gate_statuses["G1_regswap_c"],
                "expected_roles": {
                    "W0-W3_gather": 40,
                    "W4-W7_math": 128,
                    "W8-W15_reduce": 128,
                    "W16-W19_special": 56,
                },
                "pool_registers": REGSWAP_C_POOL,
                "expected_usetmaxreg_multiset": [
                    {"kind": "DEALLOC", "value": 40, "count": 1},
                    {"kind": "DEALLOC", "value": 56, "count": 1},
                    {"kind": "TRY_ALLOC", "value": 128, "count": 2},
                ],
                "observed": usetmaxreg,
            },
            "G2_special_role_spills": {
                "status": gate_statuses["G2_special_role_spills"],
                "condition": (
                    "candidate W16/W17/W18 each has no positive LDL or STL "
                    "delta versus matched v12"
                ),
                "candidate": candidate_role_data,
                "reference_v12": reference_role_data,
                "delta": role_deltas,
                "leader_regswap_fingerprint": leader_fingerprint,
                "leader_fingerprint_condition": "candidate W16 total < v12 W16 total",
                "note": (
                    "Counts are static instructions in the top-level role "
                    "branch, not dynamic execution counts."
                ),
            },
            "G3_publish_and_l2x_bulk": {
                "status": gate_statuses["G3_publish_and_l2x_bulk"],
                "condition": (
                    "two math publish waves, four STSM each and no STS.U16 "
                    "fallback; exactly 2 S2G globally/in math and exactly "
                    "three compiler-lowered clones of the five-site G2S "
                    "source block, all inside W18"
                ),
                "math_window": {
                    "start": first_and_last_address(math_instructions)[0],
                    "end_inclusive": first_and_last_address(math_instructions)[1],
                    "STSM": math_stsm,
                    "STS.U16_diagnostic_whole_math": math_sts_u16,
                    "UBLKCP.G.S": len(s2g_instructions),
                },
                "publish_waves": waves,
                "global_main_kernel": {
                    "UBLKCP.G.S": all_s2g_count,
                    "UBLKCP.S.G": all_g2s_count,
                },
                "W18": {
                    "UBLKCP.S.G": w18_g2s,
                    "G2S_sites_per_source_block": G2S_SITES_PER_SOURCE_BLOCK,
                    "static_clone_count_expected": G2S_STATIC_CLONES_EXPECTED,
                    "static_clone_groups": g2s_clone_group_report,
                    "window": candidate_role_data["W18"],
                },
                "opcode_direction": {
                    "UBLKCP.G.S": "S2G (global destination, shared source)",
                    "UBLKCP.S.G": "G2S (shared destination, global source)",
                },
            },
            "G4_variant_b_reduce": {
                "status": "N/A",
                "reason": (
                    "Default run is REGSWAP variant C; the variant-B "
                    "reduce=120 drain<=23 gate is not applicable."
                ),
            },
        },
        "limitations": [
            (
                "G0's 231424-byte value is source-specialized evidence unless "
                "--dynamic-smem-bytes is supplied by the capture runner."
            ),
            (
                "Role spill counts use stable top-level warp-id dispatch "
                "windows and count static LDL/STL, not runtime spill traffic."
            ),
            (
                "CUTLASS 4.6.1 clones the five-call W18 G2S source block "
                "three times in SASS; G3 therefore requires exactly three "
                "separate five-site static groups, not five opcodes globally."
            ),
            (
                "The analyzer is pinned to 2960a6a and intentionally fails "
                "provenance for later source revisions."
            ),
        ],
    }
    json_path = output_dir / "stage0_gate_report.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    def gate_row(name: str, observed: str) -> str:
        return f"| {name} | {gate_statuses[name]} | {observed} |"

    md = [
        "# v15 rev3 — Stage-0 hard gates",
        "",
        f"- Overall: **{report['overall_status']}**",
        (
            "- Downstream: **CONTINUE**"
            if overall_pass
            else "- Downstream: **STOP**"
        ),
        f"- Target revision: `{TARGET_REVISION}`",
        f"- Candidate SHA256: `{candidate_sha}`",
        f"- Matched v12 SHA256: `{reference_sha or 'not in manifest'}`",
        "",
        "| Gate | Status | Observed |",
        "|---|---|---|",
        gate_row(
            "P0_provenance",
            (
                f"candidate_match={provenance_matches}; "
                f"reference_match={reference_contract_ok}; "
                f"codegen_match={codegen_contract_ok}"
            ),
        ),
        gate_row(
            "G0_smem",
            (
                f"dynamic={dynamic_smem}; cap={DYNAMIC_SMEM_LIMIT}; "
                f"static={resources['STATIC_SHARED']}"
            ),
        ),
        gate_row(
            "G1_regswap_c",
            (
                "USETMAXREG="
                + ", ".join(
                    f"{item['kind']}:{item['value']}@{item['address']}"
                    for item in usetmaxreg
                )
                + f"; pool={REGSWAP_C_POOL}"
            ),
        ),
        gate_row(
            "G2_special_role_spills",
            (
                "candidate/ref totals "
                + ", ".join(
                    f"{role}={candidate_role_data[role]['local_total']}/"
                    f"{reference_role_data[role]['local_total']}"
                    for role in ("W16", "W17", "W18")
                )
                + f"; leader={leader_fingerprint}"
            ),
        ),
        gate_row(
            "G3_publish_and_l2x_bulk",
            (
                f"math STSM={math_stsm}; S2G={all_s2g_count}; "
                f"W18 G2S={w18_g2s} "
                f"({[len(group) for group in g2s_clone_groups]} clones); waves="
                f"{[(wave['STSM'], wave['STS.U16']) for wave in waves]}"
            ),
        ),
        gate_row("G4_variant_b_reduce", "REGSWAP C default; not applicable"),
        "",
        "## Evidence interpretation",
        "",
        (
            f"- G0 uses `{dynamic_smem}` bytes with "
            f"`{DYNAMIC_SMEM_LIMIT - dynamic_smem if dynamic_smem is not None else 'unknown'}` "
            "bytes headroom. The `SHARED=1024` resource field is static "
            "shared memory and is reported separately."
        ),
        (
            "- G2 counts static `LDL`/`STL` inside the compiler's unique "
            "W16/W17/W18 top-level branch windows. Positive deltas are a "
            "hard failure; W16 strictly below v12 is the REGSWAP fired "
            "fingerprint."
        ),
        (
            "- G3 checks software issue sites only: `UBLKCP.G.S`/`.S.G` "
            "show S2G/G2S enqueue instructions, not asynchronous completion."
        ),
        "",
        "## Known limitation",
        "",
        (
            "Without `--dynamic-smem-bytes`, G0 is tied to the pinned "
            "2960a6a source hash and its compile-time struct assert. A future "
            "runner should emit the launch dynamic-SMEM value explicitly."
        ),
    ]
    (output_dir / "stage0_gate_report.md").write_text("\n".join(md) + "\n")

    generated = sorted(
        path for path in output_dir.iterdir() if path.is_file()
    )
    manifest = {
        "schema_version": 1,
        "generated": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in generated
        ],
    }
    (output_dir / "stage0_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if overall_pass else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
