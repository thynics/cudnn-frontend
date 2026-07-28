---
name: validate-dsa-b200
description: Run the repository's mandatory one-click B200 validation after completing changes to a registered SM100 DeepSeek sparse_attention_backward implementation, including canonical v0/v1 and user-named v... experimental copies. Use when DSA kernel development is ready for correctness, release performance, baseline/candidate IKET capture, span aggregation, or final validation; also use when a previous one-click run failed and the implementation has been fixed.
---

# Validate DSA on B200

Run exactly one repository command after the implementation is committed and
pushed:

```bash
./benchmark/dsa/run_b200_pipeline.sh --impl vxxx
```

Replace `vxxx` with the exact registered implementation token. Use `v0` or
`v1` for the canonical implementations. Run from the repository root. A
registered experimental token resolves to:

```text
python/cudnn/deepseek_sparse_attention/sparse_attention_backward/dsa_bwd_sm100_2cta_<token>.py
```

The selected implementation must exist, be tracked, and have no uncommitted
changes. Commit and push it before invoking the command so the locked remote
`git pull --ff-only` sees the same bytes.

Do not upload or SCP source, inspect or acquire a B200 manually, manage
allocations or containers, invoke harness stages separately, precompile the
kernel, or aggregate traces by hand. The command owns:

- the global lock and its release on every exit path;
- `git pull --ff-only` of the fixed Computelab worktree;
- creation or reuse of the manager-owned four-hour B200 service;
- the long-lived Docker container and mounted scratch dependencies;
- correctness, uninstrumented baseline/candidate performance, both IKET
  captures, span aggregation, and the two comparison tables;
- lightweight result download and exact failure-code propagation.

Do not copy an experimental candidate over canonical v1 for validation. Pass
its token directly with `--impl`; the pipeline maps that exact module into its
private harness snapshot and records its path and SHA256.

Wait for the command to exit. Success requires exit code zero and
`DSA_PIPELINE_PASSED`. Report the four-pattern correctness result, baseline
and candidate release latency, ratio, and the emitted `DSA_TABLES`,
`DSA_TABLES_JSON`, and `DSA_VALIDATION` paths.

On failure, preserve the reported stage, error, and exit code. Read only the
status files and bounded log tail downloaded by the command. Fix the
implementation, commit and push it, then invoke the same one-click command
again. Do not replace the failed stage with an ad hoc command.

Treat IKET spans as diagnostic software-annotation intervals. Use the
uninstrumented performance result—not trace timing—for performance decisions.
