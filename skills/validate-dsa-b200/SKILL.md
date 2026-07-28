---
name: validate-dsa-b200
description: Run the repository's mandatory one-click B200 validation after completing changes to the SM100 DeepSeek sparse_attention_backward baseline, v0, or v1 implementation. Use when DSA kernel development is ready for correctness, release performance, baseline/candidate IKET capture, span aggregation, or final validation; also use when a previous one-click run failed and the implementation has been fixed.
---

# Validate DSA on B200

Run exactly one repository command after the implementation is committed and
pushed:

```bash
./benchmark/dsa/run_b200_pipeline.sh --impl v1
```

Use `--impl v0` only when validating v0. Run from the repository root.

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
