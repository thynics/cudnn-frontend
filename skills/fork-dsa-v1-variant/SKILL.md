---
name: fork-dsa-v1-variant
description: Create an isolated, user-named experimental SM100 DSA backward variant as an exact copy of the repository's current v1 implementation. Use when a user asks to explore, develop, optimize, or try a code such as “基于 v1 探索 vxxx”, “从 v1 创建 vfoo”, or “fork current v1 as a new candidate”; the user supplies the `v...` variant token.
---

# Fork a DSA v1 Variant

Create the experiment before changing kernel code:

```bash
python3 skills/fork-dsa-v1-variant/scripts/fork_v1_variant.py vxxx
```

Run from the repository root and replace `vxxx` with the user's token. Accept
the script's lowercase normalization. The token must start with `v` and
contain only letters, digits, or underscores.

The script copies:

```text
python/cudnn/deepseek_sparse_attention/sparse_attention_backward/dsa_bwd_sm100_2cta_v1.py
```

to:

```text
python/cudnn/deepseek_sparse_attention/sparse_attention_backward/dsa_bwd_sm100_2cta_<token>.py
```

It refuses `v0`, `v1`, an invalid token, or an existing target. Never overwrite
an existing experiment. Report the emitted source path, target path, and
SHA256. Immediately after creation the two hashes must match.

Keep the copied contents byte-for-byte identical at the fork point. In
particular, retain the internal v1 class names and harness aliases initially;
the Python module namespace isolates them, and preserving them avoids an
unrelated mechanical change. The new module filename is the experiment name.

After the fork:

1. Treat the new variant file as the only implementation under development.
2. Do not edit baseline, v0, v1, dispatch, or public interfaces merely to
   create or develop the experiment.
3. Apply every requested optimization, review, and diagnostic change to the
   variant file. Compare against the captured source SHA when lineage matters.
4. Do not claim the variant was tested if a command actually selected the
   unchanged canonical v1.

After changing the variant, commit and push it, read
`skills/validate-dsa-b200/SKILL.md`, and validate the exact token with:

```bash
./benchmark/dsa/run_b200_pipeline.sh --impl vxxx
```

The validation entry resolves the token to the new module, maps it into a
private harness snapshot, and records its source path and SHA256. Never copy
the experiment over canonical v1 merely to test it, and never substitute
manual B200, Docker, synchronization, or harness-stage commands.

Creating an untouched fork is setup, not a completed kernel modification; do
not consume a B200 validation run solely for the byte-identical copy.
