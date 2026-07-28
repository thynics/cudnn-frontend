---
name: merge-dsa-canonical-locked
description: Promote or merge an experimental SM100 DSA backward implementation into canonical v0 or v1 under a per-target exclusive lock. Use whenever a user asks to “合入 v1 主文件”, “merge/promote a candidate into canonical v0/v1”, or otherwise modify a canonical DSA implementation from an experiment; serialize the complete edit, commit, push, and mandatory B200 validation transaction.
---

# Merge DSA Canonical with a Lock

Acquire the canonical target lock before reading it for a merge or changing
it. Use `v0` or `v1`, matching the file being promoted into:

```bash
python3 skills/merge-dsa-canonical-locked/scripts/hold_dsa_canonical_lock.py v1 --hold
```

Run the command from the repository root in a persistent interactive process.
Wait for `DSA_CANONICAL_LOCK_ACQUIRED` before doing any merge work. Keep that
process alive throughout the transaction. Never treat
`DSA_CANONICAL_LOCK_WAIT` as ownership.

The locks are independent: a v0 promotion does not block v1, while every v1
promotion on the host uses the same v1 lock. The OS releases the lock if the
holder exits unexpectedly. An unlocked file may retain stale owner text after
an ungraceful exit; lock ownership, not file contents, is authoritative.

## Locked transaction

After acquisition:

1. Re-read `git status`, the canonical file, and the candidate. State may have
   changed while waiting. If the canonical file already has overlapping
   uncommitted edits, preserve them and stop for direction rather than
   overwriting them.
2. Apply only the intended candidate delta to:

   ```text
   python/cudnn/deepseek_sparse_attention/sparse_attention_backward/dsa_bwd_sm100_2cta_<v0-or-v1>.py
   ```

   Use a focused patch. Do not replace the whole canonical module merely to
   validate an experiment.
3. Run proportional static checks and inspect the final diff.
4. Commit only the in-scope promotion, then push it.
5. Read `skills/validate-dsa-b200/SKILL.md` and run its one-command pipeline
   with the canonical token. Keep this lock held until the command exits with
   `DSA_PIPELINE_PASSED`. If validation fails and a fix is in scope, keep the
   lock while fixing, committing, pushing, and rerunning the same command.
6. Release the holder by sending exactly:

   ```text
   release
   ```

   Wait for `DSA_CANONICAL_LOCK_RELEASED`. Release before asking the user for
   blocking input or ending the task; never leave an idle holder behind.

For a single non-editing command that must be serialized, the script can run
it directly:

```bash
python3 skills/merge-dsa-canonical-locked/scripts/hold_dsa_canonical_lock.py v1 -- command arg
```

This command form is not sufficient for a multi-step promotion; use
`--hold` for the complete transaction.
