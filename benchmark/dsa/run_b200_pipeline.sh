#!/usr/bin/env bash
set -Eeuo pipefail

# Stable compatibility entrypoint.  The all-in-one runner is authoritative:
# it owns source identity checks, v2 gpu-pool acquisition/guardian lifetime,
# correctness, fair release performance, optional IKET, and result validation.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
exec "${script_dir}/run_v_w3_2_allinone.sh" "$@"
