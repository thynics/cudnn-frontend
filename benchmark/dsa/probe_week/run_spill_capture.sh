#!/usr/bin/env bash
set -Eeuo pipefail

repo="${DSA_REPO:-/home/scratch.longcheng_gpu/cudnn-frontend-thynics}"
impl="${DSA_SPILL_IMPL:-vkq6v}"
out="${DSA_SPILL_OUT:-/home/scratch.longcheng_gpu/dsa-vkq6v-spill/${impl}_release45}"
python_bin="/home/scratch.longcheng_gpu/cudnn-frontend/.venv/bin/python"
parser="/home/scratch.longcheng_gpu/dsa-vkq6v-spill/tools/sass_spill_to_py_locs.py"

mkdir -p -- "${out}/dump" "${out}/cache"
cd -- "${repo}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${repo}/python:${repo}/test/python:${repo}"
export CUTE_DSL_KEEP=ptx,cubin
export CUTE_DSL_LINEINFO=True
export CUTE_DSL_DUMP_DIR="${out}/dump"
export CUTE_DSL_CACHE_DIR="${out}/cache"
export CUTE_DSL_NO_CACHE=True
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
# The managed worker is long-lived; never inherit another baseline experiment.
export DSA_BL_QDO_STAGE=1
export DSA_BL_K_STAGE=1
export DSA_BL_HALFK=0
export DSA_BL_KSTAGE2=0
export DSA_BL_OVPAD=0
unset DSA_DEV_IKET DKG_IKET_INSTRUMENTATION_METHOD IKET_STANDALONE_SITE_PACKAGES

printf 'SPILL_CAPTURE_ENV DSA_BL_KSTAGE2=%s CUTE_DSL_NO_CACHE=%s\n' \
  "${DSA_BL_KSTAGE2}" "${CUTE_DSL_NO_CACHE}"

"${python_bin}" benchmark/dsa/allinone/compile_capture.py \
  --impl "${impl}" \
  --out "${out}" \
  --allow-cubin-only

dump_dir="${out}/logs/codegen/compile"
mapfile -d '' -t cubins < <(
  find "${dump_dir}" -maxdepth 1 -type f -name '*.cubin' -print0
)
if ((${#cubins[@]} == 0)); then
  echo "ERROR: ${impl} compile produced no CUBIN" >&2
  exit 1
fi
printf 'SPILL_CAPTURE_CUBINS count=%d\n' "${#cubins[@]}"

for cubin in "${cubins[@]}"; do
  stem="${cubin%.cubin}"
  sass="${stem}.lineinfo.sass"
  /usr/local/cuda/bin/nvdisasm -g -c "${cubin}" >"${sass}"
  /usr/local/cuda/bin/cuobjdump --dump-resource-usage "${cubin}" \
    >"${stem}.resource_usage.txt"
  readelf -S -W "${cubin}" >"${stem}.sections.txt"
  readelf -Ws -W "${cubin}" >"${stem}.symbols.txt"
  grep -qE '//## File "[^"]+", line [0-9]+' "${sass}"
  grep -qE '/\*(0x)?[[:xdigit:]]+\*/' "${sass}"
  python3 "${parser}" "${sass}" --source-root "${repo}" \
    -o "${stem}.spill_product.json" \
    | tee "${stem}.spill_product.log"
  python3 "${parser}" "${sass}" --source-root "${repo}" \
    --include-non-python -o "${stem}.spill_full.json" \
    | tee "${stem}.spill_full.log"
done

echo "SPILL_CAPTURE_OK ${out}"
