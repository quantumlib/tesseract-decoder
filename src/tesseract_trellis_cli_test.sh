#!/usr/bin/env bash
set -euo pipefail

binary="$(find -L "${TEST_SRCDIR}" -type f -path '*/src/tesseract_trellis' -perm -111 | head -n 1)"
if [[ -z "${binary}" ]]; then
  echo "Failed to find tesseract_trellis binary in runfiles." >&2
  exit 1
fi

work_dir="${TEST_TMPDIR}/trellis_cli"
mkdir -p "${work_dir}"

one_obs_dem="${work_dir}/one_obs.dem"
cat >"${one_obs_dem}" <<'EOF'
error(0.1) D0 L0
detector(0, 0, 0) D0
EOF

two_obs_dem="${work_dir}/two_obs.dem"
cat >"${two_obs_dem}" <<'EOF'
error(0.1) D0 L0
error(0.1) D0 L1
detector(0, 0, 0) D0
EOF

shots_01="${work_dir}/shots.01"
cat >"${shots_01}" <<'EOF'
0
1
1
EOF

stats_json="${work_dir}/stats.json"
stdout_txt="${work_dir}/stdout.txt"
"${binary}" \
  --dem "${one_obs_dem}" \
  --in "${shots_01}" \
  --in-format 01 \
  --max-errors 1 \
  --threads 1 \
  --stats-out "${stats_json}" \
  >"${stdout_txt}"
grep -q 'num_shots = 3' "${stdout_txt}"
! grep -q 'num_errors' "${stdout_txt}"
grep -q '"num_errors":null' "${stats_json}"

for ranking_mode in mass future-detcost future-active-detcost; do
  probs_bin="${work_dir}/${ranking_mode}.bin"
  "${binary}" \
    --dem "${one_obs_dem}" \
    --in "${shots_01}" \
    --in-format 01 \
    --threads 1 \
    --ranking-mode "${ranking_mode}" \
    --obs-probs-out "${probs_bin}" \
    >"${work_dir}/${ranking_mode}.stdout"
  [[ "$(wc -c <"${probs_bin}" | tr -d ' ')" == "24" ]]
done

set +e
"${binary}" \
  --dem "${one_obs_dem}" \
  --in "${shots_01}" \
  --in-format 01 \
  --dem-out "${work_dir}/unsupported.dem" \
  >"${work_dir}/dem_out.stdout" \
  2>"${work_dir}/dem_out.stderr"
dem_out_status=$?
set -e
[[ "${dem_out_status}" -ne 0 ]]
grep -q -- '--dem-out is not supported by tesseract_trellis' "${work_dir}/dem_out.stderr"

set +e
"${binary}" \
  --dem "${two_obs_dem}" \
  --in "${shots_01}" \
  --in-format 01 \
  --threads 1 \
  >"${work_dir}/multi_obs.stdout" \
  2>"${work_dir}/multi_obs.stderr"
multi_obs_status=$?
set -e
[[ "${multi_obs_status}" -ne 0 ]]
grep -q 'supports at most one observable' "${work_dir}/multi_obs.stderr"

set +e
"${binary}" \
  --dem "${one_obs_dem}" \
  --in "${shots_01}" \
  --in-format 01 \
  --out "${work_dir}/missing_dir/predictions.01" \
  --out-format 01 \
  >"${work_dir}/out_open.stdout" \
  2>"${work_dir}/out_open.stderr"
out_open_status=$?
set -e
[[ "${out_open_status}" -ne 0 ]]
grep -q 'Could not open the file' "${work_dir}/out_open.stderr"
