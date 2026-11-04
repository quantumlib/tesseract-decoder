# Tesseract-BP-OSD & Tesseract-Cascade Benchmarking Suite

This directory holds the reproducible benchmark runners and the consolidated
JSON datasets used for the tradeoff plots and the paper. It covers four code
families:

1. **Color Codes**: superdense color code, $d \in \{3,5,7,9,11\}$, $r=d$, SI1000 $p=0.001$
2. **Bivariate Bicycle (BB) Codes**: `[[72,12,6]]` ($r=6$, $q=144$) and the **Gross code** `[[144,12,12]]` ($r=12$, $q=288$)
3. **Mitten Code**: `[[150,30,10]]` ($r=10$, $q=270$), from a DEM only
4. **Hyperbolic Octagon Cat-Flagged**: `iter=32` $[[144,8]]$, `iter=40` $[[288,12]]$, `iter=58` $[[1152,36]]$, `iter=109` $[[3600,100]]$, $r=16$, $p=0.001$, $p_\text{seam}=0.01$

---

## Decoder Codes

Each code has exactly one definition in every dataset, and there is at most one
record per `(tag, decoder_code)`.

| Code | Label | Binary | Flags |
| :---: | :--- | :--- | :--- |
| `[B]` | Baseline Tesseract | `bazel-bin/src/tesseract` | `--beam 20 --beam-climbing --no-revisit-dets --num-det-orders 16 --pqlimit 200000|1000000` (the exact command is stored in each record) |
| `[G]` | Gari | `bazel-bin/src/tesseract` | `--dem <gari.dem> --detector-orders <gari_ord.json> --beam 5 --beam-climbing [--no-revisit-dets] --pqlimit 1000000` |
| `[M]` | Multi-Pass Tesseract (canonical) | `bazel-bin/src/tesseract` | `--dem <mp.dem> --multipass --num-passes 2 --beam 8 --beam-climbing --no-revisit-dets --pqlimit 100000` |
| `[M5]` | Multi-Pass Tesseract, beam 5 (ablation) | `bazel-bin/src/tesseract` | same as `[M]` but `--beam 5` |
| `[M20]` | Multi-Pass Tesseract, beam 20 (ablation) | `bazel-bin/src/tesseract` | same as `[M]` but `--beam 20 --pqlimit 1000000` (or `200000`) |
| `[O]` | Tesseract-BP-OSD | `bazel-bin/src/bp` | `--multipass --num-passes 2 --random-schedule --max-iter 20 --normalization-factor 0.75 --osd-order 10 --osd-weight 2` |
| `[C]` | Tesseract-Cascade | `bazel-bin/src/bp` | `[O]` flags + `--tesseract-fallback` |
| `[R]`, `[T]` | Relay-BP, Telescoping-BP | external | external reference points (color codes and Mitten only) |
| `[AQ]` | AlphaQubit reference | n/a | hyperbolic only |

`<mp.dem>` is built with `demutil.annotate_detector_bases(circuit.detector_error_model(decompose_errors=False, flatten_loops=True, allow_gauge_detectors=True, approximate_disjoint_errors=1.0))`.

---

## Layout

```text
benchmarking/
├── README.md
├── data/
│   ├── color_codes.json          # d=3..11 tradeoff records + distance_sweep (copies of [O]/[C] at d=5..11)
│   ├── bivariate_bicycle.json    # bb_d6 and bb_d12_gross
│   ├── mitten.json               # mitten_d10
│   ├── hyperbolic_octagon.json   # iter=32/40/58/109: [AQ], cluster [O]/[C] + local stage telemetry
│   └── raw/<family>/<tag>_<code>.json   # raw `--stats-out` JSON + command line of every fresh run
└── runners/
    ├── bench_common.py           # decoder registry, stage-breakdown math, record/upsert helpers
    ├── run_color_codes.py        # fresh [O]/[C]/[M] runs → color_codes.json
    ├── run_bivariate_bicycle.py  # fresh [O]/[C]/[M] runs → bivariate_bicycle.json
    ├── run_mitten.py             # fresh [O]/[C]/[M] runs → mitten.json
    ├── run_hyperbolic_octagon.py # [AQ] + cluster headline rows, local [O]/[C] telemetry
    └── normalize_datasets.py     # recode/dedupe/validate all datasets (run last)
```

## Provenance Rules

* **Fresh runs** (`source: "fresh run via benchmarking/runners ..."`): the
  headline numbers (`num_shots`, `num_errors`, `cpu_time_seconds`) and the
  `stage_breakdown` come from the **same** `--stats-out` JSON
  (`stage_breakdown.telemetry_same_run_as_headline == true`). The raw JSON is in
  `raw_stats_path`.
* **Legacy records**: color-code `[B]/[G]/[M5]/[M20]/[R]/[T]` are imported from
  `benchmarking/multipass/results.json`. BB and Mitten `[B]/[G]/[M5]/[M20]/[R]/[T]` come
  from earlier local runs, and each record keeps its exact `command_line`. These records
  have no `stage_breakdown`.
* **Hyperbolic `[O]/[C]`**: the headline is the 330k-shot `dripto3` cluster
  suite. The exact cluster command was not archived. The `stage_breakdown` comes from a
  **separate** local run (`telemetry_same_run_as_headline == false`), and its own
  `telemetry_command_line`, `telemetry_num_shots`, `telemetry_num_errors`,
  `telemetry_seed`, and `telemetry_cpu_seconds` are stored with it. Quote stage percentages
  from that telemetry run only.
* `ler_per_round` uses the sinter formula $0.5\,(1-(1-2p_\text{shot})^{1/r})$
  in every file. Hyperbolic records also carry `ler_per_round_per_logical`
  (pieces $= r\cdot k$).
* `shot_error_rate = (num_errors + num_low_confidence) / num_shots`. A shot the decoder
  reports as low-confidence (no prediction) counts as a failure. This is nonzero only for
  some legacy `[T]` records (e.g. cc_d7, cc_d9).

## Running

### External circuit inputs (not committed)

The hyperbolic and Mitten inputs are not in git (about 24 MB in total). Copy them into the repo root
before running the hyperbolic or Mitten runners:

| Path expected by the runners | Source |
| :--- | :--- |
| `cat-flagged-circuits-octagon-L2-aug13-2026/` | x20: `/google/data/rw/users/ar/ariash/tesseract-bp/benchmarks/circuits/` |
| `testdata/mittencodes/` (`mitten_mp.dem`, `mitten_gari.dem`, `mitten_gari_ord.json`) | https://github.com/a7b/yarn, also mirrored at the x20 path above |

Color-code and BB circuits are already in `testdata/`.

Per `AGENTS.md`: builds use **one core**, and benchmarks use all cores (`--threads 64`).
The runners need `stim` and `tesseract_decoder`, for example via the Bazel venv:

```bash
bazel build --jobs=1 //src:bp //src:tesseract //src/py:multi_pass_bindings_test
export RUNFILES_DIR=$PWD/bazel-bin/src/py/multi_pass_bindings_test.runfiles
PY=$RUNFILES_DIR/_main/src/py/_multi_pass_bindings_test.venv/bin/python3

$PY benchmarking/runners/run_color_codes.py --codes O,C,M --threads 64
$PY benchmarking/runners/run_bivariate_bicycle.py --codes O,C,M --threads 64
$PY benchmarking/runners/run_mitten.py --codes O,C,M --threads 64
$PY benchmarking/runners/run_hyperbolic_octagon.py --codes O,C --threads 64
$PY benchmarking/runners/normalize_datasets.py          # must report 0 validation errors
```

Each runner upserts its records by `(tag, decoder_code)`. Use `--tags`, `--codes`, or
`--shots` to refresh a subset.

---

## Per-Stage Telemetry (`bazel-bin/src/bp --stats-out`)

The raw counters for `[O]` (`--multipass`) and `[C]` (`--multipass --tesseract-fallback`) are:

- `stage1_bp_converged_shots`: both BP passes converged without OSD (`lc0 == 0 && lc1 == 0`).
- `stage1_osd_shots`: at least one pass needed OSD. Together with the previous counter this sums to `num_shots`.
- `num_stage2_triggered`: shots flagged low-confidence after Stage 1. This is always 0 for `[O]`.
- `num_stage2_by_reason[0..5]`: Stage-2 triggers split by low-confidence reason code. Reason 5 is
  the only one raised on BP-converged shots and is also emitted as `num_stage2_from_bp_converged`.
- `num_fallback_triggered`: Stage-2 shots escalated to Stage 3 (Tesseract). This is a subset of Stage 2.
- `num_fallback_substituted`: shots whose observable prediction was replaced by Stage 2/3.
- `stage1_time_seconds + stage2_time_seconds + stage3_time_seconds == total_time_seconds` (CPU time).
- `t_bp`, `t_post`, `t_construct`, `t_deinterleave`: Stage-1 component profile (BP vs OSD).

The derived `stage_breakdown` in every dataset contains:

- Inclusive percentages: `pct_shots_stage1_bp`, `pct_shots_stage1_osd`, `pct_shots_stage2`, `pct_shots_stage3`.
- An exact exclusive partition by the stage that resolved each shot: `resolved_stage1_bp_shots`,
  `resolved_stage1_osd_shots`, `resolved_stage2_shots`, `resolved_stage3_shots`, plus the
  `pct_shots_resolved_*` versions. The four parts sum to the shot count.
- A CPU split that sums to 100%: `pct_cpu_stage1_bp`, `pct_cpu_stage1_osd`, `pct_cpu_stage2`,
  `pct_cpu_stage3`. Stage-1 time is divided between BP and OSD in the ratio `t_bp : t_post`.
