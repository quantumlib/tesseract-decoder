# Detector-order semantics benchmark

This directory makes the numerical claims in
[PR #306](https://github.com/quantumlib/tesseract-decoder/pull/306)
reproducible. It compares the PR's merge commit against the exact `main`
parent merged into it:

- baseline: `4fd104d88205b9c5424c94b2da28e684c113ebcf`
- candidate: `9d62d5d317892c706c73d899dae87644911f199f`

The benchmark uses the three circuits, two detector-order methods, seeds, and
decoder settings from the PR description. It writes the native stats from
every invocation before producing a Markdown table and an SVG comparison
plot. The checked-in files under `reference/` are regenerated from the counts
and timings reported in the PR; they are a test fixture, not a new benchmark
run.

## Reproduce

From a clean repository checkout:

```bash
benchmarking/detector_order_semantics/reproduce.sh \
  --output /path/to/new-run \
  --hardware-description "CPU model and machine description"
```

The default is the full 100,000-shot benchmark. For a plumbing check, add
`--shots 10`. The script:

1. resolves and archives both pinned source revisions;
2. builds `//src:tesseract` from each archive with Bazel;
3. hashes the source revisions, binaries, and circuits;
4. runs baseline and candidate with identical sampling/order seeds;
5. retains native JSON, stdout, stderr, command lines, and external wall time;
6. writes `results.jsonl`, `results.md`, and `comparison.svg`.

`--threads` defaults to 1. Keep the thread count fixed when interpreting timing
comparisons, and record enough hardware detail to identify the machine used for
the run.

An already-built pair of binaries can be used without source builds:

```bash
benchmarking/detector_order_semantics/reproduce.sh \
  --output /path/to/new-run \
  --hardware-description "..." \
  --baseline-bin /path/to/baseline/tesseract \
  --candidate-bin /path/to/candidate/tesseract
```

Both binary options must be supplied together. Their SHA-256 digests are
recorded, but a run using external binaries cannot independently bind those
binaries to the pinned source commits; this limitation is explicit in the
manifest.

To regenerate only the report from an existing run:

```bash
python3 benchmarking/detector_order_semantics/report.py \
  --input-dir /path/to/run/raw \
  --output-dir /path/to/run
```

## Outputs and interpretation

`results.md` reports failures as `num_errors + num_low_confidence` and Wilson
95% intervals for each shot failure probability. Relative error reduction is
`1 - p_candidate / p_baseline`; its interval is the log-relative-risk interval
used by the original attachment. The two-sided p-value is Fisher's exact test
treating the aggregate samples as independent.

Identical sample seeds make the underlying circuit samples paired, but the
native aggregate stats do not retain per-shot discordance. A paired McNemar
test therefore cannot be reconstructed from these artifacts. The report says
"Fisher (independent)" rather than implying a paired significance test.

The native `total_time_seconds` field is the sum of measured per-shot decode
durations. It is not process CPU time. The report calls it "summed decode
time" and separately preserves external process wall time and throughput.
