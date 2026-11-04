#!/usr/bin/env python3
"""Reproducible runner: Mitten code [[150,30,10]] (d=10, r=10, q=270, p=0.001).

Only a DEM is available (testdata/mittencodes/mitten_mp.dem, basis-annotated),
so shots are sampled once with `stim.DetectorErrorModel.compile_sampler(seed)`
into b8 files and fed to every decoder with `--in/--obs_in`.

Fresh-runs [O] Tesseract-BP-OSD, [C] Tesseract-Cascade and [M] Multi-Pass
Tesseract (beam 8) and upserts them into benchmarking/data/mitten.json.
Headline numbers and `stage_breakdown` come from the same `--stats-out` run.

  bazel build --jobs=1 //src:bp //src:tesseract
  python3 benchmarking/runners/run_mitten.py --codes O,C,M --threads 64
  python3 benchmarking/runners/normalize_datasets.py
"""

import argparse
import pathlib

import bench_common as bc

DATA_OUT = bc.DATA_DIR / "mitten.json"
FAMILY = "mitten"
DEM = "testdata/mittencodes/mitten_mp.dem"
META = {
    "tag": "mitten_d10",
    "label": "mitten [[150,30,10]], d=10, q=270",
    "family": FAMILY,
    "d": 10,
    "r": 10,
    "q": 270,
    "n": 150,
    "k": 30,
    "p": 0.001,
    "circuit_path": DEM,
}
SHOTS = {"[O]": 100000, "[C]": 100000, "[M]": 20000}


def write_shots(n: int, seed: int, out_dir: pathlib.Path):
    import stim

    dem = stim.DetectorErrorModel.from_file(bc.REPO_ROOT / DEM)
    dets, obs, _ = dem.compile_sampler(seed=seed).sample(shots=n, bit_packed=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = out_dir / f"mitten_dets_n{n}_s{seed}.b8"
    of = out_dir / f"mitten_obs_n{n}_s{seed}.b8"
    stim.write_shot_data_file(data=dets, path=str(df), format="b8", num_detectors=dem.num_detectors)
    stim.write_shot_data_file(data=obs, path=str(of), format="b8", num_observables=dem.num_observables)
    return df, of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default="O,C,M")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--shot-dir", default="/tmp/tesseract_bench_shots")
    args = ap.parse_args()

    commit = bc.git_commit()
    ds = bc.load_dataset(DATA_OUT)
    ds.update(
        {
            "family": FAMILY,
            "description": "Mitten code [[150,30,10]], r=10, SI1000 p=0.001 (DEM-sampled shots).",
            "git_commit": commit,
        }
    )
    for code in [f"[{c.strip()}]" for c in args.codes.split(",")]:
        shots = args.shots or SHOTS[code]
        df, of = write_shots(shots, args.seed, pathlib.Path(args.shot_dir))
        cmd, stats = bc.run_decoder(
            code,
            ["--dem", DEM],
            ["--in", str(df), "--in-format", "b8", "--obs_in", str(of), "--obs-in-format", "b8"],
            args.threads,
        )
        rec = bc.make_record(META, code, cmd, stats, args.seed, commit)
        rec["shot_generation"] = (
            f"stim.DetectorErrorModel.from_file('{DEM}').compile_sampler(seed={args.seed})"
            f".sample(shots={shots}) -> b8"
        )
        rec["raw_stats_path"] = bc.archive_raw(FAMILY, META["tag"], code, cmd, stats)
        bc.upsert(ds, rec)
        bc.save_dataset(DATA_OUT, ds, bc.record_order([META["tag"]]))
        print(
            f"  mitten {code}: {rec['num_errors']}/{rec['num_shots']} cpu={rec['cpu_time_seconds']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
