#!/usr/bin/env python3
"""Reproducible runner: Bivariate Bicycle codes (SI1000, p=0.001).

  bb_d6        [[72,12,6]]   r=6,  q=144
  bb_d12_gross [[144,12,12]] r=12, q=288 (Gross code)

Fresh-runs [O] Tesseract-BP-OSD, [C] Tesseract-Cascade and [M] Multi-Pass
Tesseract (beam 8) and upserts them into benchmarking/data/bivariate_bicycle.json.
Headline numbers and `stage_breakdown` come from the same `--stats-out` run.
[B]/[G]/[M5]/[M20] records were produced earlier with the command lines stored
in each record and are not re-run by default (pass e.g. --codes M --tags
bb_d12_gross to refresh one).

  bazel build --jobs=1 //src:bp //src:tesseract
  python3 benchmarking/runners/run_bivariate_bicycle.py --codes O,C,M --threads 64
  python3 benchmarking/runners/normalize_datasets.py
"""

import argparse
import pathlib

import bench_common as bc
from run_color_codes import MP_DEM_RECIPE, write_mp_dem

DATA_OUT = bc.DATA_DIR / "bivariate_bicycle.json"
FAMILY = "bivariate_bicycle"
CIRCUITS = [
    {
        "tag": "bb_d6",
        "label": "bivariate bicycle [[72,12,6]], d=6, q=144",
        "d": 6,
        "r": 6,
        "q": 144,
        "n": 72,
        "k": 12,
        "path": "testdata/bivariatebicyclecodes/r=6,d=6,p=0.001,noise=si1000,c=bivariate_bicycle_Z,nkd=[[72,12,6]],q=144,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim",
    },
    {
        "tag": "bb_d12_gross",
        "label": "Gross bivariate bicycle [[144,12,12]], d=12, q=288",
        "d": 12,
        "r": 12,
        "q": 288,
        "n": 144,
        "k": 12,
        "path": "testdata/bivariatebicyclecodes/r=12,d=12,p=0.001,noise=si1000,c=bivariate_bicycle_Z,nkd=[[144,12,12]],q=288,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim",
    },
]
SHOTS = {
    "[O]": {"bb_d6": 100000, "bb_d12_gross": 50000},
    "[C]": {"bb_d6": 100000, "bb_d12_gross": 50000},
    "[M]": {"bb_d6": 100000},  # Gross [M] (beam 8, 50k shots) already recorded.
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default="O,C,M")
    ap.add_argument("--tags", default=",".join(c["tag"] for c in CIRCUITS))
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--dem-dir", default="/tmp/tesseract_bench_dems")
    args = ap.parse_args()

    commit = bc.git_commit()
    codes = [f"[{c.strip()}]" for c in args.codes.split(",")]
    tags = args.tags.split(",")
    ds = bc.load_dataset(DATA_OUT)
    ds.update({"family": FAMILY, "git_commit": commit})
    for c in CIRCUITS:
        if c["tag"] not in tags:
            continue
        meta = {k: v for k, v in c.items() if k != "path"}
        meta.update({"family": FAMILY, "p": 0.001, "circuit_path": c["path"]})
        for code in codes:
            shots = args.shots or SHOTS[code].get(c["tag"])
            if not shots:
                continue
            extra = []
            if code == "[M]":
                dem_p = pathlib.Path(args.dem_dir) / f"{c['tag']}_mp.dem"
                write_mp_dem(c["path"], dem_p)
                extra = ["--dem", str(dem_p)]
            cmd, stats = bc.run_decoder(
                code,
                ["--circuit", c["path"]],
                ["--sample-num-shots", str(shots), "--sample-seed", str(args.seed)],
                args.threads,
                extra,
            )
            rec = bc.make_record(meta, code, cmd, stats, args.seed, commit)
            if code == "[M]":
                rec["dem_recipe"] = MP_DEM_RECIPE
            rec["raw_stats_path"] = bc.archive_raw(FAMILY, c["tag"], code, cmd, stats)
            bc.upsert(ds, rec)
            bc.save_dataset(DATA_OUT, ds, bc.record_order([x["tag"] for x in CIRCUITS]))
            print(
                f"  {c['tag']} {code}: {rec['num_errors']}/{rec['num_shots']} "
                f"cpu={rec['cpu_time_seconds']:.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
