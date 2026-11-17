#!/usr/bin/env python3
"""Run all Baseline, GARI, and Multi-Pass benchmarks and generate tradeoff plots."""

import argparse
import glob
import json
import pathlib
import shutil
import subprocess
import sys

RF = "bazel-bin/benchmarking/sparsify_errors/plot.runfiles"
sys.path[:0] = [f"{RF}/_main/src", f"{RF}/_main/src/py"] + glob.glob(
    f"{RF}/*/site-packages"
)

import stim
from tesseract_decoder import demutil

import make_plots

SCRATCH = pathlib.Path(__file__).resolve().parent
DEM_CACHE = SCRATCH / "dem_cache"

CIRCUITS = [
    {
        "tag": "cc_d3",
        "label": "cc, d=3",
        "family": "cc",
        "d": 3,
        "r": 3,
        "q": 13,
        "path": "testdata/colorcodes/r=3,d=3,p=0.001,noise=si1000,c=superdense_color_code_Z,q=13,gates=cz.stim",
        "shots": {
            "baseline_b20": 30000,
            "gari_b5": 50000,
            "mp_2p_b5": 50000,
            "mp_2p_b20": 50000,
        },
    },
    {
        "tag": "cc_d5",
        "label": "cc, d=5",
        "family": "cc",
        "d": 5,
        "r": 5,
        "q": 37,
        "path": "testdata/colorcodes/r=5,d=5,p=0.001,noise=si1000,c=superdense_color_code_Z,q=37,gates=cz.stim",
        "shots": {
            "baseline_b20": 20000,
            "gari_b5": 30000,
            "mp_2p_b5": 150000,
            "mp_2p_b20": 100000,
        },
    },
    {
        "tag": "cc_d7",
        "label": "cc, d=7",
        "family": "cc",
        "d": 7,
        "r": 7,
        "q": 73,
        "path": "testdata/colorcodes/r=7,d=7,p=0.001,noise=si1000,c=superdense_color_code_Z,q=73,gates=cz.stim",
        "shots": {
            "baseline_b20": 8000,
            "gari_b5": 50000,
            "mp_2p_b5": 200000,
            "mp_2p_b20": 100000,
        },
    },
    {
        "tag": "cc_d9",
        "label": "cc, d=9",
        "family": "cc",
        "d": 9,
        "r": 9,
        "q": 121,
        "path": "testdata/colorcodes/r=9,d=9,p=0.001,noise=si1000,c=superdense_color_code_Z,q=121,gates=cz.stim",
        "shots": {
            "baseline_b20": 18000,
            "gari_b5": 50000,
            "mp_2p_b5": 150000,
            "mp_2p_b20": 150000,
        },
    },
    {
        "tag": "bb_d6",
        "label": "bb, d=6, q=144",
        "family": "bb",
        "d": 6,
        "r": 6,
        "q": 144,
        "path": "testdata/bivariatebicyclecodes/r=6,d=6,p=0.001,noise=si1000,c=bivariate_bicycle_Z,nkd=[[72,12,6]],q=144,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim",
        "shots": {
            "baseline_b20": 6000,
            "gari_b5": 50000,
            "mp_2p_b5": 150000,
            "mp_2p_b20": 50000,
        },
    },
]


def main():
    parser = argparse.ArgumentParser(
        description="Run all Baseline, GARI, and Multi-Pass benchmarks and generate plots."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=46,
        help="Number of worker threads for shot decoding (default: 46).",
    )
    args = parser.parse_args()

    DEM_CACHE.mkdir(parents=True, exist_ok=True)
    out_file = SCRATCH / "results.json"
    results = []

    try:
        for cinfo in CIRCUITS:
            tag = cinfo["tag"]
            ckt_path = cinfo["path"]
            ckt = stim.Circuit.from_file(ckt_path)

            gari_dem = demutil.gari.circuit_to_gari(
                ckt,
                prior_function=demutil.gari.tesseract_xor_prior_probabilities,
            )
            gari_dem_p = DEM_CACHE / f"{tag}_gari.dem"
            gari_ord_p = DEM_CACHE / f"{tag}_gari_ord.json"
            gari_dem.to_file(gari_dem_p)
            gari_ord_p.write_text(
                json.dumps(
                    demutil.gari.build_detector_orders(ckt, gari_dem, 1, seed=0)
                )
            )

            src_dem = ckt.detector_error_model(
                decompose_errors=False,
                flatten_loops=True,
                allow_gauge_detectors=True,
                approximate_disjoint_errors=1,
            )
            mp_dem = demutil.annotate_detector_bases(src_dem)
            mp_dem_p = DEM_CACHE / f"{tag}_mp.dem"
            mp_dem.to_file(mp_dem_p)

            modes = [
                (
                    "baseline_b20",
                    ["--beam", "20", "--beam-climbing", "--no-revisit-dets"],
                ),
                (
                    "gari_b5",
                    [
                        "--dem",
                        str(gari_dem_p),
                        "--detector-orders",
                        str(gari_ord_p),
                        "--beam",
                        "5",
                        "--beam-climbing",
                    ],
                ),
                (
                    "mp_2p_b5",
                    [
                        "--dem",
                        str(mp_dem_p),
                        "--multipass",
                        "--num-passes",
                        "2",
                        "--beam",
                        "5",
                        "--beam-climbing",
                        "--no-revisit-dets",
                    ],
                ),
                (
                    "mp_2p_b20",
                    [
                        "--dem",
                        str(mp_dem_p),
                        "--multipass",
                        "--num-passes",
                        "2",
                        "--beam",
                        "20",
                        "--beam-climbing",
                        "--no-revisit-dets",
                    ],
                ),
            ]

            for mode, extra_args in modes:
                nshots = cinfo["shots"][mode]
                cmd = [
                    "bazel-bin/src/tesseract",
                    "--circuit",
                    ckt_path,
                    "--sample-num-shots",
                    str(nshots),
                    "--sample-seed",
                    "12345",
                    "--threads",
                    str(args.threads),
                    "--pqlimit",
                    "1000000",
                    "--stats-out",
                    "-",
                ] + extra_args
                out = json.loads(subprocess.check_output(cmd, text=True))
                r = cinfo["r"]
                f = out["num_errors"] + out["num_low_confidence"]
                p_shot = f / out["num_shots"]
                ler_round = (
                    0.5 * (1.0 - (1.0 - 2.0 * p_shot) ** (1.0 / r))
                    if p_shot < 0.5
                    else 0.5
                )
                t_round = out["total_time_seconds"] / (out["num_shots"] * r)
                rec = {
                    "tag": tag,
                    "label": cinfo["label"],
                    "family": cinfo["family"],
                    "d": cinfo["d"],
                    "r": r,
                    "q": cinfo["q"],
                    "mode": mode,
                    "num_shots": out["num_shots"],
                    "num_errors": out["num_errors"],
                    "num_low_confidence": out["num_low_confidence"],
                    "total_time_seconds": out["total_time_seconds"],
                    "time_per_round": t_round,
                    "ler_per_round": ler_round,
                }
                results.append(rec)
                out_file.write_text(json.dumps(results, indent=2))
                print(
                    f"{tag:6s} | {mode:13s} | shots={out['num_shots']:7d} errs={out['num_errors']:4d} low_c={out['num_low_confidence']:2d} | LER/r={ler_round:.4e} | t/r={t_round:.4e}s",
                    flush=True,
                )
    finally:
        shutil.rmtree(DEM_CACHE, ignore_errors=True)

    make_plots.main()


if __name__ == "__main__":
    main()
