# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


CIRCUIT = (
    "testdata/surfacecodes/"
    "r=3,d=3,p=0.001,noise=si1000,c=surface_code_X,q=17,gates=cz.stim"
)


def _runfile(relative_path: str) -> Path:
    return (
        Path(os.environ["TEST_SRCDIR"]) / os.environ["TEST_WORKSPACE"] / relative_path
    )


class StatsIntegrationTest(unittest.TestCase):
    def test_stats_capture_runtime_model_counts(self) -> None:
        binary = _runfile("src/tesseract")
        circuit = _runfile(CIRCUIT)
        with tempfile.TemporaryDirectory() as directory:
            stats_path = Path(directory) / "stats.json"
            subprocess.run(
                [
                    str(binary),
                    "--circuit",
                    str(circuit),
                    "--sample-num-shots",
                    "1",
                    "--max-errors",
                    "1",
                    "--sample-seed",
                    "123",
                    "--threads",
                    "1",
                    "--num-det-orders",
                    "1",
                    "--det-order-index",
                    "--pqlimit",
                    "1000000",
                    "--sparsify-errors",
                    "--sparsify-base-degree",
                    "2",
                    "--sparsify-reactivate-limit",
                    "8",
                    "--stats-out",
                    str(stats_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            stats = json.loads(stats_path.read_text(encoding="utf-8"))

        self.assertEqual(stats["det_order_method"], "index")
        self.assertTrue(stats["merge_errors"])
        self.assertGreater(stats["num_detectors"], 0)
        self.assertGreater(stats["num_raw_dem_errors"], 0)
        self.assertGreater(stats["num_compiled_errors"], 0)
        self.assertEqual(
            stats["num_mandatory_errors"] + stats["num_optional_errors"],
            stats["num_compiled_errors"],
        )


if __name__ == "__main__":
    unittest.main()
