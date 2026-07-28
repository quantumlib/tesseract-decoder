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
import re
import unittest
from pathlib import Path

from benchmarking.sparsify_errors.benchmark_data import (
    canonical_jsonl,
    enrich_rows,
    numerical_content_sha256,
    read_jsonl,
)


class SchemaTest(unittest.TestCase):
    def test_committed_aggregate_is_strict_and_preserves_exact_circuits(self) -> None:
        path = Path(__file__).with_name("aggregated_results.jsonl")
        rows = read_jsonl(path)

        self.assertEqual(len(rows), 840)
        keys = {
            (
                row["circuit_path"],
                row["sparsify_errors"],
                row["sparsify_base_degree"],
                row["sparsify_max_degree"],
                row["sparsify_reactivate_limit"],
            )
            for row in rows
        }
        self.assertEqual(len(keys), len(rows))
        self.assertEqual({row["basis"] for row in rows}, {"X", "Z"})

        # Resolve through Bazel's data-file symlink so the repository root and
        # circuit files have the same canonical root during sandboxed tests.
        repo_root = path.resolve().parents[2]
        self.assertEqual(
            canonical_jsonl(enrich_rows(rows, repo_root)),
            path.read_text(encoding="utf-8"),
        )

        provenance = json.loads(
            Path(__file__).with_name("provenance.json").read_text(encoding="utf-8")
        )
        numerical = provenance["numerical_results"]
        self.assertEqual(numerical["rows"], len(rows))
        self.assertEqual(
            numerical["rows_over_10000000_shots"],
            sum(row["num_shots"] > 10_000_000 for row in rows),
        )
        self.assertEqual(
            numerical["maximum_num_shots"], max(row["num_shots"] for row in rows)
        )
        self.assertEqual(
            numerical["preserved_numerical_content_sha256"],
            numerical_content_sha256(rows),
        )
        self.assertRegex(
            provenance["metadata_enrichment"]["circuit_source_commit"],
            re.compile(r"[0-9a-f]{40}"),
        )


if __name__ == "__main__":
    unittest.main()
