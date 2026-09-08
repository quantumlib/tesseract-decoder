# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import pytest
import stim
import tesseract_decoder

_DETECTOR_ERROR_MODEL = stim.DetectorErrorModel("""
error(0.125) D0
error(0.375) D0 D1
error(0.25) D1
""")

_DETECTOR_ERROR_MODEL_10 = stim.DetectorErrorModel(
    "\n".join(f"error(0.1) D{i}" for i in range(10))
)


def test_module_has_global_constants():
    assert tesseract_decoder.utils.EPSILON <= 1e-7
    assert not math.isfinite(tesseract_decoder.utils.INF)


def test_detector_order_method_names_and_compatibility_aliases():
    utils = tesseract_decoder.utils
    assert utils.DetOrder is utils.DetectorOrderMethod
    assert utils.DetOrder.DetBFS == utils.DetectorOrderMethod.BFS
    assert utils.DetOrder.DetIndex == utils.DetectorOrderMethod.Index
    assert utils.DetOrder.DetCoordinate == utils.DetectorOrderMethod.Coordinate
    assert utils.DetectorOrderMethod.Literal.name == "Literal"


def test_literal_detector_order_method_is_not_generated():
    with pytest.raises(ValueError, match="Literal detector orders cannot be generated"):
        tesseract_decoder.utils.build_det_orders(
            _DETECTOR_ERROR_MODEL,
            num_det_orders=1,
            method=tesseract_decoder.utils.DetectorOrderMethod.Literal,
        )


def test_get_detector_coords():
    assert tesseract_decoder.utils.get_detector_coords(_DETECTOR_ERROR_MODEL) == []


def test_get_detector_coords_sparse():
    dem = stim.DetectorErrorModel("detector(1, 2, 3) D1\nerror(0.1) D0 D1\n")
    assert tesseract_decoder.utils.get_detector_coords(dem) == [[], [1.0, 2.0, 3.0]]


def test_build_detector_graph():
    assert tesseract_decoder.utils.build_detector_graph(_DETECTOR_ERROR_MODEL) == [
        [1],
        [0],
    ]


def test_build_detector_graph_uses_positive_parity_reduced_symptoms():
    dem = stim.DetectorErrorModel("""
        error(0) D0 D1
        error(0.1) D0 D0 D1
        error(0.2) D1 D2 D3
    """)
    assert tesseract_decoder.utils.build_detector_graph(dem) == [
        [],
        [2, 3],
        [1, 3],
        [1, 2],
    ]


def test_build_det_orders_default_index():
    res = tesseract_decoder.utils.build_det_orders(
        _DETECTOR_ERROR_MODEL_10, num_det_orders=1, seed=0
    )
    expected_asc = list(range(10))
    expected_desc = list(range(9, -1, -1))
    assert res == [expected_asc] or res == [expected_desc]


def test_build_det_orders_bfs():
    path_dem = stim.DetectorErrorModel("""
        error(0.1) D0 D4
        error(0.1) D4 D1
        error(0.1) D1 D3
        error(0.1) D3 D2
    """)
    graph = tesseract_decoder.utils.build_detector_graph(path_dem)
    orders = tesseract_decoder.utils.build_det_orders(
        path_dem,
        num_det_orders=16,
        method=tesseract_decoder.utils.DetectorOrderMethod.BFS,
        seed=0,
    )
    for order in orders:
        assert sorted(order) == list(range(5))
        distance = [None] * len(graph)
        distance[order[0]] = 0
        frontier = [order[0]]
        for detector in frontier:
            for neighbor in graph[detector]:
                if distance[neighbor] is None:
                    distance[neighbor] = distance[detector] + 1
                    frontier.append(neighbor)
        assert [distance[detector] for detector in order] == sorted(
            distance[detector] for detector in order
        )


def test_build_det_orders_bfs_empty_dem():
    assert tesseract_decoder.utils.build_det_orders(
        stim.DetectorErrorModel(),
        num_det_orders=3,
        method=tesseract_decoder.utils.DetectorOrderMethod.BFS,
        seed=0,
    ) == [[], [], []]


def test_build_det_orders_coordinate():
    dem = stim.DetectorErrorModel("""
        detector(2) D3
        detector(0) D0
        detector(3) D1
        detector(1) D2
    """)
    order = tesseract_decoder.utils.build_det_orders(
        dem,
        num_det_orders=1,
        method=tesseract_decoder.utils.DetectorOrderMethod.Coordinate,
        seed=0,
    )[0]
    assert order in ([0, 2, 3, 1], [1, 3, 2, 0])


def test_detector_coords_are_keyed_and_allow_missing_or_short_coordinates():
    dem = stim.DetectorErrorModel("""
        detector(2, 20) D2
        detector(0) D0
        detector(99) D2
        error(0.1) D3
    """)
    assert tesseract_decoder.utils.get_detector_coords(dem) == [
        [0],
        [],
        [2, 20],
        [],
    ]
    order = tesseract_decoder.utils.build_det_orders(
        dem,
        num_det_orders=1,
        method=tesseract_decoder.utils.DetectorOrderMethod.Coordinate,
        seed=0,
    )[0]
    assert order[2:] == [1, 3]


def test_build_det_orders_coordinate_sparse():
    dem = stim.DetectorErrorModel("detector(1, 2, 3) D1\nerror(0.1) D0 D1\n")
    orders = tesseract_decoder.utils.build_det_orders(
        dem,
        num_det_orders=1,
        method=tesseract_decoder.utils.DetectorOrderMethod.Coordinate,
        seed=0,
    )
    assert len(orders) == 1
    assert len(orders[0]) == 2


def test_build_det_orders_index():
    res = tesseract_decoder.utils.build_det_orders(
        _DETECTOR_ERROR_MODEL_10,
        num_det_orders=1,
        method=tesseract_decoder.utils.DetectorOrderMethod.Index,
        seed=0,
    )
    expected_asc = list(range(10))
    expected_desc = list(range(9, -1, -1))
    assert res == [expected_asc] or res == [expected_desc]


def test_get_errors_from_dem():
    expected = "Error{cost=1.945910, symptom=Symptom{detectors=[0], observables=[]}}, Error{cost=0.510826, symptom=Symptom{detectors=[0 1], observables=[]}}, Error{cost=1.098612, symptom=Symptom{detectors=[1], observables=[]}}"
    assert (
        ", ".join(
            map(str, tesseract_decoder.utils.get_errors_from_dem(_DETECTOR_ERROR_MODEL))
        )
        == expected
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
