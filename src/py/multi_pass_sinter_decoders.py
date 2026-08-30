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

"""Sinter integration for Tesseract's two-component multi-pass decoder."""

from collections.abc import Callable, Sequence
from numbers import Integral

import sinter
import stim
import tesseract_decoder as _core

from _tesseract_py_util.detector_basis import (
    DetectorBasis,
    DetectorBasisClassifier,
    automatic_detector_basis_classifier,
    classify_detector_bases,
)


LegacyDetectorClassifier = Callable[[int, Sequence[float], str], int]


class MultiPassSinterDecoder(sinter.Decoder):
    """A Sinter decoder for one- or two-pass, two-component Tesseract.

    Detectors are classified once in Python and the resulting explicit X/Z
    component vector is passed into native code. By default, the shared
    automatic detector-basis classifier handles supported metadata and the
    Chromobius fourth-coordinate convention.

    Preferred ``detector_basis_classifier`` callables use the shared protocol
    and return ``"X"``, ``"Z"``, or ``None``. The compatibility keyword
    ``detector_classifier`` retains the old protocol of returning arbitrary
    nonnegative integer component labels; exactly two labels must occur.

    ``merge_errors=False`` is supported with one pass. Two-pass reweighting is
    defined on aggregate component symptoms and therefore requires merging.
    """

    def __init__(
        self,
        num_passes: int = 2,
        detector_basis_classifier: DetectorBasisClassifier | None = None,
        strategy=_core.SchedulingStrategy.Causal,
        *,
        detector_classifier: LegacyDetectorClassifier | None = None,
        det_beam: int = 5,
        beam_climbing: bool = False,
        no_revisit_dets: bool = True,
        verbose: bool = False,
        merge_errors: bool = True,
        pqlimit: int = 200_000,
        det_penalty: float = 0.0,
        create_visualization: bool = False,
        sparsify_errors: bool = False,
        sparsify_base_degree: int = -1,
        sparsify_max_degree: int = -1,
        sparsify_reactivate_limit: int = -1,
        det_orders: Sequence[Sequence[int]] | None = None,
        num_det_orders: int = 1,
        det_order_method=_core.utils.DetOrder.DetIndex,
        seed: int = 0,
    ):
        if num_passes not in (1, 2):
            raise ValueError("num_passes must be 1 or 2.")
        if detector_basis_classifier is not None and detector_classifier is not None:
            raise ValueError(
                "Specify at most one of detector_basis_classifier and detector_classifier."
            )
        if strategy not in (
            _core.SchedulingStrategy.Static,
            _core.SchedulingStrategy.Causal,
        ):
            raise ValueError("strategy must be SchedulingStrategy.Static or Causal.")

        self.num_passes = num_passes
        self.detector_basis_classifier = (
            detector_basis_classifier
            if detector_basis_classifier is not None
            else automatic_detector_basis_classifier
        )
        self._legacy_detector_classifier = detector_classifier
        self.strategy = strategy

        self.det_beam = det_beam
        self.beam_climbing = beam_climbing
        self.no_revisit_dets = no_revisit_dets
        self.verbose = verbose
        self.merge_errors = merge_errors
        self.pqlimit = pqlimit
        self.det_penalty = det_penalty
        self.create_visualization = create_visualization
        self.sparsify_errors = sparsify_errors
        self.sparsify_base_degree = sparsify_base_degree
        self.sparsify_max_degree = sparsify_max_degree
        self.sparsify_reactivate_limit = sparsify_reactivate_limit
        self.det_orders = (
            [list(order) for order in det_orders] if det_orders is not None else []
        )
        self.num_det_orders = num_det_orders
        self.det_order_method = det_order_method
        self.seed = seed

    @property
    def detector_classifier(
        self,
    ) -> DetectorBasisClassifier | LegacyDetectorClassifier:
        """Legacy integer-label classifier, or the preferred X/Z classifier."""

        return (
            self._legacy_detector_classifier
            if self._legacy_detector_classifier is not None
            else self.detector_basis_classifier
        )

    @detector_classifier.setter
    def detector_classifier(self, value: LegacyDetectorClassifier | None) -> None:
        self._legacy_detector_classifier = value

    def _classify_detector_components(self, dem: stim.DetectorErrorModel) -> list[int]:
        if self._legacy_detector_classifier is None:
            bases = classify_detector_bases(
                dem,
                detector_basis_classifier=self.detector_basis_classifier,
            )
            return [0 if basis == "X" else 1 for basis in bases]

        labels = [0] * dem.num_detectors
        label_to_basis: dict[int, DetectorBasis] = {}

        def legacy_adapter(index, coordinates, tag):
            label = self._legacy_detector_classifier(index, coordinates, tag)
            if isinstance(label, bool) or not isinstance(label, Integral) or label < 0:
                raise ValueError(
                    f"Detector D{index} could not be classified: detector_classifier "
                    f"returned {label!r}, expected a nonnegative integer label."
                )
            label = int(label)
            if label not in label_to_basis:
                if len(label_to_basis) == 2:
                    raise ValueError(
                        "Multi-pass decoding requires exactly 2 detector components; "
                        "detector_classifier produced more than 2 labels."
                    )
                label_to_basis[label] = "X" if not label_to_basis else "Z"
            labels[index] = label
            return label_to_basis[label]

        classify_detector_bases(dem, detector_basis_classifier=legacy_adapter)
        if len(label_to_basis) != 2:
            raise ValueError(
                "Multi-pass decoding requires exactly 2 detector components; "
                f"detector_classifier produced {len(label_to_basis)}."
            )
        return labels

    def compile_decoder_for_dem(
        self, *, dem: stim.DetectorErrorModel
    ) -> sinter.CompiledDecoder:
        detector_components = self._classify_detector_components(dem)
        base_config = _core.tesseract.TesseractConfig(
            det_beam=self.det_beam,
            beam_climbing=self.beam_climbing,
            no_revisit_dets=self.no_revisit_dets,
            verbose=self.verbose,
            merge_errors=self.merge_errors,
            pqlimit=self.pqlimit,
            det_penalty=self.det_penalty,
            create_visualization=self.create_visualization,
            sparsify_errors=self.sparsify_errors,
            sparsify_base_degree=self.sparsify_base_degree,
            sparsify_max_degree=self.sparsify_max_degree,
            sparsify_reactivate_limit=self.sparsify_reactivate_limit,
            det_orders=self.det_orders,
        )
        # The Python TesseractConfig constructor synthesizes orders when given
        # an empty list. Restore the caller's empty list so native multipass
        # compilation uses num_det_orders/method/seed for each component.
        base_config.det_orders = self.det_orders
        return _core._compile_multi_pass_decoder_for_dem(
            dem=dem,
            detector_components=detector_components,
            num_passes=self.num_passes,
            base_config=base_config,
            num_det_orders=self.num_det_orders,
            det_order_method=self.det_order_method,
            seed=self.seed,
            strategy=self.strategy,
        )


def get_sinter_decoders() -> dict[str, sinter.Decoder]:
    """Returns the stable long-beam monolithic and multi-pass registry."""

    common_kwargs = dict(
        det_beam=20,
        beam_climbing=True,
        no_revisit_dets=True,
        merge_errors=True,
        pqlimit=1_000_000,
        num_det_orders=21,
        det_order_method=_core.utils.DetOrder.DetIndex,
        seed=2_384_753,
    )
    return {
        "tesseract-long-beam-mono": _core.TesseractSinterDecoder(**common_kwargs),
        "tesseract-long-beam-multipass-1pass": MultiPassSinterDecoder(
            num_passes=1,
            strategy=_core.SchedulingStrategy.Causal,
            **common_kwargs,
        ),
        "tesseract-long-beam-multipass-2pass": MultiPassSinterDecoder(
            num_passes=2,
            strategy=_core.SchedulingStrategy.Causal,
            **common_kwargs,
        ),
    }


__all__ = ["MultiPassSinterDecoder", "get_sinter_decoders"]
