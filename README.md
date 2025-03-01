# Tesseract Decoder

A most-likely-error decoder for quantum error correction

[![Licensed under the Apache 2.0 open-source
license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/tesseract-decoder/blob/main/LICENSE)

## Introduction

The implementation of [quantum error
correction](https://en.wikipedia.org/wiki/Quantum_error_correction) (QEC)
requires fast and accurate decoders to achieve low logical error rates. Decoding
is an NP-hard optimization problem in the worst case, but there exists a variety
of partial solutions for specific error-correcting codes. The Tesseract Decoder
takes a novel approach: rather than building an algorithm with a polynomial
runtime and using heuristics to make it more accurate, we begin with an
exponential-time algorithm that always identifies the most likely error and use
heuristics to make it faster. The decoder uses [A*
search](https://en.wikipedia.org/wiki/A*_search_algorithm) along with a variety
of pruning heuristics.

<!-- ## Installation -->

<!-- ## Usage -->

<!-- ## Citing Tesseract Decoder<a name="how-to-cite-tesseract"> -->

## Contact

For any questions or concerns not addressed here, please email
<quantum-oss-maintainers@google.com>.

## Disclaimer

Tesseract Decoder is not an official Google product.
Copyright 2025 Google LLC.
