# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="KBmkwvhmupn-"
# # Tesseract Tutorial
#
# - We will also, partly, explain how to use features of Stim and PyMatching
# - Stim is a dependency of Tesseract but you can also use other sources of data
# - This is not a comprehensive introduction.

# %% [markdown] id="jaZcr-NevBSB"
# ## Installation

# %% id="i6_88o7kKOVJ"
# !pip install --quiet --upgrade stim galois tesseract-decoder pymatching python-sat -U

# %% [markdown] id="RLXXX3eMT_LR"
# ## Getting a Surface Code Circuit

# %% id="8zcmVHFFUPq2"
import stim

d = 11
p = 0.005
circuit = stim.Circuit.generated(
    code_task="surface_code:rotated_memory_x",
    distance=d,
    rounds=d,
    after_clifford_depolarization=p,
    before_round_data_depolarization=p,
    before_measure_flip_probability=p,
    after_reset_flip_probability=p
)

# %% [markdown] id="UBMIlXY9U30Y"
# ## Sample

# %% id="GCkUlTJZU2T_"
sampler = circuit.compile_detector_sampler()

num_shots = 10000
detector_outcomes, actual_observables = sampler.sample(shots=num_shots, separate_observables=True)

# %% [markdown] id="m9x8pivTVCir"
# ## Decode with uncorrelated matching

# %% colab={"base_uri": "https://localhost:8080/"} id="-5W0AX8nVEyU" outputId="06a66e39-b33b-4a17-a1e1-2b91997b1d40"
import pymatching
import numpy as np

dem = circuit.detector_error_model(decompose_errors=True)
matching = pymatching.Matching.from_detector_error_model(model=dem)
predicted_observables = matching.decode_batch(shots=detector_outcomes)
num_errors = np.sum(np.any(predicted_observables != actual_observables, axis=1))

print(f"Logical error rate: {num_errors}/{num_shots}")

# %% [markdown] id="Xp7MyK0XVs_6"
# ## Decode with new correlated matching!

# %% colab={"base_uri": "https://localhost:8080/"} id="vufQ8G5iVx7b" outputId="3b0517a3-e65e-42b7-eb25-fbb068c4a912"
dem = circuit.detector_error_model(decompose_errors=True)
matching_corr = pymatching.Matching.from_detector_error_model(
    model=dem, enable_correlations=True
    )
predicted_observables_corr = matching_corr.decode_batch(
    shots=detector_outcomes,
    enable_correlations=True
    )
num_errors_corr = np.sum(np.any(predicted_observables_corr != actual_observables, axis=1))

print(f"Logical error rate: {num_errors_corr}/{num_shots}")

# %% [markdown] id="a-AMqTUeuqOe"
# ## Getting a Color Code Circuit

# %% colab={"base_uri": "https://localhost:8080/"} id="W7fU_MYJCRen" outputId="da1b1571-8160-440a-ce1a-f38de9db82e4"
# !curl 'https://raw.githubusercontent.com/quantumlib/tesseract-decoder/refs/heads/main/testdata/colorcodes/r%3D5%2Cd%3D5%2Cp%3D0.001%2Cnoise%3Dsi1000%2Cc%3Dsuperdense_color_code_Z%2Cq%3D37%2Cgates%3Dcz.stim' > d5r5colorcode_p001.stim

# %% [markdown] id="E-vXEhbaTeQI"
# # Visualizing with Stim

# %% colab={"base_uri": "https://localhost:8080/", "height": 343} id="2jTOVijwKPXm" outputId="b2b3ecc8-3491-44e7-e200-238dae9b7f36"
import stim

circuit = stim.Circuit.from_file('d5r5colorcode_p001.stim')
circuit.diagram('timeline-3d')

# %% [markdown] id="YDnwv2dacTbf"
# # Estimating code distance with Stim

# %% colab={"base_uri": "https://localhost:8080/"} id="r2MFDDBMvkq3" outputId="6216021d-5506-4304-e4e5-d10ab170fff7"
distance_estimate = len(circuit.search_for_undetectable_logical_errors(
    dont_explore_detection_event_sets_with_size_above=6,
    dont_explore_edges_with_degree_above=3,
    dont_explore_edges_increasing_symptom_degree=False))
print(f'estimated distance: {distance_estimate}')

# %% [markdown] id="UuYvEwq9cgYc"
# # Create DEM, detection events and observables with Stim

# %% [markdown] id="6oCW5jRsXy8-"
# ### Can't decode with pymatching...

# %% colab={"base_uri": "https://localhost:8080/"} id="x6TQbGZ7b06k" outputId="4767d883-cbc7-4b2c-a42e-e503d0dc6332"
import traceback

try:
  # decompose_errors=True needed for DEM to be matchable
  circuit.detector_error_model(decompose_errors=True)
except:
  traceback.print_exc()

# %% [markdown] id="ye5W7BJHX8DJ"
# No need to decompose errors using tesseract:

# %% id="AVu7idoTYAdM"
dem = circuit.detector_error_model()

# %% id="vFDn06Xach0_"
num_shots = 1000
sampler = circuit.compile_detector_sampler()
dets, obs = sampler.sample(num_shots, separate_observables=True)

# %% [markdown] id="JrX13vNQcrm3"
# # Decoding with Tesseract and ILP decoder

# %% id="Uds8S04a-z-G"
import tesseract_decoder
import tesseract_decoder.tesseract as tesseract
import numpy as np
import time

# Helper functions for benchmarking

def print_results(result):
  print("Tesseract Decoder Stats:")
  print(f"   Number of Errors / num_shots: {results['num_errors']} / {results['num_shots']}")
  print(f"   Time: {results['time_seconds']:.4f} s")
  print()

def run_tesseract_decoder(decoder, dets, obs):
  # Run and time the Tesseract decoder
  num_errors = 0
  start_time = time.time()
  obs_predicted = decoder.decode_batch(dets)
  num_errors = np.sum(np.any(obs_predicted != obs, axis=1))
  end_time = time.time()

  return {
      'num_errors': num_errors,
      'num_shots': len(dets),
      'time_seconds': end_time - start_time,
  }



# %% colab={"base_uri": "https://localhost:8080/"} id="D0Tx2eY3ctFw" outputId="f419743f-f692-4490-fecc-6b7a831dc586"
# setup the tesseract decoder configuration
tesseract_config = tesseract.TesseractConfig(
    dem=dem,
    pqlimit=10000,
    no_revisit_dets=True,
    # verbose=True,
    det_orders=tesseract_decoder.utils.build_det_orders(
        dem, num_det_orders=1,
        method=tesseract_decoder.utils.DetOrder.DetIndex,
        seed=2384753),
)
print(f'Tesseract decoder configurations --> {tesseract_config}\n')

tesseract_dec = tesseract_config.compile_decoder()

results = run_tesseract_decoder(tesseract_dec, dets, obs)
print_results(results)

# %% [markdown] id="INvMKs7zc5T_"
# #Decoding with ILP decoder

# %% colab={"base_uri": "https://localhost:8080/"} id="9Npo7ibac4x5" outputId="c687e15b-e167-449b-9508-33701a9e1944"
simplex_config = tesseract_decoder.simplex.SimplexConfig(
  dem=dem, parallelize=True
)
print(f'ILP decoder configurations --> {simplex_config}')
ilp_dec = simplex_config.compile_decoder()

start_time = time.time()

# Run and time ILP decoder -- so slow!
num_shots_to_decode = 10 # Only decoding 10 shots because it's soooo slow
obs_predicted = ilp_dec.decode_batch(dets[0:num_shots_to_decode])
num_errors = np.sum(np.any(obs_predicted != obs[0:num_shots_to_decode], axis=1))

end_time = time.time()
print(f'ILP stats:\n   Estimated time for full shots {num_shots/num_shots_to_decode * (end_time - start_time)} s')
print(f"   Number of Errors / num_shots: {num_errors} / {num_shots_to_decode}")

# %% [markdown] id="VQqlMqFRIZ2J"
# # Tesseract Config and impact of heuristic
# You can tune tesseract decoder through the Config that is passed to the decoder with this set of parameters:
# Explanation of configuration arguments:
#
# * `pqlimit` - An integer that sets a limit on the number of nodes in the priority queue. This can be used to constrain the memory usage of the decoder. The default value is `sys.maxsize`, which means the size is effectively unbounded.
#
#

# %% colab={"base_uri": "https://localhost:8080/"} id="0pExdmuPQuGr" outputId="8e8ff3fd-bde9-4e31-d1e2-4f97c77ce43d"
tesseract_config1 = tesseract.TesseractConfig(
    dem=dem,
    pqlimit=100,
    no_revisit_dets=True,
    det_orders=tesseract_decoder.utils.build_det_orders(
        dem, num_det_orders=5,
        method=tesseract_decoder.utils.DetOrder.DetIndex,
        seed=2384753),
)

print ("Smaller pqlimit")
results = run_tesseract_decoder(tesseract_config1.compile_decoder(), dets, obs)
print_results(results)


tesseract_config2 = tesseract.TesseractConfig(
    dem=dem,
    pqlimit=20000,
    no_revisit_dets=True,
    det_orders=tesseract_decoder.utils.build_det_orders(
        dem, num_det_orders=5,
        method=tesseract_decoder.utils.DetOrder.DetIndex,
        seed=2384753),
)
print ("Larger pqlimit")
results = run_tesseract_decoder(tesseract_config2.compile_decoder(), dets, obs)
print_results(results)

# %% [markdown] id="ru-MRctAIq5-"
# #More heurisitcs
# * `det_beam` - This integer value represents the beam search cutoff. It specifies a threshold for the number of "residual detection events" a node can have before it is pruned from the search. A lower `det_beam` value makes the search more aggressive, potentially sacrificing accuracy for speed. The default value `INF_DET_BEAM` means no beam cutoff is applied.
# * `beam_climbing` - A boolean flag that, when set to `True`, enables a heuristic called "beam climbing." This optimization causes the decoder to try different `det_beam` values (up to a maximum) to find a good decoding path. This can improve the decoder's chance of finding the most likely error, even with an initial narrow beam search.
# * Try replacing `DetIndex` with `DetBFS` or `DetCoordinate` -- this enables different heuristics to assign the ordering of detectors for the traversal, leading to different results.
#

# %% colab={"base_uri": "https://localhost:8080/"} id="cyctTUyzQ-cQ" outputId="8bc31b95-90a0-43d6-bd53-7176ec58b32f"
tesseract_config1 = tesseract.TesseractConfig(
    dem=dem,
    pqlimit=1000,
    det_beam=3,
    beam_climbing=True,
    det_orders=tesseract_decoder.utils.build_det_orders(
        dem, num_det_orders=5,
        method=tesseract_decoder.utils.DetOrder.DetIndex,
        seed=2384753),
)

print ("Smaller det_beam")
results = run_tesseract_decoder(tesseract_config1.compile_decoder(), dets, obs)
print_results(results)

tesseract_config2 = tesseract.TesseractConfig(
    dem=dem,
    pqlimit=1000,
    det_beam=5,
    beam_climbing=True,
    det_orders=tesseract_decoder.utils.build_det_orders(
        dem, num_det_orders=5,
        method=tesseract_decoder.utils.DetOrder.DetIndex,
        seed=2384753),
)
print ("Larger det_beam")
results = run_tesseract_decoder(tesseract_config2.compile_decoder(), dets, obs)
print_results(results)

# %% [markdown] id="VJiBCWpUQ9sf"
# #Even More Heuristics
# * `no_revisit_dets` - A boolean flag that, when `True`, activates a heuristic to prevent the decoder from revisiting nodes that have the same set of leftover detection events as a node it has already visited. This can help to reduce search redundancy and improve decoding speed.
#
# * `det_orders` - A list of lists of integers, where each inner list represents an ordering of the detectors. This is used for "ensemble reordering," an optimization that tries different detector orderings to improve the search's convergence. The default is an empty list, meaning a single, fixed ordering is used.
# * `det_penalty` - A floating-point value that adds a cost for each residual detection event. This encourages the decoder to prioritize paths that resolve more detection events, steering the search towards more complete solutions. The default value is `0.0`, meaning no penalty is applied.

# %% colab={"base_uri": "https://localhost:8080/"} id="0VrW2z8sSXtN" outputId="de264ea5-becb-46f5-f98b-644448fbb773"
tesseract_config1 = tesseract.TesseractConfig(
    dem=dem,
    pqlimit=1000,
    # no_revisit_dets=True,
    det_penalty = 10,
    # det_orders=tesseract_decoder.utils.build_det_orders(
    #     dem, num_det_orders=2,
    #     method=tesseract_decoder.utils.DetOrder.DetIndex,
    #     seed=2384753),
)

print ("First version")
results = run_tesseract_decoder(tesseract_config1.compile_decoder(), dets, obs)
print_results(results)


tesseract_config2 = tesseract.TesseractConfig(
    dem=dem,
    pqlimit=1000,
    # no_revisit_dets=False,
    det_penalty = False,
    # det_orders=tesseract_decoder.utils.build_det_orders(
    #     dem, num_det_orders=2,
    #     method=tesseract_decoder.utils.DetOrder.DetIndex,
    #     seed=2384753),
)
print ("Second version")
results = run_tesseract_decoder(tesseract_config2.compile_decoder(), dets, obs)
print_results(results)

# %% [markdown] id="BoEALeo3OYGp"
# # Decoding Wild Stabilizer Codes under Code Capacity Noise with Tesseract
#
#
#
# *   checkout https://www.codetables.de/ for a qubit stabilizer code
# *   full table of qubit codes: [here](https://codetables.de/QECC/Tables_color.php?q=4&n0=1&n1=256&k0=0&k1=256)
# *   copy the stabilizer matrix for a code, such as: [this one used below](https://codetables.de/QECC/QECC.php?q=4&n=21&k=8)
#
#

# %% id="pJ1gEKAgPbHO"
import time
import tesseract_decoder
import stim
import numpy as np
import numpy.typing as npt
from galois import GF2
from typing import List, Tuple


def paulis_from_symplectic_matrix(check_matrix: npt.NDArray[np.uint8]) -> List[stim.PauliString]:
    n = check_matrix.shape[1] // 2
    paulis = []
    for i in range(check_matrix.shape[0]):
        paulis.append(
            stim.PauliString.from_numpy(
                xs=check_matrix[i, :n].astype(bool), zs=check_matrix[i, n:].astype(bool)
            )
        )
    return paulis

def rank(H):
  return np.linalg.matrix_rank(GF2(H))

def stabilizer_code_logical_operators(
    check_matrix: npt.NDArray[np.uint8]) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    check_matrix = np.array(check_matrix, dtype=np.uint8)

    r = rank(check_matrix)
    n = check_matrix.shape[1] // 2

    stabilisers = paulis_from_symplectic_matrix(check_matrix=check_matrix)

    tableau = stim.Tableau.from_stabilizers(
        stabilizers=stabilisers, allow_underconstrained=True, allow_redundant=True
    )

    x2x, x2z, z2x, z2z, x_signs, z_signs = tableau.to_numpy()

    num_logicals = n - r

    Lx = np.zeros((num_logicals, check_matrix.shape[1]), dtype=np.uint8)
    Lz = np.zeros((num_logicals, check_matrix.shape[1]), dtype=np.uint8)

    Lx[:, :n] = x2x[r:]
    Lx[:, n:] = x2z[r:]
    Lz[:, :n] = z2x[r:]
    Lz[:, n:] = z2z[r:]
    return Lx.astype(np.uint8), Lz.astype(np.uint8)


def pauli_to_observable_include_target(pauli: stim.PauliString) -> List[stim.GateTarget]:
    obs_pauli_targets = []
    for i in range(len(pauli)):
        if pauli[i] != 0:
            obs_pauli_targets.append(stim.target_pauli(i, pauli[i]))
    return obs_pauli_targets


def append_observable_includes_for_paulis(circuit: stim.Circuit, paulis: List[stim.PauliString]) -> None:
    for i, obs in enumerate(paulis):
        circuit.append(
            "OBSERVABLE_INCLUDE",
            targets=pauli_to_observable_include_target(pauli=obs),
            arg=i
        )


def code_capacity_circuit(
    stabilizers: npt.NDArray[np.uint8],
    x_logicals: npt.NDArray[np.uint8],
    z_logicals: npt.NDArray[np.uint8],
    p: float
) -> stim.Circuit:
    """Generate a code capacity stim circuit for a stabilizer code

    Parameters
    ----------
    stabilizers : npt.NDArray[np.uint8]
        The stabilizer generators of the code, as a binary symplectic matrix.
        The matrix has dimensions (r, 2 * n) where r is the number of stabilizer
        generators and n is the number of physical qubits.
        `stabilizers[i, j]` is 1 if stabilizer i is X or Y on qubit j and 0 otherwise.
        `stabilizers[i, n + j]` is 1 if stabilizer i is Z or Y on qubit j and 0 otherwise.
    x_logicals : npt.NDArray[np.uint8]
        The X logical operators of the code, as a binary symplectic matrix.
        The matrix has dimensions (k, 2 * n) where k is the number of logical qubits
        and n is the number of physical qubits.
    z_logicals : npt.NDArray[np.uint8]
        The Z logical operators of the code, as a binary symplectic matrix.
        The matrix has dimensions (k, 2 * n) where k is the number of logical qubits
        and n is the number of physical qubits.
    p : float
        The strength of single-qubit depolarizing noise to use

    Returns
    -------
    stim.Circuit
        The stim circuit of the code capacity circuit
    """
    num_qubits = stabilizers.shape[1] // 2
    num_stabilizers = stabilizers.shape[0]
    stabilizer_paulis = paulis_from_symplectic_matrix(stabilizers)
    x_logicals_paulis = paulis_from_symplectic_matrix(x_logicals)
    z_logicals_paulis = paulis_from_symplectic_matrix(z_logicals)
    all_logicals_paulis = x_logicals_paulis + z_logicals_paulis

    circuit = stim.Circuit()

    append_observable_includes_for_paulis(
        circuit=circuit, paulis=all_logicals_paulis)
    circuit.append("MPP", stabilizer_paulis)
    circuit.append("DEPOLARIZE1", targets=list(range(num_qubits)), arg=p)
    circuit.append("MPP", stabilizer_paulis)

    for i in range(num_stabilizers):
        circuit.append(
            "DETECTOR",
            targets=[
                stim.target_rec(i - 2 * num_stabilizers),
                stim.target_rec(i - num_stabilizers)
            ]
        )

    append_observable_includes_for_paulis(
        circuit=circuit, paulis=all_logicals_paulis)
    return circuit


def parse_symplectic_matrix(text: str) -> npt.NDArray[np.uint8]:
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line[0] != '[' or line[-1] != ']':
            continue  # skip malformed lines
        body = line[1:-1]
        if "|" in body:
            left, right = body.split("|")
            bits = left.strip().split() + right.strip().split()
        else:
            bits = body.strip().split()
        row = [int(b) for b in bits]
        rows.append(row)
    return np.array(rows, dtype=np.uint8)



# %% colab={"base_uri": "https://localhost:8080/", "height": 343} id="pH_b3u1rBogl" outputId="8b775afe-a447-4856-f698-ab87ff752dfc"
# Example QEC code:
text = '''[1 0 0 0 0 1 1 0 1 0 1 1 0 0 1 1 1 0 0 0 0|0 0 0 0 0 1 1 1 0 0 1 1 0 0 1 0 1 0 0 0 0]
      [0 0 0 0 0 1 1 1 0 0 1 1 0 0 1 1 1 0 0 0 0|1 0 0 0 1 0 0 1 1 0 0 0 0 1 1 0 0 0 0 0 0]
      [0 1 0 0 0 1 1 0 1 1 0 1 1 0 1 0 1 0 0 0 0|0 0 0 0 1 1 0 1 0 0 1 0 1 1 0 1 0 0 0 0 0]
      [0 0 0 0 0 1 0 1 0 0 1 0 1 0 1 0 0 0 0 0 0|0 1 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 0 0 0 0]
      [0 0 1 0 0 0 0 1 1 1 0 1 1 1 0 1 1 0 0 0 0|0 0 0 0 0 0 0 1 1 0 1 0 0 1 0 0 1 0 0 0 0]
      [0 0 0 0 0 0 0 1 1 0 1 0 0 1 0 1 1 0 0 0 0|0 0 1 0 1 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0]
      [0 0 0 1 0 1 0 1 0 1 1 0 1 1 0 1 1 0 0 0 0|0 0 0 0 1 1 1 0 0 1 1 0 0 1 1 0 0 0 0 0 0]
      [0 0 0 0 0 1 1 0 0 1 1 0 0 0 0 1 0 0 0 0 0|0 0 0 1 0 0 1 1 0 0 0 0 1 1 0 1 1 0 0 0 0]
      [0 0 0 0 1 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0|0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0]
      [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0|0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
      [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0|0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
      [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0|0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
      [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1|0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]'''

H = parse_symplectic_matrix(text)

LX, LZ = stabilizer_code_logical_operators(check_matrix=H)

circuit = code_capacity_circuit(
    stabilizers=H,
    x_logicals=LX,
    z_logicals=LZ,
    p=0.025
)

circuit.diagram('timeline-3d')

# %% [markdown] id="cK2Mf2fTCAWO"
# ## Computing minimum distance with Stim + SAT Solver

# %% colab={"base_uri": "https://localhost:8080/"} id="ZdVK4Dq1Bp1B" outputId="84834d1f-6d5a-4dc3-b06e-6898c689c355"
# Note: this maxSAT solver only works for very small codes.
# For larger codes, use the solvers at https://maxsat-evaluations.github.io/2024/
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

wcnf = WCNF(from_string=circuit.shortest_error_sat_problem())

with RC2(wcnf) as rc2:
  rc2.compute()
  print(f'Distance of code: {rc2.cost}')

# %% [markdown] id="GQjQkhD4C4rK"
# ## Sample new data for this stabilizer code

# %% id="7iOIl7vjC3uG"
num_shots = 1000
dem = circuit.detector_error_model()
sampler = circuit.compile_detector_sampler(seed=23845386)
dets, obs = sampler.sample(num_shots, separate_observables=True)

# %% [markdown] id="63xjagbBCj8x"
# ## Decode code capacity noise data with ILP and Tesseract

# %% colab={"base_uri": "https://localhost:8080/"} id="IM7W37cHaKfT" outputId="eb33df9c-30bd-44c5-85f3-ce0919cb8c47"
tesseract_config = tesseract_decoder.tesseract.TesseractConfig(
    dem=dem,
    pqlimit=1000,
    det_beam=10,
    det_orders=tesseract_decoder.utils.build_det_orders(
        dem, num_det_orders=10,
        method=tesseract_decoder.utils.DetOrder.DetIndex,
        seed=2384753),
    # no_revisit_dets=True,
)

results = run_tesseract_decoder(tesseract_config.compile_decoder(), dets, obs)
print_results(results)

# Run and time ILP decoder
ilp_dec = tesseract_decoder.simplex.SimplexConfig(
    dem=dem, parallelize=True).compile_decoder()
start_time = time.time()
obs_predicted = ilp_dec.decode_batch(dets)
num_errors_ilp = np.sum(np.any(obs_predicted != obs, axis=1))
end_time = time.time()
print(
    f'ILP: num_errors / num_shots = {num_errors_ilp} / {len(dets)} time {end_time - start_time} s')

# %% [markdown] id="K0QvSpXQIwgf"
# # Visualize the Tesseract's decoding
# For visualizing tesseract we use the `verbose` flag to get the decoding information.
# ## [Link to visualizer](https://quantumlib.github.io/tesseract-decoder/viz/)
# * `verbose` - A boolean flag that, when `True`, enables verbose logging. This is useful for debugging and understanding the decoder's internal behavior, as it will print information about the search process.

# %% colab={"base_uri": "https://localhost:8080/"} id="DzWRL1cNjyix" outputId="ef6600a9-1ade-4208-ce50-fc77b6b00db7"
# Remove the existing directory and its contents
# !rm -rf tesseract-decoder
# Clone the repository
# !git clone https://github.com/quantumlib/tesseract-decoder.git

# %% colab={"base_uri": "https://localhost:8080/"} id="ZNKaqvN8dE-X" outputId="34ccf8b2-a1fc-4f1c-e1d9-1ef3f99d594e"
# !curl 'https://raw.githubusercontent.com/quantumlib/tesseract-decoder/refs/heads/main/testdata/colorcodes/r%3D9%2Cd%3D9%2Cp%3D0.002%2Cnoise%3Dsi1000%2Cc%3Dsuperdense_color_code_X%2Cq%3D121%2Cgates%3Dcz.stim' > d9r9colorcode_p002.stim


# %% id="Cdo-oenEdF1-"
import stim

circuit = stim.Circuit.from_file('d9r9colorcode_p002.stim')

# %% colab={"base_uri": "https://localhost:8080/"} id="awJYxAOMTc3t" outputId="3ee82ddf-8cb0-4a31-ef95-8d7f3b090c33"
import tesseract_decoder
import tesseract_decoder.tesseract as tesseract
import numpy as np
import time
import contextlib
import io

num_shots = 100
dem = circuit.detector_error_model()
dets, obs = circuit.compile_detector_sampler().sample(num_shots, separate_observables=True)

tesseract_config1 = tesseract.TesseractConfig(
    dem=dem,
    pqlimit=10000,
    verbose=False,
    create_visualization=True,
    det_orders=tesseract_decoder.utils.build_det_orders(
        dem, num_det_orders=2,
        method=tesseract_decoder.utils.DetOrder.DetIndex,
        seed=2384753),
)

tesseract_dec = tesseract.TesseractDecoder(tesseract_config1)

# Run and time the Tesseract decoder
num_errors = 0
start_time = time.time()
for shot in range(len(dets)):
  obs_predicted = tesseract_dec.decode(dets[shot])
  obs_actual = obs[shot]
  if np.any(obs_predicted != obs_actual):
    num_errors += 1
end_time = time.time()
print(f'Tesseract: num_errors / num_shots = {num_errors} / {len(dets)} \n time {end_time - start_time} s')

# Print with the visualizer
tesseract_dec.visualizer.write('/content/tmp.txt')

# %% id="MuQb8XQlpvU6"
# !cat tmp.txt | grep -E 'Error|Detector|activated_errors|activated_detectors' > logfile.txt

# %% colab={"base_uri": "https://localhost:8080/"} id="WExtQ3x4j_Md" outputId="7c5f2373-5ecb-450c-ad3f-c9915458d9a6"
# !python tesseract-decoder/viz/to_json.py logfile.txt -o logfile.json

# %% [markdown] id="HSdTwXBINjkH"
# copy the json file and upload it [here to see the visualizaion](https://quantumlib.github.io/tesseract-decoder/viz/)

# %% [markdown] id="QehTGJcB7-Ca"
# # Accuracy Comparison between Tesseract and ILP

# %% colab={"base_uri": "https://localhost:8080/"} id="GOY0hHYx79HC" outputId="a1a35b20-255f-40ab-d71a-202b7ad9edf3"
circuit = stim.Circuit.from_file('d5r5colorcode_p001.stim')
dem = circuit.detector_error_model()

tesseract_dec = tesseract_decoder.tesseract.TesseractConfig(
    dem=dem,
    pqlimit=10000,
    det_beam=5,
    det_orders=tesseract_decoder.utils.build_det_orders(
        dem, num_det_orders=10,
        method=tesseract_decoder.utils.DetOrder.DetIndex,
        seed=2384753),
    no_revisit_dets=True,
).compile_decoder()

ilp_dec = tesseract_decoder.simplex.SimplexConfig(
    dem=dem, parallelize=True).compile_decoder()

num_shots = 1000
dets, obs = circuit.compile_detector_sampler(seed=237435).sample(num_shots, separate_observables=True)

num_errors_tesseract = 0
num_errors_tesseract_no_error_ilp = 0
start_time = time.time()
for shot in range(len(dets)):
  obs_predicted = tesseract_dec.decode(dets[shot])
  obs_actual = obs[shot]
  if np.any(obs_predicted != obs_actual):
    num_errors_tesseract += 1
    obs_predicted_ilp = ilp_dec.decode(dets[shot])
    if not np.any(obs_predicted_ilp != obs_actual):
      num_errors_tesseract_no_error_ilp += 1

end_time = time.time()
print(f'Tesseract: num_errors / num_shots = {num_errors_tesseract} / {len(dets)}')
print(f'num_errors_tesseract_no_error_ilp = {num_errors_tesseract_no_error_ilp}')
print(f'time {end_time - start_time} s')

# %% id="CijALG4TXFaP"
