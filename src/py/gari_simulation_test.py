import stim
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
import glob
import os
from _tesseract_py_util.gari_dem_utils import get_detector_types, dem_to_check_matrices, matrices_to_dem, gari_transform, assign_prior_weights, get_detector_orderings

def get_target_path(relative_path):
    workspace_root = os.environ.get("BUILD_WORKSPACE_DIRECTORY", "")
    return os.path.join(workspace_root, relative_path)

def testdata_one_basis_circuit(circuit_og,z_basis=True):
    circuit = circuit_og.flattened().copy()
    for i in range(len(circuit)-1,-1,-1):
        if circuit[i].name == "DETECTOR":
            args = circuit[i].gate_args_copy()
            if z_basis and len(args) > 3 and args[3] <= 2:  
                # remove x detectors
                circuit.pop(i)

            if not z_basis and len(args) > 3 and args[3] >= 3:
                circuit.pop(i)

    return circuit

def load_circuit(codename,d,r,p,obs_basis,noise='si1000'):
    fname=""
    if codename == "surfacecodes":
        fname = f"testdata/{codename}/r={r},d={d},p={p},noise={noise},c=surface_code_{obs_basis}"
    if codename == "bivariatebicyclecodes":
        fname = f"testdata/{codename}/r={r},d={d},p={p},noise={noise},c=bivariate_bicycle_{obs_basis}"
    if codename == "colorcodes":
        fname = f"testdata/{codename}/r={r},d={d},p={p},noise={noise},c=superdense_color_code_{obs_basis}"
    circuit = None
    try:
        fname = get_target_path(fname)
        fname = glob.glob(fname + '*.stim')[0]
        circuit = stim.Circuit.from_file(fname)
        print("successful")
    except:
        raise("could not find the circuit")
    
    return circuit



def test_gari_transform():
    print("Reading Circuit...")
    import sys
    if len(sys.argv) > 1:
        circuit = stim.Circuit.from_file(sys.argv[1])
    else:
        circuit = load_circuit("surfacecodes", d=3, r=3, p=0.001, obs_basis='Z')
        # circuit = load_circuit("bivariatebicyclecodes", d=10, r=10, p=0.001, obs_basis='Z')
        # circuit = load_circuit("colorcodes", d=7, r=7, p=0.001, obs_basis='Z')

    dem = circuit.detector_error_model()
    print("Extracting DEM...")
    dem = circuit.detector_error_model(
        decompose_errors=False, 
        flatten_loops=True, 
        ignore_decomposition_failures=True
    )
    # dem2 = circuit.detector_error_model(
    #     decompose_errors=True, 
    #     flatten_loops=True, 
    #     ignore_decomposition_failures=True
    # )
    # print(dem)
    # print("decomposed",dem2)
    det_types = get_detector_types(circuit)
    print(f"Total Detectors: {dem.num_detectors}")
    print(f"X Detectors: {np.sum(det_types == 1)}, Z Detectors: {np.sum(det_types == 3)}")
    
    print("Building original Check Matrix...")
    H, L, priors, errors = dem_to_check_matrices(dem)
    # H, L, priors, errors = dem_to_check_matrices(dem2)
    print(f"Original Check Matrix Shape: {H.shape}")
    print(f"Original Observables Matrix Shape: {L.shape}")
    
    # print("Applying Gari Transform (matrices)...")
    # H_gari, L_gari, priors_gari, L_ez_prime, L_ex_prime = gari_transform(H, L, det_types, priors, return_dem=False)
    # print(f"Gari Matrix Shape: {H_gari.shape}")
    # print(f"Gari Observables Matrix Shape: {L_gari.shape}")
    # print(f"Gari Priors Shape: {priors_gari.shape}")
    

    print("Applying Gari Transform (return_dem=False)...")
    gari_structure = gari_transform(H, L, det_types)
    gari_matrix = gari_structure["gari_matrix"]
    gari_obs_matrix = gari_structure["gari_obs_matrix"]
    gari_obs_matrix_og = gari_structure["gari_obs_matrix_og"]
    
    p_agg = assign_prior_weights(gari_structure, "modeA", priors)
    dem_agg = matrices_to_dem(gari_matrix, gari_obs_matrix, p_agg)
    
    modes_to_test = {
        "Mode N: ez,ex,ey Keep, ez',ex' XOR Aggregated": "modeN",
        "Mode P: LP based Weight Distribution (Fixed Epsilon)": "modeP",
        "Mode Q: LP based Weight Distribution (No Lambda)": "modeQ",
        "Mode R: LP based Weight Distribution (Uniform Lambda)": "modeR",
        "Mode S: LP based Weight Distribution (Weighted Lambda)": "modeS",
        "Mode U: Max-Min Formulation (Zero-Cost Reals allowed, Lambda = 1)": "modeU",
        "Mode V: Max-Min Formulation (Zero-Cost Reals allowed, Topologically Weighted Lambda)": "modeV",
        "Mode S2: Max-Min Formulation (g_real >= t, Topologically Weighted Lambda)": "modeS2",
        "Mode SO: Max-Min Formulation (g_real >= t, Original Weighted Lambda, Safe Lambda Reg)": "modeSO",
        "Mode SO2: Max-Min Formulation (g_real >= t, Topologically Weighted Lambda, Safe Lambda Reg)": "modeSO2"
    }
    
    prior_modes = {}
    for pretty_name, mode_id in modes_to_test.items():
        try:
            p_weights = assign_prior_weights(gari_structure, mode_id, priors)
            prior_modes[pretty_name] = matrices_to_dem(gari_matrix, gari_obs_matrix, p_weights)
        except NotImplementedError:
            pass
            
    print(f"Gari Matrix Shape: {gari_matrix.shape}")
    print(f"Gari DEM has {dem_agg.num_detectors} detectors and {dem_agg.num_errors} errors.")
    orig_row_weights = np.sum(H.toarray(), axis=1)
    print(f"Original Avg Row Weight: {np.mean(orig_row_weights):.2f}")
    
    print("Success! Gari matrix generated and verified.")

    import time
    from tesseract_decoder.tesseract_sinter_compat import make_tesseract_sinter_decoders_dict
    import tesseract_decoder.tesseract as tesseract
    import tesseract_decoder.utils
    has_tesseract = True

    if has_tesseract:
        print("\n--- Running Tesseract Decoder Comparison ---")

        num_shots = 100
        sampler = circuit.compile_detector_sampler(seed=0)
        shots, obs_shots = sampler.sample(shots=num_shots, separate_observables=True)
        
        sinter_decoders = make_tesseract_sinter_decoders_dict()
        short_beam_decoder_obj = sinter_decoders["tesseract-long-beam"]
        base_config = short_beam_decoder_obj.compile_decoder_for_dem(dem=dem).decoder.config
        base_config.det_orders = [list(range(dem.num_detectors))]
        
        print("\nCompiling Tesseract for Original DEM (Forced 1 Order)...")
        decoder_orig = tesseract.TesseractDecoder(base_config)
        
        print("Decoding Original DEM (1 Order)...")
        start_time = time.time()
        predicted_obs_orig = decoder_orig.decode_batch(shots)
        time_orig = time.time() - start_time
        
        correct_orig = np.sum(np.all(predicted_obs_orig == obs_shots, axis=1))
        print(f"Original DEM (1 Order): {correct_orig}/{num_shots} correct, Time: {time_orig:.4f}s")
        
        orders = {
            "order2": get_detector_orderings(gari_structure, det_types, "order2"),
            "order4": get_detector_orderings(gari_structure, det_types, "order4"),
            "order7": get_detector_orderings(gari_structure, det_types, "order7"),
            "order9": get_detector_orderings(gari_structure, det_types, "order9"),
            "order10": get_detector_orderings(gari_structure, det_types, "order10"),
        }
        
        is_x_det = (det_types == 1)
        is_z_det = (det_types == 3)
        num_virtual = dem_agg.num_detectors - dem.num_detectors
        x_shots = shots[:, is_x_det]
        z_shots = shots[:, is_z_det]
        virtual_shots = np.zeros((num_shots, num_virtual), dtype=bool)
        gari_shots = np.concatenate([x_shots, z_shots, virtual_shots], axis=1)
        
        print("\nTesting Priority Modes on Gari DEM ...")
        
        for mode_name, target_dem in prior_modes.items():
            print(f"\n>> {mode_name}")
            for name, order in orders.items():
                base_gari_config = short_beam_decoder_obj.compile_decoder_for_dem(dem=target_dem).decoder.config
                base_gari_config.det_orders = [order]
                base_gari_config.no_revisit_dets = False

                decoder_gari = tesseract.TesseractDecoder(base_gari_config)
                
                start_time = time.time()
                predicted_obs_gari = decoder_gari.decode_batch(gari_shots)
                time_gari = time.time() - start_time
                correct_gari = np.sum(np.all(predicted_obs_gari == obs_shots, axis=1))
                print(f"{name[:30]:30s} | {correct_gari}/{num_shots} correct | {time_gari:.4f}s")

if __name__ == "__main__":
    # plot_gari_structure()
    test_gari_transform()


