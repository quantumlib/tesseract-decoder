import os
import json
import math
import re
import statistics
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm
import numpy as np
from scipy.stats import binomtest
import stim

_E_CACHE = {}

# Fallback mapping extracted directly from your provided output.
_FALLBACK_DETECT_MAP = {
    'testdata/bivariatebicyclecodes/r=10,d=10,p=0.001,noise=si1000,c=bivariate_bicycle_X,nkd=[[108,8,10]],q=216,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim': (42012, 1090),
    'testdata/bivariatebicyclecodes/r=10,d=10,p=0.001,noise=si1000,c=bivariate_bicycle_X,nkd=[[90,8,10]],q=180,iscolored=True,A_poly=x^9+y+y^2,B_poly=x^7+1+x^2.stim': (35010, 910),
    'testdata/bivariatebicyclecodes/r=10,d=10,p=0.001,noise=si1000,c=bivariate_bicycle_Z,nkd=[[108,8,10]],q=216,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim': (41958, 1090),
    'testdata/bivariatebicyclecodes/r=10,d=10,p=0.001,noise=si1000,c=bivariate_bicycle_Z,nkd=[[90,8,10]],q=180,iscolored=True,A_poly=x^9+y+y^2,B_poly=x^7+1+x^2.stim': (34965, 910),
    'testdata/bivariatebicyclecodes/r=10,d=10,p=0.002,noise=si1000,c=bivariate_bicycle_X,nkd=[[108,8,10]],q=216,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim': (42012, 1090),
    'testdata/bivariatebicyclecodes/r=10,d=10,p=0.002,noise=si1000,c=bivariate_bicycle_X,nkd=[[90,8,10]],q=180,iscolored=True,A_poly=x^9+y+y^2,B_poly=x^7+1+x^2.stim': (35010, 910),
    'testdata/bivariatebicyclecodes/r=10,d=10,p=0.002,noise=si1000,c=bivariate_bicycle_Z,nkd=[[108,8,10]],q=216,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim': (41958, 1090),
    'testdata/bivariatebicyclecodes/r=10,d=10,p=0.002,noise=si1000,c=bivariate_bicycle_Z,nkd=[[90,8,10]],q=180,iscolored=True,A_poly=x^9+y+y^2,B_poly=x^7+1+x^2.stim': (34965, 910),
    'testdata/bivariatebicyclecodes/r=12,d=12,p=0.001,noise=si1000,c=bivariate_bicycle_X,nkd=[[144,12,12]],q=288,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim': (67824, 1740),
    'testdata/bivariatebicyclecodes/r=12,d=12,p=0.001,noise=si1000,c=bivariate_bicycle_Z,nkd=[[144,12,12]],q=288,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim': (67752, 1740),
    'testdata/bivariatebicyclecodes/r=12,d=12,p=0.002,noise=si1000,c=bivariate_bicycle_X,nkd=[[144,12,12]],q=288,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim': (67824, 1740),
    'testdata/bivariatebicyclecodes/r=12,d=12,p=0.002,noise=si1000,c=bivariate_bicycle_Z,nkd=[[144,12,12]],q=288,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim': (67752, 1740),
    'testdata/bivariatebicyclecodes/r=6,d=6,p=0.001,noise=si1000,c=bivariate_bicycle_X,nkd=[[72,12,6]],q=144,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim': (16200, 438),
    'testdata/bivariatebicyclecodes/r=6,d=6,p=0.001,noise=si1000,c=bivariate_bicycle_Z,nkd=[[72,12,6]],q=144,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim': (16164, 438),
    'testdata/bivariatebicyclecodes/r=6,d=6,p=0.002,noise=si1000,c=bivariate_bicycle_X,nkd=[[72,12,6]],q=144,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim': (16200, 438),
    'testdata/bivariatebicyclecodes/r=6,d=6,p=0.002,noise=si1000,c=bivariate_bicycle_Z,nkd=[[72,12,6]],q=144,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim': (16164, 438),
    'testdata/colorcodes/r=11,d=11,p=0.001,noise=si1000,c=superdense_color_code_X,q=181,gates=cz.stim': (43298, 1001),
    'testdata/colorcodes/r=11,d=11,p=0.001,noise=si1000,c=superdense_color_code_Z,q=181,gates=cz.stim': (41044, 1001),
    'testdata/colorcodes/r=11,d=11,p=0.002,noise=si1000,c=superdense_color_code_X,q=181,gates=cz.stim': (43298, 1001),
    'testdata/colorcodes/r=11,d=11,p=0.002,noise=si1000,c=superdense_color_code_Z,q=181,gates=cz.stim': (41044, 1001),
    'testdata/colorcodes/r=3,d=3,p=0.001,noise=si1000,c=superdense_color_code_X,q=13,gates=cz.stim': (420, 21),
    'testdata/colorcodes/r=3,d=3,p=0.001,noise=si1000,c=superdense_color_code_Z,q=13,gates=cz.stim': (358, 21),
    'testdata/colorcodes/r=3,d=3,p=0.002,noise=si1000,c=superdense_color_code_X,q=13,gates=cz.stim': (420, 21),
    'testdata/colorcodes/r=3,d=3,p=0.002,noise=si1000,c=superdense_color_code_Z,q=13,gates=cz.stim': (358, 21),
    'testdata/colorcodes/r=5,d=5,p=0.001,noise=si1000,c=superdense_color_code_X,q=37,gates=cz.stim': (3128, 95),
    'testdata/colorcodes/r=5,d=5,p=0.001,noise=si1000,c=superdense_color_code_Z,q=37,gates=cz.stim': (2788, 95),
    'testdata/colorcodes/r=5,d=5,p=0.002,noise=si1000,c=superdense_color_code_X,q=37,gates=cz.stim': (3128, 95),
    'testdata/colorcodes/r=5,d=5,p=0.002,noise=si1000,c=superdense_color_code_Z,q=37,gates=cz.stim': (2788, 95),
    'testdata/colorcodes/r=7,d=7,p=0.001,noise=si1000,c=superdense_color_code_X,q=73,gates=cz.stim': (9941, 259),
    'testdata/colorcodes/r=7,d=7,p=0.001,noise=si1000,c=superdense_color_code_Z,q=73,gates=cz.stim': (9143, 259),
    'testdata/colorcodes/r=7,d=7,p=0.002,noise=si1000,c=superdense_color_code_X,q=73,gates=cz.stim': (9941, 259),
    'testdata/colorcodes/r=7,d=7,p=0.002,noise=si1000,c=superdense_color_code_Z,q=73,gates=cz.stim': (9143, 259),
    'testdata/colorcodes/r=9,d=9,p=0.001,noise=si1000,c=superdense_color_code_X,q=121,gates=cz.stim': (22713, 549),
    'testdata/colorcodes/r=9,d=9,p=0.001,noise=si1000,c=superdense_color_code_Z,q=121,gates=cz.stim': (21277, 549),
    'testdata/colorcodes/r=9,d=9,p=0.002,noise=si1000,c=superdense_color_code_X,q=121,gates=cz.stim': (22713, 549),
    'testdata/colorcodes/r=9,d=9,p=0.002,noise=si1000,c=superdense_color_code_Z,q=121,gates=cz.stim': (21277, 549),
    'testdata/surfacecodes/r=11,d=11,p=0.001,noise=si1000,c=surface_code_X,q=241,gates=cz.stim': (24483, 1331),
    'testdata/surfacecodes/r=11,d=11,p=0.001,noise=si1000,c=surface_code_Z,q=241,gates=cz.stim': (24485, 1331),
    'testdata/surfacecodes/r=11,d=11,p=0.002,noise=si1000,c=surface_code_X,q=241,gates=cz.stim': (24483, 1331),
    'testdata/surfacecodes/r=11,d=11,p=0.002,noise=si1000,c=surface_code_Z,q=241,gates=cz.stim': (24485, 1331),
    'testdata/surfacecodes/r=3,d=3,p=0.001,noise=si1000,c=surface_code_X,q=17,gates=cz.stim': (219, 27),
    'testdata/surfacecodes/r=3,d=3,p=0.001,noise=si1000,c=surface_code_Z,q=17,gates=cz.stim': (221, 27),
    'testdata/surfacecodes/r=3,d=3,p=0.002,noise=si1000,c=surface_code_X,q=17,gates=cz.stim': (219, 27),
    'testdata/surfacecodes/r=3,d=3,p=0.002,noise=si1000,c=surface_code_Z,q=17,gates=cz.stim': (221, 27),
    'testdata/surfacecodes/r=5,d=5,p=0.001,noise=si1000,c=surface_code_X,q=49,gates=cz.stim': (1677, 125),
    'testdata/surfacecodes/r=5,d=5,p=0.001,noise=si1000,c=surface_code_Z,q=49,gates=cz.stim': (1679, 125),
    'testdata/surfacecodes/r=5,d=5,p=0.002,noise=si1000,c=surface_code_X,q=49,gates=cz.stim': (1677, 125),
    'testdata/surfacecodes/r=5,d=5,p=0.002,noise=si1000,c=surface_code_Z,q=49,gates=cz.stim': (1679, 125),
    'testdata/surfacecodes/r=7,d=7,p=0.001,noise=si1000,c=surface_code_X,q=97,gates=cz.stim': (5471, 343),
    'testdata/surfacecodes/r=7,d=7,p=0.001,noise=si1000,c=surface_code_Z,q=97,gates=cz.stim': (5473, 343),
    'testdata/surfacecodes/r=7,d=7,p=0.002,noise=si1000,c=surface_code_X,q=97,gates=cz.stim': (5471, 343),
    'testdata/surfacecodes/r=7,d=7,p=0.002,noise=si1000,c=surface_code_Z,q=97,gates=cz.stim': (5473, 343),
    'testdata/surfacecodes/r=9,d=9,p=0.001,noise=si1000,c=surface_code_X,q=161,gates=cz.stim': (12705, 729),
    'testdata/surfacecodes/r=9,d=9,p=0.001,noise=si1000,c=surface_code_Z,q=161,gates=cz.stim': (12707, 729),
    'testdata/surfacecodes/r=9,d=9,p=0.002,noise=si1000,c=surface_code_X,q=161,gates=cz.stim': (12705, 729),
    'testdata/surfacecodes/r=9,d=9,p=0.002,noise=si1000,c=surface_code_Z,q=161,gates=cz.stim': (12707, 729)
}

def get_circuit_metrics(circuit_path):
    if circuit_path in _E_CACHE:
        return _E_CACHE[circuit_path]
        
    for k, v in _FALLBACK_DETECT_MAP.items():
        if circuit_path.endswith(k) or k.endswith(circuit_path):
            _E_CACHE[circuit_path] = v
            return v
            
    actual_path = circuit_path
    if not os.path.exists(actual_path):
        alt_path = os.path.join('testdata', os.path.basename(circuit_path))
        if os.path.exists(alt_path):
            actual_path = alt_path
        else:
            _E_CACHE[circuit_path] = (1, 1)
            return (1, 1)
            
    try:
        circuit = stim.Circuit.from_file(actual_path)
        dem = circuit.detector_error_model(decompose_errors=False)
        count = 0
        for inst in dem:
            if inst.type == "error":
                targets = inst.targets_copy()
                seps = sum(1 for t in targets if t.is_separator())
                count += (1 + seps)
        if count == 0: count = 1
        _E_CACHE[circuit_path] = (count, circuit.num_detectors)
    except Exception:
        _E_CACHE[circuit_path] = (1, 1)
    return _E_CACHE[circuit_path]

def get_optimal_reactivate_limit(num_detectors, base_degree, c_type):
    """
    The robust M-scaling heuristic.
    Scales exponentially with base degree k, linearly with num_detectors.
    M = round( (4.5^(k-2) / 3) * num_detectors )
    """
    k = base_degree
    if k == -1:
        # Fallback to logical defaults if run with sparsify_errors=False
        k = 2 if c_type == 'surfacecodes' else 3
        
    target_m = (4.5 ** (max(2, k) - 2) / 3.0) * num_detectors
    return max(8, round(target_m))

def extract_circuit_info(circuit_path):
    info = { 'type': 'unknown', 'r': 1, 'd': 1, 'p': 0.0, 'q': 1, 'is_x_or_z': 'unknown', 'E': 1, 'num_detectors': 1 }
    if 'surfacecodes' in circuit_path: info['type'] = 'surfacecodes'
    elif 'colorcodes' in circuit_path: info['type'] = 'colorcodes'
    elif 'bivariatebicyclecodes' in circuit_path: info['type'] = 'bivariatebicyclecodes'
        
    m_r = re.search(r'r=(\d+)', circuit_path)
    if m_r: info['r'] = max(1, int(m_r.group(1)))
    m_d = re.search(r'd=(\d+)', circuit_path)
    if m_d: info['d'] = int(m_d.group(1))
    m_p = re.search(r'p=([\d\.]+)', circuit_path)
    if m_p: info['p'] = float(m_p.group(1))
    m_q = re.search(r'q=(\d+)', circuit_path)
    if m_q: info['q'] = int(m_q.group(1))
        
    if '_X' in circuit_path: info['is_x_or_z'] = 'X'
    elif '_Z' in circuit_path: info['is_x_or_z'] = 'Z'
        
    info['E'], info['num_detectors'] = get_circuit_metrics(circuit_path)
    return info

def process_data(filepath):
    data_groups = {}
    if not os.path.exists(filepath):
        print(f"Input file not found: {filepath}")
        return data_groups
        
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            row = json.loads(line)
            info = extract_circuit_info(row.get('circuit_path', ''))
            
            p_val, c_type, d_val, q_val, r_val = info['p'], info['type'], info['d'], info['q'], info['r']
            E_val, D_val = info['E'], info['num_detectors']
            
            M_val = row.get('sparsify_reactivate_limit', 0)
            if not row.get('sparsify_errors', True):
                M_val = float('inf')
            k_val = row.get('sparsify_base_degree', -1)
            key = (p_val, c_type, d_val, q_val, r_val, M_val, k_val)
            
            if key not in data_groups:
                data_groups[key] = {
                    'total_time_seconds': 0.0, 'num_shots': 0, 'num_errors': 0, 
                    'num_low_confidence': 0, 'E_sum': 0.0, 'D_sum': 0.0, 'E_count': 0
                }
                
            data_groups[key]['total_time_seconds'] += row.get('total_time_seconds', 0.0)
            data_groups[key]['num_shots'] += row.get('num_shots', 0)
            data_groups[key]['num_errors'] += row.get('num_errors', 0)
            data_groups[key]['num_low_confidence'] += row.get('num_low_confidence', 0)
            data_groups[key]['E_sum'] += E_val
            data_groups[key]['D_sum'] += D_val
            data_groups[key]['E_count'] += 1
    return data_groups

def compute_metrics(data_groups):
    metrics = []
    for key, agg in data_groups.items():
        p_val, c_type, d_val, q_val, r_val, M_val, k_val = key
        shots = agg['num_shots']
        if shots == 0: continue
            
        errs = agg['num_errors'] + agg['num_low_confidence']
        time_sec = agg['total_time_seconds']
        
        avg_E = agg['E_sum'] / agg['E_count']
        avg_D = agg['D_sum'] / agg['E_count']
        p_raw = errs / shots
        ci = binomtest(k=errs, n=shots).proportion_ci(confidence_level=0.95)
            
        metrics.append({
            'p': p_val, 'type': c_type, 'd': d_val, 'q': q_val, 'r': r_val, 
            'M': M_val, 'k': k_val, 'E': avg_E, 'D': avg_D,
            'ler': p_raw / r_val,
            'ler_err_low': (p_raw - ci.low) / r_val,
            'ler_err_high': (ci.high - p_raw) / r_val,
            'time_per_round': time_sec / shots / r_val,
            'shots': shots
        })
    return metrics

def get_M_alpha(M):
    if M == float('inf'): return 1.0
    logM = math.log2(M) if M > 0 else 0
    return min(max((logM - 4) / 8.0, 0.2), 1.0)

def interpolate_required_M(pareto, target_ler):
    valid_pts = [p for p in pareto if p['M'] > 0 and p['M'] != float('inf')]
    if len(valid_pts) == 0:
        return float('inf') if (len(pareto) > 0 and pareto[-1]['ler'] <= target_ler) else float('nan')
        
    for i in range(len(valid_pts) - 1):
        p1, p2 = valid_pts[i], valid_pts[i+1]
        if p1['ler'] >= target_ler >= p2['ler']:
            if p1['ler'] == p2['ler']: return p2['M']
            log_m1, log_m2 = math.log2(p1['M']), math.log2(p2['M'])
            ratio = (target_ler - p2['ler']) / (p1['ler'] - p2['ler'])
            return 2 ** (log_m2 + ratio * (log_m1 - log_m2))
            
    if valid_pts[0]['ler'] <= target_ler: return valid_pts[0]['M']
    return float('inf')

def fit_power_law(x_vals, y_vals):
    valid_pairs = [(x, y) for x, y in zip(x_vals, y_vals) if x > 0 and y > 0 and y != float('inf') and not math.isnan(y)]
    if len(valid_pairs) < 2: return float('nan'), float('nan'), float('nan')
        
    log_x = [math.log2(x) for x, y in valid_pairs]
    log_y = [math.log2(y) for x, y in valid_pairs]
    mean_lx, mean_ly = sum(log_x) / len(log_x), sum(log_y) / len(log_y)
    
    num = sum((lx - mean_lx) * (ly - mean_ly) for lx, ly in zip(log_x, log_y))
    den = sum((lx - mean_lx)**2 for lx in log_x)
    if den == 0: return float('nan'), float('nan'), float('nan')
        
    k = num / den
    log_c = mean_ly - k * mean_lx
    c = 2 ** log_c
    
    ss_tot = sum((ly - mean_ly)**2 for ly in log_y)
    ss_res = sum((ly - (k * lx + log_c))**2 for lx, ly in zip(log_x, log_y))
    return k, c, 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

def extract_fit_data(metrics, p_filter):
    filtered = [m for m in metrics if m['p'] == p_filter]
    code_types = list(set([m['type'] for m in filtered]))
    fit_data = {}
    for c_type in code_types:
        c_metrics = [m for m in filtered if m['type'] == c_type]
        circuits = {}
        for m in c_metrics:
            ckey = (m['d'], m['q'])
            if ckey not in circuits: circuits[ckey] = []
            circuits[ckey].append(m)
            
        data = {'E': [], 'M0': [], 'M5': [], 'M10': [], 'min_ler': [], 'ckey': []}
        for ckey in sorted(circuits.keys()):
            pts = circuits[ckey]
            pts_sorted = sorted(pts, key=lambda x: x['time_per_round'])
            pareto, best_ler = [], float('inf')
            avg_E = pts[0]['E']
            
            for pt in pts_sorted:
                if pt['ler'] < best_ler:
                    pareto.append(pt)
                    best_ler = pt['ler']
                    
            if len(pareto) == 0: continue
            data['E'].append(avg_E)
            data['M0'].append(interpolate_required_M(pareto, best_ler * 1.0001))
            data['M5'].append(interpolate_required_M(pareto, best_ler * 1.05))
            data['M10'].append(interpolate_required_M(pareto, best_ler * 1.10))
            data['min_ler'].append(best_ler)
            data['ckey'].append(ckey)
        fit_data[c_type] = data
    return fit_data

def evaluate_scaling_ansatz(metrics, p_filter):
    fit_data = extract_fit_data(metrics, p_filter)
    if not fit_data: return
        
    print(f"\n{'='*80}\n ERROR COUNT SCALING ANALYSIS: M vs DEM Errors (E) [p = {p_filter}]\n{'='*80}")
    for c_type, data in fit_data.items():
        print(f"\n--- {c_type.upper()} ---")
        print(f"{'d':<4} | {'q':<5} | {'Avg Errors (E)':<14} | {'Min LER':<9} | {'+0% M':<10} | {'+5% M':<10} | {'+10% M':<10}")
        print("-" * 79)
        for i in range(len(data['E'])):
            c_d, c_q = data['ckey'][i]
            m0, m5, m10 = data['M0'][i], data['M5'][i], data['M10'][i]
            sm0  = f"{m0:.1f}" if m0 != float('inf') and not math.isnan(m0) else "inf"
            sm5  = f"{m5:.1f}" if m5 != float('inf') and not math.isnan(m5) else "inf"
            sm10 = f"{m10:.1f}" if m10 != float('inf') and not math.isnan(m10) else "inf"
            print(f"{c_d:<4} | {c_q:<5} | {data['E'][i]:<14.1f} | {data['min_ler'][i]:.2e} | {sm0:<10} | {sm5:<10} | {sm10:<10}")
            
        print("-" * 79)
        k0, c0, r0 = fit_power_law(data['E'], data['M0'])
        k5, c5, r5 = fit_power_law(data['E'], data['M5'])
        k10, c10, r10 = fit_power_law(data['E'], data['M10'])
        
        print(f"POWER LAW FIT (M = c * E^k)")
        print(f"{'+0% penalty':<15} | {k0:<15.4f} | {c0:<15.4e} | {r0:<10.4f}")
        print(f"{'+5% penalty':<15} | {k5:<15.4f} | {c5:<15.4e} | {r5:<10.4f}")
        print(f"{'+10% penalty':<15} | {k10:<15.4f} | {c10:<15.4e} | {r10:<10.4f}\n")

def plot_power_law_fits(metrics, p_filter, filename, title):
    fit_data = extract_fit_data(metrics, p_filter)
    if not fit_data: return
    code_types = ['surfacecodes', 'colorcodes', 'bivariatebicyclecodes']
    present_types = [ct for ct in code_types if ct in fit_data]
    if len(present_types) == 0: return
        
    display_names = {'surfacecodes': 'Surface Codes', 'colorcodes': 'Color Codes', 'bivariatebicyclecodes': 'Bicycle Codes'}
    fig, axes = plt.subplots(nrows=1, ncols=len(present_types), figsize=(6 * len(present_types), 6))
    if len(present_types) == 1: axes = [axes]
        
    colors, markers = {'+0%': 'black', '+5%': 'red', '+10%': 'orange'}, {'+0%': 'o', '+5%': 's', '+10%': '^'}
    for idx, c_type in enumerate(present_types):
        ax = axes[idx]
        data = fit_data[c_type]
        for penalty, key in [('+0%', 'M0'), ('+5%', 'M5'), ('+10%', 'M10')]:
            valid_x = [x for x, y in zip(data['E'], data[key]) if y > 0 and y != float('inf') and not math.isnan(y)]
            valid_y = [y for y in data[key] if y > 0 and y != float('inf') and not math.isnan(y)]
            if len(valid_x) > 0:
                ax.scatter(valid_x, valid_y, color=colors[penalty], marker=markers[penalty], alpha=0.7, label=f"{penalty} Data")
                if len(valid_x) > 1:
                    k, c, r2 = fit_power_law(valid_x, valid_y)
                    if not math.isnan(k):
                        fit_x = np.linspace(min(valid_x), max(valid_x), 100)
                        ax.plot(fit_x, c * (fit_x ** k), color=colors[penalty], linestyle='--', alpha=0.8, label=f"{penalty} Fit (k={k:.2f})")
                        
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2)
        ax.grid(True, which='both', linestyle='--', alpha=0.4)
        ax.set_xlabel('Average DEM Errors (E)')
        if idx == 0: ax.set_ylabel('Required M limit')
        ax.set_title(f"{display_names[c_type]}")
        ax.legend(fontsize=8)
        
    fig.suptitle(title, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.savefig(filename.replace('png','pdf'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_tradeoff_arrows(metrics, p_filter, filename, title):
    plt.figure(figsize=(10, 8))
    filtered = [m for m in metrics if m['p'] == p_filter or p_filter == 'both']
    circuits = {}
    for m in filtered:
        ckey = (m['type'], m['d'], m['q'], m['p'])
        if ckey not in circuits: circuits[ckey] = []
        circuits[ckey].append(m)
        
    color_map = {'surfacecodes': '#5D95E8', 'colorcodes': '#F6C644', 'bivariatebicyclecodes': 'fuchsia'}
    unique_qd = sorted(list(set((c[1], c[2]) for c in circuits.keys())))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', '<', '>']
    marker_map = {unique_qd[i]: markers[i % len(markers)] for i in range(len(unique_qd))}

    for ckey, points in circuits.items():
        c_type, c_d, c_q, c_p = ckey
        base_color = color_map.get(c_type, 'black')
        marker = marker_map.get((c_d, c_q), 'o')
        
        before_pts = [p for p in points if p['M'] == float('inf')]
        if not before_pts: continue
        before_pt = before_pts[0]
        
        k_vals = [p['k'] for p in points if p['k'] != -1]
        k_val = k_vals[0] if len(k_vals) > 0 else -1
        avg_D = sum(p['D'] for p in points) / len(points)
        opt_M = get_optimal_reactivate_limit(avg_D, k_val, c_type)
        
        valid_pts = [p for p in points if p['M'] > 0 and p['M'] != float('inf')]
        if not valid_pts: continue
            
        after_pt = min(valid_pts, key=lambda x: abs(x['M'] - opt_M))
        
        x0, y0 = before_pt['time_per_round'], before_pt['ler']
        x1, y1 = after_pt['time_per_round'], after_pt['ler']
        
        if x0 <= 0 or x1 <= 0 or y0 <= 0 or y1 <= 0: continue
            
        is_p002 = (c_p == 0.002)
        fc = 'white' if is_p002 else base_color
        ls = '--' if is_p002 else '-'
        ec_before = 'black'
        ec_after = base_color if is_p002 else 'none'
        lw_after = 1.5 if is_p002 else 0
        
        # --- Add Error Bars ---
        y0_err = [[before_pt['ler_err_low']], [before_pt['ler_err_high']]]
        y1_err = [[after_pt['ler_err_low']], [after_pt['ler_err_high']]]
        
        plt.errorbar([x0], [y0], yerr=y0_err, fmt='none', ecolor=ec_before, alpha=0.3, zorder=1)
        plt.errorbar([x1], [y1], yerr=y1_err, fmt='none', ecolor=base_color, alpha=0.7, zorder=2)
        
        # Plot Arrow
        plt.annotate('', xy=(x1, y1), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle="->", color=base_color, lw=1.5, ls=ls, shrinkA=8, shrinkB=8), zorder=3)
        
        # Base Points
        plt.scatter([x0], [y0], facecolors=fc, edgecolors=ec_before, marker=marker, s=80, linewidths=1.5, alpha=0.4, zorder=4)
        plt.scatter([x1], [y1], facecolors=fc, edgecolors=ec_after, marker=marker, s=80, linewidths=lw_after, alpha=1.0, zorder=5)
        
        # Geometric Midpoint Annotations
        speedup = x0 / x1 if x1 > 0 else 1
        
        # --- Calculate LER Uncertainty Bounds using log-relative-risk and Delta method ---
        # Recover observed error counts
        k0 = before_pt['ler'] * before_pt['r'] * before_pt['shots']
        n0 = before_pt['shots']
        k1 = after_pt['ler'] * after_pt['r'] * after_pt['shots']
        n1 = after_pt['shots']
        
        # Apply pseudo-counts to ensure robustness against rare 0-error runs
        k0_a, n0_a = k0 + 0.5, n0 + 0.5
        k1_a, n1_a = k1 + 0.5, n1 + 0.5
        
        r_adj = (k1_a / n1_a) / (k0_a / n0_a)
        se_log_r = math.sqrt(max(0, 1/k1_a - 1/n1_a + 1/k0_a - 1/n0_a))
        
        r_low = r_adj * math.exp(-1.96 * se_log_r)
        r_high = r_adj * math.exp(1.96 * se_log_r)
        
        if round(r_low, 2) == round(r_high, 2):
            ler_str = f"{r_low:.2f}x err"
        else:
            ler_str = f"{r_low:.2f}-{r_high:.2f}x err"
        
        mid_x = math.exp((math.log(x0) + math.log(x1)) / 2)
        mid_y = math.exp((math.log(y0) + math.log(y1)) / 2)
        
        plt.text(mid_x, mid_y * 1.05, f"{speedup:.1f}x spd\n{ler_str}", 
                 fontsize=6, color='black', ha='center', va='bottom', 
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7), zorder=6)

    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.4) 
    
    valid_lers = [p['ler'] for p in filtered if p['ler'] > 0]
    if valid_lers: plt.ylim(bottom=min(valid_lers) / 5.0) 
        
    plt.xlabel('Time per round (seconds)')
    plt.ylabel('Logical Error Rate per round')
    plt.title(title)
    
    legend_elements = []
    display_names = {'surfacecodes': 'Surface Codes', 'colorcodes': 'Color Codes', 'bivariatebicyclecodes': 'Bicycle Codes'}
    
    for c_type in ['surfacecodes', 'colorcodes', 'bivariatebicyclecodes']:
        type_qds = sorted(list(set([(c[1], c[2]) for c in circuits.keys() if c[0] == c_type])))
        if type_qds:
            c_color = color_map.get(c_type, 'black')
            k_set = set([m['k'] for m in filtered if m['type'] == c_type and m['k'] != -1])
            k_str = f" (k={list(k_set)[0]})" if len(k_set) == 1 else ""
            legend_elements.append(mlines.Line2D([0], [0], color='none', label=f"  {display_names[c_type]}{k_str}"))
            for qd in type_qds:
                legend_elements.append(mlines.Line2D([0], [0], color='none', marker=marker_map[qd], markerfacecolor=c_color, 
                                                     markeredgecolor='none', markersize=8, label=f"d={qd[0]}, q={qd[1]}"))

    legend_elements.append(mlines.Line2D([0], [0], color='none', label=""))
    if p_filter == 'both' or p_filter == 0.001:
        legend_elements.append(mlines.Line2D([0], [0], color='gray', linestyle='-', lw=2, marker='o', markerfacecolor='gray', markeredgecolor='none', label='p=0.001 (Solid Line, Filled)'))
    if p_filter == 'both' or p_filter == 0.002:
        legend_elements.append(mlines.Line2D([0], [0], color='gray', linestyle='--', lw=2, marker='o', markerfacecolor='white', markeredgecolor='gray', markeredgewidth=1.5, label='p=0.002 (Dashed Line, Hollow)'))
        
    legend_elements.extend([
        mlines.Line2D([0], [0], color='gray', marker='o', linestyle='None', markerfacecolor='gray', markeredgecolor='black', markersize=8, markeredgewidth=1.5, alpha=0.4, label='Before (sparsify_errors=False)'),
        mlines.Line2D([0], [0], color='gray', marker='o', linestyle='None', markerfacecolor='gray', markeredgecolor='none', markersize=8, alpha=1.0, label='After (Heuristic M applied)')
    ])
                                          
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, labelspacing=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.savefig(filename.replace('png', 'pdf'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_ler_vs_time(metrics, p_filter, filename, title, highlight_heuristic=False):
    plt.figure(figsize=(10, 8))
    filtered = [m for m in metrics if m['p'] == p_filter or p_filter == 'both']
    circuits = {}
    for m in filtered:
        ckey = (m['type'], m['d'], m['q'], m['p'])
        if ckey not in circuits: circuits[ckey] = []
        circuits[ckey].append(m)
        
    color_map = {'surfacecodes': '#5D95E8', 'colorcodes': '#F6C644', 'bivariatebicyclecodes': 'fuchsia'}
    unique_qd = sorted(list(set((c[1], c[2]) for c in circuits.keys())))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', '<', '>']
    marker_map = {unique_qd[i]: markers[i % len(markers)] for i in range(len(unique_qd))}
        
    for ckey, points in circuits.items():
        points.sort(key=lambda x: x['M'])
        c_type, c_d, c_q, c_p = ckey
        base_color = color_map.get(c_type, 'black')
        marker = marker_map.get((c_d, c_q), 'o')
        
        is_p002 = (c_p == 0.002)
        line_style = '--' if is_p002 else '-'
        
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            seg_alpha = (get_M_alpha(p1['M']) + get_M_alpha(p2['M'])) / 2.0
            plt.plot([p1['time_per_round'], p2['time_per_round']], [p1['ler'], p2['ler']], 
                     color=base_color, linestyle=line_style, alpha=seg_alpha, linewidth=1.5, zorder=1)
        
        for p in points:
            M, alpha = p['M'], get_M_alpha(p['M'])
            sz = 80 if M == float('inf') else 50
            
            fc = 'white' if is_p002 else base_color
            ec = 'black' if M == float('inf') else (base_color if is_p002 else 'none')
            lw = 1.5 if (M == float('inf') or is_p002) else 0
                
            y_err_asym = [[p['ler_err_low']], [p['ler_err_high']]]
            plt.errorbar(p['time_per_round'], p['ler'], yerr=y_err_asym, fmt='none', ecolor=base_color, alpha=alpha, zorder=2)
            plt.scatter([p['time_per_round']], [p['ler']], facecolors=fc, edgecolors=ec, linewidths=lw, 
                        marker=marker, alpha=alpha, s=sz, zorder=3)
            
        if highlight_heuristic and len(points) > 0:
            avg_D = sum(p['D'] for p in points) / len(points)
            k_val = [p['k'] for p in points if p['k'] != -1]
            opt_M = get_optimal_reactivate_limit(avg_D, k_val[0] if k_val else -1, c_type)
            
            valid_points = [p for p in points if p['M'] > 0 and p['M'] != float('inf')]
            if valid_points:
                best_p = min(valid_points, key=lambda x: abs(x['M'] - opt_M))
                plt.scatter([best_p['time_per_round']], [best_p['ler']], facecolors='none', edgecolors='red', linewidths=2.5, marker='o', s=250, zorder=4)

    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.4) 
    
    valid_lers = [p['ler'] for p in filtered if p['ler'] > 0]
    if valid_lers: plt.ylim(bottom=min(valid_lers) / 5.0) 
        
    plt.xlabel('Time per round (seconds)')
    plt.ylabel('Logical Error Rate per round')
    plt.title(title)
    
    legend_elements = []
    display_names = {'surfacecodes': 'Surface Codes', 'colorcodes': 'Color Codes', 'bivariatebicyclecodes': 'Bicycle Codes'}
    
    for c_type in ['surfacecodes', 'colorcodes', 'bivariatebicyclecodes']:
        type_qds = sorted(list(set([(c[1], c[2]) for c in circuits.keys() if c[0] == c_type])))
        if type_qds:
            c_color = color_map.get(c_type, 'black')
            k_set = set([m['k'] for m in filtered if m['type'] == c_type and m['k'] != -1])
            k_str = f" (k={list(k_set)[0]})" if len(k_set) == 1 else ""
            legend_elements.append(mlines.Line2D([0], [0], color='none', label=f"  {display_names[c_type]}{k_str}"))
            for qd in type_qds:
                legend_elements.append(mlines.Line2D([0], [0], color='none', marker=marker_map[qd], markerfacecolor=c_color, 
                                                     markeredgecolor='none', markersize=8, label=f"d={qd[0]}, q={qd[1]}"))

    legend_elements.append(mlines.Line2D([0], [0], color='none', label=""))
    if p_filter == 'both' or p_filter == 0.001:
        legend_elements.append(mlines.Line2D([0], [0], color='gray', linestyle='-', lw=2, marker='o', markerfacecolor='gray', markeredgecolor='none', label='p=0.001 (Solid Line, Filled)'))
    if p_filter == 'both' or p_filter == 0.002:
        legend_elements.append(mlines.Line2D([0], [0], color='gray', linestyle='--', lw=2, marker='o', markerfacecolor='white', markeredgecolor='gray', markeredgewidth=1.5, label='p=0.002 (Dashed Line, Hollow)'))
        
    legend_elements.extend([
        mlines.Line2D([0], [0], color='gray', marker='o', linestyle='None', markersize=8, alpha=0.3, label='Lower M'),
        mlines.Line2D([0], [0], color='gray', marker='o', linestyle='None', markersize=10, markeredgecolor='black', markeredgewidth=1.5, label='M=inf (sparsify_errors=false)')
    ])
    
    if highlight_heuristic:
        legend_elements.append(mlines.Line2D([0], [0], color='none', label=""))
        legend_elements.append(mlines.Line2D([0], [0], marker='o', color='none', markeredgecolor='red', markerfacecolor='none', markersize=12, markeredgewidth=2.5, label='Heuristic Optimal M'))
                                          
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, labelspacing=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.savefig(filename.replace('png', 'pdf'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_stacked_ler_vs_M(metrics, p_filter, filename, title, log_y=False):
    filtered = [m for m in metrics if m['p'] == p_filter or p_filter == 'both']
    circuits = {}
    for m in filtered:
        ckey = (m['type'], m['d'], m['q'])
        if ckey not in circuits: circuits[ckey] = []
        circuits[ckey].append(m)
        
    sorted_keys = sorted(list(circuits.keys()), key=lambda x: (x[0], x[2]))
    num_subplots = len(sorted_keys)
    if num_subplots == 0: return
        
    fig, axes = plt.subplots(nrows=num_subplots, ncols=1, sharex=True, figsize=(8, 2.5 * num_subplots))
    if num_subplots == 1: axes = [axes]
        
    color_map = {'surfacecodes': '#5D95E8', 'colorcodes': '#F6C644', 'bivariatebicyclecodes': 'fuchsia'}
    display_names = {'surfacecodes': 'Surface Codes', 'colorcodes': 'Color Codes', 'bivariatebicyclecodes': 'Bicycle Codes'}
        
    for i in range(num_subplots):
        ckey = sorted_keys[i]
        c_type, c_d, c_q = ckey
        all_points = circuits[ckey]
        ax = axes[i]
        
        p_groups = {}
        for p in all_points:
            p_val = p['p']
            if p_val not in p_groups: p_groups[p_val] = []
            p_groups[p_val].append(p)
            
        finite_Ms = [p['M'] for p in all_points if p['M'] != float('inf') and p['M'] > 0]
        max_M = max(finite_Ms) if finite_Ms else 1
        min_M = min(finite_Ms) if finite_Ms else 1
        color = color_map.get(c_type, 'black')
        
        for p_val, pts in p_groups.items():
            pts.sort(key=lambda x: x['M'])
            x_vals, y_vals, y_errs_low, y_errs_high = [], [], [], []
            inf_x, inf_y, inf_err_low, inf_err_high = None, None, None, None
            zero_x, zero_y, zero_err_low, zero_err_high = None, None, None, None
            
            for pt in pts:
                if pt['M'] == float('inf'):
                    inf_x, inf_y = max_M * 4, pt['ler']
                    inf_err_low, inf_err_high = pt['ler_err_low'], pt['ler_err_high']
                elif pt['M'] == 0:
                    zero_x, zero_y = min_M / 4, pt['ler']
                    zero_err_low, zero_err_high = pt['ler_err_low'], pt['ler_err_high']
                elif pt['M'] > 0:
                    x_vals.append(pt['M']) 
                    y_vals.append(pt['ler'])
                    y_errs_low.append(pt['ler_err_low'])
                    y_errs_high.append(pt['ler_err_high'])
                    
            is_p002 = (p_val == 0.002)
            ls = '--' if is_p002 else '-'
            marker = 's' if is_p002 else 'o'
            mfc = 'white' if is_p002 else color
            
            k_set = set([p['k'] for p in pts if p['k'] != -1])
            k_str = f" (k={list(k_set)[0]})" if len(k_set) == 1 else ""
            lbl = f"{display_names.get(c_type, c_type)}{k_str} (d={c_d}, q={c_q})"
            if len(p_groups) > 1: lbl += f", p={p_val}"
            
            ax.errorbar(x_vals, y_vals, yerr=[y_errs_low, y_errs_high], fmt=f'{ls}{marker}', color=color, markerfacecolor=mfc, capsize=3, label=lbl)
            
            if inf_x is not None: ax.errorbar([inf_x], [inf_y], yerr=[[inf_err_low], [inf_err_high]], fmt='*', color=color, markerfacecolor=mfc, markersize=10, markeredgecolor='black', capsize=3)
            if zero_x is not None: ax.errorbar([zero_x], [zero_y], yerr=[[zero_err_low], [zero_err_high]], fmt='X', color=color, markerfacecolor=mfc, markersize=8, markeredgecolor='black', capsize=3)
                
        if log_y: ax.set_yscale('log')
        ax.set_xscale('log', base=2)
        ax.grid(True, which='both', linestyle='--', alpha=0.4) 
        ax.axvline(x=max_M * 2, color='gray', linestyle=':')
        ax.axvline(x=min_M / 2, color='gray', linestyle=':')
        ax.set_ylabel('LER / round')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        
    axes[-1].set_xlabel('M (log2 scale, X = 0, * = inf)')
    fig.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.savefig(filename.replace('png', 'pdf'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_mq_scaling_meta_analysis(metrics, p_filter, filename, title):
    filtered = [m for m in metrics if m['p'] == p_filter or p_filter == 'both']
    if len(filtered) == 0: return

    code_types = ['surfacecodes', 'colorcodes', 'bivariatebicyclecodes']
    display_names = {'surfacecodes': 'Surface Codes', 'colorcodes': 'Color Codes', 'bivariatebicyclecodes': 'Bicycle Codes'}
    present_types = [ct for ct in code_types if any(m['type'] == ct for m in filtered)]
    if len(present_types) == 0: return

    fig, axes = plt.subplots(nrows=len(present_types), ncols=1, figsize=(10, 6 * len(present_types)), sharex=False)
    if len(present_types) == 1: axes = [axes]

    for idx, c_type in enumerate(present_types):
        ax1 = axes[idx]
        ax2 = ax1.twinx()

        c_metrics = [m for m in filtered if m['type'] == c_type]
        circuits = {}
        for m in c_metrics:
            ckey = (m['d'], m['q'], m['p'])
            if ckey not in circuits: circuits[ckey] = []
            circuits[ckey].append(m)

        sorted_keys = sorted(list(circuits.keys()))
        all_finite_fractions = [pt['M'] / pt['E'] for pts in circuits.values() for pt in pts if pt['M'] > 0 and pt['M'] != float('inf')]
        
        if all_finite_fractions:
            min_frac, max_frac = min(all_finite_fractions), max(all_finite_fractions)
            if min_frac >= max_frac: min_frac *= 0.5; max_frac *= 2.0
        else:
            min_frac, max_frac = 0.5, 2.0
            
        x_zero, x_inf = min_frac / 4.0, max_frac * 4.0
        colors = cm.viridis([i/max(1, len(sorted_keys)-1) for i in range(len(sorted_keys))])
        
        for c_idx, ckey in enumerate(sorted_keys):
            c_d, c_q, c_p = ckey
            pts = circuits[ckey]
            
            pts_sorted = sorted(pts, key=lambda x: x['time_per_round'])
            pareto, best_ler = [], float('inf')
            for pt in pts_sorted:
                if pt['ler'] < best_ler:
                    pareto.append(pt)
                    best_ler = pt['ler']
                    
            if not pareto: continue
                
            min_ler_val = max(1e-12, best_ler)
            ref_time_val = max(1e-12, pareto[-1]['time_per_round'])
            x_vals, y1_vals, y2_vals, y1_err_low, y1_err_high = [], [], [], [], []
            
            for pt in pareto:
                if pt['M'] == 0: x = x_zero
                elif pt['M'] == float('inf'): x = x_inf
                else: x = pt['M'] / pt['E']
                    
                x_vals.append(x)
                y1_vals.append(pt['ler'] / min_ler_val)
                y2_vals.append(pt['time_per_round'] / ref_time_val)
                y1_err_low.append(pt['ler_err_low'] / min_ler_val)
                y1_err_high.append(pt['ler_err_high'] / min_ler_val)
                
            color = colors[c_idx]
            is_p002 = (c_p == 0.002)
            mfc = 'white' if is_p002 else color
            ls = '--' if is_p002 else '-'
            lbl = f"d={c_d}, q={c_q}" + (f", p={c_p}" if p_filter == 'both' else "")
            
            ax1.plot(x_vals, y1_vals, linestyle=ls, color=color, label=lbl, markersize=0)
            ax1.errorbar(x_vals, y1_vals, yerr=[y1_err_low, y1_err_high], fmt='o', color=color, markerfacecolor=mfc, capsize=4, markersize=6)
            ax2.plot(x_vals, y2_vals, marker='s', linestyle=':', color=color, markerfacecolor=mfc, alpha=0.7, markersize=5)
            
            for i, pt in enumerate(pareto):
                if pt['M'] == 0:
                    ax1.scatter(x_vals[i], y1_vals[i], marker='X', facecolor=mfc, edgecolor='black', s=80, zorder=5)
                    ax2.scatter(x_vals[i], y2_vals[i], marker='X', facecolor=mfc, edgecolor='black', s=70, alpha=0.7, zorder=5)
                elif pt['M'] == float('inf'):
                    ax1.scatter(x_vals[i], y1_vals[i], marker='*', facecolor=mfc, edgecolor='black', s=120, zorder=5)
                    ax2.scatter(x_vals[i], y2_vals[i], marker='*', facecolor=mfc, edgecolor='black', s=100, alpha=0.7, zorder=5)

        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log')
        ax1.set_ylim(bottom=0.9)
        ax1.axhline(1.00, color='black', linestyle='-', alpha=0.4, linewidth=1)
        ax1.axhline(1.05, color='red', linestyle=':', alpha=0.8, label='+5% LER Penalty')
        ax1.axhline(1.10, color='orange', linestyle=':', alpha=0.8, label='+10% LER Penalty')
        ax1.axvline(x_zero * 2, color='gray', linestyle=':')
        ax1.axvline(x_inf / 2, color='gray', linestyle=':')
        ax1.set_ylabel('Normalized LER (Solid Line, Circles)', color='black', fontweight='bold')
        ax2.set_ylabel('Normalized Time (Dotted Line, Squares)', color='gray', fontweight='bold')
        ax1.set_title(f"{display_names[c_type]} (p={p_filter})")
        
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.10, 1))

    axes[-1].set_xlabel('Fraction: M / E [log2 scale, X = 0, * = inf]')
    fig.suptitle(title, y=1.02, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.savefig(filename.replace('png', 'pdf'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    INPUT_FILE = 'aggregated_results.jsonl'
    OUTPUT_DIR = 'plots'
    
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
        
    data_groups = process_data(INPUT_FILE)
    metrics = compute_metrics(data_groups)
    
    if len(metrics) > 0:
        print(f"Loaded {len(metrics)} aggregated data points. Running analysis and generating plots...")
        
        for p_val in [0.001, 0.002]:
            evaluate_scaling_ansatz(metrics, p_val)
            
            plot_ler_vs_time(metrics, p_val, os.path.join(OUTPUT_DIR, f'ler_vs_time_p{p_val}.png'), f'LER vs Time per Shot (p={p_val})')
            plot_ler_vs_time(metrics, p_val, os.path.join(OUTPUT_DIR, f'ler_vs_time_highlighted_p{p_val}.png'), f'LER vs Time per Shot - Heuristic Target Highlighted (p={p_val})', highlight_heuristic=True)
            plot_tradeoff_arrows(metrics, p_val, os.path.join(OUTPUT_DIR, f'tradeoff_arrows_p{p_val}.png'), f'Before vs After: Sparsification Tradeoffs (p={p_val})')
            
            plot_stacked_ler_vs_M(metrics, p_val, os.path.join(OUTPUT_DIR, f'ler_vs_M_stacked_p{p_val}.png'), f'LER vs M Stacked (p={p_val})')
            plot_stacked_ler_vs_M(metrics, p_val, os.path.join(OUTPUT_DIR, f'ler_vs_M_stacked_logy_p{p_val}.png'), f'LER vs M Stacked [Log Y] (p={p_val})', log_y=True)
            
            plot_mq_scaling_meta_analysis(metrics, p_val, os.path.join(OUTPUT_DIR, f'mq_scaling_meta_p{p_val}.png'), f'M/E Scaling Efficiency Meta-Analysis (p={p_val})')
            plot_power_law_fits(metrics, p_val, os.path.join(OUTPUT_DIR, f'power_law_fits_p{p_val}.png'), f'Power Law Extrapolations (p={p_val})')
        
        # Combined versions
        plot_ler_vs_time(metrics, 'both', os.path.join(OUTPUT_DIR, 'ler_vs_time_combined.png'), 'LER vs Time per Shot (Combined)')
        plot_ler_vs_time(metrics, 'both', os.path.join(OUTPUT_DIR, 'ler_vs_time_highlighted_combined.png'), 'LER vs Time per Shot - Heuristic Target Highlighted (Combined)', highlight_heuristic=True)
        plot_tradeoff_arrows(metrics, 'both', os.path.join(OUTPUT_DIR, 'tradeoff_arrows_combined.png'), 'Before vs After: Sparsification Tradeoffs (Combined)')
        plot_stacked_ler_vs_M(metrics, 'both', os.path.join(OUTPUT_DIR, 'ler_vs_M_stacked_combined.png'), 'LER vs M Stacked (Combined)')
        plot_stacked_ler_vs_M(metrics, 'both', os.path.join(OUTPUT_DIR, 'ler_vs_M_stacked_logy_combined.png'), 'LER vs M Stacked [Log Y] (Combined)', log_y=True)
        
        print("\nDone! Plots saved successfully in the 'plots/' directory.")
    else:
        print("No valid data found to plot.")
