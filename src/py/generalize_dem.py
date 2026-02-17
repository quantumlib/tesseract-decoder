import stim
import numpy as np
from typing import List

def get_dets_logicals(error: stim.DemInstruction):
  dets = set()
  logicals = set()
  for t in error.targets_copy():
    if t.is_logical_observable_id():
      logicals = logicals.symmetric_difference({t.val})
    elif t.is_relative_detector_id():
      dets = dets.symmetric_difference({t.val})
  return dets, logicals

def spatial_key(detector_coords: dict, min_t_coord:float, max_t_coord: float, error: stim.DemInstruction):
  dets, logicals = get_dets_logicals(error)
  d_coords = sorted([tuple(detector_coords[d]) for d in dets])
  min_d_coord = d_coords[0]
  relative_d_coords = [tuple(np.array(c)-np.array(min_d_coord)) for c in d_coords]
  min_xy = (min_d_coord[0], min_d_coord[1])
  min_t_error = min(c[2] for c in d_coords)
  max_t_error = min(c[2] for c in d_coords)
  is_begin = bool(min_t_error == min_t_coord)
  is_end = bool(max_t_error == max_t_coord)
  rel_coords = tuple(sorted(relative_d_coords))
  return (min_xy, rel_coords, tuple(logicals), is_begin, is_end)
  # return (min_xy, rel_coords, tuple(logicals))

def get_detector_coords(dem: stim.DetectorErrorModel):
  detector_coords = {}
  for inst in dem.flattened():
    if inst.type != 'detector':
      continue
    coords = np.array(inst.args_copy())
    dets = inst.targets_copy()
    D = dets[0].val
    detector_coords[D] = coords[:3]
  min_t_coord = min(c[2] for c in detector_coords.values())
  max_t_coord = max(c[2] for c in detector_coords.values())
  return detector_coords, min_t_coord, max_t_coord

def get_key_to_probabilities(spatial_data, template, verbose=False):
  key_to_probabilities = {}
  for inst in template.flattened():
    if inst.type != 'error':
      continue
    probability = inst.args_copy()[0]
    key = spatial_key(*spatial_data, inst)
    if key not in key_to_probabilities:
      key_to_probabilities[key] = []
    key_to_probabilities[key].append(probability)
  if verbose:
    print(f'identified {len(key_to_probabilities)} distinct errors out of {template.num_errors}')
  return key_to_probabilities

def merge_concat(dictionaries: List[dict]):
  merged = {}
  for d in dictionaries:
    for k in d:
      if k not in merged:
        merged[k] = []
      merged[k] = np.concatenate([merged[k], d[k]])
  return merged
    

def generalize(templates: List[stim.DetectorErrorModel], scaffold: stim.DetectorErrorModel, verbose: bool=False) -> stim.DetectorErrorModel:
  # Get detector coords for all detectors
  spatial_data_scaffold = get_detector_coords(scaffold)
  # Build a lookup table from unique key to probabilities
  all_key_to_probabilities = [
    get_key_to_probabilities(get_detector_coords(template), template, verbose=verbose)
    for template in templates
  ]
  key_to_probabilities = merge_concat(all_key_to_probabilities)
  key_to_probability = {
    key: float(np.mean(probabilities))
    for key, probabilities in key_to_probabilities.items()
  }
  output_dem = stim.DetectorErrorModel()
  for inst in scaffold.flattened():
    if inst.type == 'error':
      # update the probability
      key = spatial_key(*spatial_data_scaffold, inst)
      inst = stim.DemInstruction(type='error', args=[key_to_probability[key]], targets=inst.targets_copy())
    output_dem.append(inst)

  return output_dem


def call_generalize(template_fnames: List[str], scaffold_fname: str, output_fname: str, verbose: bool=False):
  template_dems = [stim.DetectorErrorModel.from_file(template_fname) for template_fname in template_fnames]
  scaffold_dem = stim.DetectorErrorModel.from_file(scaffold_fname)
  output_dem = generalize(template_dems, scaffold_dem, verbose)
  if output_fname == '-':
      print(output_dem)
  else:
      output_dem.to_file(output_fname)

def main():
  import argparse
  parser = argparse.ArgumentParser(description="Generalize detector error models using templates and scaffold.")
  parser.add_argument(
      '--template', 
      required=True, 
      action='append',
      help="Template file names (at least one required)"
  )
  parser.add_argument(
      '--scaffold', 
      required=True, 
      help="Scaffold file name"
  )
  parser.add_argument(
      '--out', 
      required=True, 
      help="Output file name (use '-' for stdout)"
  )
  parser.add_argument(
      '--verbose',
      action='store_true',
  )
  args = parser.parse_args()
  call_generalize(args.template, args.scaffold, args.out, verbose=args.verbose)

if __name__ == '__main__':
  main()
