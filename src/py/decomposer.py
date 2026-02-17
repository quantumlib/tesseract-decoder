import stim


def do_decomposition_last_coordinate_index(dem):
  detector_coords = {}

  for instruction in dem.flattened():
    if instruction.type != 'detector':
      continue
    for d in instruction.targets_copy():
      detector_coords[d.val] = instruction.args_copy()

  output_dem = stim.DetectorErrorModel()
  for instruction in dem.flattened():
    if instruction.type != 'error':
      output_dem.append(instruction)
      continue
    # Make a new instruction where the detectors are decomposed
    targets_by_basis = [[],[]]
    observables = []
    for d in instruction.targets_copy():
      if d.is_separator():
        # Ignore the existing decomposition, if present
        continue
      if d.is_relative_detector_id():
        coord = detector_coords[d.val]
        assert coord[-1] == int(coord[-1]) and coord[-1] in {0, 1}, 'invalid tail coordinate'
        targets_by_basis[int(coord[-1])].append(d)
      else:
        # Logical observables are placed in component 0
        assert d.is_logical_observable_id()
        observables.append(d)

    all_targets = []
    for i, targets in enumerate(targets_by_basis):
      if len(all_targets) and len(targets):
        all_targets.append(stim.target_separator())
      all_targets += targets
    all_targets += observables
    probability = instruction.args_copy()[0]
    new_instruction = stim.DemInstruction(
      type='error',
      args=[probability],
      targets=all_targets,
    )
    output_dem.append(new_instruction)
  return output_dem

def apply_fix(in_fname, out_fname, fix_func):
  dem = stim.DetectorErrorModel.from_file(in_fname)
  fixed = fix_func(dem)
  fixed.to_file(out_fname)



if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser('Error mechanism decomposition script')
  parser.add_argument(
    '--method', required=True, type=str,
    help='Decomposition strategy. Must be last-coordinate-index')
  parser.add_argument('--in', dest='in_fname', help='DEM file to ingest', required=True, type=str)
  parser.add_argument('--out', dest='out_fname', help='DEM file to produce', required=True, type=str)
  args = parser.parse_args()
  if args.method == 'last-coordinate-index':
    apply_fix(args.in_fname, args.out_fname, do_decomposition_last_coordinate_index)
  # Todo -- add Stim surface code coordinate convention, optionally add connected-components strategy
  else:
    raise ValueError(f'invalid method: {args.method}')
