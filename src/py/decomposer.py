import stim


def get_dets_logicals(error: stim.DemInstruction):
  dets = set()
  logicals = set()
  for t in error.targets_copy():
    if t.is_logical_observable_id():
      logicals = logicals.symmetric_difference({t.val})
    elif t.is_relative_detector_id():
      dets = dets.symmetric_difference({t.val})
  return dets, logicals


def _decompose_by_detector_partition(dem, partition_func):
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

    # Make a new instruction where the detectors are decomposed.
    targets_by_basis = [[], []]
    observables = []
    dets, logicals = get_dets_logicals(instruction)
    for det in sorted(dets):
      coord = detector_coords[det]
      targets_by_basis[partition_func(coord)].append(stim.target_relative_detector_id(det))
    for obs in sorted(logicals):
      # Logical observables are placed in component 0.
      observables.append(stim.target_logical_observable_id(obs))

    all_targets = []
    for targets in targets_by_basis:
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


def do_decomposition_last_coordinate_index(dem):
  def partition_func(coord):
    assert coord[-1] == int(coord[-1]) and coord[-1] in {0, 1}, 'invalid tail coordinate'
    return int(coord[-1])

  return _decompose_by_detector_partition(dem, partition_func)


def do_decomposition_stim_surface_code(dem):
  def partition_func(coord):
    assert len(coord) >= 2, 'surface-code decomposition requires detector x/y coordinates'
    x, y = coord[0], coord[1]
    assert x == int(x) and y == int(y), 'surface-code decomposition requires integer detector x/y coordinates'
    return ((int(x) - int(y)) % 4) // 2

  return _decompose_by_detector_partition(dem, partition_func)


def apply_fix(in_fname, out_fname, fix_func):
  dem = stim.DetectorErrorModel.from_file(in_fname)
  fixed = fix_func(dem)
  fixed.to_file(out_fname)


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser('Error mechanism decomposition script')
  parser.add_argument(
    '--method',
    required=True,
    type=str,
    help='Decomposition strategy. Must be one of: last-coordinate-index, stim-surface-code',
  )
  parser.add_argument('--in', dest='in_fname', help='DEM file to ingest', required=True, type=str)
  parser.add_argument('--out', dest='out_fname', help='DEM file to produce', required=True, type=str)
  args = parser.parse_args()
  if args.method == 'last-coordinate-index':
    apply_fix(args.in_fname, args.out_fname, do_decomposition_last_coordinate_index)
  elif args.method == 'stim-surface-code':
    apply_fix(args.in_fname, args.out_fname, do_decomposition_stim_surface_code)
  else:
    raise ValueError(f'invalid method: {args.method}')
