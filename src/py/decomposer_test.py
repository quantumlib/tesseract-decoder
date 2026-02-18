import sys
from pathlib import Path

import pytest
import stim

sys.path.append(str(Path(__file__).resolve().parent))
from decomposer import do_decomposition_last_coordinate_index, do_decomposition_stim_surface_code


def test_do_decomposition_last_coordinate_index():
  # Convention for this decomposition mode is (x, y, t, basis), where
  # basis=0 and basis=1 identify the two detector partitions.
  dem = stim.DetectorErrorModel('''
      detector(1, 2, 0, 0) D0
      detector(3, 4, 0, 1) D1
      detector(5, 6, 1, 0) D2
      detector(7, 8, 1, 1) D3
      error(0.125) D0 D1 ^ D2 D3 L0
  ''')

  actual = do_decomposition_last_coordinate_index(dem)

  expected = stim.DetectorErrorModel('''
      detector(1, 2, 0, 0) D0
      detector(3, 4, 0, 1) D1
      detector(5, 6, 1, 0) D2
      detector(7, 8, 1, 1) D3
      error(0.125) D0 D2 ^ D1 D3 L0
  ''')
  assert actual == expected


def test_do_decomposition_stim_surface_code():
  dem = stim.DetectorErrorModel('''
      detector(2, 0, 0) D0
      detector(2, 2, 0) D1
      detector(6, 4, 0) D2
      detector(0, 4, 0) D3
      error(0.125) D0 D1 D2 D3
  ''')

  actual = do_decomposition_stim_surface_code(dem)

  expected = stim.DetectorErrorModel('''
      detector(2, 0, 0) D0
      detector(2, 2, 0) D1
      detector(6, 4, 0) D2
      detector(0, 4, 0) D3
      error(0.125) D1 D3 ^ D0 D2
  ''')
  assert actual == expected


def test_do_decomposition_stim_surface_code_requires_xy_coords():
  dem = stim.DetectorErrorModel('''
      detector(1) D0
      error(0.125) D0
  ''')

  with pytest.raises(AssertionError, match='requires detector x/y coordinates'):
    do_decomposition_stim_surface_code(dem)


def test_do_decomposition_last_coordinate_index_dedupes_mod_2_targets():
  dem = stim.DetectorErrorModel('''
      detector(0, 0, 0, 0) D0
      detector(1, 0, 0, 1) D1
      error(0.125) D0 D0 D1 D1 L0 L0
  ''')

  actual = do_decomposition_last_coordinate_index(dem)

  expected = stim.DetectorErrorModel('''
      detector(0, 0, 0, 0) D0
      detector(1, 0, 0, 1) D1
      error(0.125)
  ''')
  assert actual == expected
