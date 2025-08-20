# viz/to_json.py
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

import json
import argparse
import re
import numpy as np

INT_RE = re.compile(r'-?\d+')

def parse_int_list(text):
    """Extract all integers from a string, tolerant to spaces/commas/brackets."""
    return [int(x) for x in INT_RE.findall(text)]

def parse_implicit_list(line, prefix):
    if not line.startswith(prefix):
        raise ValueError(f"Expected line to start with '{prefix}', got: {line}")
    list_part = line[len(prefix):].strip().rstrip(',')
    if not list_part:
        return []
    # Be tolerant: accept "1, 2, 3" or "1 2 3"
    return parse_int_list(list_part)

def parse_logfile(filepath):
    detector_coords = {}
    error_to_detectors = []
    frames = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not any(line.startswith(s) for s in ['Error', 'Detector', 'activated_errors', 'activated_detectors']):
            i += 1
            continue

        if line.startswith("Detector D"):
            # Example: "Detector D123 coordinate (1.0, 2.0, 3.0)"
            match = re.match(
                r'Detector D(\d+)\s+coordinate\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)',
                line
            )
            if match:
                idx = int(match.group(1))
                coord = tuple(float(match.group(j)) for j in range(2, 5))
                detector_coords[idx] = coord

        elif line.startswith("Error{"):
            # New format: Error{..., symptom=Symptom{detectors=[75 89 93 100], observables=[...]}}
            # Fallback: old format with "D###" tokens inside Symptom{...}
            dets = []

            m_detlist = re.search(r'detectors=\[([^\]]*)\]', line)
            if m_detlist:
                dets = parse_int_list(m_detlist.group(1))
            else:
                # Old fallback: scrape Symptom{...} and look for D###
                m_sym = re.search(r'Symptom\{([^}]*)\}', line)
                if m_sym:
                    tokens = m_sym.group(1).split()
                    dets = [int(t[1:]) for t in tokens if t.startswith('D') and t[1:].isdigit()]

            # Store (even if empty—we keep the index alignment with errors)
            error_to_detectors.append(dets)

        elif line.startswith("activated_errors"):
            try:
                error_line = lines[i].strip()
                det_line = lines[i + 1].strip()

                activated_errors = parse_implicit_list(error_line, "activated_errors =")
                activated_dets = parse_implicit_list(det_line, "activated_detectors =")

                frames.append({
                    "activated": activated_dets,
                    "activated_errors": activated_errors
                })

                # We consumed two lines in this block
                i += 2
                continue  # skip the unconditional i+=1 below (already advanced)
            except Exception as e:
                print(f"\n⚠️ Error parsing frame at lines {i}-{i+1}: {e}")
                print(f"  {lines[i].strip()}")
                print(f"  {lines[i+1].strip() if i+1 < len(lines) else ''}")

        i += 1

    if not detector_coords:
        raise RuntimeError("No detectors parsed!")

    # Center detector coordinates
    coords_array = np.array(list(detector_coords.values()))
    mean_coord = coords_array.mean(axis=0)
    for k in detector_coords:
        detector_coords[k] = (np.array(detector_coords[k]) - mean_coord).tolist()

    # Error coordinates as mean of their detectors (if known)
    error_coords = {}
    for ei, det_list in enumerate(error_to_detectors):
        try:
            pts = np.array([detector_coords[d] for d in det_list if d in detector_coords])
            if len(pts) > 0:
                error_coords[ei] = pts.mean(axis=0).tolist()
        except KeyError as e:
            print(f"⚠️ Skipping error {ei}: unknown detector {e}")

    error_to_detectors_dict = {str(i): dets for i, dets in enumerate(error_to_detectors)}

    return {
        "detectorCoords": {str(k): v for k, v in detector_coords.items()},
        "errorCoords": {str(k): v for k, v in error_coords.items()},
        "errorToDetectors": error_to_detectors_dict,
        "frames": frames
    }

def main():
    parser = argparse.ArgumentParser(description="Convert a tesseract decoder logfile to a 3D visualization JSON.")
    parser.add_argument("logfile", help="Path to the logfile.txt")
    parser.add_argument("-o", "--output", default="tesseract_visualization.json", help="Output JSON filename")
    args = parser.parse_args()

    data = parse_logfile(args.logfile)

    with open(args.output, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✅ JSON written to {args.output} with {len(data['frames'])} frames and {len(data['errorCoords'])} error coords.")

if __name__ == "__main__":
    main()
