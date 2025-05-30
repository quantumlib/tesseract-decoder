import json
import argparse
import re
import numpy as np

def parse_implicit_list(line, prefix):
    if not line.startswith(prefix):
        raise ValueError(f"Expected line to start with '{prefix}', got: {line}")
    list_part = line[len(prefix):].strip().rstrip(',')
    if not list_part:
        return []
    return [int(x.strip()) for x in list_part.split(',') if x.strip()]

def parse_logfile(filepath):
    detector_coords = {}
    error_to_detectors = []
    frames = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not any(line.startswith(s) for s in ['Error', 'Detector', 'activated_errors', 'activated_dets']):
          continue

        if line.startswith("Detector D"):
            match = re.match(r'Detector D(\d+) coordinate \(([-\d.]+), ([-\d.]+), ([-\d.]+)\)', line)
            if match:
                idx = int(match.group(1))
                coord = tuple(float(match.group(j)) for j in range(2, 5))
                detector_coords[idx] = coord

        elif line.startswith("Error{"):
            match = re.search(r'Symptom\{([^\}]+)\}', line)
            if match:
                dets = match.group(1).split()
                det_indices = [int(d[1:]) for d in dets if d.startswith('D')]
                error_to_detectors.append(det_indices)

        elif line.startswith("activated_errors"):
            try:
                error_line = lines[i].strip()
                det_line = lines[i + 1].strip()

                activated_errors = parse_implicit_list(error_line, "activated_errors =")
                activated_dets = parse_implicit_list(det_line, "activated_dets =")

                frame = {
                    "activated": activated_dets,
                    "activated_errors": activated_errors
                }
                frames.append(frame)
                i += 1
            except Exception as e:
                print(f"\n⚠️ Error parsing frame at lines {i}-{i+1}: {e}")
                print(f"  {lines[i].strip()}")
                print(f"  {lines[i+1].strip() if i+1 < len(lines) else ''}")
        i += 1

    if not detector_coords:
        raise RuntimeError("No detectors parsed!")

    coords_array = np.array(list(detector_coords.values()))
    mean_coord = coords_array.mean(axis=0)
    for k in detector_coords:
        detector_coords[k] = (np.array(detector_coords[k]) - mean_coord).tolist()

    error_coords = {}
    for i, det_list in enumerate(error_to_detectors):
        try:
            pts = np.array([detector_coords[d] for d in det_list if d in detector_coords])
            if len(pts) > 0:
                error_coords[i] = pts.mean(axis=0).tolist()
        except KeyError as e:
            print(f"⚠️ Skipping error {i}: unknown detector {e}")

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
