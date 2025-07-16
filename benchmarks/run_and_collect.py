import subprocess

def parse_output_line(log_line):
    """
    Parses a log line containing key-value pairs and returns a dictionary.

    Args:
        log_line (str): The input string (e.g., "num_shots = 10 ...").

    Returns:
        dict: A dictionary containing the parsed fields and their values.
    """
    parsed_data = {}

    parts = log_line.split()

    for i in range(0, len(parts), 3):
        if i + 2 < len(parts) and parts[i+1] == '=':
            key = parts[i]
            value_str = parts[i+2]
            
            if key in ["num_shots", "num_low_confidence", "num_errors"]:
                try:
                    parsed_data[key.replace("num_", "")] = int(value_str)
                except ValueError:
                    print(f"Warning: Could not convert '{value_str}' to int for '{key}'. Storing as string.")
                    parsed_data[key.replace("num_", "")] = value_str
            elif key == "total_time_seconds":
                try:
                    parsed_data["seconds"] = float(value_str)
                except ValueError:
                    print(f"Warning: Could not convert '{value_str}' to float for '{key}'. Storing as string.")
                    parsed_data["seconds"] = value_str
            else:
                parsed_data[key] = value_str
                
    return parsed_data



import re
import os

def parse_stim_path(path_string):
    """
    Parses a given file path string to extract 'r', 'd', 'p' values
    and a custom 'code_type'.

    Args:
        path_string (str): The input file path string.

    Returns:
        dict: A dictionary containing the parsed 'r', 'd', 'p' values (as integers/float)
              and the 'code_type' (as a string).
              Returns None if parsing fails for essential parts.
    """
    data = {}

    param_match = re.search(r'r=(\d+),d=(\d+),p=([\d.]+)', path_string)
    if param_match:
        data['r'] = int(param_match.group(1))
        data['d'] = int(param_match.group(2))
        data['p'] = float(param_match.group(3))
    else:
        print(f"Warning: Could not parse r, d, p from: {path_string}")
        return None

    code_type_mapping = {
        "colorcodes": "color",
        "surfacecodes": "surface",
        "bivariatebicyclecodes": "bicycle",
        "bivariatebicyclecodes_nlr5wb": "bicycle_nlr5",
        "bivariatebicyclecodes_nlr10wb": "bicycle_nlr10",
        "surface_code_trans_cx_circuits": "trans_cx"
    }

    normalized_path = os.path.normpath(path_string)
    path_parts = normalized_path.split(os.sep)

    try:
        testdata_index = path_parts.index("testdata")
        raw_code_type_dir = path_parts[testdata_index + 1]
        data['code_type'] = code_type_mapping.get(raw_code_type_dir, "unknown")
    except ValueError:
        print(f"Warning: 'testdata' not found in path: {path_string}")
        data['code_type'] = "unknown"
    except IndexError:
        print(f"Warning: Code type directory not found after 'testdata' in path: {path_string}")
        data['code_type'] = "unknown"

    return data



def run_and_collect(file_path, shots, at_most_two_errors_per_detector=False):
    arguments = ['python3', 'run.py', file_path, f"--shots={shots}"] if not at_most_two_errors_per_detector else ['python3', 'run.py', file_path, f"--shots={shots}", "--at-most-two-errors-per-detector"]
    result = subprocess.run(arguments, capture_output=True, text=True, check=True)

    output_lines = result.stdout.strip().split('\n')

    if output_lines:
        last_line = output_lines[-1]
        print(last_line)
    else:
        print("No output captured from the script.")


    benchmark_data = parse_output_line(last_line)
    stim_data = parse_stim_path(file_path)
    return stim_data | benchmark_data

