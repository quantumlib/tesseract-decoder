import os
import sys

SHOTS_FLAG = "--shots"
DETECTORS_FLAG = "--detectors"
PQ_LIMIT_FLAG = "--pqlimit"
AT_MOST_TWO_ERRORS_PER_DETECTOR_FLAG = "--at-most-two-errors-per-detector"
BEAM_FLAG = "--beam"
THREADS_FLAG = "--threads"

def main():
    file_path = sys.argv[1]

    shots = 1
    for arg in sys.argv:
        if arg.startswith(f"{SHOTS_FLAG}="):
            shots = int(arg.split("=")[1])
            break

    detectors = 1
    for arg in sys.argv:
        if arg.startswith(f"{DETECTORS_FLAG}="):
            detectors = int(arg.split("=")[1])
            break

    pqlimit = 200000
    for arg in sys.argv:
        if arg.startswith(f"{PQ_LIMIT_FLAG}="):
            pqlimit = int(arg.split("=")[1])
            break

    beam = 15
    for arg in sys.argv:
        if arg.startswith(f"{BEAM_FLAG}="):
            beam = int(arg.split("=")[1])
            break

    threads = 1
    for arg in sys.argv:
        if arg.startswith(f"{THREADS_FLAG}="):
            threads = int(arg.split("=")[1])
            break


    at_most_two_errors_per_detector = AT_MOST_TWO_ERRORS_PER_DETECTOR_FLAG if AT_MOST_TWO_ERRORS_PER_DETECTOR_FLAG in sys.argv else ""

    command = (f"../bazel-bin/src/tesseract --circuit {file_path} --threads={threads} --sample-num-shots={shots} "
               f"--beam={beam} --num-det-orders={detectors} --pqlimit={pqlimit} --no-revisit-dets "
               f"--sample-seed=123 --det-order-seed=123 --print-stats {at_most_two_errors_per_detector}")


    print(command)
    os.system(command)


if __name__ == '__main__':
    main()
