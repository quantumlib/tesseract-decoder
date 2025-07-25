#Visualization

This tool displays the detectors and errors from a Tesseract decoding run in 3D.

## Generating the JSON

Use the `--verbose` flag when running the decoder and filter the output to
include only the lines used by the converter script:

```bash
bazel build src:all && \
./bazel-bin/src/tesseract \
  --sample-num-shots 1 --det-order-seed 13267562 --pqlimit 10000 --beam 1 --num-det-orders 20 \
  --circuit testdata/colorcodes/r\=9\,d\=9\,p\=0.002\,noise\=si1000\,c\=superdense_color_code_X\,q\=121\,gates\=cz.stim \
  --sample-seed 717347 --threads 1 --verbose | \
  grep -E 'Error|Detector|activated_errors|activated_dets' > logfile.txt

python viz/to_json.py logfile.txt -o logfile.json
```


The `--no-det-order-bfs` flag is compatible with visualization logs. BFS-based
detector ordering is now enabled by default, so include this flag only if you
want to disable it. Make sure `--verbose` is enabled so the detector
coordinates are printed for `to_json.py` to parse.

The `to_json.py` script produces `logfile.json`, which contains the detector
coordinates and animation frames for the viewer.

## Viewing

Open `viz/index.html` in a modern browser. It will automatically try to load
`logfile.json` from the same directory. If the file picker is used, any JSON
produced by `to_json.py` can be visualized.
