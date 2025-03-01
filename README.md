Simplex and Tesseract are both most-likely-error decoders.

Simplex is based on Integer Linear Programming. The latest version of the Simplex decoder [lives in Pyle](https://github.com/qh-lab/pyle/blob/0645d873e6093b29a0460040c01c68f02041416a/src/pyle/dataking/error_correction/tools/simplex_decoder/simplexlib.py#L217) and is being maintained by Nathan Lacroix.
We have an older (but still correct) version here only for convenience since we don't want to have a dependency on Pyle and we need the Simplex decoder for use in developing Tesseract -- since the weights output by both decoders should match identically, we can do 'weight fuzzing' just like MWPM-based decoders.

Tesseract is based on A* search.
