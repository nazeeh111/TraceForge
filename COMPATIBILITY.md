# TraceForge preservation record

## Change boundary

The repository presentation, documentation entry points and source comments were updated. Existing executable entry points, algorithms, defaults, filenames, formats and numerical operations remain unchanged. The serialized `core:author` metadata now uses `nazeeh111`; this is the sole intentional non-numerical output change. First-party licensing is MIT under nazeeh111. Separate bundled-component terms and external submodule notices are retained.

## Verified locally

All 59 Python files passed parsing. Executable syntax trees match the baseline after allowing only the `core:author` string change in `emcap.py`. Two standalone metrics were compared against the baseline on 200 deterministic array cases with exact equality. This narrow check used NumPy 2.3.5. The full pinned TensorFlow 1.14/Keras 2.2.5, worker, capture and model-training environment was not installed or executed.

The source comparison checks Python syntax trees without comments or source locations, so changes in documentation do not obscure computational changes. This is an equivalence check against the supplied source, not a claim that every experiment is correct or portable. Existing invalid-escape warnings in legacy Python strings also occur in the baseline and were not changed.

## Execution boundary

No RF transmission, signal acquisition, connected-device commands, firmware changes or live-target experiments were performed. No external experimental datasets were downloaded. Build and reproduction requirements remain those documented by the source; absent dependencies have not been silently replaced with new algorithms.
