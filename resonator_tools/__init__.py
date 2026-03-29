import os as _os

# Pin BLAS to single-threaded to ensure deterministic floating-point
# results across runs and platforms.  Must be set before numpy is imported.
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("BLIS_NUM_THREADS", "1")
