"""Make onnxruntime-gpu find the CUDA libraries that PyTorch already ships.

onnxruntime's CUDA provider is a separate .so that dlopen()s libcublasLt,
libcudnn and friends by soname. Those libraries exist inside the venv (torch
vendors them under nvidia/*/lib) but are not on the dynamic loader's search
path, so the provider silently fails to load and onnxruntime falls back to CPU
-- roughly 10x slower for face detection, with no error unless you inspect the
session's actual providers.

LD_LIBRARY_PATH is read once at process start and cannot be set from inside
Python in time. Loading the libraries explicitly with RTLD_GLOBAL puts them in
the global symbol namespace, so the provider's own dlopen finds them already
resolved.
"""
from __future__ import annotations

import contextlib
import ctypes
import glob
import os
import sys
import threading

_done = False
_lock = threading.Lock()

# Order matters -- each library must be resolvable when the next one loads.
# libcudart is the CUDA runtime everything else links against, so it goes first.
# This is the union of what `ldd libonnxruntime_providers_cuda.so` reports plus
# the cuDNN chain the provider dlopen()s lazily at session creation.
_WINDOWS = sys.platform == "win32"

# On Windows the same libraries are DLLs with version-tagged names, and torch
# vendors them under torch/lib rather than nvidia/*/lib.
_PRELOAD_WIN = ["cudart64", "cublasLt64", "cublas64", "curand64", "cufft64",
                "cusparse64", "cudnn64", "cudnn_graph64", "cudnn_ops64"]

_PRELOAD = [
    "libcudart.so",
    "libnvJitLink.so",
    "libcublasLt.so",
    "libcublas.so",
    "libcurand.so",
    "libcufft.so",
    "libcusparse.so",
    "libcusolver.so",
    "libcudnn_graph.so",
    "libcudnn_ops.so",
    "libcudnn_cnn.so",
    "libcudnn.so",
]


def _lib_dirs() -> list[str]:
    dirs = []
    for site in sys.path:
        if not site.endswith("site-packages"):
            continue
        if _WINDOWS:
            dirs += sorted(glob.glob(os.path.join(site, "torch", "lib")))
            dirs += sorted(glob.glob(os.path.join(site, "nvidia", "*", "bin")))
        else:
            dirs += sorted(glob.glob(os.path.join(site, "nvidia", "*", "lib")))
    return dirs


def preload() -> bool:
    """Load vendored CUDA libraries into the global namespace. Idempotent.
    Returns True if anything was loaded."""
    global _done
    if _done:
        return True
    with _lock:
        if _done:
            return True
        loaded = False
        dirs = _lib_dirs()
        if _WINDOWS:
            # Since Python 3.8 Windows only searches directories added here.
            for directory in dirs:
                with contextlib.suppress(OSError, AttributeError):
                    os.add_dll_directory(directory)
        for soname in (_PRELOAD_WIN if _WINDOWS else _PRELOAD):
            for directory in dirs:
                pattern = soname + ("*.dll" if _WINDOWS else "*")
                matches = sorted(glob.glob(os.path.join(directory, pattern)))
                if not matches:
                    continue
                try:
                    ctypes.CDLL(matches[-1], mode=ctypes.RTLD_GLOBAL)
                    loaded = True
                    break
                except OSError:
                    continue
        _done = True
        return loaded
