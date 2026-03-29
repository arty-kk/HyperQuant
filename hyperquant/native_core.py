# Copyright 2026 Сацук Артём Венедиктович (Satsuk Artem)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np


_NATIVE_LOCK = threading.Lock()
_NATIVE_LIB: ctypes.CDLL | None = None
_NATIVE_PATH: Path | None = None
_NATIVE_ERROR: str | None = None


def _package_root() -> Path:
    return Path(__file__).resolve().parent


def _source_path() -> Path:
    return _package_root() / "_native" / "fwht.c"


def _library_name() -> str:
    if sys.platform.startswith("linux"):
        return "libhyperquant_fwht.so"
    if sys.platform == "darwin":
        return "libhyperquant_fwht.dylib"
    if sys.platform.startswith("win"):
        return "hyperquant_fwht.dll"
    raise RuntimeError(f"unsupported platform for native core: {sys.platform}")


def _compiler_candidates() -> list[str]:
    requested = os.environ.get("CC")
    candidates = [requested] if requested else []
    candidates.extend(["gcc", "clang"])
    return [candidate for candidate in candidates if candidate]


def _find_compiler() -> str | None:
    for candidate in _compiler_candidates():
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def build_native_fwht(*, force: bool = False, build_dir: str | Path | None = None) -> Path | None:
    source = _source_path()
    if not source.exists():
        return None

    compiler = _find_compiler()
    if compiler is None:
        return None

    target_dir = Path(build_dir) if build_dir is not None else Path(tempfile.gettempdir()) / "hyperquant_native"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / _library_name()
    if target.exists() and not force:
        return target

    if sys.platform.startswith("linux"):
        cmd = [compiler, "-O3", "-shared", "-fPIC", str(source), "-o", str(target), "-lm"]
    elif sys.platform == "darwin":
        cmd = [compiler, "-O3", "-dynamiclib", str(source), "-o", str(target), "-lm"]
    elif sys.platform.startswith("win"):
        cmd = [compiler, "-O3", "-shared", str(source), "-o", str(target)]
    else:  # pragma: no cover
        return None

    subprocess.run(cmd, check=True, capture_output=True)
    return target


def _load_native_fwht(*, auto_build: bool = True) -> ctypes.CDLL | None:
    global _NATIVE_LIB, _NATIVE_PATH, _NATIVE_ERROR
    with _NATIVE_LOCK:
        if _NATIVE_LIB is not None:
            return _NATIVE_LIB
        if _NATIVE_ERROR is not None and not auto_build:
            return None
        try:
            source_dir_target = _package_root() / _library_name()
            lib_path = source_dir_target if source_dir_target.exists() else None
            if lib_path is None and auto_build:
                built = build_native_fwht(force=False)
                lib_path = built
            if lib_path is None or not lib_path.exists():
                return None
            lib = ctypes.CDLL(str(lib_path))
            lib.hq_fwht_rows.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_longlong, ctypes.c_int]
            lib.hq_fwht_rows.restype = None
            _NATIVE_LIB = lib
            _NATIVE_PATH = lib_path
            _NATIVE_ERROR = None
            return lib
        except Exception as exc:  # pragma: no cover - environment-specific
            _NATIVE_ERROR = str(exc)
            _NATIVE_LIB = None
            return None



def native_fwht_available(*, auto_build: bool = True) -> bool:
    return _load_native_fwht(auto_build=auto_build) is not None



def native_fwht_status(*, auto_build: bool = False) -> dict[str, object]:
    lib = _load_native_fwht(auto_build=auto_build)
    return {
        "available": lib is not None,
        "path": str(_NATIVE_PATH) if _NATIVE_PATH is not None else None,
        "error": _NATIVE_ERROR,
    }



def fwht_rows_numpy(data: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(np.asarray(data, dtype=np.float32))
    if values.ndim != 2:
        raise ValueError("fwht_rows_numpy expects a 2D float32 array")
    dim = int(values.shape[1])
    if dim <= 0 or dim & (dim - 1):
        raise ValueError("FWHT requires a power-of-two dimension")

    transformed = values.copy()
    h = 1
    while h < dim:
        transformed = transformed.reshape(-1, dim // (2 * h), 2, h)
        left = transformed[:, :, 0, :].copy()
        right = transformed[:, :, 1, :].copy()
        transformed[:, :, 0, :] = left + right
        transformed[:, :, 1, :] = left - right
        transformed = transformed.reshape(-1, dim)
        h *= 2
    transformed /= np.float32(np.sqrt(dim))
    return transformed.astype(np.float32, copy=False)



def fwht_rows_inplace(data: np.ndarray, *, prefer_native: bool = True) -> bool:
    if data.ndim != 2 or data.dtype != np.float32 or not data.flags.c_contiguous:
        raise ValueError("fwht_rows_inplace expects a C-contiguous 2D float32 array")
    dim = int(data.shape[1])
    if dim <= 0 or dim & (dim - 1):
        raise ValueError("FWHT requires a power-of-two dimension")

    lib = _load_native_fwht(auto_build=prefer_native) if prefer_native else None
    if lib is not None:
        ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        lib.hq_fwht_rows(ptr, int(data.shape[0]), dim)
        return True

    data[:] = fwht_rows_numpy(data)
    return False



def fwht_rows(data: np.ndarray, *, prefer_native: bool = True) -> tuple[np.ndarray, bool]:
    values = np.ascontiguousarray(np.asarray(data, dtype=np.float32))
    native_used = fwht_rows_inplace(values, prefer_native=prefer_native)
    return values, native_used
