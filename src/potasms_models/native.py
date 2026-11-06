"""ctypes binding to the vendored Unicode15/C++ runtime; no host Unicode fallback."""
from __future__ import annotations

import ctypes as C
from functools import lru_cache
import os
from pathlib import Path

import numpy as np

CONTRACT = "unicode15-nfkc-casefold-fnv1a-char-2-5-l2-v2"
LEGACY_CONTRACT = "nfkc-lower-unicode-scalar-fnv1a-char-2-5-l2-v1"


class Feature(C.Structure):
    _fields_ = [("bucket", C.c_uint32), ("value", C.c_float)]


@lru_cache(maxsize=1)
def library():
    root = Path(__file__).resolve().parents[2] / "build/native"
    override = os.environ.get("POTASMS_NATIVE_LIB")
    paths = [Path(override)] if override else [root / name for name in ("libpotasms.so", "libpotasms.dylib", "Release/potasms.dll", "potasms.dll")]
    path = next((p for p in paths if p.is_file()), None)
    if path is None:
        raise RuntimeError("Build the POTASMS C++ core or set POTASMS_NATIVE_LIB; host Unicode fallback is intentionally disabled")
    lib = C.CDLL(str(path))
    lib.pota_version.restype = C.c_char_p
    lib.pota_feature_contract.restype = C.c_char_p
    lib.pota_free.argtypes = [C.c_void_p]
    lib.pota_normalize.argtypes = [C.c_char_p, C.c_size_t, C.c_int, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)]
    for name, typ in [("pota_features", Feature), ("pota_bucket_ids", C.c_uint32)]:
        getattr(lib, name).argtypes = [C.c_char_p, C.c_size_t, C.c_int, C.c_uint32, C.c_uint32,
                                      C.c_uint32, C.POINTER(C.POINTER(typ)), C.POINTER(C.c_size_t)]
    lib.pota_logit.argtypes = [C.POINTER(Feature), C.c_size_t, C.POINTER(C.c_int8), C.c_size_t,
                               C.c_float, C.c_float, C.POINTER(C.c_float), C.c_float, C.POINTER(C.c_float)]
    lib.pota_sgd_sample.argtypes = [C.POINTER(Feature), C.c_size_t, C.POINTER(C.c_int8), C.c_size_t,
                                    C.c_float, C.c_float, C.POINTER(C.c_float), C.POINTER(C.c_float),
                                    C.c_uint8, C.c_float, C.c_float, C.c_float]
    lib.pota_sigmoid.argtypes = [C.c_float]
    lib.pota_sigmoid.restype = C.c_float
    if lib.pota_feature_contract().decode() != CONTRACT:
        raise RuntimeError("Incompatible native feature contract")
    return lib


def mode(contract: str) -> int:
    if contract == CONTRACT:
        return 2
    if contract == LEGACY_CONTRACT:
        return 1
    raise ValueError(f"Unknown feature contract: {contract}")


def check(status: int) -> None:
    if status:
        raise ValueError(f"Native core rejected input (status={status})")


def normalize(text: str, *, fold: bool = False, contract: str = CONTRACT) -> str:
    raw = str(text).encode("utf-8")
    out, n = C.c_void_p(), C.c_size_t()
    lib = library()
    check(lib.pota_normalize(raw, len(raw), mode(contract) if fold else 0, C.byref(out), C.byref(n)))
    try:
        return C.string_at(out, n.value).decode("utf-8")
    finally:
        lib.pota_free(out)


def bucket_ids(text: str, *, buckets: int = 65536, min_n: int = 2, max_n: int = 5,
               contract: str = CONTRACT) -> list[int]:
    if not 1 <= buckets <= 1048576 or min_n != 2 or max_n != 5:
        raise ValueError("Unsupported feature dimensions")
    raw, out, n = str(text).encode(), C.POINTER(C.c_uint32)(), C.c_size_t()
    lib = library()
    check(lib.pota_bucket_ids(raw, len(raw), mode(contract), buckets, min_n, max_n, C.byref(out), C.byref(n)))
    try:
        return list(out[:n.value])
    finally:
        lib.pota_free(out)


def features(text: str, *, buckets: int = 65536, contract: str = CONTRACT) -> list[tuple[int, float]]:
    if not 1 <= buckets <= 1048576:
        raise ValueError("Unsupported feature dimensions")
    raw, out, n = str(text).encode(), C.POINTER(Feature)(), C.c_size_t()
    lib = library()
    check(lib.pota_features(raw, len(raw), mode(contract), buckets, 2, 5, C.byref(out), C.byref(n)))
    try:
        return [(out[i].bucket, out[i].value) for i in range(n.value)]
    finally:
        lib.pota_free(out)


def score(text: str, weights: np.ndarray, scale: float, bias: float, *,
          contract: str = CONTRACT, delta: dict[int, float] | None = None, bias_delta: float = 0) -> float:
    w = np.ascontiguousarray(weights, dtype=np.int8)
    rows = features(text, buckets=len(w), contract=contract)
    f = (Feature * len(rows))(*(Feature(*r) for r in rows))
    d = (C.c_float * len(rows))(*((delta or {}).get(r[0], 0) for r in rows))
    value = C.c_float()
    lib = library()
    check(lib.pota_logit(f, len(rows), w.ctypes.data_as(C.POINTER(C.c_int8)), len(w), scale, bias, d, bias_delta, C.byref(value)))
    return float(lib.pota_sigmoid(value))


def fit_delta(texts: list[str], labels: list[int], weights: np.ndarray, scale: float, bias: float, *,
              epochs: int = 12, rate: float = .12, l2: float = .0005, maximum_delta: float = 4,
              contract: str = CONTRACT) -> tuple[dict[int, float], float]:
    if len(texts) != len(labels) or not 1 <= epochs <= 100 or any(v not in (0, 1) for v in labels):
        raise ValueError("Invalid training examples or epochs")
    w = np.ascontiguousarray(weights, dtype=np.int8)
    dense, db = np.zeros(len(w), dtype=np.float32), C.c_float(0)
    prepared = []
    for text, label in zip(texts, labels):
        rows = features(text, buckets=len(w), contract=contract)
        prepared.append(((Feature * len(rows))(*(Feature(*r) for r in rows)), len(rows), label))
    lib = library()
    for _ in range(epochs):
        for f, n, label in prepared:
            check(lib.pota_sgd_sample(f, n, w.ctypes.data_as(C.POINTER(C.c_int8)), len(w), scale, bias,
                                     dense.ctypes.data_as(C.POINTER(C.c_float)), C.byref(db), label, rate, l2, maximum_delta))
    return {int(i): float(dense[i]) for i in np.flatnonzero(np.abs(dense) >= .0001)}, float(db.value)
