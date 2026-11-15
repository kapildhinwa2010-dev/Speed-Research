# Speed-Research - Empirical scaling of compression runtimes: timing plus complexity fit
# Part of the Kapildhinwa2010-dev Toolkit.

"""SpeedResearch - compression runtime scaling experiments.

Lineage: speed-layer repos. Times compression methods across matrix
sizes, fits log(time) = exponent * log(size) + c per method, and reports
the empirical complexity exponent. Exponents are measured on one machine
with one BLAS; they are indicative, not certified.
"""

from __future__ import annotations

import time

import numpy as np

__all__ = ["time_method", "scaling_table", "complexity_exponent"]


def _fourier_basis(n_cols: int, num_coeffs: int) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n_cols, endpoint=False)
    cols = [np.ones_like(t)]
    for k in range(1, num_coeffs + 1):
        cols.append(np.sin(k * t))
        cols.append(np.cos(k * t))
    return np.stack(cols, axis=1)


def _methods(matrix: np.ndarray) -> dict:
    def svd4():
        u, s, vt = np.linalg.svd(matrix, full_matrices=False)
        rank = min(4, s.size)
        return (u[:, :rank] * s[:rank]) @ vt[:rank, :]

    def quant8():
        lo, hi = float(matrix.min()), float(matrix.max())
        levels = 2 ** 8 - 1
        scale = (hi - lo) / levels if hi > lo else 1.0
        codes = np.clip(np.round((matrix - lo) / scale), 0, levels)
        return codes * scale + lo

    def fourier4():
        basis = _fourier_basis(matrix.shape[1], 4)
        coeffs, _, _, _ = np.linalg.lstsq(basis, matrix.T, rcond=None)
        return coeffs.T @ basis.T

    return {"svd_rank4": svd4, "quantize8": quant8, "fourier4": fourier4}


def time_method(func, repeats: int = 3) -> float:
    """Best-of-N wall-clock seconds for one call of func."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    best = None
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        if best is None or elapsed < best:
            best = elapsed
    return best


def scaling_table(sizes=(32, 64, 128, 256), repeats: int = 3, seed: int = 0) -> list:
    """Best-of-N seconds per method per size, on random matrices."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    rng = np.random.default_rng(seed)
    rows = []
    for size in sizes:
        if size < 2:
            raise ValueError("sizes must be >= 2")
        matrix = rng.standard_normal((size, size))
        for name, func in _methods(matrix).items():
            rows.append({"size": int(size), "method": name,
                         "seconds": time_method(func, repeats)})
    return rows


def complexity_exponent(rows: list, method: str) -> float:
    """Fit log(seconds) = exponent * log(size) + c for one method.

    An exponent near 3 suggests cubic scaling (as expected for SVD), near
    2 quadratic, near 1 linear. Needs at least 2 sizes.
    """
    samples = [(r["size"], r["seconds"]) for r in rows if r["method"] == method]
    if len(samples) < 2:
        raise ValueError("need at least 2 sizes for method {}".format(method))
    x = np.log([float(s) for s, _ in samples])
    y = np.log([max(float(t), 1e-12) for _, t in samples])
    exponent, _ = np.polyfit(x, y, 1)
    return float(exponent)
