"""SYNTHETIC demonstration data. Never for scientific publication."""
from pathlib import Path
import numpy as np
from .config import CAPACITY_NOMINAL_AH, N_POINTS


def make_demo(processed_dir):
    path = Path(processed_dir)
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(734)
    for name, n, fade in (("B0005", 75, .0062),
                          ("B0006", 80, .0068),
                          ("B0007", 86, .0052)):
        x, aux, capacities = [], [], []
        for i in range(n):
            capacity = 1.99 - fade * i
            if i >= 2:
                capacity += rng.normal(0, .002)
            duration = 3650 * capacity / 2
            t = np.linspace(0, 1, N_POINTS)
            v = 4.14 - (1.30 + .08 * i / n) * (t ** (0.65 + .1 * i / n))
            amp = -1.85 + 0.015 * np.sin(4 * np.pi * t)
            temp = 24 + (6 + 2 * i / n) * t
            signal = np.column_stack([v, np.full_like(t, 0) + amp, temp])
            signal += rng.normal(0, .005, signal.shape)
            x.append(signal)
            aux.append([np.log1p(duration), i + 1])
            capacities.append(capacity)
        np.savez_compressed(str(path / (name + ".npz")),
                            x=np.asarray(x, dtype=np.float32),
                            aux=np.asarray(aux, dtype=np.float32),
                            capacity=np.asarray(capacities, dtype=np.float32))
    (path / "DEMO_NOT_REAL_DATA.txt").write_text(
        "SYNTHETIC DATA ONLY. NOT NASA. NOT FOR RESEARCH RESULTS.\n", encoding="utf-8")
    return path
