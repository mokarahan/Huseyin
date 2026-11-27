"""Read the NASA .mat files and build observations without future information.

For each completed discharge cycle: 128 measurements (V, I, T) + duration +
known discharge number. Capacity is recorded ONLY for the
labels; it is never placed in X.
"""
from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat
from .config import (BATTERIES, SIGNALS, CAPACITY_NOMINAL_AH,
                     EOL_FRACTION, N_POINTS)


def resample_discharge(time, voltage, current, temperature, points=N_POINTS):
    """Interpolation over relative time; keeps the duration in seconds."""
    channels = [np.asarray(v, dtype=np.float64).ravel()
                for v in (time, voltage, current, temperature)]
    if len({len(v) for v in channels}) != 1:
        raise ValueError("Incompatible sizes between Time, V, I and T")
    if len(channels[0]) < 2:
        raise ValueError("Discharge with fewer than two measurements")
    matrix = np.column_stack(channels)
    matrix = matrix[np.isfinite(matrix).all(axis=1)]
    if len(matrix) < 2:
        raise ValueError("Fewer than two finite measurements")
    matrix = matrix[np.argsort(matrix[:, 0], kind="stable")]
    matrix = matrix[np.r_[True, np.diff(matrix[:, 0]) > 0]]
    if len(matrix) < 2 or matrix[-1, 0] <= matrix[0, 0]:
        raise ValueError("No positive discharge duration")
    duration = float(matrix[-1, 0] - matrix[0, 0])
    x_time = (matrix[:, 0] - matrix[0, 0]) / duration
    target_time = np.linspace(0.0, 1.0, points)
    x = np.column_stack([np.interp(target_time, x_time, matrix[:, j])
                         for j in (1, 2, 3)]).astype(np.float32)
    return x, duration


def _cycles(struct):
    """MATLAB: cycle can be a scalar or an array of mat_struct."""
    cycles = getattr(struct, "cycle", None)
    if cycles is None:
        raise ValueError("MATLAB field 'cycle' not found")
    return np.asarray(cycles, dtype=object).ravel()


def load_nasa_battery(path):
    """Direct extraction from the NASA file; no artificial labels."""
    path = Path(path)
    name = path.stem.upper()
    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    if name not in mat:
        raise ValueError("Missing MATLAB key: " + name)
    traces, durations, capacities, warnings = [], [], [], []
    for operation, cycle in enumerate(_cycles(mat[name]), start=1):
        if str(getattr(cycle, "type", "")).strip() != "discharge":
            continue
        data = getattr(cycle, "data", None)
        if data is None:
            warnings.append("operation %d : data absent" % operation)
            continue
        try:
            capacity = float(np.asarray(getattr(data, "Capacity")).ravel()[0])
            if not np.isfinite(capacity) or capacity <= 0 or capacity > 2.5:
                raise ValueError("Implausible capacity (%s Ah)" % capacity)
            x, duration = resample_discharge(
                getattr(data, "Time"),
                *(getattr(data, signal) for signal in SIGNALS))
        except (ValueError, AttributeError, TypeError, IndexError) as exc:
            warnings.append("operation %d : %s" % (operation, str(exc)))
            continue
        traces.append(x)
        durations.append(duration)
        capacities.append(capacity)
    if not traces:
        raise ValueError("No valid discharge in " + str(path))
    n = len(traces)
    # The discharge index follows valid records, not charge or
    # impedance operations; exclusions are documented.
    aux = np.column_stack([np.log1p(durations), np.arange(1, n + 1)])
    return {"x": np.stack(traces).astype(np.float32),
            "aux": aux.astype(np.float32),
            "capacity": np.asarray(capacities, dtype=np.float32),
            "warnings": warnings, "name": name}


def eol_index(capacity, threshold=CAPACITY_NOMINAL_AH * EOL_FRACTION):
    """First observed crossing. None = censored, no invented RUL."""
    hits = np.flatnonzero(np.asarray(capacity) <= threshold)
    return int(hits[0]) if len(hits) else None


def prepare(raw_dir, processed_dir):
    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    if (processed_dir / "DEMO_NOT_REAL_DATA.txt").exists():
        raise ValueError("DEMO dataset detected. Empty data/processed before importing NASA data so the sources are not mixed.")
    report = {"nominal_ah": CAPACITY_NOMINAL_AH,
              "eol_threshold_ah": CAPACITY_NOMINAL_AH * EOL_FRACTION,
              "n_points": N_POINTS, "batteries": {}}
    for name in BATTERIES:
        candidates = list(raw_dir.rglob(name + ".mat"))
        if len(candidates) != 1:
            report["batteries"][name] = {"status": "missing_or_ambiguous", "found": len(candidates)}
            continue
        try:
            cell = load_nasa_battery(candidates[0])
            path = processed_dir / (name + ".npz")
            np.savez_compressed(str(path), x=cell["x"], aux=cell["aux"],
                                capacity=cell["capacity"])
            at = eol_index(cell["capacity"])
            report["batteries"][name] = {
                "status": "observed_eol" if at is not None else "censored",
                "n_discharge": len(cell["capacity"]),
                "eol_cycle_1based": at + 1 if at is not None else None,
                "excluded_records": cell["warnings"],
                "raw_path": str(candidates[0])}
        except (ValueError, OSError) as exc:
            report["batteries"][name] = {"status": "error", "reason": str(exc)}
    (processed_dir / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def read_cells(processed_dir):
    """Batteries without an EOL crossing remain described but unlabeled."""
    result = {}
    for file in sorted(Path(processed_dir).glob("B*.npz")):
        if file.stem not in BATTERIES:
            continue
        with np.load(str(file), allow_pickle=False) as z:
            result[file.stem] = {k: z[k] for k in ("x", "aux", "capacity")}
    return result


def windows(cell, window=10):
    """Target at k: SOH(k+1) and RUL(k), without access to the signals of k+1.

    Excludes the EOL cycle: the SOH(k+1) label thus never requires
    a post-EOL measurement. RUL=0 is not part of the common supervised set.
    """
    x, aux, capacity = (cell[k] for k in ("x", "aux", "capacity"))
    at = eol_index(capacity)
    if at is None:
        raise ValueError("Censored battery: no exact RUL observable")
    if len(x) != len(aux) or len(x) != len(capacity):
        raise ValueError("Inconsistent number of cycles")
    if x.shape[1:] != (N_POINTS, 3) or aux.shape[1:] != (2,):
        raise ValueError("Input shape does not conform")
    ends = np.arange(window - 1, at, dtype=int)
    if not len(ends):
        raise ValueError("Not enough cycles before EOL for window=%d" % window)
    signal = np.stack([x[k - window + 1:k + 1] for k in ends]).astype(np.float32)
    features = np.stack([aux[k - window + 1:k + 1] for k in ends]).astype(np.float32)
    target = np.column_stack([100 * capacity[ends + 1] / CAPACITY_NOMINAL_AH,
                              at - ends]).astype(np.float32)
    return {"x": signal, "aux": features, "y": target,
            "cycle": ends + 1, "eol_cycle": at + 1}
