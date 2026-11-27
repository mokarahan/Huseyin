"""The tests express scientific invariants, not invented scores."""
import numpy as np
import pytest
from scipy.io import savemat
from src.config import N_POINTS
from src.data import eol_index, load_nasa_battery, resample_discharge, windows
from src.scaling import TrainScaler
from src.demo import make_demo
from src.data import read_cells
from src.experiment import build_fold, metrics, split_cells


def test_resampling_keeps_duration_and_order():
    t = [5.0, 1.0, 3.0, 3.0, 7.0]
    v = [3.0, 4.0, 3.5, 3.5, 2.8]
    i = [-2.0] * 5
    temp = [27, 24, 25, 25, 30]
    signal, duration = resample_discharge(t, v, i, temp)
    assert signal.shape == (N_POINTS, 3)
    assert duration == 6.0
    assert signal[0, 0] == pytest.approx(4.0)
    assert signal[-1, 0] == pytest.approx(2.8)


def test_eol_first_crossing_and_censor():
    assert eol_index([2.0, 1.7, 1.6, 1.62, 1.5]) == 2
    assert eol_index([1.9, 1.8, 1.7]) is None


def test_windows_future_soh_and_no_target_in_x():
    x = np.zeros((7, 128, 3), np.float32)
    for t in range(7):
        x[t] = t
    aux = np.column_stack([np.ones(7), np.arange(1, 8)]).astype(np.float32)
    capacities = np.array([2., 1.95, 1.92, 1.85, 1.75, 1.60, 1.55])
    data = windows({"x": x, "aux": aux, "capacity": capacities}, window=3)
    assert data["cycle"].tolist() == [3, 4, 5]
    assert data["eol_cycle"] == 6
    assert data["y"][0, 0] == pytest.approx(92.5)  # SOH at cycle 4
    assert data["y"][0, 1] == pytest.approx(3.0)
    assert data["x"][0, -1, 0, 0] == pytest.approx(2.0)  # cycle 3 only
    assert data["x"][0].shape == (3, 128, 3)


def test_train_stats_never_use_test_cell(tmp_path):
    make_demo(tmp_path)
    cells = read_cells(tmp_path)
    train, val, test, scaler, groups = build_fold(cells, "B0005", 10)
    assert groups["test"] not in groups["train"]
    assert groups["validation"] not in groups["train"]
    assert groups["test"] != groups["validation"]
    direct = TrainScaler().fit(train)
    assert np.allclose(direct.parameters["x"][0], 0, atol=0.03)
    # Val and test are not independently normalized to zero mean.
    assert not np.allclose(test["aux"].mean(axis=(0, 1)), [0, 0], atol=0.01)
    assert val["x"].shape[-1] == 3


def test_matlab_reader_smoke(tmp_path):
    def operation(cap):
        return {"type": "discharge", "data": {
            "Time": np.array([0, 1, 2]), "Voltage_measured": [4.0, 3.5, 3.0],
            "Current_measured": [-2, -2, -2],
            "Temperature_measured": [24, 25, 26], "Capacity": cap}}
    path = tmp_path / "B0005.mat"
    savemat(str(path), {"B0005": {"cycle": [operation(2.0), operation(1.58)]}})
    result = load_nasa_battery(path)
    assert result["x"].shape == (2, N_POINTS, 3)
    assert eol_index(result["capacity"]) == 1


def test_error_units():
    result = metrics(np.array([[90., 40.], [80., 20.]]),
                     np.array([[92., 42.], [77., 16.]]))
    assert result["rul_mae_cycles"] == pytest.approx(3.)
    assert result["soh_mae_pct_points"] == pytest.approx(2.5)
    assert split_cells(["B0005", "B0006", "B0007"], "B0005")[2] == "B0005"


def test_scaler_inverse_restores_original_targets():
    from src.scaling import TrainScaler
    b = {"x": np.random.RandomState(0).randn(6, 3, 128, 3).astype(np.float32),
         "aux": np.random.RandomState(1).randn(6, 3, 2).astype(np.float32),
         "y": np.array([[92., 40.], [91., 35.], [90., 30.],
                        [89., 25.], [88., 20.], [87., 15.]], dtype=np.float32)}
    scaler = TrainScaler().fit(b)
    assert np.allclose(scaler.inverse_y(scaler.transform(b)["y"]), b["y"], atol=1e-4)


def test_offline_nested_nasa_archive_extraction(tmp_path):
    import io
    from zipfile import ZipFile
    from scipy.io import savemat
    from src.download import download_nasa
    raw = tmp_path / "raw"
    mat = tmp_path / "B0005.mat"
    savemat(str(mat), {"B0005": {"cycle": []}})
    inner = io.BytesIO()
    with ZipFile(inner, "w") as z:
        z.write(str(mat), arcname="BatteryAgingARC-FY08Q4/B0005.mat")
    original = tmp_path / "nasa_official_style.zip"
    with ZipFile(original, "w") as z:
        z.writestr("BatteryAgingARC-FY08Q4.zip", inner.getvalue())
    manifest = download_nasa(raw, archive_path=original)
    assert (raw / "B0005.mat").exists()
    assert "B0005" in manifest["mat_sha256"]
    assert "B0006" in manifest["missing"]
