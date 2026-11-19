from src.config import Experiment
from src.demo import make_demo
from src.data import read_cells
from src.experiment import train_one, run_multiseed, predict_checkpoint


def test_end_to_end_smoke(tmp_path):
    make_demo(tmp_path / "input")
    cells = read_cells(tmp_path / "input")
    report = train_one(cells, "B0005", "hybrid",
                       Experiment(window=10, epochs=2, patience=2,
                                  batch=64, seed=7, hidden=16, heads=4),
                       tmp_path / "run", device_name="cpu")
    assert report["group"]["test"] == "B0005"
    assert (tmp_path / "run" / "model.pt").exists()
    assert (tmp_path / "run" / "soh_prediction.png").exists()
    prediction = predict_checkpoint(tmp_path / "run" / "model.pt",
                                    tmp_path / "input", "B0005", cycle=20)
    assert prediction["observed_cycle"] == 20
    assert isinstance(prediction["rul_cycles_raw"], float)
    assert "bias_ttest" in report and "rul" in report["bias_ttest"]


def test_multiseed_summary(tmp_path):
    make_demo(tmp_path / "input")
    summary = run_multiseed(
        tmp_path / "input", tmp_path / "ms", ["ridge", "cnn"], ["B0005"],
        Experiment(window=10, epochs=2, patience=2, batch=64, hidden=16, heads=4, aux_mode="no_cycle"),
        seeds=[1, 2], device="cpu",
    )
    assert (tmp_path / "ms" / "multiseed_per_battery.csv").exists()
    assert "ci95" in summary["macro_over_seeds"]["cnn"]["rul_mae_cycles"]
