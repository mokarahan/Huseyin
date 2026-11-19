"""Expériences reproductibles : séparation par batterie, entraînement, métriques.

Aucune statistique ni décision d'arrêt ne doit utiliser la batterie test.
"""
import copy
import csv
from dataclasses import replace
import json
import random
import time
from pathlib import Path
import numpy as np
from scipy import stats
import torch
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset
from .config import Experiment
from .data import read_cells, windows
from .models import MODELS, BatteryPredictor
from .scaling import TrainScaler
from .plots import plot_prediction_file


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Les expériences CUDA restent dépendantes des opérations disponibles.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_cells(ids, test_id):
    """Fold externe tenu à l'écart. La validation vient d'une AUTRE batterie."""
    ids = sorted(ids)
    if len(ids) < 3 or test_id not in ids:
        raise ValueError("Au moins trois batteries avec EOL observe sont requises")
    other = [name for name in ids if name != test_id]
    # Choix fixe et reproductible, jamais fondé sur le score test.
    validation = other[-1]
    training = other[:-1]
    assert not (set(training) & {validation, test_id})
    return training, validation, test_id


def concatenate(batches):
    return {key: np.concatenate([batch[key] for batch in batches], axis=0)
            for key in ("x", "aux", "y")}


def _mask_aux(batch, aux_mode):
    """Neutralise à 0 la/les variables auxiliaires ciblées pour l'ablation."""
    if aux_mode == "all":
        return batch
    out = dict(batch)
    aux = np.array(batch["aux"], copy=True)
    if aux_mode in ("no_duration", "none"):
        aux[..., 0] = 0.0
    if aux_mode in ("no_cycle", "none"):
        aux[..., 1] = 0.0
    out["aux"] = aux
    return out


def build_fold(cells, test_id, window, aux_mode="all"):
    from .data import eol_index
    eligible = {name: value for name, value in cells.items()
                if eol_index(value["capacity"]) is not None
                and eol_index(value["capacity"]) >= window}
    train_ids, val_id, test_id = split_cells(eligible, test_id)
    train = _mask_aux(concatenate([windows(eligible[name], window) for name in train_ids]), aux_mode)
    val = _mask_aux(windows(eligible[val_id], window), aux_mode)
    test = _mask_aux(windows(eligible[test_id], window), aux_mode)
    scaler = TrainScaler().fit(train)
    return (scaler.transform(train), scaler.transform(val),
            scaler.transform(test), scaler,
            {"train": train_ids, "validation": val_id, "test": test_id})


def metrics(y, prediction):
    """Erreurs en unités physiques : SOH en points de %, RUL en cycles."""
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return {"soh_mae_pct_points": float(np.mean(np.abs(error[:, 0]))),
            "soh_rmse_pct_points": float(np.sqrt(np.mean(error[:, 0] ** 2))),
            "soh_r2": float(1.0 - np.sum(error[:, 0] ** 2) / np.sum((y[:, 0] - np.mean(y[:, 0])) ** 2)) if np.var(y[:, 0]) > 1e-12 else None,
            "rul_mae_cycles": float(np.mean(np.abs(error[:, 1]))),
            "rul_rmse_cycles": float(np.sqrt(np.mean(error[:, 1] ** 2)))}


def paired_bias_ttest(truth, prediction):
    """Test t apparié (prédit vs réel) : t>0 surestimation, t<0 sous-estimation."""
    out = {}
    for idx, name in enumerate(("soh", "rul")):
        diff = np.asarray(prediction[:, idx], dtype=np.float64) - np.asarray(truth[:, idx], dtype=np.float64)
        t_stat, p_val = stats.ttest_rel(prediction[:, idx], truth[:, idx])
        out[name] = {"mean_bias": float(np.mean(diff)),
                     "t_stat": float(t_stat) if np.isfinite(t_stat) else 0.0,
                     "p_value": float(p_val) if np.isfinite(p_val) else 1.0}
    return out


def _tensor_set(batch):
    return TensorDataset(*(torch.as_tensor(batch[key], dtype=torch.float32)
                           for key in ("x", "aux", "y")))


def _model_predict(model, batch, device, batch_size=256):
    model.eval()
    out = []
    with torch.no_grad():
        for x, aux, _ in DataLoader(_tensor_set(batch), batch_size=batch_size):
            out.append(model(x.to(device), aux.to(device)).cpu().numpy())
    return np.concatenate(out)


def _ridge_features(batch):
    """Une référence simple : statistique des signaux et métadonnées du passé.

    Par cycle : moyenne V/I/T + V initial/final + caractéristiques connues.
    L'aplatissement ne porte que sur la fenêtre des cycles passés.
    """
    x, a = batch["x"], batch["aux"]
    descriptors = np.concatenate([x.mean(axis=2), x[:, :, 0, :1],
                                  x[:, :, -1, :1], a], axis=2)
    return descriptors.reshape(len(x), -1)


def _multi_task_loss(prediction, target, alpha):
    """Pertes compatibles avec les deux cibles z-score indépendantes."""
    soh = torch.nn.functional.mse_loss(prediction[:, 0], target[:, 0])
    rul = torch.nn.functional.smooth_l1_loss(prediction[:, 1], target[:, 1], beta=1.0)
    return alpha * soh + (1.0 - alpha) * rul


def _fit_neural(train, val, variant, cfg, device):
    """Entraîne sans jamais voir la cellule de test."""
    model = BatteryPredictor(variant, cfg.hidden, cfg.heads, cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    loader = DataLoader(_tensor_set(train), batch_size=cfg.batch,
                        shuffle=True, generator=torch.Generator().manual_seed(cfg.seed))
    val_loader = DataLoader(_tensor_set(val), batch_size=cfg.batch)
    best_loss, best_state, epoch_best, stale = float("inf"), None, -1, 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_sum = 0.0
        for x, aux, y in loader:
            x, aux, y = x.to(device), aux.to(device), y.to(device)
            loss = _multi_task_loss(model(x, aux), y, cfg.alpha_soh)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_sum += loss.item() * len(x)

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for x, aux, y in val_loader:
                x, aux, y = x.to(device), aux.to(device), y.to(device)
                val_sum += _multi_task_loss(model(x, aux), y, cfg.alpha_soh).item() * len(x)
        val_loss = val_sum / len(val["y"])
        history.append({"epoch": epoch, "train_loss": train_sum / len(train["y"]),
                        "validation_loss": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epoch_best, stale = epoch, 0
        else:
            stale += 1
        if stale >= cfg.patience:
            break
    model.load_state_dict(best_state)
    return model, epoch_best, history


def train_one(cells, test_id, variant, cfg, out_dir, device_name="auto"):
    """Orchestre une cellule test, les fichiers de résultats et le checkpoint."""
    cfg.validate()
    seed_everything(cfg.seed)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    train, val, test, scaler, group = build_fold(cells, test_id, cfg.window, cfg.aux_mode)
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    history = []

    if variant == "ridge":
        # Baseline déterministe, sans early stopping ni dépendance au test.
        model = Ridge(alpha=10.0)
        model.fit(_ridge_features(train), train["y"])
        prediction = scaler.inverse_y(model.predict(_ridge_features(test)))
        np.savez_compressed(str(folder / "ridge_coefficients.npz"),
                            coef=model.coef_, intercept=model.intercept_)
        param_count = int(model.coef_.size + model.intercept_.size)
        epoch_best = 0
    elif variant in MODELS:
        model, epoch_best, history = _fit_neural(train, val, variant, cfg, device)
        prediction = scaler.inverse_y(_model_predict(model, test, device))
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Le checkpoint contient les prétraitements, les groupes et la config.
        torch.save({"state_dict": model.cpu().state_dict(),
                    "variant": variant, "config": vars(cfg),
                    "scaler": scaler.to_dict(), "group": group},
                   str(folder / "model.pt"))
    else:
        raise ValueError("Modele indisponible : " + variant)

    duration = time.perf_counter() - started
    truth = scaler.inverse_y(test["y"])
    scores = metrics(truth, prediction)
    result = {"variant": variant, "seed": cfg.seed, "window": cfg.window,
              "group": group, "best_epoch": epoch_best,
              "train_samples": len(train["y"]), "validation_samples": len(val["y"]),
              "test_samples": len(test["y"]), "parameters": param_count,
              "training_seconds": round(duration, 3), "metrics": scores,
              "bias_ttest": paired_bias_ttest(truth, prediction)}
    (folder / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if history:
        _write_csv(folder / "history.csv", history)
    # Il faut les numéros des cycles réellement testés pour interpréter
    # un graphique, et non seulement une seule métrique.
    end_cycles = windows(cells[test_id], cfg.window)["cycle"]
    rows = [{"battery": test_id, "observed_cycle": int(end_cycles[i]),
             "true_next_soh_pct": float(truth[i, 0]),
             "pred_next_soh_pct": float(prediction[i, 0]),
             "true_rul_cycles": float(truth[i, 1]),
             "pred_rul_cycles_raw": float(prediction[i, 1]),
             "pred_rul_cycles_display": max(0.0, float(prediction[i, 1]))}
            for i in range(len(truth))]
    _write_csv(folder / "predictions.csv", rows)
    plot_prediction_file(folder / "predictions.csv")
    return result


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(processed, out_dir, variants, test_ids, cfg, device="auto"):
    from .data import eol_index
    cells = read_cells(processed)
    eligible = sorted(name for name, cell in cells.items()
                      if eol_index(cell["capacity"]) is not None
                      and eol_index(cell["capacity"]) >= cfg.window)
    if len(eligible) < 3:
        raise ValueError("Trois batteries avec EOL observe et suffisamment de cycles sont necessaires."
                         " Executer 'download', puis 'prepare'.")
    tests = eligible if test_ids == ["all"] else test_ids
    results = []
    for battery in tests:
        if battery not in eligible:
            raise ValueError("Batterie test non admissible : " + battery)
        for variant in variants:
            place = Path(out_dir) / ("seed_%d" % cfg.seed) / battery / variant
            print("[TRAIN]", battery, variant, "train/validation/test independants")
            result = train_one(cells, battery, variant, cfg, place, device)
            results.append({"test_battery": battery, "variant": variant,
                            **result["metrics"]})
            print("  RUL MAE = %.3f cycles; SOH MAE = %.3f points" %
                  (result["metrics"]["rul_mae_cycles"],
                   result["metrics"]["soh_mae_pct_points"]))
    _write_csv(Path(out_dir) / "comparison.csv", results)
    # Moyennes macro : chaque batterie test compte une fois, pas chaque cycle.
    summary = {}
    for variant in variants:
        selected = [row for row in results if row["variant"] == variant]
        summary[variant] = {
            metric: {"mean": float(np.mean([row[metric] for row in selected])),
                     "std": float(np.std([row[metric] for row in selected]))}
            for metric in ("soh_mae_pct_points", "soh_rmse_pct_points",
                           "rul_mae_cycles", "rul_rmse_cycles")}
    (Path(out_dir) / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return results


def _ci95_stats(values):
    """Moyenne, écart-type (ddof=1) et demi-largeur IC 95 % (loi de Student)."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    mean, std = float(np.mean(arr)), float(np.std(arr, ddof=1)) if n > 1 else 0.0
    ci = float(stats.t.ppf(0.975, df=n - 1) * std / np.sqrt(n)) if n > 1 else 0.0
    return {"mean": mean, "std": std, "ci95": ci}


def run_multiseed(processed, out_dir, variants, test_ids, cfg, seeds, device="auto"):
    """Exécute `run_experiment` sur plusieurs graines et agrège IC 95 % + t-test de biais."""
    out = Path(out_dir)
    all_runs = []
    for seed in seeds:
        all_runs.extend(
            {"seed": seed, **r}
            for r in run_experiment(processed, out, variants, test_ids, replace(cfg, seed=seed), device)
        )

    batteries = sorted({r["test_battery"] for r in all_runs})
    metric_keys = ("soh_mae_pct_points", "soh_rmse_pct_points", "soh_r2", "rul_mae_cycles", "rul_rmse_cycles")
    per_battery_rows, ens_rows = [], []

    for bat in batteries:
        for var in variants:
            sub = [r for r in all_runs if r["test_battery"] == bat and r["variant"] == var]
            # 1) Agrégation par graine (moyenne ± std ± IC95 %)
            row = {"test_battery": bat, "variant": var, "n_seeds": len(seeds)}
            for k in metric_keys:
                st = _ci95_stats([r[k] for r in sub if r[k] is not None])
                row[f"{k}_mean"], row[f"{k}_std"], row[f"{k}_ci95"] = st["mean"], st["std"], st["ci95"]

            # 2) Prédiction d'ensemble (moyenne sur les graines) + t-test apparié de biais
            preds, truth = [], None
            sig_rul_seeds = 0
            for s in seeds:
                fold = out / f"seed_{s}" / bat / var
                rep = json.loads((fold / "report.json").read_text(encoding="utf-8"))
                if rep["bias_ttest"]["rul"]["p_value"] < 0.05:
                    sig_rul_seeds += 1
                with (fold / "predictions.csv").open(encoding="utf-8", newline="") as f:
                    csv_rows = list(csv.DictReader(f))
                if truth is None:
                    truth = np.array([[float(x["true_next_soh_pct"]), float(x["true_rul_cycles"])] for x in csv_rows])
                preds.append(np.array([[float(x["pred_next_soh_pct"]), float(x["pred_rul_cycles_raw"])] for x in csv_rows]))

            ens_pred = np.mean(preds, axis=0)
            ens_m = metrics(truth, ens_pred)
            ens_bias = paired_bias_ttest(truth, ens_pred)
            ens_rows.append({"test_battery": bat, "variant": var, **ens_m})
            row.update({
                "ens_rul_mae": ens_m["rul_mae_cycles"], "ens_rul_rmse": ens_m["rul_rmse_cycles"],
                "ens_soh_mae": ens_m["soh_mae_pct_points"], "ens_soh_r2": ens_m["soh_r2"],
                "rul_mean_bias": ens_bias["rul"]["mean_bias"],
                "rul_t_stat": ens_bias["rul"]["t_stat"], "rul_p_value": ens_bias["rul"]["p_value"],
                "sig_rul_bias_seeds": f"{sig_rul_seeds}/{len(seeds)}",
                "soh_mean_bias": ens_bias["soh"]["mean_bias"],
                "soh_t_stat": ens_bias["soh"]["t_stat"], "soh_p_value": ens_bias["soh"]["p_value"],
            })
            per_battery_rows.append(row)

    _write_csv(out / "multiseed_per_battery.csv", per_battery_rows)

    # 3) Résumé macro inter-cellules : moyenne des 4 batteries par graine, puis IC 95 % sur les graines
    summary = {"seeds": list(seeds), "batteries": batteries, "macro_over_seeds": {}, "ensemble_macro": {}}
    for var in variants:
        seed_macros = {
            k: [float(np.mean([r[k] for r in all_runs if r["seed"] == s and r["variant"] == var and r[k] is not None]))
                for s in seeds]
            for k in metric_keys
        }
        summary["macro_over_seeds"][var] = {k: _ci95_stats(v) for k, v in seed_macros.items()}
        ens_sub = [r for r in ens_rows if r["variant"] == var]
        summary["ensemble_macro"][var] = {k: float(np.mean([r[k] for r in ens_sub])) for k in metric_keys}

    (out / "multiseed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def predict_checkpoint(checkpoint, processed, battery, cycle, device="cpu"):
    """Inférence après un cycle complet; aucune étiquette n'entre dans le modèle."""
    try:
        content = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    except TypeError:  # torch antérieur à weights_only
        content = torch.load(str(checkpoint), map_location="cpu")
    cfg = content["config"]
    model = BatteryPredictor(content["variant"], cfg["hidden"], cfg["heads"],
                             cfg["dropout"])
    model.load_state_dict(content["state_dict"])
    model.eval()
    cells = read_cells(processed)
    if battery not in cells:
        raise ValueError("Batterie absente des donnees preparees")
    cell = cells[battery]
    window = cfg["window"]
    if cycle < window or cycle > len(cell["x"]):
        raise ValueError("Cycle non admissible : attendre au moins %d decharges" % window)
    sample = _mask_aux({"x": cell["x"][cycle - window:cycle][None],
                        "aux": cell["aux"][cycle - window:cycle][None],
                        "y": np.zeros((1, 2), dtype=np.float32)},
                       cfg.get("aux_mode", "all"))
    scaler = TrainScaler.from_dict(content["scaler"])
    sample = scaler.transform(sample)
    with torch.no_grad():
        y = model(torch.from_numpy(sample["x"]),
                  torch.from_numpy(sample["aux"])).numpy()
    raw = scaler.inverse_y(y)[0]
    return {"battery": battery, "observed_cycle": cycle,
            "next_soh_pct_raw": float(raw[0]),
            "rul_cycles_raw": float(raw[1]),
            "rul_cycles_display": float(max(0, raw[1])),
            "warning": "Estimation de recherche, pas une garantie de securite BMS"}
