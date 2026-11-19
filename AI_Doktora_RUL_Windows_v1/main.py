"""Entrée unique de AI-Doktora; exécuter `python main.py --help`."""
import argparse
import json
from pathlib import Path
from src.config import Experiment
from src.data import prepare, read_cells, eol_index
from src.demo import make_demo
from src.download import download_nasa
from src.experiment import run_experiment, run_multiseed, predict_checkpoint

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="AI-Doktora : futur SOH et RUL batterie NASA")
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help="Télécharger les fichiers NASA officiels")
    dl.add_argument("--archive", type=Path, help="ZIP NASA officiel déjà téléchargé")
    sub.add_parser("prepare", help="Convertir MAT => NPZ et générer l'audit")
    sub.add_parser("audit", help="Afficher événements EOL et nombre de cycles")
    sub.add_parser("demo", help="Générer un dataset SYNTHETIQUE pour les tests seulement")
    train = sub.add_parser("train", help="Entraîner une architecture sur un fold séparé par batterie")
    comparison = sub.add_parser("compare", help="Comparer toutes les architectures sur plusieurs folds")
    for p in (train, comparison):
        p.add_argument("--test", choices=["all", "B0005", "B0006", "B0007", "B0018"],
                       default="all")
        p.add_argument("--window", type=int, default=10)
        p.add_argument("--epochs", type=int, default=100)
        p.add_argument("--batch", type=int, default=32)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--patience", type=int, default=15)
        p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        p.add_argument("--aux-mode", choices=["all", "no_cycle", "no_duration", "none"],
                       default="all", help="Ablation des variables auxiliaires")
        p.add_argument("--out", type=Path, default=ROOT / "results" / "experiments")
    train.add_argument("--model", choices=["ridge", "cnn", "cnn_lstm", "cnn_tcn", "hybrid"],
                       default="hybrid")
    comparison.add_argument("--seeds", type=int, nargs="+",
                            help="Liste de graines pour évaluation multi-graines avec IC 95 %% et t-test")
    inference = sub.add_parser("predict", help="Inférence sans consulter les étiquettes")
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--battery", choices=["B0005", "B0006", "B0007", "B0018"],
                           required=True)
    inference.add_argument("--cycle", type=int, required=True,
                           help="Nombre de cycles de décharge achevés et observés")
    args = parser.parse_args()
    raw = ROOT / "data" / "raw"
    processed = ROOT / "data" / "processed"
    if args.command == "download":
        answer = download_nasa(raw, args.archive)
    elif args.command == "prepare":
        answer = prepare(raw, processed)
    elif args.command == "demo":
        if list(processed.glob("B*.npz")):
            parser.error("data/processed contient deja des batteries. Effacer ou sauvegarder ce dossier avant demo.")
        make_demo(processed)
        answer = {"message": "DEMONSTRATION SYNTHETIQUE; aucune valeur n'est un resultat NASA"}
    elif args.command == "audit":
        cells = read_cells(processed)
        answer = {name: {"cycles": len(v["capacity"]),
                         "eol_80pct_cycle": None if eol_index(v["capacity"]) is None
                         else eol_index(v["capacity"]) + 1}
                  for name, v in cells.items()}
        if (processed / "DEMO_NOT_REAL_DATA.txt").exists():
            answer["warning"] = "DONNEES SYNTHETIQUES; PAS NASA"
    elif args.command in ("train", "compare"):
        if (processed / "DEMO_NOT_REAL_DATA.txt").exists():
            print("AVERTISSEMENT: donnees synthetiques; ne pas publier ces scores")
        cfg = Experiment(window=args.window, epochs=args.epochs, batch=args.batch,
                         seed=args.seed, patience=args.patience, aux_mode=args.aux_mode)
        variants = [args.model] if args.command == "train" else [
            "ridge", "cnn", "cnn_lstm", "cnn_tcn", "hybrid"]
        if args.command == "compare" and args.seeds:
            answer = run_multiseed(processed, args.out, variants, [args.test], cfg,
                                   args.seeds, device=args.device)
        else:
            answer = run_experiment(processed, args.out, variants, [args.test], cfg,
                                    device=args.device)
    elif args.command == "predict":
        answer = predict_checkpoint(args.checkpoint, processed,
                                    args.battery, args.cycle)
    else:
        parser.error("Commande inconnue")
    print(json.dumps(answer, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
