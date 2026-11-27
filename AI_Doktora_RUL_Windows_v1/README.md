# AI-Doktora | Battery SOH & RUL Research Laboratory

**Scientific reconstruction of the NASA Battery project — Windows / Python 3.9**  
**Status:** research pipeline that runs and is tested on synthetic data. NASA scores are **not** provided, because the original archive contained no NASA `.mat` files.  
**Guide language:** English (French version: [README_FR.md](README_FR.md)). Code and variables are documented in short technical English.

---

## 1. Scientific question: what exactly does this project predict?

A lithium-ion cell loses capacity over its discharges. The goal is neither to classify "good/bad battery" nor to directly recover a capacity already given as input, but to use **only the observations available after past discharges** to produce two estimates:

1. **Next-cycle SOH**: health of the cell at the *next* discharge, expressed as a % of nominal capacity.
2. **RUL at the observed cycle**: number of discharges still needed before the **first** crossing of 80 % of nominal capacity.

This version gives a rigorous answer to one point in the historical documentation: the old README only mentioned RUL with a 70 % threshold, while the methodological notes `NASA_Batarya_SOH_RUL_Tahmin_Yontemi` clearly describe joint prediction of future SOH + RUL and a common 80 % threshold. This project's protocol adopts **80 %**, fixed and declared before the experiments, because according to these notes the crossing would be observable for B0005, B0006 and B0007. This must be checked automatically on the real files.

### Physical definition of the targets

For the discharge with index `k` (internal indices from 0 to N-1), `C_k` is the measured capacity, and `C_nom = 2.0 Ah`:

```text
SOH(k+1) [%] = 100 × C_(k+1) / C_nom
k_EOL       = min { j : C_j <= 1.6 Ah }
RUL(k)      = k_EOL - k                         [remaining discharges]
```

A training observation ends at cycle `k`, with `k < k_EOL`, and uses **no measurement from cycle `k+1` as input**. It carries a future SOH(k+1) target and a RUL(k) target. Since these targets have different units, they are normalized **separately on the training set**.

**Note on RUL = 0.** To keep common SOH(k+1) and RUL(k) targets up to the EOL limit, the pipeline stops at the cycle preceding the crossing. The EOL cycle with RUL = 0 is therefore not a multitask training observation. It remains mathematically defined; this must be stated in any publication.

**Operational note:** a discharge curve must be **complete** before prediction. This is not an "at any moment" on-board vehicle prediction. Extrapolation to partial measurements or to a real BMS is not proven.

## 2. Dataset: NASA PCoE, not an invented dataset

- Official source: **B. Saha and K. Goebel (2007), Battery Data Set, NASA Ames Prognostics Data Repository**.
- Archive URL: https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip
- NASA portal: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- Target cells: `B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat` if the data are present.
- Three signals: `Voltage_measured`, `Current_measured`, `Temperature_measured`.
- `Time` is used for resampling and for the physical duration; `Capacity` is used **exclusively** to build and check the labels, never as network input.
- Charge and impedance operations are excluded in this protocol; using them would require a new experimental hypothesis.

**The starting archive contained NO `.mat` file.** It would be scientifically incorrect to create files resembling NASA measurements. This ZIP therefore includes: a downloader pointing to NASA, a MAT importer, a data audit, and a generator of **synthetic data for software testing only**. The official raw files must be fetched on the user's machine.

## 3. Clean Windows installation (Python 3.9)

### Prerequisites

- Windows 10/11, 64-bit.
- Python **3.9.x 64-bit** with the `py.exe` launcher: https://www.python.org/downloads/release/python-3913/ (or another 3.9.x).
- Internet connection to install the packages and fetch the NASA data (unless you use an already downloaded official archive).
- 8 GB of RAM recommended; **CPU** training is possible. CUDA is optional, if PyTorch and the drivers are compatible.

In PowerShell, opened **in the extracted folder**:

```powershell
# Avoids changing Windows' global execution policies.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\win\installer.ps1
```

The installation creates `.venv` using Python 3.9 and then runs the tests. If `py -3.9` does not exist, install 64-bit Python 3.9 before continuing. Do not copy a Python interpreter into the project by hand: a virtual environment is safer and easier to maintain.

Dependencies are pinned in `requirements.txt` with versions compatible with Python 3.9. Recent GPUs may require adapting **only** the `torch` line according to the official PyTorch compatibility matrix.

### Getting the real data

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\win\nasa.ps1
```

The downloader opens the **official** NASA source, keeps the archive under `data/raw/`, looks for the `.mat` files in its nested ZIPs, extracts the selected cells and writes `data/raw/provenance.json` with the SHA-256 checksums. Preparation builds the `data/processed/Bxxxx.npz` files and `data/processed/audit.json`.

**Mode without automatic download:** download the official archive yourself, then:

```powershell
.\.venv\Scripts\python.exe main.py download --archive "D:\data\5. Battery Data Set.zip"
.\.venv\Scripts\python.exe main.py prepare
.\.venv\Scripts\python.exe main.py audit
```

If the NASA source does not respond, **do not automatically substitute an unknown mirror**. Obtain the archive from a reliable source, check its provenance and use `--archive`. The three folders `data/raw`, `data/processed` and `results` do not need absolute paths, `HOME`, WAMP or a Linux terminal.

### Running the checks and a first experiment

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests
.\.venv\Scripts\python.exe main.py audit
.\.venv\Scripts\python.exe main.py train --model hybrid --test B0005 --epochs 100 --window 10 --seed 42
```

**Compare the methods**, each with the same battery groups and the same window:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\win\comparer.ps1 -Epochs 100 -Window 10 -Seed 42
```

Or directly:

```powershell
.\.venv\Scripts\python.exe main.py compare --test all --epochs 100 --window 10 --seed 42
```

`--test all` takes each eligible battery in turn as an **entire test battery**, keeps another one for validation and the remaining batteries for training. With three cells, there is only **one training cell** per fold, so performance will be uncertain. With four eligible cells, there are two. Each test cell remains independent.

### Reading a prediction

```powershell
.\.venv\Scripts\python.exe main.py predict `
  --checkpoint "results\experiments\seed_42\B0005\hybrid\model.pt" `
  --battery B0005 --cycle 30
```

The command outputs a real number of remaining cycles and the SOH (%) of the next discharge. The **raw** RUL prediction is kept even if it is negative; a second value, clipped at 0, is provided **for display only**. Do not recompute the score using this clipped value as if it were the raw prediction.

### Purely technical demonstration (without NASA)

If you want to check the code before downloading the data, in a **clean folder** where `data/processed/` contains no `B*.npz`:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\win\demo.ps1
```

**WARNING:** this creates **synthetic** laboratory signals, with the `DEMO_NOT_REAL_DATA.txt` warning file. NO score, plot or weight produced by the demonstration may be presented as performance on NASA. Before the real study, move/delete the demonstration `.npz` files and their warning file, then run `download` and `prepare`.

## 4. Studied architecture and activation functions

The hybrid variant is **CNN → causal MHA → causal TCN → FCL → 2 linear heads**. The order follows the methodological notes accompanying the received ZIP.

```text
W completed discharge cycles     X : [batch, W, 128, 3]       A : [batch, W, 2]
             │                              │                          │
             │                         Per-cycle 1D CNN               │
             │                     Conv(3→16,k=5), ReLU                │
             │                     MaxPool(2), Conv(16→32,k=3), ReLU   │
             │                     AdaptiveAvgPool(4), Linear(128→32) │
             │                              └──────────┬───────────────┘
             │                          Fusion+ReLU [batch,W,32]
             │                                      │
             │                  Causal MHA, 4 heads, residual, LayerNorm
             │                                      │
             │                 Causal residual TCN, 2 blocks (d=1,2)
             │                                      │
             └──────────────────── Last observable cycle ─────────────┘
                                                    │
                                      Linear(32→16), ReLU, Dropout
                                            ┌───────┴────────┐
                                      Linear(16→1)     Linear(16→1)
                                      SOH(k+1)            RUL(k)
                                        Z-score           Z-score
```

The auxiliary input has **two observable variables**: `log(1 + discharge duration in seconds)` and the **index of the completed discharge**. The absolute duration would be lost after temporal resampling onto [0,1]. The discharge index is known at estimation time, but may produce a statistical shortcut; removing it must be the subject of a sensitivity experiment before any conclusion about the architecture.

| Layer / activation | Why here? |
|---|---|
| 1D CNN with 2 convolutions, kernels 5 and 3 | Detect local variations of the three signals without imposing a large network on small series. |
| ReLU after convolutions and projection | Cheap non-linearity, easy to reproduce; hypothesis to test against GELU if needed. |
| Pooling 2 then adaptive pooling 4 | Reduce the 128-point measurements to a compact representation, without fixing batch values. |
| MHA with 4 heads, dimension 32 | Learn relations between past cycles; the upper-triangular mask forbids looking at later cycles. |
| TCN with 2 blocks, dilations 1 and 2 | Capture trends over several cycles with causal convolutions; theoretical receptive field of 7 cycles after the two blocks (kernel 3), without promising that it summarizes the whole window W. |
| FCL `32→16`, ReLU, dropout 0.10 | Small shared head to limit over-parameterization. |
| 2 **linear** outputs, no activation | The **z-score targets can be negative**; ReLU/Softplus would be unsuitable during training. |

**Why not 8 heads, 6 layers or a large Transformer?** The starting protocol involves three to four cells, not thousands of independent aging runs. Overlapping windows do not create new independent cells. A heavier architecture needs experimental justification and a credible validation budget, not just a higher theoretical capacity.

## 5. Preparation without information leakage

1. Load the valid `discharge` operations from the `.mat` files.
2. Check time, current, voltage, temperature and capacity; record all exclusions.
3. Sort the time stamps and resample **each complete discharge** to 128 points over its own relative time.
4. Determine the **observed EOL per cell** at the fixed threshold of 1.6 Ah; if it is never reached, mark it `censored` and **do not invent** an exact RUL.
5. Build sequences of W discharges ending at cycle k, never beyond.
6. Split **by cell**, with an exclusive training/validation/test role.
7. Fit **the three standardizations**: signals (3 channels), auxiliaries (2 variables), targets (SOH and RUL), on the **train windows only**.
8. Apply these same parameters, without recomputing them, to the validation and test windows.
9. Save the real units, the index of each prediction and the results per battery.

The SOH(k+1) label uses future data only in the **ground truth**, never as a feature. Current or future capacity is not concatenated to the input vector. The only way to check generalization to unseen cells remains strict separation of the cells at test time.

## 6. Evaluation: what to compare and what to publish?

The available models are:

| CLI name | Model | Question being controlled |
|---|---|---|
| `ridge` | Ridge regression, descriptors of the W cycles | Is simplicity enough? |
| `cnn` | CNN of the most recent cycle | Is a sequential history even needed? |
| `cnn_lstm` | CNN + one-layer LSTM | Is recurrence suited to small data? |
| `cnn_tcn` | CNN + two-block TCN | Value of temporal convolution without attention? |
| `hybrid` | CNN + causal MHA + TCN | Does attention really do better than CNN-TCN? |

Same window, same seed and same groups for every comparison. Network training uses **AdamW** (rate `1e-3`, weight decay `1e-4`), clipped gradients and early stopping based on validation only. The outputs are normalized separately. The loss function is:

```text
L = 0.5 × MSE( zSOH_pred, zSOH_true )
  + 0.5 × SmoothL1( zRUL_pred, zRUL_true ; beta=1 )
```

MSE penalizes large SOH errors; SmoothL1/Huber limits the influence of very large RUL errors. The 0.5 / 0.5 weighting is **a starting choice**, not an optimized value. Sensitivity must be tested on the development batteries, never on the test results.

**Primary scores:** MAE and RMSE of RUL in **cycles** and of SOH in **percentage points**. SOH R² is provided as a complement. The figures show ground truth and **raw** prediction in the same units. With only 3 or 4 cells, prefer **per-cell** results and the **per-cell macro** average, without pseudo-significance computed on correlated windows.

**Do not claim a SOTA improvement** from this project alone. Several seeds must be evaluated (`42, 123, 2027` for example), variability quantified, baselines implemented on the same data compared, and then, ideally, results confirmed on an external cohort (CALCE/MATR) without changing the protocol based on the test.

## 7. Folder layout and file responsibilities

```text
AI_Doktora_RUL_Windows/
├── README.md                 Goal, usage, hypotheses and interpretation
├── README_FR.md              French version of this README
├── requirements.txt          Pinned Python 3.9 versions
├── main.py                   CLI entry point (prepare/train/compare/predict)
├── src/
│   ├── config.py             EOL, model and training constants
│   ├── data.py               MATLAB reading, filtering, targets, windows
│   ├── download.py           Official acquisition and SHA-256
│   ├── scaling.py            TRAIN-only standardization
│   ├── models.py             CNN, LSTM, TCN, attention, hybrid
│   ├── experiment.py         Per-cell validation, losses and metrics
│   ├── plots.py              Real PNG plots from the CSV
│   └── demo.py               Synthetic data for TESTING only
├── data/
│   ├── raw/                 NASA .mat files + provenance (after download)
│   └── processed/           .npz + audit (after prepare)
├── win/
│   ├── installer.ps1         Python 3.9 environment
│   ├── nasa.ps1              Acquisition + conversion
│   ├── comparer.ps1          Baselines and ablations
│   └── demo.ps1              Synthetic software test
├── tests/                    Unit and integration tests
├── docs/
│   ├── PROTOCOLE_SCIENTIFIQUE.md      Limits, decisions, audit and references (French)
│   ├── PROTOCOLE_SCIENTIFIQUE_EN.md   English version
│   ├── AUDIT_ORIGINAL.md              Audit of the original archive (French)
│   └── AUDIT_ORIGINAL_EN.md           English version
└── results/                  Experimental outputs, not versioned
```

### Outputs of a fold

```text
results/experiments/
  comparison.csv                metrics per cell and variant
  comparison_summary.json       macro cross-cell mean and standard deviation
  seed_42/B0005/hybrid/
    model.pt                    weights + configuration + TRAIN scaler
    report.json                 groups, parameters, timings, metrics
    history.csv                 train / validation losses per epoch
    predictions.csv             each cycle: true, predicted and unit
    soh_prediction.png          true / predicted future SOH
    rul_prediction.png          true / predicted RUL
```

The Ridge model does not produce `model.pt`: its coefficients are saved as `.npz`, with no Ridge loading mechanism for `predict`. The `predict` command applies to the four PyTorch models only.

## 8. Self-assessment and currently known limitations

**Facts verified on this delivery:** the automated tests pass and the models train and predict with **synthetic** inputs. The official `.mat` files are not provided; their import can only be considered validated end to end once the NASA archives have been downloaded and checked. The download may depend on the availability of the NASA server and the local connection.

**Unresolved scientific limitations:** a tiny cohort, strong dependence between windows, different discharge protocols and cut-off voltages, dependence on the cycle index, a common but imperfect SOH(k+1) and RUL(k) definition at the EOL cycle, no uncertainty intervals and no BMS validation. The 80 % threshold is an **experimental criterion** of this protocol, not a universal battery safety rule.

**GWO (Grey Wolf Optimizer):** mentioned in the historical notes, **not implemented here**. Its cost, search space and benefit must be compared with random search on the development cells only. Adding a sophisticated optimizer before the usefulness of attention has been demonstrated is a source of unjustified complexity.

For any academic report, read [docs/PROTOCOLE_SCIENTIFIQUE_EN.md](docs/PROTOCOLE_SCIENTIFIQUE_EN.md) before interpreting the results.

## 9. Selected references

- Saha, B. & Goebel, K. (2007), *Battery Data Set*, NASA Ames PCoE Repository: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- Vaswani et al. (2017), *Attention Is All You Need*, NeurIPS: https://arxiv.org/abs/1706.03762
- Bai, Kolter & Koltun (2018), *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*: https://arxiv.org/abs/1803.01271
- Zhang (2025), *Robustness of CNN-augmented sequential models for Li-ion battery RUL prediction under data scarcity*, PLOS ONE: https://doi.org/10.1371/journal.pone.0339528
- *Lithium-ion battery RUL prediction based on optimized VMD-SSA-PatchTST algorithm* (2025), Scientific Reports: https://www.nature.com/articles/s41598-025-11934-7
- Zhu et al. (2026), *Leakage-Resistant and Censor-Aware Evaluation of Capacity-Only Online Remaining-Useful-Life Prediction...*, Batteries: https://doi.org/10.3390/batteries12110484

**Scope of the references:** they justify precautions and possible mechanisms. They do not prove that this repository's hybrid variant outperforms the literature. Different protocols, metrics and thresholds prevent direct numerical comparisons.
