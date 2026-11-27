# Audit of the received AI-Doktora-main.zip archive

**Purpose:** verifiable justification of the changes made to the project, without confusing original code, scientific intent and confirmed functionality.

## Sources examined

- historical `README.md` and `requirements.txt`;
- `src/DataExporter.py`, `src/DataNormalizer.py`, `src/dataset/BatteryDataset.py`;
- `src/models/{CNN,LSTM,TCN,MHA}.py` and the four `*Handler.py` files;
- Windows PowerShell and Linux Shell scripts;
- provided notes: `docs/pdf/001-NASA_Batarya_SOH_RUL_Tahmin_Yontemi.pdf`, `002-NASA_128_Nokta_Egitim_Notu.pdf`, `003-NASA_Batarya_Z_Score_Standardizasyon.pdf`.

## Specific findings

| Finding in the original archive | Consequence | Decision in the reconstruction |
|---|---|---|
| README: RUL on NASA, 70 % threshold (1.4 Ah); note 001: two targets, future SOH / RUL, 80 % threshold (1.6 Ah). | The scientific questions and the labels do not describe exactly the same study. | Common objective defined in the README, 80 % threshold fixed and audited. |
| No `.mat` file among the 92 entries of the ZIP. | NASA training cannot be run from this archive alone. | Official downloader + controlled import, no invented NASA data. |
| `DataNormalizer.py` uses `EOL_IDX` with no guarantee that a cycle crosses the threshold. | Risk of an exception and of a false label on a censored cell. | `eol_index` returns `None` and censored cells are excluded from supervised learning. |
| `DataNormalizer.py` computes global extrema while iterating over the cells. | Risk of contaminating preprocessing with the test cells; confusion between MinMax and z-score. | Three z-score transformations computed after the split, on train only. |
| `BatteryDataset.py` applies an individual z-score to the CSV columns. | Physical shifts between cycles may be erased, contrary to note 003. | Global per-channel standardization on training data; durations preserved. |
| The RUL was encoded in the **CSV file name**. | Fragile metadata, mixing of identifiers and labels; risk of errors in sorting and evaluation. | `.npz` files and explicit label matrices, cycle number saved. |
| The CNN model consumes `[B,128,4]` but the preprocessor from note 002 returns `[128,3]` V/I/T. | Inconsistency between the truly informative channels and the number of inputs. | V/I/T signals = 3; duration/index = 2 separate auxiliaries. |
| The historical TCN class specifies `input_shape='NLC'` and selects `tcn_out[:,:, -1]`. | Risk of selecting a **channel dimension** rather than the last time step, depending on the package contract. | Native `Conv1d` TCN, documented axes, last step read explicitly. |
| Historical attention parameterized with `pos_embedding(1,128,d_model)`. | Rigid coupling to 128 internal steps and confusion between samples within a cycle and historical cycles. | Attention after per-cycle encoding, sequence of W cycles, causal mask. |
| Four `*Handler` classes repeat almost the entire train/validation/test loop. | Tedious maintenance and potential divergences in metrics/early stopping. | One training loop / one metrics pipeline / one variant factory. |
| Scripts reference `HOME/Dev/AI-Doktora` and personal venv paths. | Non-portable Windows execution. | `win/*.ps1` scripts computing the root folder via `$PSScriptRoot`. |
| Note 001 considers GWO without experiments justifying it. | Costly hyperparameter search, risk of selection on test. | GWO reserved for a study with an explicit budget and controls, not added artificially. |

## Verification scope

The reconstruction has tests for shapes, label generation, resampling, MHA/TCN causality, normalization, cell separation, reading synthetic MATLAB files, extraction of nested ZIPs, as well as a full run with synthetic data.

**What the audit does not claim:** empirical validation of the model on real NASA curves, reproduction of the results in the PDFs, or proof of MHA superiority. These conclusions require running on the official files, an audit of the excluded records, several folds and an appropriate statistical protocol.
