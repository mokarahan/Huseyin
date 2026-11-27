# Verifiable scientific protocol — AI-Doktora

## Positioning and purpose

This project does not claim to have invented the CNN, the self-attention mechanism, the TCN or their combination. It examines a **falsifiable hypothesis**: on the available NASA discharge data, does a local CNN representation followed by causal attention across cycles and a causal temporal convolution enable better **out-of-battery** prediction of future SOH and RUL than a simpler model?

The conclusion "the CNN–MHA–TCN combination is useful" is justified **only if** its out-of-battery performance and its stability exceed those of the relevant controls. Worse performance is a valid scientific result; the combination must not be defended a priori.

## Formal definition

Let `b` be a cell with `N_b` valid observed discharge cycles, sorted in their experimental order. For cycle `k`:

- `V_{b,k}(t), I_{b,k}(t), T_{b,k}(t)`: measurements over a completed discharge;
- `C_{b,k}`: capacity measured during this discharge (Ah);
- `D_{b,k}`: duration in seconds, extracted from the time series;
- `C_nom = 2 Ah`: nominal capacity in this protocol;
- `C_EOL = 0.8 × C_nom = 1.6 Ah`: EOL threshold fixed before testing.

The first observed crossing index is:

`e_b = min{k | C_{b,k} <= C_EOL}`.

It may not exist: a cell without a crossing is **right-censored**. The last recorded cycle of a censored cell **is not an EOL event** and is never artificially given a RUL=0 label.

For each `k < e_b`, the target vector is:

`y_(b,k) = (100 × C_(b,k+1)/C_nom, e_b - k)`.

**Important:** a discharge is a unit of observation; a RUL value of 20 means "20 discharges remaining until the first experimental crossing of the threshold", not 20 hours or 20 days. The future SOH at `k+1` must never appear in the input.

The input of each example is:

`X_(b,k) = [ S_(b,k-W+1), …, S_(b,k) ] ∈ R^(W×128×3)`.

`S` represents the V/I/T signals interpolated over 128 fractions of the discharge time **specific to that cycle**, without using future cycles. The complement `A_(b,k) ∈ R^(W×2)` contains the observed durations (`log1p`) and the discharge indices. **No capacity appears in X or A.**

### Characterization of delay and causality

Each example is built right after discharge `k`. `D_(b,k)` is then observable, since the whole curve is complete. The model cannot be used in real time during a partial discharge without new adaptation and validation.

For the sequence layers, every future position `j > i` is masked in the attention and ignored by the causal temporal convolution. The final predictor uses only the representation of cycle `k`. Masking alone does **not** fix a leak caused elsewhere (label, normalization, model selection).

## Pipeline and exact layer choices

### Per-cycle CNN

The encoder shares its weights across all cycles:

1. Conv1D `in_channels=3`, `out_channels=16`, `kernel=5`, symmetric padding 2 in the internal time **of the cycle**; ReLU.
2. MaxPool1D, size 2: 128 → 64 samples.
3. Conv1D 16→32, kernel 3, padding 1; ReLU.
4. AdaptiveAvgPool1D to 4 positions: 32 × 4 = 128 features.
5. Linear 128→32, ReLU; concatenation with the two observable auxiliaries; Linear 34→32, ReLU.

The CNN uses the whole signal of the **completed cycle**. It is not causal **within** that discharge, which is intentional: the decision is only made at the end of the cycle. It is causal **across cycles** since future cycles are absent from the input.

**Why ReLU rather than sigmoid/tanh/GELU?** ReLU reduces the risk of saturation and the cost per layer, and two convolutions are enough as a starting point. The LSTM uses its internal sigmoid/tanh gates, imposed by the standard cell definition. The SOH/RUL outputs remain linear because their standardized values are not constrained to be positive.

### MHA across cycles (hybrid only)

Total dimension `d_model=32`, `num_heads=4`, `8` dimensions per head; a single PyTorch attention block. The mask `M[i,j]=True` for `j>i` prevents any attention to a future cycle. The output is added through a residual connection and then normalized by `LayerNorm`. No artificial positional encoding is added: order is provided by the known index in A and by the cycle sequence. This decision should be ablated in an in-depth study.

For each head: `Attention(Q,K,V) = softmax(QK^T / sqrt(d_head) + mask) V`, with weights learned on past representations. This mechanism **can learn** long-range links; using it does not automatically demonstrate biological interpretability or an SOH improvement.

### Causal temporal TCN (cnn_tcn and hybrid)

Two residual blocks. In each, Conv1D 32→32, kernel `3`, dilation `d=1` then `d=2`; left padding only, ReLU, dropout 0.10, residual addition. The temporal receptive field is `1 + (3-1)(1+2) = 7` cycles. The attention of the hybrid variant can, for its part, integrate all W cycles, if the learned parameters exploit them.

### FCL and two heads

Representation of the last cycle → `Linear 32→16` → ReLU → Dropout 0.10. Two separate projections `Linear 16→1` produce next-cycle SOH and current RUL, each as a z-score. De-standardization is done at the output, using the means and standard deviations of **train**, never those of test.

### Number of layers: justification and limits

Two CNN layers, one MHA, two TCN blocks and one FCL hidden layer form a **compact and verifiable** starting point, not a demonstrated optimum. Relevant future comparisons: 1 versus 2 CNN convolutions, 1 versus 2 TCN blocks, 0 versus 1 MHA block, `W∈{10,20,30}`. Each comparison requires a justification before the test results are used. Adding layers without evidence is discouraged.

## Loss, standardization and optimization

For each target `j` and each signal `c`, the transformation is `z=(x - μ_train)/σ_train`. The statistics are fitted on TRAIN observations exclusively; overlapping windows naturally influence their distribution and must be declared. No individual per-cycle normalization is applied, since it could erase the level variations related to wear.

`L = α×MSE(zSOH_pred,zSOH_true) + (1-α)×Huber_δ=1(zRUL_pred,zRUL_true)`, with `α=0.5` for the initial protocol. AdamW (`lr=0.001`, `weight_decay=1e-4`), gradient clipping at norm 1.0, mini-batches of 32, seed 42, patience 15 and 100 epochs **maximum**.

The model restored after early stopping is the one with the best **validation** score. Test values play no role in the parameters, the scaling, the number of epochs, the selection of architectures or the selection of hyperparameters.

## Mandatory experimental plan

### Phase 0 — Audit of the real files

For each cell: number of discharges, minimum/maximum capacity, EOL observed or not, number of excluded records, dimensions, duration of each discharge, SHA-256 hashes. Check the physical consistency of the durations and the quality of the series. A failed load must not trigger automatic generation of synthetic data.

### Phase 1 — Simple controls

Compare Ridge, CNN, CNN-LSTM, CNN-TCN. Heavier architectures are only admissible if they bring a non-trivial improvement over these baselines. The effect of the **cycle index** and the **duration** must also be controlled: remove them separately in a complementary study in order to know what actually drives the performance.

### Phase 2 — Hybrid model hypothesis

Compare `hybrid` versus `cnn_tcn`: the only major architectural addition is the MHA (with residual normalization). Document the cost: number of parameters, CPU/GPU time, prediction time (specific measurement not yet integrated), memory.

### Phase 3 — Out-of-battery validity

One entire cell is test, another is validation, the others form train. Repeat the independent folds on B0005, B0006, B0007 and B0018 *if EOL is observed*. Primarily evaluate RUL MAE/RMSE in cycles and future SOH MAE/RMSE in percentage points. Report each fold, then the macro aggregates. Windows from the same cell are not independent replications.

**Limitation of the current protocol:** for each test, the validation cell is chosen in alphabetical order among the remaining cells. This is reproducible but not optimal for estimating selection variability. A confirmatory doctoral study should create outer folds and several **inner validation rotations**, without ever testing several models selected on the same outer fold multiple times. Results obtained with 3 cells do not allow strong statistical inference.

### Phase 4 — Confirm or reject the addition of GWO

GWO is **not** an automatic source of originality. A complementary study must define a compact search space, an equivalent budget between GWO and random search, an objective function measured **only on inner validation**, multiple seeds and a CPU/GPU cost. If GWO does not bring a robust improvement, it must be abandoned and the decision justified. No GWO script is added in this delivery so as not to simulate an undemonstrated contribution.

### Phase 5 — Limits and external generalization

- Variations in chemistry, temperature, current, cut-off voltage: NASA is not enough to prove industrial robustness.
- Evaluate, if available, another dataset (CALCE, Oxford, MATR) with a harmonized EOL definition and compatible protocols.
- A right-censored cell requires a survival model / censoring treatment, not a forced RUL label.
- For real-world use, establish prediction intervals, calibration, domain checks, handling of incomplete measurements and validation on BMS hardware.
- Do not mix **SOH at k+1** prediction, **SOH at k** estimation, **full trajectory** extrapolation and **RUL at k** forecasting in comparison tables.

## Traceability of decisions and status

| Decision | Status | Reason |
|---|---|---|
| NASA B0005/6/7/18, `discharge` | Implemented | Matches the project sources |
| EOL at 80 % (1.6 Ah) | Implemented | Common threshold proposed by the notes, to be verified on the .mat files |
| V/I/T input, 128 points + duration + index | Implemented | Verifiable representation after a complete discharge |
| Multitask SOH(k+1) + RUL(k) | Implemented | Non-trivial, temporally aligned targets |
| Z-score TRAIN only | Implemented and tested | No leakage through standardization |
| Independent test battery | Implemented and tested | Out-of-cell generalization |
| CNN/MHA/TCN/2 heads | Implemented and technically tested | Predictive contribution still to be demonstrated |
| GWO | Not implemented | No justification from baselines |
| Certified prediction on NASA | **Not established** | .mat files absent from the provided archive |
| Performance on a real BMS battery | **Not established** | NASA test-bench data only |

## Main scientific sources

1. Saha and Goebel, NASA Battery Data Set (2007): https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
2. Vaswani et al., Attention Is All You Need (2017): https://arxiv.org/abs/1706.03762
3. Bai, Kolter, Koltun, Temporal Convolutional Networks (2018): https://arxiv.org/abs/1803.01271
4. Zhang, Robustness of CNN-augmented sequential models for Li-ion battery RUL prediction under data scarcity (PLOS ONE, 2025): https://doi.org/10.1371/journal.pone.0339528
5. Lithium-ion battery RUL prediction based on optimized VMD-SSA-PatchTST algorithm (Scientific Reports, 2025): https://www.nature.com/articles/s41598-025-11934-7
6. Zhu et al., Leakage-Resistant and Censor-Aware Evaluation ... (Batteries, 2026): https://doi.org/10.3390/batteries12110484

**Bibliographic warning:** published protocols differ in their inputs, horizons, EOL thresholds and splits. Figures from articles cannot be compared directly with the project's values until the experiments are harmonized.
