
# ML Signal Improvement Design

**Date:** 2026-02-18
**Status:** Approved

## Context

Baseline training run (2026-02-17) produced:

| Model    | Accuracy | Top-Decile Win Rate | Lift  | Training Samples |
|----------|----------|---------------------|-------|-----------------|
| TabPFN   | 53.2%    | 48.2%               | 1.66x | 10,000 (capped) |
| LightGBM | 50.1%    | 47.7%               | 1.64x | 175,889         |

Key issues identified:
- TabPFN hard-capped at 10k samples by design
- LightGBM underperforms despite full data access (needs tuning)
- TIMEOUT class (46.3% of data) dominates predictions, diluting WIN signal
- Very few high-confidence (>60%) signals (14–17 samples), limiting actionability
- LOSS recall for TabPFN is extremely low (0.108)

## Goal

**Primary:** Improve top-decile precision — fewer but higher-quality WIN signals.

## Approach: Three-Phase Pipeline

### Phase 1: Binary Reformulation + Model Expansion

**Rationale:** The 3-class problem is muddied by the TIMEOUT majority class. Collapsing LOSS+TIMEOUT into NOT_WIN sharpens focus on what matters: detecting wins.

**Changes to `scripts/train_ml_model.py`:**
- Add `--binary` flag: collapses labels so `WIN=1`, `LOSS+TIMEOUT=0`
- Add `"catboost"` and `"xgboost"` to `--model` choices
- `--model all` trains all 4 models in one run and outputs `metrics_comparison.json`
- Add `--profit-target`, `--stop-loss`, `--hold-days` CLI args (used by Phase 2)
- Top-decile precision becomes headline metric alongside top-decile win rate

**New file: `scripts/tune_lightgbm.py`:**
- Standalone Optuna hyperparameter tuning script
- 50 trials, objective = top-decile precision on validation set
- Outputs best params to `data/ml/lgbm_best_params.json`
- Best params auto-loaded by `train_ml_model.py` when present

**What stays the same:**
- `feature_engineering.py` — untouched in Phase 1
- `MLPredictor` inference class — binary output still surfaces `win_prob`
  (TIMEOUT just folds into `loss_prob`; inference API unchanged)
- Model save/load mechanism

**Success criteria:**
- Binary LightGBM top-decile precision ≥ 3-class TabPFN baseline (48.2%)
- At least one model produces >20 high-confidence (>60%) signals on validation set

---

### Phase 2: Label Parameter Ablation

**Rationale:** The 5%/3%/7-day triple-barrier config is a default, not an optimized choice. Different configs produce different class balances and signal sharpness.

**New file: `scripts/label_ablation.py`:**

Grid search over 27 configurations:
```
profit_target: [0.03, 0.05, 0.07]
stop_loss:     [0.02, 0.03, 0.05]
hold_days:     [5, 7, 10]
```

For each config:
1. Re-apply `apply_triple_barrier_labels()` with those parameters on existing OHLCV cache
2. Train binary LightGBM (fixed hyperparams for comparability, no Optuna)
3. Log: top-decile precision, lift, WIN%, LOSS%, TIMEOUT%, val accuracy

**Output:** `data/ml/label_ablation_results.csv` — sorted by top-decile precision.

**Integration:** The winning config's parameters become the new defaults passed to
`train_ml_model.py`. The ablation script itself never saves a production model.

**Success criteria:**
- At least one config improves top-decile precision by ≥1% over Phase 1 baseline

---

### Phase 3: Ensemble

**Rationale:** Models trained with different inductive biases (TabPFN's meta-learning vs. LightGBM's gradient boosting vs. CatBoost) tend to make uncorrelated errors. Averaging their probabilities reduces variance.

**New file: `scripts/ensemble_models.py`:**

1. Load saved model files for any subset of Phase 1 models
2. **Soft voting**: average `win_prob` across models, evaluate top-decile precision on val
3. **Optional stacking**: fit a logistic regression meta-model on top of base model
   probability vectors — trained on a held-out fold, evaluated on val set
4. Save ensemble config to `data/ml/ensemble_config.json`:
   ```json
   {
     "models": ["data/ml/tabpfn_model_tabpfn.pkl", "data/ml/tabpfn_model_lightgbm.pkl"],
     "weights": [0.5, 0.5],
     "stacking_model": null
   }
   ```

**`EnsemblePredictor` class in `tradingagents/ml/predictor.py`:**
- Subclass of `MLPredictor` compatible with same inference API
- Loads all models from config, averages `win_prob` at inference time
- Falls back to first available model if others fail to load

**Success criteria:**
- Ensemble top-decile precision ≥ best individual Phase 1 model + 1%

---

## Execution Order

```
Phase 1 → Phase 2 → Phase 3
```

Phase 2 depends on Phase 1's infrastructure (binary labels, CLI args).
Phase 3 depends on Phase 1's trained models.
Phases 2 and 3 can be run concurrently once Phase 1 is complete.

## Files Changed / Created

| File | Action |
|------|--------|
| `scripts/train_ml_model.py` | Modify — add `--binary`, CatBoost/XGBoost, `--profit-target/stop-loss/hold-days` |
| `scripts/tune_lightgbm.py` | Create — Optuna tuning script |
| `scripts/label_ablation.py` | Create — label parameter grid search |
| `scripts/ensemble_models.py` | Create — soft voting + stacking ensemble |
| `tradingagents/ml/predictor.py` | Modify — add `EnsemblePredictor` class |
| `tradingagents/ml/feature_engineering.py` | No change in Phase 1 |

## Non-Goals

- No changes to the data fetching pipeline (`build_ml_dataset.py`)
- No new features in Phase 1 (reserved for future iteration)
- No changes to the live inference integration (filter.py etc.) until ensemble is validated
