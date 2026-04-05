# Model Training

This project trains several model families for multilabel cascade prediction. All models consume features produced by the data preparation pipeline.

## Prerequisites

Run the data preparation pipeline to generate serialized train/test splits:

```bash
cd src
python prepare_data.py --split_type chronological --filter_cascades False
```

This produces `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`, and `metadata.pkl` in a subdirectory under `data/`.

### Arguments

| Flag | Options | Description |
|------|---------|-------------|
| `--split_type` | `random`, `chronological` | Random stratified split or time-based split (train on past, test on future) |
| `--filter_cascades` | `True`, `False` | If `True`, only events that triggered at least one cascade are included |

## Models

### XGBoost Multilabel (Notebook 02)

Binary Relevance strategy — trains one XGBoost classifier per secondary event type with `scale_pos_weight` to handle class imbalance.

- Per-label threshold tuning to maximize macro F1
- SHAP feature importance analysis
- Outputs serialized models to `models/`

### Sklearn Multilabel Baselines (Notebook 03)

- **OneVsRest Logistic Regression**: Independent binary classifiers per label
- **Classifier Chains**: Exploits label correlations by chaining classifiers

### Neural Networks (Notebook 04)

Five architectures with Optuna hyperparameter tuning:

- MLP baseline
- Shared-bottom multi-task network
- Weather-embedding network
- Architectures optimized for the high class-imbalance setting

### Equity Analysis (Notebook 05)

Not a prediction model, but analyzes model outputs and cascade patterns through an equity lens using CDC SVI data.

## Evaluation Metrics

All models report:

| Metric | Description |
|--------|-------------|
| Macro F1 | Average F1 across all secondary event types |
| Per-label Precision / Recall | Breakdown by cascade type |
| ROC-AUC | Area under ROC curve (per-label and macro) |
| PR-AUC | Area under Precision-Recall curve |

## Output

Trained models and metadata are saved to `models/` (gitignored). Feature importance plots and confusion matrices are generated inline in the notebooks.
