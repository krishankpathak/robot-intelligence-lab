# Reports Directory

This directory stores structured JSON and CSV training run logs.

- `exp_*.json`: Detailed experiment records containing timestamp, model type, data partition strategy, hyperparameters, system platform, and evaluation metrics (Accuracy, Precision, Recall, F1, PR-AUC, ROC-AUC, confusion matrix).
- Never overwrites historical records; each training run generates a unique timestamped experiment log.
