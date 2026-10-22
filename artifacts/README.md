# Artifacts Directory

This directory stores trained scikit-learn model pipelines and cryptographic metadata.

- `*.joblib`: Serialized `Pipeline` objects containing fitted preprocessing transformations and estimators.
- `*_metadata.json`: Provenance metadata capturing feature names, evaluation metrics, and SHA-256 integrity checksums.
- `model_registry.py` automatically checks SHA-256 hashes before loading artifacts to prevent corrupted or untrusted execution.
