"""
Script to Train, Evaluate, and Save Verified Models on the Kaggle Dataset
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PRIMARY_SENSOR_FEATURES, ARTIFACTS_DIR, REPORTS_DIR
from src.data.dataset_loader import load_motor_level_dataset
from src.preprocessing.cleaning import clean_telemetry_data
from src.preprocessing.data_splitting import split_telemetry_data
from src.models.model_training import train_all_models
from src.evaluation.classification_metrics import build_comparison_leaderboard
from src.models.model_registry import save_model_artifact
from src.reporting.experiment_report import log_experiment_run
from src.explainability.feature_importance import extract_tree_importance, compute_held_out_permutation_importance
from src.utils.logging_config import setup_logger

logger = setup_logger("train_verified_models")


def main():
    logger.info("Loading real Kaggle predictive-maintenance dataset...")
    df_raw = load_motor_level_dataset()
    logger.info(f"Loaded raw dataset shape: {df_raw.shape}")

    logger.info("Cleaning telemetry data and treating extreme negative glitches...")
    df_clean, clean_rep = clean_telemetry_data(df_raw, handle_glitches=True)
    logger.info(f"Cleaning report: {clean_rep}")

    logger.info("Executing Grouped Run Splitting (preventing contiguous time-series leakage)...")
    split_res = split_telemetry_data(df_clean, strategy="grouped", test_size=0.25, random_state=42)
    train_df = split_res["train_df"]
    test_df = split_res["test_df"]

    logger.info(f"Train partition: {len(train_df)} rows across runs: {split_res['train_runs']}")
    logger.info(f"Test partition: {len(test_df)} rows across runs: {split_res['test_runs']}")
    logger.info(f"Train class distribution: {split_res['train_classes']}")
    logger.info(f"Test class distribution: {split_res['test_classes']}")

    logger.info("Training and evaluating all candidate models...")
    all_results = train_all_models(
        train_df=train_df,
        test_df=test_df,
        feature_names=PRIMARY_SENSOR_FEATURES,
        target_col="label",
        scaler_type="standard",
        random_state=42,
    )

    leaderboard = build_comparison_leaderboard({k: v["metrics"] for k, v in all_results.items()})
    print("\n" + "=" * 80)
    print("MODEL EVALUATION LEADERBOARD ON HELD-OUT GROUPED RUNS")
    print("=" * 80)
    print(leaderboard.to_string(index=False))
    print("=" * 80 + "\n")

    # Select best model based on F1-Score / Recall on fault class
    best_model_name = leaderboard.iloc[0]["Model"]
    best_pipeline = all_results[best_model_name]["pipeline"]
    best_metrics = all_results[best_model_name]["metrics"]
    logger.info(f"Selected best model: {best_model_name} (F1: {best_metrics['f1']:.4f}, Recall: {best_metrics['recall']:.4f})")

    # Feature Importance
    tree_imp = extract_tree_importance(best_pipeline, PRIMARY_SENSOR_FEATURES)
    print("Feature Importances:")
    print(tree_imp.to_string(index=False))

    # Save artifact
    artifact_path = save_model_artifact(
        pipeline=best_pipeline,
        model_name=best_model_name,
        feature_names=PRIMARY_SENSOR_FEATURES,
        metrics=best_metrics,
        metadata={
            "train_runs": split_res["train_runs"],
            "test_runs": split_res["test_runs"],
            "split_strategy": "grouped",
            "cleaning_report": clean_rep,
        },
        save_dir=ARTIFACTS_DIR,
    )

    # Save all other models as well so users can switch models in the dashboard
    for name, res in all_results.items():
        if name != best_model_name:
            save_model_artifact(
                pipeline=res["pipeline"],
                model_name=name,
                feature_names=PRIMARY_SENSOR_FEATURES,
                metrics=res["metrics"],
                save_dir=ARTIFACTS_DIR,
            )

    # Log experiment
    log_file = log_experiment_run(
        model_name=best_model_name,
        feature_names=PRIMARY_SENSOR_FEATURES,
        metrics=best_metrics,
        split_strategy="grouped",
        train_shape=train_df.shape,
        test_shape=test_df.shape,
        random_seed=42,
        threshold=0.5,
        artifact_path=str(artifact_path),
        reports_dir=REPORTS_DIR,
    )

    logger.info(f"Experiment log saved to: {log_file}")
    logger.info("Model training and artifact persistence successfully completed!")


if __name__ == "__main__":
    main()
