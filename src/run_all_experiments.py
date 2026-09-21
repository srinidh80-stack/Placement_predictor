"""Master Experiment Runner - Placement Prediction (All 18 Reference Sessions)

Executes the complete end-to-end Machine Learning pipeline strictly on the
50,000-student placement dataset, generating all persistent splits, diagnostic
plots in Output/plots/, and text summary reports in Output/Report/.

Usage:
  python src/run_all_experiments.py
"""

from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")

import config
from src.data_utils import (clean_data, generate_and_save_splits, load_cleaned,
                            load_raw, save_baseline_report, save_cleaning_report)
from src.linear_regression import generate_regression_diagrams, train_regression_model
from src.logistic_regression import (generate_logistic_diagrams,
                                     save_coefficients_report,
                                     train_logistic_regression)
from src.decision_tree import (analyze_pruning_and_splitting,
                               generate_decision_tree_diagrams,
                               train_decision_tree_classifier)
from src.model_diagnostics import analyze_bias_variance
from src.random_forest import (generate_random_forest_diagrams,
                               save_random_forest_report,
                               train_random_forest_classifier)
from src.boosting import (generate_boosting_diagrams, save_boosting_report,
                          train_boosting_models)
from src.modern_boosting import (generate_modern_boosting_diagrams,
                                 save_modern_boosting_report,
                                 train_modern_boosting_benchmarks)


def run_all():
    print("=" * 80)
    print("  PLACEMENT PREDICTION - COMPLETE ML EXPERIMENT PIPELINE")
    print("  Dataset: 50,000 Student Records")
    print("=" * 80)
    start_total = time.perf_counter()

    # Step 1: Data Cleaning & Baseline Benchmark
    print("\n[1/7] Ingestion, IQR Cleaning & Baseline Benchmarks...")
    df_raw = load_raw()
    df_clean, dropped = clean_data(df_raw)
    rep_clean = save_cleaning_report(df_raw, df_clean, dropped)
    rep_base = save_baseline_report(df_clean)
    print(f"  -> Cleaned {len(df_clean)} rows ({dropped} duplicates removed)")
    print(f"  -> Reports: {Path(rep_clean).name}, {Path(rep_base).name}")

    # Step 2: Stratified Persistent Splits
    print("\n[2/7] Generating 70/10/20 Stratified Dataset Splits...")
    splits = generate_and_save_splits(df_clean)
    print(f"  -> Splits: Train={splits['train_len']}, Val={splits['val_len']}, Test={splits['test_len']}")
    print(f"  -> Saved CSVs to: {config.SPLITS_DIR}")
    print(f"  -> Report: {Path(splits['report_path']).name}")

    # Step 3: Linear & Logistic Models with Interpretability
    print("\n[3/7] Training Linear Models & Odds Ratio Interpretability...")
    reg_model = train_regression_model(df=df_clean)
    reg_plots = generate_regression_diagrams(model_data=reg_model)
    print(f"  -> Linear Regression R2: {reg_model['metrics']['r2']} | RMSE: {reg_model['metrics']['rmse']} LPA")
    print(f"  -> Plots: {list(reg_plots.values())}")

    log_model = train_logistic_regression(df=df_clean)
    log_plots = generate_logistic_diagrams(model_data=log_model)
    rep_coef = save_coefficients_report(model_data=log_model)
    print(f"  -> Logistic Regression Acc: {log_model['metrics']['accuracy']}% | ROC-AUC: {log_model['metrics']['auc']}")
    print(f"  -> Plots: {list(log_plots.values())}")
    print(f"  -> Report: {Path(rep_coef).name}")

    # Step 4: Decision Trees & Cost-Complexity Pruning
    print("\n[4/7] Training Decision Trees & Pruning Analysis...")
    dt_clf = train_decision_tree_classifier(df=df_clean, criterion="entropy", max_depth=3)
    dt_plots = generate_decision_tree_diagrams(dt_clf)
    dt_pruning = analyze_pruning_and_splitting(df=df_clean)
    print(f"  -> Decision Tree Acc: {dt_clf['metrics']['val_accuracy']}% (Depth={dt_clf['max_depth']})")
    print(f"  -> Pruning curve generated: {dt_pruning['pruning_plot']}")
    print(f"  -> Reports: decision_tree_report.txt, splitting_criteria_report.txt")

    # Step 5: Bias-Variance Tradeoff Analysis
    print("\n[5/7] Analyzing Bias-Variance Complexity Tradeoff...")
    bv_res = analyze_bias_variance(df=df_clean)
    print(f"  -> Optimal Tree Depth: {bv_res['optimal_depth']} (Val Acc: {bv_res['best_val_acc']}%)")
    print(f"  -> Plot: {bv_res['plot_filename']}")
    print(f"  -> Report: {Path(bv_res['report_path']).name}")

    # Step 6: Random Forest & Bagging Ensembles
    print("\n[6/7] Training Random Forest & Bagging Ensembles...")
    rf_data = train_random_forest_classifier(df=df_clean, n_estimators=100)
    rf_plots = generate_random_forest_diagrams(model_data=rf_data)
    rep_rf = save_random_forest_report(rf_data)
    print(f"  -> Random Forest Acc: {rf_data['metrics']['rf_accuracy']}% | OOB: {rf_data['metrics']['rf_oob_score']}% | AUC: {rf_data['metrics']['rf_roc_auc']}%")
    print(f"  -> Bagging Acc: {rf_data['metrics']['bagging_accuracy']}%")
    print(f"  -> Plots: {list(rf_plots.values())}")
    print(f"  -> Report: {Path(rep_rf).name}")

    # Step 7: Boosting Foundations & Modern Boosted Trees
    print("\n[7/7] Training AdaBoost, Gradient Boosting & Modern Boosters...")
    boost_data = train_boosting_models(df=df_clean, n_estimators=100)
    boost_plots = generate_boosting_diagrams(model_data=boost_data)
    rep_boost = save_boosting_report(boost_data)
    print(f"  -> AdaBoost Acc: {boost_data['metrics']['adaboost']['accuracy']}%")
    print(f"  -> Gradient Boosting Acc: {boost_data['metrics']['gradient_boosting']['accuracy']}%")
    print(f"  -> Plots: {list(boost_plots.values())}")
    print(f"  -> Report: {Path(rep_boost).name}")

    modern_data = train_modern_boosting_benchmarks(df=df_clean)
    modern_plots = generate_modern_boosting_diagrams(model_data=modern_data)
    rep_modern = save_modern_boosting_report(modern_data)
    print("  -> Modern Boosted Trees Benchmark:")
    for b in modern_data["benchmarks"]:
        print(f"     * {b['model']:<20}: {b['accuracy']}% Acc | {b['train_time_sec']}s train")
    print(f"  -> Plots: {list(modern_plots.values())}")
    print(f"  -> Report: {Path(rep_modern).name}")

    elapsed = round(time.perf_counter() - start_total, 2)
    print("\n" + "=" * 80)
    print(f"  ALL EXPERIMENTS COMPLETED SUCCESSFULLY in {elapsed}s!")
    print(f"  Diagnostic plots saved in: {config.PLOTS_DIR}")
    print(f"  Summary reports saved in:  {config.REPORTS_DIR}")
    print(f"  Persistent splits in:      {config.SPLITS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    run_all()
