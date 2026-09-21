from pathlib import Path
import sys
from uuid import uuid4

from flask import (Flask, flash, redirect, render_template, request,
                   send_from_directory, session, url_for)
from werkzeug.utils import secure_filename

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data_utils import calculate_baseline_metrics, clean_data, load_raw as load_data
from src.EDA import (PLOT_FILENAMES, generate_all_plots, get_bivariate_stats, get_correlation_stats,
                     get_multivariate_stats, get_overview_stats, get_univariate_stats)
from src.feature_eng import MISSING_VALUE_CONCEPTS, build_encoded_splits, build_scaled_splits, scaling_columns
from src.linear_regression import (calculate_salary_prediction, generate_regression_diagrams,
                                   train_regression_model)
from src.logistic_regression import (generate_logistic_diagrams, predict_placement_status,
                                     train_logistic_regression)
from src.regularization import (generate_regularization_diagrams, predict_placement_regularized,
                                predict_salary_regularized, select_features_with_elasticnet,
                                select_features_with_lasso, train_regularized_classification,
                                train_regularized_regression)
from src.decision_tree import (generate_decision_tree_diagrams, predict_placement_tree,
                               train_decision_tree_classifier, train_decision_tree_regressor)
from src.random_forest import (generate_random_forest_diagrams, predict_placement_rf,
                               train_random_forest_classifier)
from src.model_diagnostics import analyze_bias_variance
from src.boosting import (generate_boosting_diagrams, predict_placement_boosting,
                          train_boosting_models)
from src.modern_boosting import (generate_modern_boosting_diagrams, predict_placement_modern,
                                 train_modern_boosting_benchmarks)
from src.kmeans_clustering import (KMEANS_FEATURES, generate_kmeans_diagrams,
                                   train_kmeans_sweep)
from src.hierarchical_clustering import (HIERARCHICAL_FEATURES, generate_hierarchical_diagrams,
                                         train_hierarchical_clustering)

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["UPLOAD_FOLDER"] = Path(app.root_path) / "uploads"
app.config["UPLOAD_FOLDER"].mkdir(exist_ok=True)

# In-memory model cache to avoid redundant 50k-row retraining on every HTTP request
_MODEL_CACHE = {}


def get_cached_clustering(df, dataset_key):
    cache_key = ("clustering", dataset_key)
    if cache_key not in _MODEL_CACHE:
        km_data = train_kmeans_sweep(df=df)
        km_diagrams = generate_kmeans_diagrams(km_data)
        hier_data = train_hierarchical_clustering(df=df)
        hier_diagrams = generate_hierarchical_diagrams(hier_data)
        _MODEL_CACHE[cache_key] = (km_data, km_diagrams, hier_data, hier_diagrams)
    return _MODEL_CACHE[cache_key]


def get_dataset_cache_key():
    return session.get("dataset_path", "default")


def get_cached_rf(df, dataset_key):
    cache_key = ("rf", dataset_key)
    if cache_key not in _MODEL_CACHE:
        rf_data = train_random_forest_classifier(df=df, n_estimators=100)
        rf_diagrams = generate_random_forest_diagrams(model_data=rf_data)
        _MODEL_CACHE[cache_key] = (rf_data, rf_diagrams)
    return _MODEL_CACHE[cache_key]


def get_cached_dt(df, dataset_key, criterion, max_depth):
    cache_key = ("dt", dataset_key, criterion, max_depth)
    if cache_key not in _MODEL_CACHE:
        dt_clf = train_decision_tree_classifier(df=df, criterion=criterion, max_depth=max_depth)
        dt_reg = train_decision_tree_regressor(df=df, max_depth=max_depth)
        dt_diagrams = generate_decision_tree_diagrams(dt_clf, dt_reg)
        _MODEL_CACHE[cache_key] = (dt_clf, dt_reg, dt_diagrams)
    return _MODEL_CACHE[cache_key]


def get_cached_bv(df, dataset_key):
    cache_key = ("bv", dataset_key)
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = analyze_bias_variance(df=df)
    return _MODEL_CACHE[cache_key]


def get_cached_boosting(df, dataset_key):
    cache_key = ("boosting", dataset_key)
    if cache_key not in _MODEL_CACHE:
        b_data = train_boosting_models(df=df, n_estimators=100)
        b_diagrams = generate_boosting_diagrams(model_data=b_data)
        _MODEL_CACHE[cache_key] = (b_data, b_diagrams)
    return _MODEL_CACHE[cache_key]


def get_cached_modern_boosting(df, dataset_key):
    cache_key = ("modern_boosting", dataset_key)
    if cache_key not in _MODEL_CACHE:
        m_data = train_modern_boosting_benchmarks(df=df)
        m_diagrams = generate_modern_boosting_diagrams(model_data=m_data)
        _MODEL_CACHE[cache_key] = (m_data, m_diagrams)
    return _MODEL_CACHE[cache_key]


def current_dataset_path():
    """Use the uploaded CSV for this browser session, otherwise use the default CSV."""
    uploaded_path = session.get("dataset_path")
    if uploaded_path and Path(uploaded_path).is_file():
        return Path(uploaded_path)
    return None


@app.route("/")
def index_page():
    return redirect(url_for("load_page"))


@app.route("/plots/<path:filename>")
def serve_plot(filename):
    """Serve dynamically generated analysis and model plots from Output/plots directory."""
    return send_from_directory(config.PLOTS_DIR, filename)


@app.route("/load", methods=["GET", "POST"])
def load_page():
    if request.method == "POST":
        dataset = request.files.get("dataset")
        if not dataset or not dataset.filename:
            flash("Choose a CSV file before uploading.", "error")
        elif not dataset.filename.lower().endswith(".csv"):
            flash("Only CSV files are supported.", "error")
        else:
            filename = f"{uuid4().hex}_{secure_filename(dataset.filename)}"
            saved_path = app.config["UPLOAD_FOLDER"] / filename
            dataset.save(saved_path)
            try:
                load_data(saved_path)
            except Exception as error:
                saved_path.unlink(missing_ok=True)
                flash(f"The CSV could not be read: {error}", "error")
            else:
                session["dataset_path"] = str(saved_path)
                session["dataset_name"] = dataset.filename
                session.pop("plots_dataset", None)
                _MODEL_CACHE.clear()
                flash("Dataset uploaded successfully.", "success")
                return redirect(url_for("load_page"))

    raw_df = load_data(current_dataset_path())
    return render_template(
        "load.html",
        dataset_name=session.get("dataset_name", "Default placement dataset"),
        shape=raw_df.shape,
        columns=raw_df.columns.tolist(),
        preview=raw_df.head(15).to_dict(orient="records"),
    )


@app.route("/eda")
@app.route("/eda/<section>")
def eda_page(section="overview"):
    raw_df = load_data(current_dataset_path())
    uploaded_path = current_dataset_path()
    if uploaded_path and session.get("plots_dataset") != str(uploaded_path):
        generate_all_plots(raw_df)
        session["plots_dataset"] = str(uploaded_path)
    elif not uploaded_path and not all((config.PLOTS_DIR / plot).exists() for plot in PLOT_FILENAMES):
        generate_all_plots(raw_df)

    plots = [url_for("serve_plot", filename=filename) for filename in PLOT_FILENAMES]
    return render_template(
        "eda.html",
        dataset_name=session.get("dataset_name", "Placement prediction dataset"),
        section=section,
        overview=get_overview_stats(raw_df),
        univariate=get_univariate_stats(raw_df),
        bivariate=get_bivariate_stats(raw_df),
        multivariate=get_multivariate_stats(raw_df),
        correlation=get_correlation_stats(raw_df),
        plots=plots,
    )


@app.route("/feature-engg", methods=["GET", "POST"])
def feature_engg_page():
    raw_df = load_data(current_dataset_path())

    missing_percent = raw_df.isna().mean().mul(100)
    missing_summary = [
        {
            "column": column,
            "missing_count": int(raw_df[column].isna().sum()),
            "missing_percent": round(float(percent), 2),
        }
        for column, percent in missing_percent[missing_percent > 0].sort_values(ascending=False).items()
    ]
    cleaned_df, duplicate_count = clean_data(raw_df, save=current_dataset_path() is None)
    feature_columns = scaling_columns(cleaned_df)

    scaler_outputs = []
    for method, label in [
        ("min_max", "Min-Max scaling"),
        ("standard", "Standard / Z-score scaling"),
        ("robust", "Robust scaling"),
    ]:
        train_scaled, test_scaled, _ = build_scaled_splits(method, cleaned_df, feature_columns)
        preview = train_scaled[feature_columns].head(8).round(3)
        scaler_outputs.append({
            "method": method,
            "label": label,
            "train_shape": train_scaled.shape,
            "test_shape": test_scaled.shape,
            "preview_columns": preview.columns,
            "preview": preview.to_dict(orient="records"),
        })

    encoder_outputs = []
    for method, label in [
        ("one_hot", "One-hot encoding"),
        ("ordinal", "Ordinal encoding"),
        ("target", "Target encoding"),
    ]:
        train_encoded, test_encoded, encoder = build_encoded_splits(method, cleaned_df)
        if method == "one_hot":
            preview_columns = encoder["encoded_columns"]
        elif method == "target":
            preview_columns = [f"{column}_target_enc" for column in encoder["columns"]]
        else:
            preview_columns = encoder["columns"]
        preview = train_encoded[preview_columns].head(8)
        encoder_outputs.append({
            "method": method,
            "label": label,
            "train_shape": train_encoded.shape,
            "test_shape": test_encoded.shape,
            "input_columns": encoder["columns"],
            "preview_columns": preview.columns,
            "preview": preview.to_dict(orient="records"),
        })

    # --- Linear Regression Model & Live Prediction ---
    linear_model_data = train_regression_model(cleaned_df)
    default_lin_cgpa = 8.5
    default_lin_coding = 75.0
    default_lin_interview = 80.0
    default_lin_aptitude = 70.0

    req_form = request.form if request.method == "POST" else {}
    active_folder = request.args.get("folder", req_form.get("active_folder", ""))

    if req_form.get("model_type") == "linear" or ("cgpa" in req_form and "attendance" not in req_form and "reg_model" not in req_form):
        try:
            lin_cgpa = float(req_form.get("cgpa", default_lin_cgpa))
            lin_coding = float(req_form.get("coding_score", default_lin_coding))
            lin_interview = float(req_form.get("interview_score", default_lin_interview))
            lin_aptitude = float(req_form.get("aptitude_score", default_lin_aptitude))
        except (ValueError, TypeError):
            lin_cgpa, lin_coding, lin_interview, lin_aptitude = (
                default_lin_cgpa, default_lin_coding, default_lin_interview, default_lin_aptitude
            )
        active_folder = "linear-output"
    else:
        lin_cgpa, lin_coding, lin_interview, lin_aptitude = (
            default_lin_cgpa, default_lin_coding, default_lin_interview, default_lin_aptitude
        )

    linear_prediction = calculate_salary_prediction(
        cgpa=lin_cgpa,
        coding_score=lin_coding,
        interview_score=lin_interview,
        aptitude_score=lin_aptitude,
        model_data=linear_model_data,
    )
    linear_diagrams = generate_regression_diagrams(
        df=cleaned_df,
        model_data=linear_model_data,
        current_prediction=linear_prediction,
    )
    linear_inputs = {
        "cgpa": lin_cgpa,
        "coding_score": lin_coding,
        "interview_score": lin_interview,
        "aptitude_score": lin_aptitude,
    }

    # --- Logistic Regression Model & Live Prediction ---
    logistic_model_data = train_logistic_regression(cleaned_df)
    default_log_cgpa = 8.0
    default_log_coding = 70.0
    default_log_interview = 75.0
    default_log_aptitude = 70.0
    default_log_attendance = 85.0
    default_log_softskills = 4.0

    if req_form.get("model_type") == "logistic" or ("attendance" in req_form and "reg_model" not in req_form):
        try:
            log_cgpa = float(req_form.get("cgpa", default_log_cgpa))
            log_coding = float(req_form.get("coding_score", default_log_coding))
            log_interview = float(req_form.get("interview_score", default_log_interview))
            log_aptitude = float(req_form.get("aptitude_score", default_log_aptitude))
            log_attendance = float(req_form.get("attendance", default_log_attendance))
            log_softskills = float(req_form.get("softskills", default_log_softskills))
        except (ValueError, TypeError):
            log_cgpa, log_coding, log_interview, log_aptitude, log_attendance, log_softskills = (
                default_log_cgpa, default_log_coding, default_log_interview, default_log_aptitude, default_log_attendance, default_log_softskills
            )
        active_folder = "logistic-output"
    else:
        log_cgpa, log_coding, log_interview, log_aptitude, log_attendance, log_softskills = (
            default_log_cgpa, default_log_coding, default_log_interview, default_log_aptitude, default_log_attendance, default_log_softskills
        )

    logistic_inputs = {
        "CGPA": log_cgpa,
        "CodingTestScore": log_coding,
        "MockInterviewScore": log_interview,
        "AptitudeTestScore": log_aptitude,
        "AttendancePercent": log_attendance,
        "SoftSkillsRating": log_softskills,
    }
    logistic_prediction = predict_placement_status(logistic_inputs, model_data=logistic_model_data)
    logistic_diagrams = generate_logistic_diagrams(
        df=cleaned_df,
        model_data=logistic_model_data,
        current_prediction=logistic_prediction,
    )

    # --- Session 19 Regularization Models & Overfitting Diagnostics ---
    is_reg_submit = req_form.get("model_type") == "regularization" or "reg_model" in req_form
    reg_context = get_regularization_context(cleaned_df, req_form if is_reg_submit else None)
    if is_reg_submit:
        active_folder = "regularization-output"

    return render_template(
        "feature_engg.html",
        dataset_name=session.get("dataset_name", "Placement prediction dataset"),
        cleaned_shape=cleaned_df.shape,
        duplicate_count=duplicate_count,
        missing_value_concepts=MISSING_VALUE_CONCEPTS,
        missing_summary=missing_summary,
        feature_columns=feature_columns,
        scaler_outputs=scaler_outputs,
        encoder_outputs=encoder_outputs,
        linear_model_data=linear_model_data,
        linear_prediction=linear_prediction,
        linear_diagrams=linear_diagrams,
        linear_inputs=linear_inputs,
        logistic_model_data=logistic_model_data,
        logistic_prediction=logistic_prediction,
        logistic_diagrams=logistic_diagrams,
        logistic_inputs=logistic_inputs,
        active_folder=active_folder,
        raw_preview=raw_df.head(10).to_dict(orient="records"),
        raw_columns=raw_df.columns.tolist(),
        **reg_context,
    )


def get_regularization_context(cleaned_df, req_form=None):
    """Train regularized models, compute metrics/diagrams, and parse live inference."""
    req_form = req_form or {}
    reg_model_data = train_regularized_regression(cleaned_df)
    clf_model_data = train_regularized_classification(cleaned_df)
    reg_diagrams = generate_regularization_diagrams(cleaned_df, reg_model_data, clf_model_data)
    lasso_feature_info = select_features_with_lasso(cleaned_df, target_col="PlacementStatus")
    elastic_feature_info = select_features_with_elasticnet(cleaned_df, target_col="PlacementStatus")

    defaults = {
        "reg_model": "ridge",
        "cgpa": 8.5,
        "coding_score": 80.0,
        "interview_score": 75.0,
        "aptitude_score": 70.0,
        "softskills": 4.0,
        "attendance": 85.0,
        "internships": 2,
        "projects": 3,
    }

    try:
        chosen_reg_model = req_form.get("reg_model", defaults["reg_model"])
        reg_inputs = {
            "CGPA": float(req_form.get("cgpa", defaults["cgpa"])),
            "CodingTestScore": float(req_form.get("coding_score", defaults["coding_score"])),
            "MockInterviewScore": float(req_form.get("interview_score", defaults["interview_score"])),
            "AptitudeTestScore": float(req_form.get("aptitude_score", defaults["aptitude_score"])),
            "SoftSkillsRating": float(req_form.get("softskills", defaults["softskills"])),
            "AttendancePercent": float(req_form.get("attendance", defaults["attendance"])),
            "Internships": float(req_form.get("internships", defaults["internships"])),
            "Projects": float(req_form.get("projects", defaults["projects"])),
        }
    except (ValueError, TypeError):
        chosen_reg_model = defaults["reg_model"]
        reg_inputs = {
            "CGPA": defaults["cgpa"],
            "CodingTestScore": defaults["coding_score"],
            "MockInterviewScore": defaults["interview_score"],
            "AptitudeTestScore": defaults["aptitude_score"],
            "SoftSkillsRating": defaults["softskills"],
            "AttendancePercent": defaults["attendance"],
            "Internships": defaults["internships"],
            "Projects": defaults["projects"],
        }

    return {
        "reg_model_data": reg_model_data,
        "clf_model_data": clf_model_data,
        "reg_diagrams": reg_diagrams,
        "lasso_feature_info": lasso_feature_info,
        "elastic_feature_info": elastic_feature_info,
        "chosen_reg_model": chosen_reg_model,
        "reg_inputs": reg_inputs,
        "regularized_salary_pred": predict_salary_regularized(reg_inputs, model_type=chosen_reg_model, reg_bundle=reg_model_data),
        "regularized_placement_pred": predict_placement_regularized(reg_inputs, model_type=chosen_reg_model, clf_bundle=clf_model_data),
    }


@app.route("/linear-regression", methods=["GET", "POST"])
def linear_regression_page():
    return redirect(url_for("feature_engg_page", folder="linear-output"))


@app.route("/logistic-regression", methods=["GET", "POST"])
def logistic_regression_page():
    return redirect(url_for("feature_engg_page", folder="logistic-output"))


@app.route("/decision_tree", methods=["GET", "POST"])
@app.route("/decision-tree", methods=["GET", "POST"])
def decision_tree_page():
    return redirect(url_for("tree_models_page"))


@app.route("/tree_models", methods=["GET", "POST"])
@app.route("/tree-models", methods=["GET", "POST"])
def tree_models_page():
    raw_df = load_data(current_dataset_path())
    cleaned_df, _ = clean_data(raw_df, save=False)
    dataset_key = get_dataset_cache_key()
    
    req_form = request.form if request.method == "POST" else {}
    action = req_form.get("action", "")
    
    criterion = req_form.get("criterion", "entropy")
    try:
        max_depth = int(req_form.get("max_depth", 3))
    except (ValueError, TypeError):
        max_depth = 3
        
    selected_predictor = req_form.get("predictor_model", "random_forest")
    
    candidate_inputs = {
        "CGPA": float(req_form.get("CGPA", 7.8)),
        "CodingTestScore": float(req_form.get("CodingTestScore", 75.0)),
        "MockInterviewScore": float(req_form.get("MockInterviewScore", 70.0)),
        "AptitudeTestScore": float(req_form.get("AptitudeTestScore", 72.0)),
        "AttendancePercent": float(req_form.get("AttendancePercent", 85.0)),
        "SoftSkillsRating": float(req_form.get("SoftSkillsRating", 4.0)),
        "Internships": float(req_form.get("Internships", 2)),
        "Projects": float(req_form.get("Projects", 3)),
    }
    
    # 1. Decision Tree models and plots (cached by params)
    dt_clf, dt_reg, dt_diagrams = get_cached_dt(cleaned_df, dataset_key, criterion, max_depth)
    
    # 2. Random Forest and Bagging models and plots (cached)
    rf_data, rf_diagrams = get_cached_rf(cleaned_df, dataset_key)
    
    # 3. Bias-Variance Analysis (cached)
    bv_data = get_cached_bv(cleaned_df, dataset_key)
    
    # 4. Live Prediction
    tree_prediction = None
    active_folder = request.args.get("folder", "tree-output")
    
    if action == "tune_tree":
        active_folder = "tree-output"
    elif action == "predict_candidate":
        active_folder = "tree-predictor-output"
        if selected_predictor == "random_forest":
            tree_prediction = predict_placement_rf(rf_data, candidate_inputs)
        else:
            tree_prediction = predict_placement_tree(candidate_inputs, dt_clf)
            
    return render_template(
        "tree_models.html",
        dataset_name=session.get("dataset_name", "Placement prediction dataset"),
        criterion=criterion,
        max_depth=max_depth,
        dt_data=dt_clf,
        dt_reg=dt_reg,
        dt_diagrams=dt_diagrams,
        rf_data=rf_data,
        rf_diagrams=rf_diagrams,
        bv_data=bv_data,
        candidate_inputs=candidate_inputs,
        selected_predictor=selected_predictor,
        tree_prediction=tree_prediction,
        active_folder=active_folder,
    )


@app.route("/boosting_models", methods=["GET", "POST"])
@app.route("/boosting-models", methods=["GET", "POST"])
def boosting_models_page():
    raw_df = load_data(current_dataset_path())
    cleaned_df, _ = clean_data(raw_df, save=False)
    dataset_key = get_dataset_cache_key()
    
    req_form = request.form if request.method == "POST" else {}
    action = req_form.get("action", "")
    
    selected_predictor = req_form.get("predictor_model", "gradient_boosting")
    candidate_inputs = {
        "CGPA": float(req_form.get("CGPA", 7.8)),
        "CodingTestScore": float(req_form.get("CodingTestScore", 75.0)),
        "MockInterviewScore": float(req_form.get("MockInterviewScore", 70.0)),
        "AptitudeTestScore": float(req_form.get("AptitudeTestScore", 72.0)),
        "AttendancePercent": float(req_form.get("AttendancePercent", 85.0)),
        "SoftSkillsRating": float(req_form.get("SoftSkillsRating", 4.0)),
        "Internships": float(req_form.get("Internships", 2)),
        "Projects": float(req_form.get("Projects", 3)),
    }
    
    # 1. Boosting models (AdaBoost & Gradient Boosting) (cached)
    boosting_data, boosting_diagrams = get_cached_boosting(cleaned_df, dataset_key)
    
    # 2. Modern Boosted Trees (XGBoost, LightGBM, CatBoost) (cached)
    modern_data, modern_diagrams = get_cached_modern_boosting(cleaned_df, dataset_key)
    
    # 3. Live Prediction
    boosting_prediction = None
    active_folder = request.args.get("folder", "boosting-fundamentals-output")
    
    if action == "predict_candidate":
        active_folder = "boosting-predictor-output"
        if selected_predictor in ["adaboost", "gradient_boosting"]:
            boosting_prediction = predict_placement_boosting(boosting_data, candidate_inputs, model_type=selected_predictor)
        else:
            boosting_prediction = predict_placement_modern(modern_data, candidate_inputs, chosen_model=selected_predictor)
            
    return render_template(
        "boosting_models.html",
        dataset_name=session.get("dataset_name", "Placement prediction dataset"),
        boosting_data=boosting_data,
        boosting_diagrams=boosting_diagrams,
        modern_data=modern_data,
        modern_diagrams=modern_diagrams,
        candidate_inputs=candidate_inputs,
        selected_predictor=selected_predictor,
        boosting_prediction=boosting_prediction,
        active_folder=active_folder,
    )


@app.route("/unsupervised", methods=["GET", "POST"])
def unsupervised_learning_page():
    raw_df = load_data(current_dataset_path())
    cleaned_df, duplicate_count = clean_data(raw_df, save=False)
    dataset_key = get_dataset_cache_key()
    
    km_data, km_diagrams, hier_data, hier_diagrams = get_cached_clustering(cleaned_df, dataset_key)
    active_folder = request.args.get("folder", "kmeans-output")
    
    # Read text reports if available
    km_report_lines = None
    km_rep_file = config.REPORTS_DIR / "kmeans_report.txt"
    if km_rep_file.exists():
        km_report_lines = km_rep_file.read_text(encoding="utf-8").splitlines()
        
    hier_report_lines = None
    hier_rep_file = config.REPORTS_DIR / "hierarchical_report.txt"
    if hier_rep_file.exists():
        hier_report_lines = hier_rep_file.read_text(encoding="utf-8").splitlines()
        
    return render_template(
        "clustering.html",
        dataset_name=session.get("dataset_name", "Placement prediction dataset"),
        cleaned_shape=cleaned_df.shape,
        duplicate_count=duplicate_count,
        feature_columns=km_data["features"],
        km_data=km_data,
        km_diagrams=km_diagrams,
        hier_data=hier_data,
        hier_diagrams=hier_diagrams,
        active_folder=active_folder,
        km_report_lines=km_report_lines,
        hier_report_lines=hier_report_lines,
    )


if __name__ == "__main__":
    app.run(debug=True, port=8080)


        