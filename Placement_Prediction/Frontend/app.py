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
from src.data_utils import clean_data, load_raw as load_data
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

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["UPLOAD_FOLDER"] = Path(app.root_path) / "uploads"
app.config["UPLOAD_FOLDER"].mkdir(exist_ok=True)


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


@app.route("/decision-tree", methods=["GET", "POST"])
def decision_tree_page():
    raw_df = load_data(current_dataset_path())
    cleaned_df, _ = clean_data(raw_df, save=False)
    
    req_form = request.form if request.method == "POST" else {}
    criterion = req_form.get("criterion", "entropy")
    try:
        max_depth = int(req_form.get("max_depth", 3))
    except (ValueError, TypeError):
        max_depth = 3
        
    student_inputs = {
        "CGPA": float(req_form.get("cgpa", 8.0)),
        "CodingTestScore": float(req_form.get("coding_score", 75.0)),
        "MockInterviewScore": float(req_form.get("interview_score", 70.0)),
        "AptitudeTestScore": float(req_form.get("aptitude_score", 70.0)),
        "AttendancePercent": float(req_form.get("attendance", 85.0)),
        "SoftSkillsRating": float(req_form.get("softskills", 4.0)),
        "Internships": float(req_form.get("internships", 1)),
        "Projects": float(req_form.get("projects", 2)),
    }
    
    clf_bundle = train_decision_tree_classifier(
        df=cleaned_df,
        criterion=criterion,
        max_depth=max_depth
    )
    reg_bundle = train_decision_tree_regressor(
        df=cleaned_df,
        max_depth=max_depth
    )
    diagrams = generate_decision_tree_diagrams(clf_bundle, reg_bundle)
    prediction_result = predict_placement_tree(student_inputs, clf_bundle)
    
    return render_template(
        "decision_tree.html",
        dataset_name=session.get("dataset_name", "Placement prediction dataset"),
        criterion=criterion,
        max_depth=max_depth,
        clf=clf_bundle,
        reg=reg_bundle,
        diagrams=diagrams,
        inputs=student_inputs,
        prediction=prediction_result,
    )


if __name__ == "__main__":
    app.run(debug=True, port=8080)

        