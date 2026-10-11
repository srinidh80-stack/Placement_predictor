"""Central paths and dataset column definitions for the dashboard."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "Data" / "Raw Data" / "placement_predict_50k Dataset (2).csv"
CLEANED_DATA_PATH = BASE_DIR / "Data" / "processed Data" / "cleaned_data.csv"
SPLITS_DIR = BASE_DIR / "Data" / "processed Data" / "splits"
STATIC_DIR = BASE_DIR / "Frontend" / "static"
OUTPUT_DIR = BASE_DIR / "Output"
PLOTS_DIR = BASE_DIR / "Output" / "plots"
REPORTS_DIR = BASE_DIR / "Output" / "Report"
EDA_REPORT_PATH = REPORTS_DIR / "EDAsummary.txt"

# Ensure output directories exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_COLUMNS = ["Gender", "City", "Stream", "Specialisation"]
YES_NO_COLUMNS = ["Hostel", "HistoryOfBacklogs"]
CATEGORICAL_COLS = ["Gender", "City", "CollegeTier", "Stream", "Specialisation", "Hostel", "HistoryOfBacklogs", "CGPA_Tier", "ExtraCurricular"]
TEXT_COLUMNS = ["Gender", "City", "CollegeTier", "Stream", "Specialisation", "Hostel", "HistoryOfBacklogs"]
NUMERIC_COLS = [
    "CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore",
    "AttendancePercent", "SoftSkillsRating", "Internships", "Projects",
    "Workshops", "Certifications", "Publications"
]
NUMERIC_COLUMNS = NUMERIC_COLS
TARGET_COLS = ["PlacementStatus", "IsAnomaly"]
TARGET_CLASSIFICATION = "PlacementStatus"
TARGET_COLUMN = TARGET_CLASSIFICATION
TARGET_REGRESSION = "Salary Package"
ID_COL = "StudentID"
RANDOM_STATE = 42
SECRET_KEY = "placement-dashboard-development-key"

# Canonical output directory aliases
CLUSTER_AUGMENT_DIR = PLOTS_DIR
LEAKAGE_DIR = PLOTS_DIR
CV_DIR = PLOTS_DIR
NESTED_CV_DIR = PLOTS_DIR
THRESHOLD_DIR = PLOTS_DIR
ROC_PR_DIR = PLOTS_DIR
REGRESSION_METRICS_DIR = PLOTS_DIR
SEARCH_DIR = PLOTS_DIR
LEARNING_CURVE_DIR = PLOTS_DIR
BAYESOPT_DIR = PLOTS_DIR
CALIBRATION_DIR = PLOTS_DIR
KMEANS_DIR = PLOTS_DIR
HIERARCHICAL_DIR = PLOTS_DIR
DBSCAN_DIR = PLOTS_DIR
TSNE_DIR = PLOTS_DIR
UMAP_DIR = PLOTS_DIR
ANOMALY_DIR = PLOTS_DIR

