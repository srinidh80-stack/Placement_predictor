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

CATEGORICAL_COLS = ["Gender", "City", "CollegeTier", "Stream", "Specialisation", "Hostel", "HistoryOfBacklogs", "CGPA_Tier", "ExtraCurricular"]
TEXT_COLUMNS = ["Gender", "City", "CollegeTier", "Stream", "Specialisation", "Hostel", "HistoryOfBacklogs"]
NUMERIC_COLS = [
    "CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore",
    "AttendancePercent", "SoftSkillsRating", "Internships", "Projects",
    "Workshops", "Certifications", "Publications"
]
TARGET_COLS = ["PlacementStatus", "IsAnomaly"]
TARGET_CLASSIFICATION = "PlacementStatus"
TARGET_REGRESSION = "Salary Package"
ID_COL = "StudentID"
RANDOM_STATE = 42
SECRET_KEY = "placement-dashboard-development-key"
