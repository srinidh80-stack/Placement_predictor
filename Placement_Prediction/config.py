"""Central paths and dataset column definitions for the dashboard."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "Data" / "Raw Data" / "placement_predict_50k Dataset (2).csv"
CLEANED_DATA_PATH = BASE_DIR / "Data" / "processed Data" / "cleaned_data.csv"
STATIC_DIR = BASE_DIR / "Frontend" / "static"
PLOTS_DIR = BASE_DIR / "Output" / "plots"
REPORTS_DIR = BASE_DIR / "Output" / "Report"
EDA_REPORT_PATH = REPORTS_DIR / "EDAsummary.txt"

CATEGORICAL_COLS = ["Gender", "City", "CollegeTier", "Stream", "Specialisation", "Hostel", "HistoryOfBacklogs", "CGPA_Tier", "ExtraCurricular"]
TARGET_COLS = ["PlacementStatus", "IsAnomaly"]
ID_COL = "StudentID"
SECRET_KEY = "placement-dashboard-development-key"
