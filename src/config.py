from pathlib import Path

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"

DATA_FILE = PROCESSED_DIR / "injury_analysis_preprocessed.csv"
SPLIT_DATE = "2022-01-01"

LABEL_WINDOW_DAYS = 7
MIN_GAMES_PER_PLAYER = 24