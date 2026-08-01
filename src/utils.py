from pathlib import Path
import pandas as pd
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "smart_maintenance_dataset.csv"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" 
DATA_PROCESSED_PATH = DATA_PROCESSED_DIR / "processed_dataset.csv"

DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Pipeline Hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5

# Target Variables & Drops
TARGET_REGRESSION = "Failure_Prob"
TARGET_CLASSIFICATION = "Maintenance_Priority"
COLS_TO_DROP = ["Machine_ID"]


def load_raw_data() -> pd.DataFrame:
    """
    Loads raw dataset from the configured path.
    """
    if not DATA_RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data file not found at: {DATA_RAW_PATH}")
    return pd.read_csv(DATA_RAW_PATH)


def load_processed_data() -> pd.DataFrame:
    """
    Loads processed dataset from the configured path.
    """
    if not DATA_PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Raw data file not found at: {DATA_PROCESSED_PATH}")
    return pd.read_csv(DATA_PROCESSED_PATH)


def get_train_test_splits() -> Dict[str, Tuple]:
    """
    Main pipeline function that loads data, applies engineering, and returns stratified/random train-test splits for both tasks.
    
    Returns:
        dict: Containing 'regression' and 'classification' train/test tuples.
    """
    df = load_processed_data()
    
    # Drop identifier columns and target columns from feature set X
    drop_cols = COLS_TO_DROP + [TARGET_REGRESSION, TARGET_CLASSIFICATION]
    X = df.drop(columns=drop_cols)
    # Extract target series
    y_reg = df[TARGET_REGRESSION]
    y_clf = df[TARGET_CLASSIFICATION]
    
    # Split for Regression
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Split for Classification
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_clf, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_clf
    )
    
    return {
        "regression": (X_train_r, X_test_r, y_train_r, y_test_r),
        "classification": (X_train_c, X_test_c, y_train_c, y_test_c)
    }


FAILURE_PROB_MODEL_FEATURES = ["Temp_C", "Vibration_mm_s", "Acoustic_dB"]

def get_failure_prob_predictions() -> pd.Series:
    """
    Returns out-of-fold predicted Failure_Prob for every row in the processed dataset,
    using the winning regression configuration from 02_regression_model.ipynb.

    Every row's prediction comes from a fold that did not include that row during
    training, so the result is safe to use as an input feature without leakage.

    Returns:
        pd.Series: predicted Failure_Prob, indexed the same as load_processed_data().
    """
    df = load_processed_data()
    X = df[FAILURE_PROB_MODEL_FEATURES]
    y = df[TARGET_REGRESSION]
    groups = df["Machine_ID"]

    pipeline = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    gkf = GroupKFold(n_splits=N_SPLITS)
    predictions = cross_val_predict(pipeline, X, y, groups=groups, cv=gkf)

    return pd.Series(predictions, index=df.index, name="Failure_Prob_pred")