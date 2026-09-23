import json
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"


def test_model_file_exists():
    assert MODEL_PATH.exists()


def test_preprocessor_file_exists():
    assert PREPROCESSOR_PATH.exists()


def test_model_metadata_file_exists():
    assert METADATA_PATH.exists()


def test_model_loads():
    model = joblib.load(MODEL_PATH)

    assert model is not None
    assert hasattr(model, "predict")


def test_preprocessor_loads():
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    assert preprocessor is not None
    assert hasattr(preprocessor, "transform")


def test_model_is_random_forest():
    model = joblib.load(MODEL_PATH)

    assert model.__class__.__name__ == "RandomForestClassifier"


def test_model_has_two_classes():
    model = joblib.load(MODEL_PATH)

    assert list(model.classes_) == [0, 1]


def test_model_feature_count_matches_preprocessor():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    feature_count = len(preprocessor.get_feature_names_out())

    assert model.n_features_in_ == feature_count


def test_model_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    assert metadata["model_name"] == "random_forest"
    assert metadata["model_version"] == "1.0.0"
    assert metadata["n_estimators"] == 300
    assert metadata["max_depth"] == 15
    assert metadata["random_state"] == 42


def test_model_predicts_correct_shape():
    model = joblib.load(MODEL_PATH)

    # Create input with exactly the number of features
    # expected by the trained model.
    X = np.zeros((3, model.n_features_in_))

    predictions = model.predict(X)

    assert predictions.shape == (3,)


def test_model_predictions_are_binary():
    model = joblib.load(MODEL_PATH)

    X = np.zeros((5, model.n_features_in_))

    predictions = model.predict(X)

    assert set(predictions).issubset({0, 1})
