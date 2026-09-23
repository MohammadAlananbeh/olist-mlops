from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.prediction.predictor import predict

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "classification_threshold.joblib"


def create_valid_prediction_input():
    return pd.DataFrame(
        {
            "order_id": ["test_order_1"],
            "customer_id": ["test_customer_1"],
            "order_status": ["delivered"],
            "order_purchase_timestamp": ["2026-01-01 10:00:00"],
            "order_approved_at": ["2026-01-01 10:30:00"],
            "order_delivered_carrier_date": ["2026-01-03 10:00:00"],
            "order_delivered_customer_date": ["2026-01-05 10:00:00"],
            "order_estimated_delivery_date": ["2026-01-06 10:00:00"],
            "customer_unique_id": ["unique_1"],
            "customer_city": ["Sao Paulo"],
            "customer_state": ["SP"],
            "seller_state": ["SP"],
            "number_of_items": [1],
            "total_freight_value": [10.0],
            "total_price": [100.0],
            "number_of_sellers": [1],
            "number_of_products": [1],
            "number_of_payments": [1],
            "total_payment_value": [100.0],
            "number_of_payment_types": [1],
            "max_payment_installments": [1],
            "actual_delivery_days": [4.0],
            "estimated_delivery_days": [5.0],
            "delivery_difference_days": [-1.0],
            "customer_zip_code_prefix": [1000],
            "seller_zip_code_prefix": [1000],
            "seller_count": [1],
            "customer_seller_same_state": [1],
            "zip_prefix_difference": [0],
            "is_late_label": [0],
        }
    )


def test_prediction_pipeline_returns_prediction():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    classification_threshold = joblib.load(THRESHOLD_PATH)

    input_data = create_valid_prediction_input()

    prediction = predict(
        input_data,
        model,
        preprocessor,
        model_version="1.0.0",
        classification_threshold=classification_threshold,
    )

    assert prediction is not None
    # assert len(prediction) == 1
    assert "prediction" in prediction
    assert "probability" in prediction


def test_prediction_pipeline_returns_binary_prediction():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    classification_threshold = joblib.load(THRESHOLD_PATH)

    input_data = create_valid_prediction_input()

    prediction = predict(
        input_data,
        model,
        preprocessor,
        model_version="1.0.0",
        classification_threshold=classification_threshold,
    )

    # assert prediction[0] in [0, 1]
    assert prediction["prediction"] in [0, 1]


def test_prediction_rejects_invalid_input():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    classification_threshold = joblib.load(THRESHOLD_PATH)

    input_data = create_valid_prediction_input()

    # Introduce an invalid value.
    input_data.loc[0, "total_price"] = -100

    with pytest.raises(ValueError, match="Input data validation failed"):
        predict(
            input_data,
            model,
            preprocessor,
            model_version="1.0.0",
            classification_threshold=classification_threshold,
        )
