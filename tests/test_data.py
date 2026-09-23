import pandas as pd
import pytest

from src.validation.data_validation import (
    validate_input_data,
    validate_or_raise,
)
from src.validation.validator import (
    validate_feature_alignment,
    validate_input,
    validate_no_leakage_columns,
    validate_required_columns,
    validate_target,
)

# ---------------------------------------------------------------------------
# Helper test data
# ---------------------------------------------------------------------------


def create_valid_data():
    return pd.DataFrame(
        {
            "order_id": ["order_1", "order_2"],
            "customer_id": ["customer_1", "customer_2"],
            "order_status": ["delivered", "canceled"],
            "order_purchase_timestamp": [
                "2026-01-01 10:00:00",
                "2026-01-02 11:00:00",
            ],
            "order_approved_at": [
                "2026-01-01 10:30:00",
                "2026-01-02 11:30:00",
            ],
            "order_delivered_carrier_date": [
                "2026-01-03 10:00:00",
                "2026-01-04 10:00:00",
            ],
            "order_delivered_customer_date": [
                "2026-01-05 10:00:00",
                "2026-01-06 10:00:00",
            ],
            "order_estimated_delivery_date": [
                "2026-01-06 10:00:00",
                "2026-01-07 10:00:00",
            ],
            "customer_unique_id": ["unique_1", "unique_2"],
            "customer_city": ["Sao Paulo", "Rio"],
            "customer_state": ["SP", "RJ"],
            "seller_state": ["SP", "RJ"],
            "number_of_items": [1, 2],
            "total_freight_value": [10.0, 20.0],
            "total_price": [100.0, 200.0],
            "number_of_sellers": [1, 2],
            "number_of_products": [1, 2],
            "number_of_payments": [1, 2],
            "total_payment_value": [100.0, 200.0],
            "number_of_payment_types": [1, 1],
            "max_payment_installments": [1, 2],
            "actual_delivery_days": [4.0, 5.0],
            "estimated_delivery_days": [5.0, 5.0],
            "delivery_difference_days": [-1.0, 0.0],
            "customer_zip_code_prefix": [1000, 2000],
            "seller_zip_code_prefix": [1000, 2000],
            "seller_count": [1, 2],
            "customer_seller_same_state": [1, 0],
            "zip_prefix_difference": [0, 1000],
            "is_late_label": [0, 1],
        }
    )


# ---------------------------------------------------------------------------
# validate_input_data tests
# ---------------------------------------------------------------------------


def test_validate_input_data_accepts_valid_data():
    df = create_valid_data()

    result = validate_input_data(df)

    assert result["success"] is True
    assert result["failed_checks"] == 0
    assert result["passed_checks"] == result["total_checks"]


def test_validate_input_data_detects_invalid_order_status():
    df = create_valid_data()
    df.loc[0, "order_status"] = "invalid_status"

    result = validate_input_data(df)

    assert result["success"] is False
    assert result["failed_checks"] > 0

    failed_results = [
        result_item for result_item in result["results"] if not result_item["passed"]
    ]

    assert any(
        item["check"] == "allowed_categories" and item["column"] == "order_status"
        for item in failed_results
    )


def test_validate_input_data_detects_negative_values():
    df = create_valid_data()
    df.loc[0, "total_price"] = -100

    result = validate_input_data(df)

    assert result["success"] is False

    assert any(
        item["check"] == "non_negative" and item["column"] == "total_price"
        for item in result["results"]
        if not item["passed"]
    )


def test_validate_input_data_detects_invalid_range():
    df = create_valid_data()
    df.loc[0, "purchase_month"] = 13

    # purchase_month is not part of REQUIRED_COLUMNS in the current
    # data_validation.py, so add it explicitly for this range test.
    result = validate_input_data(df)

    assert result["success"] is False

    assert any(
        item["check"] == "range" and item["column"] == "purchase_month"
        for item in result["results"]
        if not item["passed"]
    )


def test_validate_input_data_detects_missing_required_column():
    df = create_valid_data()
    df = df.drop(columns=["order_id"])

    result = validate_input_data(df)

    assert result["success"] is False

    assert any(
        item["check"] == "required_column"
        and item["column"] == "order_id"
        and item["passed"] is False
        for item in result["results"]
    )


def test_validate_or_raise_accepts_valid_data():
    df = create_valid_data()

    validate_or_raise(df)


def test_validate_or_raise_rejects_invalid_data():
    df = create_valid_data()
    df.loc[0, "total_price"] = -10

    with pytest.raises(ValueError, match="Input data validation failed"):
        validate_or_raise(df)


# ---------------------------------------------------------------------------
# validator.py tests
# ---------------------------------------------------------------------------


def test_validate_target_accepts_binary_target():
    df = pd.DataFrame({"is_late_label": [0, 1, 0, 1]})

    validate_target(df, "is_late_label")


def test_validate_target_rejects_missing_target():
    df = pd.DataFrame({"is_late_label": [0, 1, None]})

    with pytest.raises(ValueError, match="contains missing values"):
        validate_target(df, "is_late_label")


def test_validate_target_rejects_non_binary_values():
    df = pd.DataFrame({"is_late_label": [0, 1, 2]})

    with pytest.raises(ValueError, match="must contain only 0 and 1"):
        validate_target(df, "is_late_label")


def test_validate_target_rejects_missing_target_column():
    df = pd.DataFrame({"some_column": [1, 2, 3]})

    with pytest.raises(ValueError, match="target column"):
        validate_target(df, "is_late_label")


def test_validate_required_columns_accepts_valid_data():
    df = pd.DataFrame(
        {
            "order_id": ["1", "2"],
            "customer_id": ["c1", "c2"],
        }
    )

    validate_required_columns(
        df,
        ["order_id", "customer_id"],
    )


def test_validate_required_columns_rejects_missing_columns():
    df = pd.DataFrame({"order_id": ["1", "2"]})

    with pytest.raises(ValueError, match="missing required columns"):
        validate_required_columns(
            df,
            ["order_id", "customer_id"],
        )


def test_validate_no_leakage_columns_accepts_clean_data():
    df = pd.DataFrame(
        {
            "total_price": [100, 200],
            "number_of_items": [1, 2],
        }
    )

    validate_no_leakage_columns(
        df,
        [
            "order_delivered_customer_date",
            "actual_delivery_days",
        ],
    )


def test_validate_no_leakage_columns_rejects_leakage():
    df = pd.DataFrame(
        {
            "total_price": [100, 200],
            "actual_delivery_days": [4, 5],
        }
    )

    with pytest.raises(ValueError, match="leakage columns"):
        validate_no_leakage_columns(
            df,
            ["actual_delivery_days"],
        )


def test_validate_feature_alignment_accepts_matching_features():
    X_train = pd.DataFrame(
        {
            "feature_a": [1, 2],
            "feature_b": [3, 4],
        }
    )

    X_val = pd.DataFrame(
        {
            "feature_a": [5, 6],
            "feature_b": [7, 8],
        }
    )

    X_test = pd.DataFrame(
        {
            "feature_a": [9, 10],
            "feature_b": [11, 12],
        }
    )

    validate_feature_alignment(
        X_train,
        X_val,
        X_test,
    )


def test_validate_feature_alignment_rejects_mismatch():
    X_train = pd.DataFrame(
        {
            "feature_a": [1, 2],
            "feature_b": [3, 4],
        }
    )

    X_val = pd.DataFrame(
        {
            "feature_a": [5, 6],
            "feature_c": [7, 8],
        }
    )

    X_test = pd.DataFrame(
        {
            "feature_a": [9, 10],
            "feature_b": [11, 12],
        }
    )

    with pytest.raises(
        ValueError,
        match="Training and validation feature columns do not match",
    ):
        validate_feature_alignment(
            X_train,
            X_val,
            X_test,
        )


def test_validate_input_accepts_valid_input():
    df = pd.DataFrame(
        {
            "order_purchase_timestamp": ["2026-01-01"],
            "order_approved_at": ["2026-01-01"],
            "order_estimated_delivery_date": ["2026-01-05"],
        }
    )

    assert validate_input(df) is True


def test_validate_input_rejects_none():
    with pytest.raises(
        ValueError,
        match="Input data cannot be None",
    ):
        validate_input(None)


def test_validate_input_rejects_empty_dataframe():
    df = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="Input data cannot be empty",
    ):
        validate_input(df)


def test_validate_input_rejects_missing_required_columns():
    df = pd.DataFrame(
        {
            "order_purchase_timestamp": ["2026-01-01"],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        validate_input(df)
