import pandas as pd

from src.features.feature_engineering import (
    convert_date_columns,
    create_date_features,
)
from src.features.feature_selection import (
    select_model_features,
)


def create_sample_data():
    """Create a small dataset for testing feature engineering."""

    return pd.DataFrame(
        {
            "order_purchase_timestamp": [
                "2026-01-10 10:00:00",
                "2026-02-15 14:30:00",
            ],
            "order_approved_at": [
                "2026-01-10 12:00:00",
                "2026-02-15 15:30:00",
            ],
            "order_estimated_delivery_date": [
                "2026-01-15 10:00:00",
                "2026-02-20 14:30:00",
            ],
            "order_delivered_carrier_date": [
                "2026-01-12",
                "2026-02-17",
            ],
            "order_delivered_customer_date": [
                "2026-01-14",
                "2026-02-19",
            ],
        }
    )


def test_convert_date_columns():
    """Test that timestamp columns are converted to datetime."""

    df = create_sample_data()

    result = convert_date_columns(df)

    assert pd.api.types.is_datetime64_any_dtype(result["order_purchase_timestamp"])

    assert pd.api.types.is_datetime64_any_dtype(result["order_approved_at"])

    assert pd.api.types.is_datetime64_any_dtype(result["order_estimated_delivery_date"])


def test_create_date_features():
    """Test that expected date features are created."""

    df = create_sample_data()

    result = create_date_features(df)

    expected_columns = [
        "purchase_year",
        "purchase_month",
        "purchase_day",
        "purchase_dayofweek",
        "purchase_hour",
        "estimated_delivery_days",
        "approval_delay_hours",
    ]

    for column in expected_columns:
        assert column in result.columns


def test_create_date_features_values():
    """Test that date feature calculations are correct."""

    df = create_sample_data()

    result = create_date_features(df)

    # First order:
    # Purchase: 10 Jan 2026 at 10:00
    assert result.loc[0, "purchase_year"] == 2026
    assert result.loc[0, "purchase_month"] == 1
    assert result.loc[0, "purchase_day"] == 10
    assert result.loc[0, "purchase_hour"] == 10

    # Approval delay = 2 hours
    assert result.loc[0, "approval_delay_hours"] == 2.0

    # Estimated delivery = 5 days
    assert result.loc[0, "estimated_delivery_days"] == 5.0


def test_select_model_features_removes_target_and_leakage():
    """Test that target and leakage columns are removed."""

    df = pd.DataFrame(
        {
            "order_id": ["A", "B"],
            "total_price": [100.0, 200.0],
            "is_late_label": [0, 1],
            "order_delivered_customer_date": [
                "2026-01-14",
                "2026-02-19",
            ],
            "order_delivered_carrier_date": [
                "2026-01-12",
                "2026-02-17",
            ],
            "actual_delivery_days": [4, 4],
            "delivery_difference_days": [1, -1],
            "order_purchase_timestamp": [
                "2026-01-10",
                "2026-02-15",
            ],
            "order_approved_at": [
                "2026-01-10",
                "2026-02-15",
            ],
            "order_estimated_delivery_date": [
                "2026-01-15 10:00:00",
                "2026-02-20 14:30:00",
            ],
        }
    )

    result = select_model_features(df)

    forbidden_columns = [
        "is_late_label",
        "order_delivered_customer_date",
        "order_delivered_carrier_date",
        "actual_delivery_days",
        "delivery_difference_days",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_estimated_delivery_date",
    ]

    for column in forbidden_columns:
        assert column not in result.columns


def test_select_model_features_keeps_valid_features():
    """Test that valid model features are preserved."""

    df = pd.DataFrame(
        {
            "order_id": ["A", "B"],
            "total_price": [100.0, 200.0],
            "number_of_items": [1, 2],
            "customer_state": ["SP", "RJ"],
            "is_late_label": [0, 1],
        }
    )

    result = select_model_features(df)

    assert "order_id" in result.columns
    assert "total_price" in result.columns
    assert "number_of_items" in result.columns
    assert "customer_state" in result.columns
