
import pandas as pd


def create_date_features(df):
    """
    Create date-based features used by the ML model.

    This function does not learn from the data.
    It only calculates deterministic features.
    """

    df = df.copy()

    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # Purchase date features
    df["purchase_year"] = (
        df["order_purchase_timestamp"].dt.year
    )

    df["purchase_month"] = (
        df["order_purchase_timestamp"].dt.month
    )

    df["purchase_day"] = (
        df["order_purchase_timestamp"].dt.day
    )

    df["purchase_dayofweek"] = (
        df["order_purchase_timestamp"].dt.dayofweek
    )

    df["purchase_hour"] = (
        df["order_purchase_timestamp"].dt.hour
    )

    # Estimated delivery duration
    df["estimated_delivery_days"] = (
        df["order_estimated_delivery_date"]
        - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    # Approval delay
    df["approval_delay_hours"] = (
        df["order_approved_at"]
        - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 3600

    return df
