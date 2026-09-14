
import pandas as pd


DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


def convert_date_columns(df):
    """
    Convert relevant timestamp columns to pandas datetime.

    No values are learned from the dataset.
    The same operation can therefore be used for
    train, validation, test, and inference data.
    """

    df = df.copy()

    for column in DATE_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column])

    return df


def create_date_features(df):
    """
    Create the date-based features used by the ML model.

    This function performs deterministic transformations only.
    It does not fit or learn anything from the data.
    """

    df = convert_date_columns(df)

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
