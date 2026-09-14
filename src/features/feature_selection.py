
import pandas as pd


TARGET_COLUMN = "is_late_label"


# Columns that represent information only available
# after the delivery process.
LEAKAGE_COLUMNS = [
    "order_delivered_customer_date",
    "order_delivered_carrier_date",
    "actual_delivery_days",
    "delivery_difference_days",
]


# Raw date columns are used to create useful features,
# but the original datetime columns are not sent to the model.
RAW_DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_estimated_delivery_date",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
]


def select_model_features(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """
    Select the columns that are allowed to enter the ML pipeline.

    Feature engineering should be performed BEFORE this function.
    """

    df = df.copy()

    columns_to_drop = [
        target_column,
        *LEAKAGE_COLUMNS,
        *RAW_DATE_COLUMNS,
    ]
    # below line is used to remove duplicate column names while preserving their original order:
    columns_to_drop = list(dict.fromkeys(columns_to_drop))
        # dict.fromkeys(columns_to_drop)
        # creates a dictionary using the list values as dictionary keys.
        # Because dictionary keys must be unique, duplicates disappear.
        # list(...) convert the dictionary keys back into a list

    existing_columns = [
        column for column in columns_to_drop
        if column in df.columns
    ]

    return df.drop(columns=existing_columns)
