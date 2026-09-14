
import pandas as pd


def validate_target(                             # Validate the target variable of a dataset. (is_late_label)
    df: pd.DataFrame,                            # The : pd.DataFrame is a type hint. It tells us: "I expect df to be a Pandas DataFrame."
    target_column: str,
    dataset_name: str = "dataset",              # "dataset" is the defaukt value if not provided
) -> None:                                      # -> None: This is another type hint. It says: This function is not expected to return a value.
    """
    Validate that the target column exists and contains valid binary values.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"{dataset_name}: target column '{target_column}' is missing."
        )

    if df[target_column].isna().any():
        raise ValueError(
            f"{dataset_name}: target column '{target_column}' contains missing values."
        )

    unique_values = set(df[target_column].unique())

    if not unique_values.issubset({0, 1}):
        raise ValueError(
            f"{dataset_name}: target column '{target_column}' "
            f"must contain only 0 and 1. Found: {unique_values}"
        )


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str = "dataset",
) -> None:
    """
    Check that all required columns exist.
    """

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name}: missing required columns: {missing_columns}"
        )


def validate_no_leakage_columns(
    df: pd.DataFrame,
    leakage_columns: list[str],
    dataset_name: str = "dataset",
) -> None:
    """
    Make sure known post-outcome/leakage columns are not
    present in the model input.
    """

    found_columns = [
        column for column in leakage_columns
        if column in df.columns
    ]

    if found_columns:
        raise ValueError(
            f"{dataset_name}: leakage columns are still present: "
            f"{found_columns}"
        )


def validate_feature_alignment(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    """
    Check that train, validation and test contain the same features.
    """

    train_columns = set(X_train.columns)
    val_columns = set(X_val.columns)
    test_columns = set(X_test.columns)

    if train_columns != val_columns:
        raise ValueError(
            "Training and validation feature columns do not match."
        )

    if train_columns != test_columns:
        raise ValueError(
            "Training and testing feature columns do not match."
        )
# ----------------------------------------------------------------------

def validate_input(df):

    if df is None:
        raise ValueError("Input data cannot be None")

    if df.empty:
        raise ValueError("Input data cannot be empty")

    required_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_estimated_delivery_date",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return True
