from __future__ import annotations

from typing import Any

import pandas as pd

# """
# Data validation for the Olist delivery prediction project.

# This module validates incoming data before it reaches the ML model.

# Great Expectations is not used because the current project environment
# uses Python 3.14, while the required Great Expectations version is not
# compatible with this Python version.

# The validation rules implemented here follow the same concepts:
# - Required columns
# - Data types
# - Missing-value rates
# - Allowed categorical values
# - Numeric ranges
# """


# ---------------------------------------------------------------------------
# Expected schema
# ---------------------------------------------------------------------------

STRING_COLUMNS = [
    "order_id",
    "customer_id",
    "order_status",
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "customer_unique_id",
    "customer_city",
    "customer_state",
    "seller_state",
]

NUMERIC_COLUMNS = [
    "number_of_items",
    "total_freight_value",
    "total_price",
    "number_of_sellers",
    "number_of_products",
    "number_of_payments",
    "total_payment_value",
    "number_of_payment_types",
    "max_payment_installments",
    "actual_delivery_days",
    "estimated_delivery_days",
    "delivery_difference_days",
    "customer_zip_code_prefix",
    "seller_zip_code_prefix",
    "seller_count",
    "customer_seller_same_state",
    "zip_prefix_difference",
]

INTEGER_COLUMNS = [
    "is_late_label",
    "purchase_year",
    "purchase_month",
    "purchase_dayofweek",
    "purchase_hour",
    "customer_zip_code_prefix",
    "seller_count",
    "customer_seller_same_state",
]


REQUIRED_COLUMNS = STRING_COLUMNS + NUMERIC_COLUMNS


# ---------------------------------------------------------------------------
# Allowed categorical values
# ---------------------------------------------------------------------------

ALLOWED_ORDER_STATUS = {
    "delivered",
    "canceled",
}

ALLOWED_BRAZILIAN_STATES = {
    "AC",
    "AL",
    "AP",
    "AM",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MT",
    "MS",
    "MG",
    "PA",
    "PB",
    "PR",
    "PE",
    "PI",
    "RJ",
    "RN",
    "RS",
    "RO",
    "RR",
    "SC",
    "SP",
    "SE",
    "TO",
}


# ---------------------------------------------------------------------------
# Missing-value expectations
# ---------------------------------------------------------------------------

# Maximum acceptable missing percentage for each column.
#
# These limits are based on the current training data quality.
MAX_MISSING_RATE = {
    "seller_zip_code_prefix": 0.05,
    "zip_prefix_difference": 0.05,
    "order_approved_at": 0.05,
    "number_of_payment_types": 0.05,
    "total_payment_value": 0.05,
    "order_delivered_carrier_date": 0.05,
    "max_payment_installments": 0.05,
}


# All columns not explicitly listed above are expected to have no missing
# values.


# ---------------------------------------------------------------------------
# Numeric expectations
# ---------------------------------------------------------------------------

# Minimum values for features where negative values do not make sense.

NON_NEGATIVE_COLUMNS = [
    "number_of_items",
    "total_freight_value",
    "total_price",
    "number_of_sellers",
    "number_of_products",
    "number_of_payments",
    "total_payment_value",
    "number_of_payment_types",
    "max_payment_installments",
    "actual_delivery_days",
    "estimated_delivery_days",
    "seller_count",
]


# Specific range expectations.

RANGE_EXPECTATIONS = {
    "is_late_label": (0, 1),
    "purchase_month": (1, 12),
    "purchase_dayofweek": (0, 6),
    "purchase_hour": (0, 23),
    "customer_seller_same_state": (0, 1),
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


# def _check_required_columns(df: pd.DataFrame) -> list[dict[str, Any]]:
def _check_required_columns(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Check that all required columns exist."""

    results = []
    columns_to_check = (
        required_columns if required_columns is not None else REQUIRED_COLUMNS
    )

    # for column in REQUIRED_COLUMNS:
    for column in columns_to_check:
        passed = column in df.columns

        results.append(
            {
                "check": "required_column",
                "column": column,
                "passed": passed,
                "details": (
                    "Column exists" if passed else "Required column is missing"
                ),
            }
        )

    return results


def _check_data_types(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Check that columns contain the expected broad data types."""

    results = []

    for column in STRING_COLUMNS:
        if column not in df.columns:
            continue

        passed = pd.api.types.is_string_dtype(
            df[column]
        ) or pd.api.types.is_object_dtype(df[column])

        results.append(
            {
                "check": "data_type",
                "column": column,
                "expected": "string",
                "actual": str(df[column].dtype),
                "passed": passed,
                "details": (
                    "Correct string type" if passed else "Expected string/object type"
                ),
            }
        )

    for column in NUMERIC_COLUMNS:
        if column not in df.columns:
            continue

        passed = pd.api.types.is_numeric_dtype(df[column])

        results.append(
            {
                "check": "data_type",
                "column": column,
                "expected": "numeric",
                "actual": str(df[column].dtype),
                "passed": passed,
                "details": (
                    "Correct numeric type" if passed else "Expected numeric type"
                ),
            }
        )

    return results


def _check_missing_rates(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Check missing-value rates against configured expectations."""

    results = []

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            continue

        actual_rate = float(df[column].isna().mean())

        max_allowed = MAX_MISSING_RATE.get(column, 0.0)

        passed = actual_rate <= max_allowed

        results.append(
            {
                "check": "missing_rate",
                "column": column,
                "actual": actual_rate,
                "expected_max": max_allowed,
                "passed": passed,
                "details": (
                    f"Missing rate {actual_rate:.4%} is within "
                    f"allowed limit {max_allowed:.4%}"
                    if passed
                    else f"Missing rate {actual_rate:.4%} exceeds "
                    f"allowed limit {max_allowed:.4%}"
                ),
            }
        )

    return results


def _check_allowed_categories(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Check categorical columns against allowed values."""

    results = []

    categorical_expectations = {
        "order_status": ALLOWED_ORDER_STATUS,
        "customer_state": ALLOWED_BRAZILIAN_STATES,
        "seller_state": ALLOWED_BRAZILIAN_STATES,
    }

    for column, allowed_values in categorical_expectations.items():
        if column not in df.columns:
            continue

        actual_values = set(df[column].dropna().unique())

        invalid_values = actual_values - allowed_values

        passed = len(invalid_values) == 0

        results.append(
            {
                "check": "allowed_categories",
                "column": column,
                "passed": passed,
                "invalid_values": sorted(invalid_values),
                "details": (
                    "All values are allowed"
                    if passed
                    else f"Invalid values found: {sorted(invalid_values)}"
                ),
            }
        )

    return results


def _check_non_negative_values(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Check that selected numeric columns contain no negative values."""

    results = []

    for column in NON_NEGATIVE_COLUMNS:
        if column not in df.columns:
            continue

        if not pd.api.types.is_numeric_dtype(df[column]):
            continue

        invalid_count = int((df[column].dropna() < 0).sum())

        passed = invalid_count == 0

        results.append(
            {
                "check": "non_negative",
                "column": column,
                "passed": passed,
                "invalid_count": invalid_count,
                "details": (
                    "No negative values found"
                    if passed
                    else f"{invalid_count} negative values found"
                ),
            }
        )

    return results


def _check_ranges(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Check columns with explicit minimum and maximum values."""

    results = []

    for column, (minimum, maximum) in RANGE_EXPECTATIONS.items():
        if column not in df.columns:
            continue

        if not pd.api.types.is_numeric_dtype(df[column]):
            continue

        invalid_mask = (df[column] < minimum) | (df[column] > maximum)

        invalid_count = int(invalid_mask.fillna(False).sum())

        passed = invalid_count == 0

        results.append(
            {
                "check": "range",
                "column": column,
                "expected": f"{minimum} to {maximum}",
                "passed": passed,
                "invalid_count": invalid_count,
                "details": (
                    "All values are within expected range"
                    if passed
                    else (f"{invalid_count} values are outside the expected range")
                ),
            }
        )

    return results


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------

# def validate_input_data(
#     df: pd.DataFrame,
# ) -> dict[str, Any]:


def validate_input_data(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
) -> dict[str, Any]:
    """
    Validate incoming data before it reaches the ML model.

    Parameters
    ----------
    df:
        Input DataFrame to validate.

    Returns
    -------
    dict
        Validation result containing:
        - success
        - total_checks
        - passed_checks
        - failed_checks
        - results
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    results = []

    # 1. Required columns
    # results.extend(_check_required_columns(df))
    results.extend(_check_required_columns(df, required_columns))

    # Only continue with other checks for columns that actually exist.
    results.extend(_check_data_types(df))
    results.extend(_check_missing_rates(df))
    results.extend(_check_allowed_categories(df))
    results.extend(_check_non_negative_values(df))
    results.extend(_check_ranges(df))

    passed_checks = sum(1 for result in results if result["passed"])

    failed_checks = sum(1 for result in results if not result["passed"])

    success = failed_checks == 0

    return {
        "success": success,
        "total_checks": len(results),
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

# def validate_or_raise(df: pd.DataFrame) -> None:


def validate_or_raise(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
) -> None:
    """
    Validate data and reject it if validation fails.

    This is the function that can be called before prediction.
    """

    # validation_result = validate_input_data(df)

    validation_result = validate_input_data(
        df,
        required_columns=required_columns,
    )

    if not validation_result["success"]:
        failed = [
            result for result in validation_result["results"] if not result["passed"]
        ]

        messages = [
            (
                f"{result['check']} failed for "
                f"{result.get('column', 'dataset')}: "
                f"{result['details']}"
            )
            for result in failed
        ]

        error_message = "Input data validation failed:\n" + "\n".join(messages)

        raise ValueError(error_message)
