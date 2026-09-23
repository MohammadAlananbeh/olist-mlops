import pandas as pd

from src.preprocessing.preprocessor import (
    build_preprocessor,
    get_feature_columns,
)


def test_get_feature_columns():
    """Test that numeric and categorical columns are identified correctly."""

    df = pd.DataFrame(
        {
            "number_of_items": [1, 2, 3],
            "total_price": [100.0, 200.0, 300.0],
            "customer_city": ["Sao Paulo", "Rio", "Curitiba"],
            "customer_state": ["SP", "RJ", "PR"],
        }
    )

    numeric_columns, categorical_columns = get_feature_columns(df)

    assert "number_of_items" in numeric_columns
    assert "total_price" in numeric_columns

    assert "customer_city" in categorical_columns
    assert "customer_state" in categorical_columns


def test_build_preprocessor():
    """Test that the preprocessing pipeline is created correctly."""

    numeric_columns = [
        "number_of_items",
        "total_price",
    ]

    categorical_columns = [
        "customer_city",
        "customer_state",
    ]

    preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns,
    )

    assert preprocessor is not None

    # Verify that the expected transformers exist.
    transformer_names = [name for name, _, _ in preprocessor.transformers]

    assert "num" in transformer_names
    assert "cat" in transformer_names


def test_preprocessor_handles_missing_values():
    """Test that the preprocessing pipeline can handle missing values."""

    df = pd.DataFrame(
        {
            "number_of_items": [1, None, 3],
            "total_price": [100.0, 200.0, None],
            "customer_city": ["Sao Paulo", None, "Curitiba"],
            "customer_state": ["SP", "RJ", None],
        }
    )

    numeric_columns = [
        "number_of_items",
        "total_price",
    ]

    categorical_columns = [
        "customer_city",
        "customer_state",
    ]

    preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns,
    )

    # Fit and transform only for testing the behavior.
    transformed = preprocessor.fit_transform(df)

    assert transformed.shape[0] == 3
