from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_COLUMNS = [
    "order_status",
    "customer_city",
    "customer_state",
    "seller_state",
]


def get_feature_columns(X):
    """
    Identify numerical columns from the training dataset.

    Categorical columns are explicitly defined because these
    are the categorical features selected for the model.
    """

    categorical_columns = CATEGORICAL_COLUMNS.copy()

    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()

    return numeric_columns, categorical_columns


#
#
#
# Notice something important:
# build_preprocessor() does NOT call .fit().
# It only builds the object.
# So:
# preprocessor = build_preprocessor(...)
# means:
# "Create the rules."
# It does not mean:
# "Learn the rules from the data."


def build_preprocessor(numeric_columns, categorical_columns):
    """
    Build the complete preprocessing pipeline.

    Nothing is fitted here.
    """

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, categorical_columns),
        ]
    )

    return preprocessor
