import time

from src.features.feature_engineering import create_date_features
from src.monitoring.prediction_logger import log_prediction
from src.utils.logger import setup_logger
from src.validation.data_validation import validate_or_raise

logger = setup_logger()


def predict(
    input_data,
    model,
    preprocessor,
    classification_threshold,
    model_version,
):
    """
    Run the production inference pipeline.

    Flow:
        Raw input
        -> Validation
        -> Feature engineering
        -> Saved preprocessor
        -> Model probability
        -> Saved classification threshold
        -> Prediction
    """
    start_time = time.perf_counter()

    logger.info(
        "Prediction request received | input=%s | model_version=%s",
        input_data,
        model_version,
    )

    try:
        # 1. Validate raw input
        validate_or_raise(
            input_data,
            required_columns=[
                "order_id",
                "customer_id",
                "order_status",
                "order_purchase_timestamp",
                "order_approved_at",
                "order_estimated_delivery_date",
                "number_of_items",
                "total_freight_value",
                "total_price",
                "number_of_sellers",
                "number_of_products",
                "number_of_payments",
                "total_payment_value",
                "number_of_payment_types",
                "max_payment_installments",
                "customer_unique_id",
                "customer_zip_code_prefix",
                "customer_city",
                "customer_state",
                "seller_state",
                "seller_zip_code_prefix",
                "seller_count",
                "customer_seller_same_state",
                "zip_prefix_difference",
                # "approval_delay_hours",
            ],
        )

        # 2. Feature engineering
        engineered_data = create_date_features(input_data)

        # 3. Apply the saved fitted preprocessor
        transformed_data = preprocessor.transform(engineered_data)

        # 4. Generate probability from the saved model
        probability = float(model.predict_proba(transformed_data)[0][1])

        # 5. Apply the saved classification threshold
        prediction = int(probability >= classification_threshold)

        # 6. Log prediction
        log_prediction(
            prediction=prediction,
            probability=probability,
            model_version=model_version,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Prediction successful | output=%s | probability=%.4f "
            "| threshold=%.4f | latency_ms=%.2f | model_version=%s",
            prediction,
            probability,
            classification_threshold,
            latency_ms,
            model_version,
        )

        return {
            "prediction": prediction,
            "probability": probability,
        }

    except ValueError as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.warning(
            "Invalid prediction input | error=%s | latency_ms=%.2f",
            exc,
            latency_ms,
        )

        raise

    except Exception:
        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.exception(
            "Prediction failed | latency_ms=%.2f | model_version=%s",
            latency_ms,
            model_version,
        )

        raise
