
import time

from utils.logger import setup_logger


logger = setup_logger()


def predict(input_data, model, preprocessor, model_version):

    start_time = time.perf_counter()

    logger.info(
        "Prediction request received | input=%s | model_version=%s",
        input_data,
        model_version,
    )

    try:
        # Validate input
        validate_input(input_data)

        # Transform input
        transformed_data = preprocessor.transform(input_data)

        # Predict
        prediction = model.predict(transformed_data)

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Prediction successful | output=%s | latency_ms=%.2f | model_version=%s",
            prediction,
            latency_ms,
            model_version,
        )

        return prediction

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


# ----------------------------------------------------------------------------

import pandas as pd

from validation.data_validation import validate_or_raise



# def predict(df: pd.DataFrame, model, preprocessor):
#     """
#     Validate input data and generate predictions.

#     Data is rejected if validation fails.
#     """

#     # ---------------------------------------------------------
#     # 1. Validate incoming data
#     # ---------------------------------------------------------
#     validate_or_raise(df)

#     # ---------------------------------------------------------
#     # 2. Preprocess using the already-fitted preprocessor
#     # ---------------------------------------------------------
#     X = preprocessor.transform(df)

#     # ---------------------------------------------------------
#     # 3. Generate prediction using the already-trained model
#     # ---------------------------------------------------------
#     predictions = model.predict(X)

#     return predictions
