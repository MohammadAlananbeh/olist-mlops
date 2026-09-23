from pydantic import BaseModel, Field


class OrderRequest(BaseModel):
    order_id: str
    customer_id: str
    order_status: str

    order_purchase_timestamp: str
    order_approved_at: str | None = None
    order_estimated_delivery_date: str

    number_of_items: int = Field(ge=1)
    total_freight_value: float = Field(ge=0)
    total_price: float = Field(ge=0)

    number_of_sellers: int = Field(ge=1)
    number_of_products: int = Field(ge=1)
    number_of_payments: int = Field(ge=1)
    total_payment_value: float = Field(ge=0)
    number_of_payment_types: int = Field(ge=1)
    max_payment_installments: int = Field(ge=1)

    customer_unique_id: str
    customer_zip_code_prefix: int = Field(ge=0)
    customer_city: str
    customer_state: str

    seller_state: str
    seller_zip_code_prefix: int | None = Field(default=None, ge=0)

    seller_count: int = Field(ge=1)
    customer_seller_same_state: int = Field(ge=0, le=1)
    zip_prefix_difference: float | None = None

    approval_delay_hours: float | None = Field(default=None, ge=0)


class PredictionResponse(BaseModel):
    prediction: int
    probability: float = Field(ge=0, le=1)
    model_version: str


class BatchPredictionRequest(BaseModel):
    orders: list[OrderRequest] = Field(min_length=1)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
