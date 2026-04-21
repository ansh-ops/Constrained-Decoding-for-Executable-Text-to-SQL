from pydantic import BaseModel, Field


class GenerateSQLRequest(BaseModel):
    question: str = Field(..., min_length=1)
    table_id: str = Field(..., min_length=1)
    headers: list[str] = Field(..., min_length=1)
    mode: str = "fine_tuned"
    temperature: float = 0.0
    max_new_tokens: int = 64


class GenerateSQLResponse(BaseModel):
    sql: str
    is_valid: bool
    mode: str
    temperature: float
    model_name: str
    constrained: bool


class ValidateSQLRequest(BaseModel):
    sql: str = Field(..., min_length=1)
    table_id: str = Field(..., min_length=1)
    headers: list[str] = Field(..., min_length=1)


class ValidateSQLResponse(BaseModel):
    is_valid: bool


class ConfigResponse(BaseModel):
    available_modes: list[str]
    default_mode: str
    default_temperature: float
    model_name: str
    constrained_available: bool


class HealthResponse(BaseModel):
    status: str
