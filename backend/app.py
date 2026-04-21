from fastapi import FastAPI, HTTPException

from backend.inference_service import generate_sql_response
from backend.model_loader import registry
from backend.schemas import (
    ConfigResponse,
    GenerateSQLRequest,
    GenerateSQLResponse,
    HealthResponse,
    ValidateSQLRequest,
    ValidateSQLResponse,
)
from backend.validation import validate_sql


app = FastAPI(title="Text-to-SQL Backend", version="1.0.0")


@app.on_event("startup")
def startup_event():
    registry.load()


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.get("/config", response_model=ConfigResponse)
def config():
    return ConfigResponse(
        available_modes=registry.available_modes,
        default_mode="fine_tuned",
        default_temperature=0.0,
        model_name=registry.model_name,
        constrained_available="fine_tuned_constrained" in registry.available_modes,
    )


@app.post("/generate-sql", response_model=GenerateSQLResponse)
def generate_sql(request: GenerateSQLRequest):
    try:
        payload = generate_sql_response(
            question=request.question,
            table_id=request.table_id,
            headers=request.headers,
            mode=request.mode,
            temperature=request.temperature,
            max_new_tokens=request.max_new_tokens,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return GenerateSQLResponse(**payload)


@app.post("/validate-sql", response_model=ValidateSQLResponse)
def validate(request: ValidateSQLRequest):
    return ValidateSQLResponse(
        is_valid=validate_sql(request.sql, request.headers, request.table_id)
    )
