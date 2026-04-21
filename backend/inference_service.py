from constrained_decoding import (
    build_grammar_logits_processor,
    grammar_constraints_available,
)
from data_utils import build_prompt
from evaluation_utils import generate_sql

from backend.model_loader import registry
from backend.validation import validate_sql


def _build_example(question: str, table_id: str, headers: list[str]) -> dict:
    return {
        "question": question,
        "table_id": table_id,
        "header": headers,
    }


def _resolve_mode(mode: str):
    normalized_mode = mode.strip().lower()
    if normalized_mode == "base":
        return registry.base_model, registry.base_tokenizer, False, "base"
    if normalized_mode == "fine_tuned":
        return registry.finetuned_model, registry.finetuned_tokenizer, False, "fine_tuned"
    if normalized_mode == "base_constrained":
        return registry.base_model, registry.base_tokenizer, True, "base_constrained"
    if normalized_mode == "fine_tuned_constrained":
        return (
            registry.finetuned_model,
            registry.finetuned_tokenizer,
            True,
            "fine_tuned_constrained",
        )
    raise ValueError(f"Unsupported mode: {mode}")


def generate_sql_response(
    question: str,
    table_id: str,
    headers: list[str],
    mode: str,
    temperature: float,
    max_new_tokens: int = 64,
):
    model, tokenizer, constrained, normalized_mode = _resolve_mode(mode)
    if constrained and not grammar_constraints_available():
        raise ValueError("Constrained decoding is not available in this environment.")

    example = _build_example(question, table_id, headers)
    prompt = build_prompt(example)
    logits_processor = None
    if constrained:
        logits_processor = build_grammar_logits_processor(
            tokenizer,
            headers,
            table_id,
            question=question,
        )

    sql = generate_sql(
        model,
        tokenizer,
        prompt,
        registry.device,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        logits_processor=logits_processor,
    )
    is_valid = validate_sql(sql, headers, table_id)
    return {
        "sql": sql,
        "is_valid": is_valid,
        "mode": normalized_mode,
        "temperature": temperature,
        "model_name": registry.model_name,
        "constrained": constrained,
    }
