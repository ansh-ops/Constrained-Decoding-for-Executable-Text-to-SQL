import re
from pathlib import Path
from typing import Iterable

from peft import PeftModel
from transformers import AutoTokenizer, T5ForConditionalGeneration

from model_config import BASE_MODEL_NAME, FINETUNED_MODEL_DIR

VALID_AGG_OPS = {"COUNT", "MAX", "MIN", "SUM", "AVG"}
VALID_COND_OPS = {"=", ">", "<"}


def load_base_model_and_tokenizer(device):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True, legacy=True)
    model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL_NAME).to(device)
    return model, tokenizer


def load_finetuned_model_and_tokenizer(device):
    return load_adapter_model_and_tokenizer(FINETUNED_MODEL_DIR, device)


def load_adapter_model_and_tokenizer(adapter_dir, device):
    adapter_dir = str(adapter_dir)
    tokenizer_source = adapter_dir if Path(adapter_dir).joinpath("tokenizer_config.json").exists() else BASE_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True, legacy=True)
    base_model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)
    model = PeftModel.from_pretrained(base_model, adapter_dir).to(device)
    return model, tokenizer


def generate_sql(
    model,
    tokenizer,
    prompt,
    device,
    temperature=0.0,
    max_new_tokens=64,
    prefix_allowed_tokens_fn=None,
    logits_processor=None,
):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
    }
    if prefix_allowed_tokens_fn is not None:
        generate_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
    if logits_processor is not None:
        generate_kwargs["logits_processor"] = [logits_processor]

    if temperature > 0:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature

    outputs = model.generate(**inputs, **generate_kwargs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def syntactic_validity_rate(flags: Iterable[bool]):
    flags = list(flags)
    if not flags:
        return 0.0
    return sum(1 for flag in flags if flag) / len(flags)


def _normalize_identifier(identifier):
    return identifier.strip().strip("`").strip('"').strip()


def normalize_sql(sql_text):
    sql_text = sql_text.strip()
    sql_text = re.sub(r"\s+", " ", sql_text)
    return sql_text.strip()


def exact_match_accuracy(predictions, golds):
    if not predictions:
        return 0.0
    matches = [
        normalize_sql(prediction) == normalize_sql(gold)
        for prediction, gold in zip(predictions, golds)
    ]
    return sum(matches) / len(matches)


def parse_sql_components(sql_text):
    sql_text = normalize_sql(sql_text)
    pattern = re.compile(
        r"^SELECT\s+(?:(COUNT|MAX|MIN|SUM|AVG)\(([^)]+)\)|(.+?))\s+FROM\s+([^\s;]+)"
        r"(?:\s+WHERE\s+(.+?))?;\s*$",
        re.IGNORECASE,
    )
    match = pattern.match(sql_text)
    if not match:
        return None

    agg_op, agg_col, plain_col, table_name, where_clause = match.groups()
    select_col = agg_col if agg_op else plain_col
    components = {
        "aggregation": (agg_op or "").upper(),
        "select_column": _normalize_identifier(select_col),
        "table_name": _normalize_identifier(table_name),
        "where_column": "",
        "where_operator": "",
        "where_value": "",
    }

    if where_clause:
        cond_match = re.match(r"^(.+?)\s*(=|>|<)\s*'((?:''|[^'])*)'$", where_clause.strip())
        if cond_match:
            where_column, where_operator, where_value = cond_match.groups()
            components["where_column"] = _normalize_identifier(where_column)
            components["where_operator"] = where_operator
            components["where_value"] = where_value

    return components


def component_accuracy(predictions, golds):
    component_names = [
        "aggregation",
        "select_column",
        "table_name",
        "where_column",
        "where_operator",
        "where_value",
    ]
    scores = {name: 0 for name in component_names}
    total = len(predictions)
    if total == 0:
        return scores

    for prediction, gold in zip(predictions, golds):
        pred_components = parse_sql_components(prediction)
        gold_components = parse_sql_components(gold)
        if pred_components is None or gold_components is None:
            continue
        for name in component_names:
            if pred_components[name] == gold_components[name]:
                scores[name] += 1

    return {name: scores[name] / total for name in component_names}


def is_sql_structurally_valid(sql_text, headers, table_name):
    sql_text = sql_text.strip()
    pattern = re.compile(
        r"^SELECT\s+(?:(COUNT|MAX|MIN|SUM|AVG)\(([^)]+)\)|(.+?))\s+FROM\s+([^\s;]+)"
        r"(?:\s+WHERE\s+(.+?))?;\s*$",
        re.IGNORECASE,
    )
    match = pattern.match(sql_text)
    if not match:
        return False

    agg_op, agg_col, plain_col, predicted_table, where_clause = match.groups()
    valid_columns = {_normalize_identifier(header) for header in headers}
    predicted_table = _normalize_identifier(predicted_table)

    if predicted_table != _normalize_identifier(table_name):
        return False

    selected_column = agg_col if agg_op else plain_col
    if _normalize_identifier(selected_column) not in valid_columns:
        return False

    if agg_op and agg_op.upper() not in VALID_AGG_OPS:
        return False

    if not where_clause:
        return True

    conditions = re.split(r"\s+AND\s+", where_clause, flags=re.IGNORECASE)
    condition_pattern = re.compile(r"^(.+?)\s*(=|>|<)\s*'((?:''|[^'])*)'$")
    for condition in conditions:
        cond_match = condition_pattern.match(condition.strip())
        if not cond_match:
            return False

        column_name, operator, _ = cond_match.groups()
        if _normalize_identifier(column_name) not in valid_columns:
            return False
        if operator not in VALID_COND_OPS:
            return False

    return True
