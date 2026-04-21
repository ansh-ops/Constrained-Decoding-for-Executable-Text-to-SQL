from datasets import load_dataset

AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
COND_OPS = ["=", ">", "<", "OP"]


def load_wikisql():
    ds = load_dataset("Salesforce/wikisql", trust_remote_code=True)
    return {
        "train": ds["train"],
        "validation": ds["validation"],
        "test": ds["test"],
    }


def _get_headers(example):
    if "table" in example and isinstance(example["table"], dict):
        if "header" in example["table"]:
            return example["table"]["header"]
        if "headers" in example["table"]:
            return example["table"]["headers"]

    if "header" in example:
        return example["header"]
    if "headers" in example:
        return example["headers"]

    raise KeyError(f"Could not find headers. Available keys: {list(example.keys())}")


def _get_table_id(example):
    if "table_id" in example:
        return example["table_id"]

    if "table" in example and isinstance(example["table"], dict):
        for key in ["id", "table_id", "name"]:
            if key in example["table"]:
                return example["table"][key]

    return "table_1"


def _normalize_conditions(conds):
    normalized = []

    # Case 1: dict of lists
    if isinstance(conds, dict):
        col_list = conds.get("column_index", [])
        op_list = conds.get("operator_index", [])
        val_list = conds.get("condition", [])

        for col_idx, op_idx, value in zip(col_list, op_list, val_list):
            normalized.append((col_idx, op_idx, value))
        return normalized

    # Case 2: list of dicts
    if isinstance(conds, list) and len(conds) > 0 and isinstance(conds[0], dict):
        for cond in conds:
            col_idx = cond["column_index"]
            op_idx = cond["operator_index"]
            value = cond["condition"]
            normalized.append((col_idx, op_idx, value))
        return normalized

    # Case 3: list of tuples/lists
    if isinstance(conds, list):
        for cond in conds:
            col_idx, op_idx, value = cond
            normalized.append((col_idx, op_idx, value))
        return normalized

    raise ValueError(f"Unsupported condition format: {type(conds)}")


def build_wikisql_sql(example):
    table_id = _get_table_id(example)
    sql = example["sql"]
    headers = _get_headers(example)

    select_col = headers[sql["sel"]]
    agg = AGG_OPS[sql["agg"]]

    if agg:
        select_clause = f"SELECT {agg}({select_col})"
    else:
        select_clause = f"SELECT {select_col}"

    from_clause = f"FROM {table_id}"

    where_clauses = []
    for col_idx, op_idx, value in _normalize_conditions(sql["conds"]):
        col_name = headers[col_idx]
        op = COND_OPS[op_idx]
        value = str(value).replace("'", "''")
        where_clauses.append(f"{col_name} {op} '{value}'")

    if where_clauses:
        return f"{select_clause} {from_clause} WHERE {' AND '.join(where_clauses)};"
    return f"{select_clause} {from_clause};"


def build_prompt(example):
    headers = _get_headers(example)
    schema = ", ".join(headers)
    question = example["question"]
    table_id = _get_table_id(example)
    return f"Translate question to SQL. Question: {question} Table: {table_id} Schema: {schema}"
