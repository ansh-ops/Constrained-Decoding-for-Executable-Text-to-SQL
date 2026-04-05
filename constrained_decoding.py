def allowed_token_texts(columns, table_name):
    keywords = [
        "SELECT", "FROM", "WHERE", "AND",
        "COUNT", "MAX", "MIN", "SUM", "AVG",
        "(", ")", ",", "*", ";", "=", ">", "<", "'"
    ]
    return keywords + columns + [table_name]


def build_allowed_token_ids(tokenizer, columns, table_name):
    allowed_ids = set()

    for term in allowed_token_texts(columns, table_name):
        token_ids = tokenizer.encode(term, add_special_tokens=False)
        for tid in token_ids:
            allowed_ids.add(tid)

    # Also allow space-prefixed variants indirectly through tokenizer pieces.
    return sorted(list(allowed_ids))


def make_prefix_allowed_tokens_fn(tokenizer, columns, table_name):
    allowed_ids = build_allowed_token_ids(tokenizer, columns, table_name)

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        return allowed_ids

    return prefix_allowed_tokens_fn