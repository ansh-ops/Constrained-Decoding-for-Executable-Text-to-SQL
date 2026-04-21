from typing import Optional
import re

try:
    from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
    from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
except ImportError:
    IncrementalGrammarConstraint = None
    GrammarConstrainedLogitsProcessor = None


SQL_OPERATORS = ["=", ">", "<"]
AGGREGATIONS = ["COUNT", "MAX", "MIN", "SUM", "AVG"]
MAX_LITERAL_CHOICES = 8
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
}


def allowed_token_texts(columns, table_name):
    keywords = [
        "SELECT",
        "FROM",
        "WHERE",
        "AND",
        "COUNT",
        "MAX",
        "MIN",
        "SUM",
        "AVG",
        "(",
        ")",
        ",",
        "*",
        ";",
        "=",
        ">",
        "<",
        "'",
    ]
    return keywords + columns + [table_name]


def build_allowed_token_ids(tokenizer, columns, table_name):
    allowed_ids = set()

    for term in allowed_token_texts(columns, table_name):
        token_ids = tokenizer.encode(term, add_special_tokens=False)
        for token_id in token_ids:
            allowed_ids.add(token_id)

    return sorted(allowed_ids)


def make_prefix_allowed_tokens_fn(tokenizer, columns, table_name, question=""):
    allowed_ids = build_allowed_token_ids(tokenizer, columns, table_name)

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        return allowed_ids

    return prefix_allowed_tokens_fn


def _escape_grammar_literal(text):
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _normalize_whitespace(text):
    return re.sub(r"\s+", " ", text.strip())


def extract_literal_choices(question):
    candidates = []
    seen = set()

    def add_candidate(value):
        value = _normalize_whitespace(value).replace("'", "''")
        key = value.lower()
        if not value or key in seen:
            return
        seen.add(key)
        candidates.append(value)

    quoted_spans = re.findall(r"'([^']+)'|\"([^\"]+)\"", question)
    for left, right in quoted_spans:
        add_candidate(left or right)

    for numeric in re.findall(r"\d+(?:[-/]\d+)*", question):
        add_candidate(numeric)

    proper_name_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
    for match in proper_name_pattern.findall(question):
        add_candidate(match)

    cleaned = re.sub(r"[^\w\s\-/']", " ", question)
    tokens = [token for token in cleaned.split() if token]
    max_ngram = min(3, len(tokens))
    for size in range(max_ngram, 0, -1):
        for start_idx in range(len(tokens) - size + 1):
            span = " ".join(tokens[start_idx : start_idx + size])
            if span.lower() in STOPWORDS:
                continue
            add_candidate(span)

    if not candidates:
        add_candidate("value")

    return candidates[:MAX_LITERAL_CHOICES]


def build_wikisql_grammar(columns, table_name, question):
    escaped_columns = " | ".join(f'"{_escape_grammar_literal(column)}"' for column in columns)
    escaped_table = _escape_grammar_literal(table_name)
    aggregations = " | ".join(f'"{agg}"' for agg in AGGREGATIONS)
    operators = " | ".join(f'"{op}"' for op in SQL_OPERATORS)
    literal_choices = extract_literal_choices(question)
    escaped_literals = " | ".join(
        f'"{_escape_grammar_literal(literal)}"' for literal in literal_choices
    )

    grammar = f"""
root ::= select_no_where | select_with_where
select_no_where ::= "SELECT " select_expr " FROM {escaped_table};"
select_with_where ::= "SELECT " select_expr " FROM {escaped_table} WHERE " condition ";"
select_expr ::= column | agg_expr
agg_expr ::= aggregation "(" column ")"
aggregation ::= {aggregations}
condition ::= column " " operator " '" value "'"
column ::= {escaped_columns}
operator ::= {operators}
value ::= {escaped_literals}
"""
    return grammar.strip()


def build_grammar_logits_processor(tokenizer, columns, table_name, question="") -> Optional[object]:
    if IncrementalGrammarConstraint is None or GrammarConstrainedLogitsProcessor is None:
        return None

    grammar_str = build_wikisql_grammar(columns, table_name, question)
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    return GrammarConstrainedLogitsProcessor(grammar)


def grammar_constraints_available():
    return (
        IncrementalGrammarConstraint is not None
        and GrammarConstrainedLogitsProcessor is not None
    )
