from evaluation_utils import is_sql_structurally_valid


def validate_sql(sql: str, headers: list[str], table_id: str) -> bool:
    return is_sql_structurally_valid(sql, headers, table_id)
