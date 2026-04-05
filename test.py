from data_utils import load_wikisql, build_prompt, build_wikisql_sql

ds = load_wikisql()
ex = ds["train"][0]

print(ex.keys())
print(ex["sql"])
print(ex["sql"]["conds"])
print(build_prompt(ex))
print(build_wikisql_sql(ex))