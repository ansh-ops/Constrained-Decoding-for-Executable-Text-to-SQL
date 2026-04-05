import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from constrained_decoding import make_prefix_allowed_tokens_fn
from data_utils import load_wikisql, build_prompt, build_wikisql_sql, _get_headers, _get_table_id

MODEL_PATH = "outputs/t5_wikisql_midterm/final"


def constrained_generate(model, tokenizer, prompt, columns, table_name, device):
    prefix_fn = make_prefix_allowed_tokens_fn(tokenizer, columns, table_name)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        prefix_allowed_tokens_fn=prefix_fn,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    dataset = load_wikisql()
    val_ds = dataset["validation"]

    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for i in range(5):
        ex = val_ds[i]
        prompt = build_prompt(ex)
        gold = build_wikisql_sql(ex)
        columns = _get_headers(ex)
        table_name = _get_table_id(ex)

        pred = constrained_generate(model, tokenizer, prompt, columns, table_name, device)

        print("QUESTION:", ex["question"])
        print("PRED:", pred)
        print("GOLD:", gold)
        print("-" * 100)


if __name__ == "__main__":
    main()