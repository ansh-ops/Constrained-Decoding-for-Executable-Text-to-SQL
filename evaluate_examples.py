import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from data_utils import load_wikisql, build_prompt, build_wikisql_sql, _get_headers, _get_table_id
from constrained_decoding import make_prefix_allowed_tokens_fn

BASE_MODEL = "t5-small"
FINETUNED_MODEL = "outputs/t5_wikisql_midterm/final"


def generate(model, tokenizer, prompt, device, max_new_tokens=64):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


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
    if not os.path.exists(FINETUNED_MODEL):
        raise FileNotFoundError(
            f"{FINETUNED_MODEL} not found. Run `python train.py` first."
        )

    dataset = load_wikisql()
    val_ds = dataset["validation"].select(range(10))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_tokenizer = T5Tokenizer.from_pretrained(BASE_MODEL)
    base_model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL).to(device)

    ft_tokenizer = T5Tokenizer.from_pretrained(FINETUNED_MODEL)
    ft_model = T5ForConditionalGeneration.from_pretrained(FINETUNED_MODEL).to(device)

    for ex in val_ds:
        prompt = build_prompt(ex)
        gold = build_wikisql_sql(ex)
        columns = _get_headers(ex)
        table_name = _get_table_id(ex)

        base_pred = generate(base_model, base_tokenizer, prompt, device)
        ft_pred = generate(ft_model, ft_tokenizer, prompt, device)
        constrained_pred = constrained_generate(ft_model, ft_tokenizer, prompt, columns, table_name, device)

        print("QUESTION:", ex["question"])
        print("GOLD:", gold)
        print("BASE:", base_pred)
        print("FINE-TUNED:", ft_pred)
        print("CONSTRAINED:", constrained_pred)
        print("=" * 120)


if __name__ == "__main__":
    main()