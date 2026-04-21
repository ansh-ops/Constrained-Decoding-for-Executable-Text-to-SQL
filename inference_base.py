import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from data_utils import load_wikisql, build_prompt, build_wikisql_sql
from model_config import BASE_MODEL_NAME


def generate_sql(model, tokenizer, prompt, device, max_new_tokens=64):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    dataset = load_wikisql()
    val_ds = dataset["validation"]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True, legacy=True)
    model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for i in range(5):
        ex = val_ds[i]
        prompt = build_prompt(ex)
        gold = build_wikisql_sql(ex)
        pred = generate_sql(model, tokenizer, prompt, device)

        print("QUESTION:", ex["question"])
        print("PROMPT:", prompt)
        print("PRED:", pred)
        print("GOLD:", gold)
        print("-" * 100)


if __name__ == "__main__":
    main()
