import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from data_utils import load_wikisql, build_prompt, build_wikisql_sql

MODEL_NAME = "t5-small"


def generate_sql(model, tokenizer, prompt, device, max_new_tokens=64):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    dataset = load_wikisql()
    val_ds = dataset["validation"]

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

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