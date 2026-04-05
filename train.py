from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from data_utils import load_wikisql, build_prompt, build_wikisql_sql

MODEL_NAME = "t5-small"
MAX_INPUT_LEN = 256
MAX_TARGET_LEN = 128
OUTPUT_DIR = "outputs/t5_wikisql_midterm"


def preprocess_function(example, tokenizer):
    prompt = build_prompt(example)
    target = build_wikisql_sql(example)

    model_inputs = tokenizer(
        prompt,
        max_length=MAX_INPUT_LEN,
        truncation=True,
        padding="max_length",
    )

    labels = tokenizer(
        target,
        max_length=MAX_TARGET_LEN,
        truncation=True,
        padding="max_length",
    )

    label_ids = labels["input_ids"]
    label_ids = [token_id if token_id != tokenizer.pad_token_id else -100 for token_id in label_ids]
    model_inputs["labels"] = label_ids

    return model_inputs


def main():
    dataset = load_wikisql()
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    base_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
    )
    model = get_peft_model(base_model, peft_config)

    train_ds = dataset["train"].select(range(2000))
    val_ds = dataset["validation"].select(range(300))

    train_ds = train_ds.map(lambda x: preprocess_function(x, tokenizer))
    val_ds = val_ds.map(lambda x: preprocess_function(x, tokenizer))

    keep_cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=keep_cols)
    val_ds.set_format(type="torch", columns=keep_cols)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=20,
        eval_steps=100,
        save_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    trainer.train()
    model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()