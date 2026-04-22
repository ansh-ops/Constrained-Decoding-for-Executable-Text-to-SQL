import json
from pathlib import Path

from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from data_utils import load_wikisql, build_prompt, build_wikisql_sql
from model_config import BASE_MODEL_NAME
from report_utils import make_run_dir, write_json

MAX_INPUT_LEN = 256
MAX_TARGET_LEN = 128
OUTPUT_DIR = "outputs/t5_wikisql_midterm"
TRAIN_EXAMPLES = 2000
VAL_EXAMPLES = 300


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
    run_dir = make_run_dir("training")
    dataset = load_wikisql()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True, legacy=True)
    base_model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
    )
    model = get_peft_model(base_model, peft_config)

    train_ds = dataset["train"].select(range(TRAIN_EXAMPLES))
    val_ds = dataset["validation"].select(range(VAL_EXAMPLES))

    train_ds = train_ds.map(lambda x: preprocess_function(x, tokenizer))
    val_ds = val_ds.map(lambda x: preprocess_function(x, tokenizer))

    keep_cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=keep_cols)
    val_ds.set_format(type="torch", columns=keep_cols)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=7,
        learning_rate=2e-4,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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

    log_history = trainer.state.log_history
    write_json(run_dir / "log_history.json", log_history)
    epoch_rows = [entry for entry in log_history if "epoch" in entry]
    write_json(run_dir / "epoch_metrics.json", epoch_rows)

    summary = {
        "base_model_name": BASE_MODEL_NAME,
        "output_dir": OUTPUT_DIR,
        "num_train_epochs": training_args.num_train_epochs,
        "train_examples": len(train_ds),
        "validation_examples": len(val_ds),
        "final_global_step": trainer.state.global_step,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
    }
    write_json(run_dir / "run_summary.json", summary)

    latest_checkpoint = None
    checkpoint_dirs = sorted(
        (
            path.name
            for path in Path(OUTPUT_DIR).iterdir()
            if path.is_dir() and path.name.startswith("checkpoint-")
        ),
        key=lambda name: int(name.split("-")[-1]),
    )
    if checkpoint_dirs:
        latest_checkpoint = checkpoint_dirs[-1]

    with open(run_dir / "README.txt", "w", encoding="utf-8") as handle:
        handle.write(f"Training run artifacts for {BASE_MODEL_NAME}\n")
        handle.write(f"Final model directory: {OUTPUT_DIR}/final\n")
        handle.write(f"Latest checkpoint: {latest_checkpoint}\n")
        handle.write("Files:\n")
        handle.write("- run_summary.json: run-level metadata\n")
        handle.write("- log_history.json: raw Trainer log history\n")
        handle.write("- epoch_metrics.json: per-epoch metrics for the report\n")


if __name__ == "__main__":
    main()
