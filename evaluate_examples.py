import os
from pathlib import Path

import torch
from data_utils import load_wikisql, build_prompt, build_wikisql_sql, _get_headers, _get_table_id
from constrained_decoding import (
    build_grammar_logits_processor,
    grammar_constraints_available,
)
from evaluation_utils import (
    component_accuracy,
    exact_match_accuracy,
    generate_sql,
    is_sql_structurally_valid,
    load_adapter_model_and_tokenizer,
    load_base_model_and_tokenizer,
    load_finetuned_model_and_tokenizer,
    syntactic_validity_rate,
)
from model_config import FINETUNED_MODEL_DIR
from report_utils import make_run_dir, write_csv, write_json

TEMPERATURES = [0.0, 0.3, 0.7, 1.0]
EVAL_EXAMPLES = 100
SAMPLE_PREVIEW_COUNT = 3


def main():
    if not os.path.exists(FINETUNED_MODEL_DIR):
        raise FileNotFoundError(
            f"{FINETUNED_MODEL_DIR} not found. Run `python train.py` first."
        )

    run_dir = make_run_dir("evaluation")
    dataset = load_wikisql()
    test_ds = dataset["test"].select(range(EVAL_EXAMPLES))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model, base_tokenizer = load_base_model_and_tokenizer(device)
    ft_model, ft_tokenizer = load_finetuned_model_and_tokenizer(device)
    grammar_available = grammar_constraints_available()

    checkpoint_dirs = sorted(
        (
            path
            for path in Path("outputs/t5_wikisql_midterm").iterdir()
            if path.is_dir() and path.name.startswith("checkpoint-")
        ),
        key=lambda path: int(path.name.split("-")[-1]),
    )

    evaluation_modes = [
        ("base", base_model, base_tokenizer, "none"),
        ("fine_tuned", ft_model, ft_tokenizer, "none"),
    ]
    if grammar_available:
        evaluation_modes.extend(
            [
                ("base_constrained", base_model, base_tokenizer, "grammar"),
                ("fine_tuned_constrained", ft_model, ft_tokenizer, "grammar"),
            ]
        )

    summary_rows = []
    sample_rows = []

    def evaluate_model_set(run_label, modes):
        print(f"RUN: {run_label}")
        for temperature in TEMPERATURES:
            print(f"TEMPERATURE: {temperature}")
            mode_results = {}

            for mode_name, model, tokenizer, constraint_type in modes:
                predictions = []
                validity_flags = []
                golds = []

                for ex in test_ds:
                    prompt = build_prompt(ex)
                    gold = build_wikisql_sql(ex)
                    columns = _get_headers(ex)
                    table_name = _get_table_id(ex)
                    prefix_fn = None
                    logits_processor = None
                    if constraint_type == "grammar":
                        logits_processor = build_grammar_logits_processor(
                            tokenizer,
                            columns,
                            table_name,
                            question=ex["question"],
                        )

                    prediction = generate_sql(
                        model,
                        tokenizer,
                        prompt,
                        device,
                        temperature=temperature,
                        prefix_allowed_tokens_fn=prefix_fn,
                        logits_processor=logits_processor,
                    )
                    predictions.append(prediction)
                    golds.append(gold)
                    validity_flags.append(
                        is_sql_structurally_valid(prediction, columns, table_name)
                    )

                validity = syntactic_validity_rate(validity_flags)
                exact_match = exact_match_accuracy(predictions, golds)
                component_scores = component_accuracy(predictions, golds)
                mode_results[mode_name] = (
                    predictions,
                    validity_flags,
                    validity,
                    exact_match,
                    component_scores,
                )
                summary_rows.append(
                    {
                        "run_label": run_label,
                        "temperature": temperature,
                        "mode": mode_name,
                        "validity": round(validity, 4),
                        "exact_match": round(exact_match, 4),
                        "aggregation_acc": round(component_scores["aggregation"], 4),
                        "select_column_acc": round(component_scores["select_column"], 4),
                        "table_name_acc": round(component_scores["table_name"], 4),
                        "where_column_acc": round(component_scores["where_column"], 4),
                        "where_operator_acc": round(component_scores["where_operator"], 4),
                        "where_value_acc": round(component_scores["where_value"], 4),
                        "num_examples": len(validity_flags),
                    }
                )
                print(
                    f"{mode_name}: validity={validity:.3f}, exact_match={exact_match:.3f} "
                    f"over {len(validity_flags)} examples"
                )

            print("SAMPLE OUTPUTS")
            for sample_idx, ex in enumerate(test_ds.select(range(SAMPLE_PREVIEW_COUNT))):
                prompt = build_prompt(ex)
                gold = build_wikisql_sql(ex)
                columns = _get_headers(ex)
                table_name = _get_table_id(ex)
                print("QUESTION:", ex["question"])
                print("PROMPT:", prompt)
                print("GOLD:", gold)
                for mode_name, _, _, _ in modes:
                    predictions, _, _, _, _ = mode_results[mode_name]
                    prediction = predictions[sample_idx]
                    is_valid = is_sql_structurally_valid(prediction, columns, table_name)
                    sample_rows.append(
                        {
                            "run_label": run_label,
                            "temperature": temperature,
                            "sample_index": sample_idx,
                            "mode": mode_name,
                            "question": ex["question"],
                            "prompt": prompt,
                            "gold": gold,
                            "prediction": prediction,
                            "is_valid": is_valid,
                        }
                    )
                    print(f"{mode_name.upper()}: {prediction}")
                    print(f"{mode_name.upper()}_VALID: {is_valid}")
                print("=" * 120)

    evaluate_model_set("final", evaluation_modes)

    for checkpoint_dir in checkpoint_dirs:
        checkpoint_model, checkpoint_tokenizer = load_adapter_model_and_tokenizer(checkpoint_dir, device)
        checkpoint_modes = [
            ("base", base_model, base_tokenizer, "none"),
            (f"{checkpoint_dir.name}", checkpoint_model, checkpoint_tokenizer, "none"),
        ]
        if grammar_available:
            checkpoint_modes.extend(
                [
                    ("base_constrained", base_model, base_tokenizer, "grammar"),
                    (f"{checkpoint_dir.name}_constrained", checkpoint_model, checkpoint_tokenizer, "grammar"),
                ]
            )
        evaluate_model_set(checkpoint_dir.name, checkpoint_modes)

    write_csv(
        run_dir / "summary.csv",
        [
            "run_label",
            "temperature",
            "mode",
            "validity",
            "exact_match",
            "aggregation_acc",
            "select_column_acc",
            "table_name_acc",
            "where_column_acc",
            "where_operator_acc",
            "where_value_acc",
            "num_examples",
        ],
        summary_rows,
    )
    write_csv(
        run_dir / "samples.csv",
        ["run_label", "temperature", "sample_index", "mode", "question", "prompt", "gold", "prediction", "is_valid"],
        sample_rows,
    )
    write_json(
        run_dir / "metadata.json",
        {
            "eval_examples": EVAL_EXAMPLES,
            "temperatures": TEMPERATURES,
            "checkpoint_runs": [path.name for path in checkpoint_dirs],
            "final_model_dir": FINETUNED_MODEL_DIR,
            "grammar_constraints_available": grammar_available,
        },
    )
    with open(run_dir / "README.txt", "w", encoding="utf-8") as handle:
        handle.write("Evaluation artifacts for the final report\n")
        handle.write("- summary.csv: validity by run/checkpoint, mode, and temperature\n")
        handle.write("- samples.csv: sample predictions for the report appendix\n")
        handle.write("- metadata.json: evaluation configuration\n")
        if not grammar_available:
            handle.write("- grammar-constrained modes were skipped because transformers-cfg is not installed\n")


if __name__ == "__main__":
    main()
