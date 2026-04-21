## Results

We evaluated four settings on 100 WikiSQL test examples across decoding temperatures 0.0, 0.3, 0.7, and 1.0:

- Base model
- Fine-tuned model
- Base model with constrained decoding
- Fine-tuned model with constrained decoding

In the initial experiment, the base model achieved 0.000 syntactic validity at every temperature. The fine-tuned model performed substantially better, reaching its best validity of 0.450 at temperature 0.0, then decreasing to 0.380 at temperature 0.3, 0.370 at temperature 0.7, and 0.190 at temperature 1.0.

These results show that fine-tuning improved the model's ability to produce SQL-shaped outputs, and that lower-temperature decoding was more effective for this structured prediction task.

## Discussion

The strongest initial result came from the fine-tuned model with greedy decoding at temperature 0.0. This is consistent with the expectation that deterministic decoding is better suited than high-temperature sampling for structured outputs such as SQL queries.

The constrained decoding implementation did not improve performance. Instead, it reduced syntactic validity to 0.000 for both the base and fine-tuned models. This happened because the method used a flat token whitelist rather than a true prefix-aware or grammar-aware constraint. The decoder could still combine allowed tokens into malformed SQL, repeated clauses, and invalid punctuation sequences.

We kept this simpler constrained decoder in the final codebase and report the negative result honestly. This is useful experimentally because it shows that not every decoding restriction helps: a weak constraint can reduce the model's effective search space without actually enforcing valid SQL structure.

## Limitations

- The current validity metric measures syntactic structure, not semantic correctness.
- A query can be counted as valid even if it selects the wrong column or uses an incorrect condition value.
- The constrained decoding implementation here is intentionally simple and should be interpreted as a failed constrained-decoding baseline rather than a full grammar-based system.

## Next Step

Rerun `python evaluate_examples.py` and use the saved results to report the constrained decoder as a negative baseline.
