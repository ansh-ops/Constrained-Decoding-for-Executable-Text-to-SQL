# Constrained Decoding for Executable Text-to-SQL

## Title and Motivation

This project studies whether constrained decoding can improve text-to-SQL generation by forcing outputs to stay closer to valid WikiSQL query structure. The motivation is practical: a fine-tuned language model may still generate malformed SQL, hallucinated schema items, or invalid clause structures. Constrained decoding is meant to reduce those failures by limiting what the model can output at generation time.

## Task Definition and Proposed Method

The task is natural-language question to SQL generation on WikiSQL. We compare four settings:

- Base model
- Fine-tuned model
- Base model with constrained decoding
- Fine-tuned model with constrained decoding

The base model is `google/flan-t5-base`. Fine-tuning is done with LoRA adapters using PEFT. The constrained method restricts decoding to a tighter WikiSQL-style SQL template so the output stays closer to `SELECT ... FROM ... [WHERE ...];` forms using schema-aware columns, the correct table id, and restricted operators.

## Experimental Setup

- Dataset: WikiSQL
- Base model: `google/flan-t5-base`
- Fine-tuning method: LoRA / PEFT
- Training subset: 2,000 train examples
- Validation subset: 300 examples
- Training epochs: 3
- Evaluation set: 100 test examples
- Temperatures: 0.0, 0.3, 0.7, 1.0

Metrics:

- Syntactic validity
- Exact match accuracy
- Component accuracies:
  - aggregation
  - selected column
  - table name
  - where column
  - where operator
  - where value

## Main Results

### Training Progress

Training improved steadily across epochs.

- Early training loss at epoch 0.1: 3.3414
- End of epoch 1 train loss: 1.0006
- End of epoch 2 train loss: 0.8968
- End of epoch 3 train loss: 0.8510

Validation loss also improved:

- Epoch 1 eval loss: 0.7646
- Epoch 2 eval loss: 0.7267
- Epoch 3 eval loss: 0.7209

This shows the model learned the task and continued improving over the full 3-epoch run.

### Final Evaluation Summary

At temperature 0.0:

- Base: validity 0.00, exact match 0.00
- Fine-tuned: validity 0.72, exact match 0.29
- Base constrained: validity 0.00, exact match 0.00
- Fine-tuned constrained: validity 0.59, exact match 0.15

At temperature 0.3:

- Base: validity 0.00, exact match 0.00
- Fine-tuned: validity 0.67, exact match 0.25
- Base constrained: validity 0.00, exact match 0.00
- Fine-tuned constrained: validity 0.59, exact match 0.16

At temperature 0.7:

- Base: validity 0.00, exact match 0.00
- Fine-tuned: validity 0.54, exact match 0.18
- Base constrained: validity 0.00, exact match 0.00
- Fine-tuned constrained: validity 0.59, exact match 0.14

At temperature 1.0:

- Base: validity 0.00, exact match 0.00
- Fine-tuned: validity 0.39, exact match 0.14
- Base constrained: validity 0.00, exact match 0.00
- Fine-tuned constrained: validity 0.56, exact match 0.13

### Detailed Fine-Tuned Performance at Temperature 0.0

Fine-tuned model component accuracies:

- Aggregation accuracy: 0.81
- Select-column accuracy: 0.79
- Table-name accuracy: 0.83
- Where-column accuracy: 0.62
- Where-operator accuracy: 0.91
- Where-value accuracy: 0.66

Fine-tuned constrained model component accuracies:

- Aggregation accuracy: 0.48
- Select-column accuracy: 0.53
- Table-name accuracy: 0.60
- Where-column accuracy: 0.37
- Where-operator accuracy: 0.56
- Where-value accuracy: 0.31

## Discussion

The clearest result is that fine-tuning substantially improves performance over the base model. The base model produced no syntactically valid outputs in this evaluation subset, while the fine-tuned model reached 72% syntactic validity and 29% exact match at temperature 0.0.

Temperature matters. For the fine-tuned model, both validity and exact match were best at temperature 0.0 and degraded as temperature increased. This is consistent with the expectation that deterministic decoding is more suitable for structured generation tasks like SQL.

Constrained decoding produced mixed outcomes. It did not outperform the fine-tuned model. However, it was more stable across temperatures: the fine-tuned constrained model stayed in the 0.56 to 0.59 validity range, while the unconstrained fine-tuned model dropped from 0.72 at temperature 0.0 to 0.39 at temperature 1.0. This suggests constrained decoding can regularize output structure, but in this implementation it restricted the model too aggressively and reduced exact correctness.

The main conclusion is that fine-tuning was the strongest contributor to performance, while constrained decoding introduced a tradeoff: somewhat more stable structural behavior across temperatures, but lower best-case validity and lower exact match than unconstrained fine-tuned decoding.

## Conclusion

This project demonstrates that:

- Fine-tuning `flan-t5-base` with LoRA is effective for WikiSQL text-to-SQL generation.
- Low-temperature decoding gives the strongest performance for structured SQL outputs.
- Constrained decoding is promising in principle, but its effectiveness depends heavily on how tightly the constraint matches the task.
- In our implementation, the best-performing configuration was the unconstrained fine-tuned model at temperature 0.0.

Overall, the project shows that model adaptation through fine-tuning is highly beneficial, while constrained decoding remains a meaningful but challenging research direction for executable text-to-SQL generation.
