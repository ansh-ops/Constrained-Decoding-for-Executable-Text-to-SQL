import torch

from constrained_decoding import grammar_constraints_available
from evaluation_utils import (
    load_base_model_and_tokenizer,
    load_finetuned_model_and_tokenizer,
)
from model_config import BASE_MODEL_NAME


class ModelRegistry:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model = None
        self.base_tokenizer = None
        self.finetuned_model = None
        self.finetuned_tokenizer = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        self.base_model, self.base_tokenizer = load_base_model_and_tokenizer(self.device)
        self.finetuned_model, self.finetuned_tokenizer = load_finetuned_model_and_tokenizer(
            self.device
        )
        self._loaded = True

    @property
    def available_modes(self):
        modes = ["base", "fine_tuned"]
        if grammar_constraints_available():
            modes.extend(["base_constrained", "fine_tuned_constrained"])
        return modes

    @property
    def model_name(self):
        return BASE_MODEL_NAME


registry = ModelRegistry()
