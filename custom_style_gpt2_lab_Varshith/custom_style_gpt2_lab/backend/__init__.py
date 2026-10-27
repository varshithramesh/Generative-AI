"""Model and data-processing functions for the Custom Style GPT-2 Lab."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_NAME = "gpt2"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = str(PROJECT_ROOT / "outputs" / "custom-gpt2")
MAX_LENGTH = 128
MIN_TRAINING_CHARACTERS = 40


def clean_text(text: str) -> str:
    """Normalize line endings and whitespace while preserving announcement lines."""
    if not isinstance(text, str):
        raise TypeError("Training text must be a string.")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def validate_training_text(text: str) -> str:
    """Clean and validate text before it is sent to the tokenizer."""
    cleaned = clean_text(text)
    if len(cleaned) < MIN_TRAINING_CHARACTERS:
        raise ValueError(
            f"Please provide at least {MIN_TRAINING_CHARACTERS} non-whitespace characters."
        )
    return cleaned


def get_device() -> torch.device:
    """Use CUDA when it is available, otherwise use the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(model_name: str = MODEL_NAME) -> GPT2Tokenizer:
    """Load a GPT-2 tokenizer and configure its required padding token."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(
    model_name: str = MODEL_NAME,
) -> tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    """Load the original pretrained GPT-2 model and tokenizer."""
    tokenizer = load_tokenizer(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(get_device())
    model.eval()
    return model, tokenizer


def checkpoint_exists(checkpoint_dir: str = CHECKPOINT_DIR) -> bool:
    """Return whether a complete locally saved model checkpoint exists."""
    checkpoint = Path(checkpoint_dir)
    return (checkpoint / "config.json").is_file() and (
        (checkpoint / "pytorch_model.bin").is_file()
        or (checkpoint / "model.safetensors").is_file()
    )


def load_fine_tuned_model(
    checkpoint_dir: str = CHECKPOINT_DIR,
) -> tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    """Load a fine-tuned model and tokenizer saved by ``fine_tune_model``."""
    if not checkpoint_exists(checkpoint_dir):
        raise FileNotFoundError(
            f"No complete fine-tuned checkpoint was found at '{checkpoint_dir}'."
        )
    tokenizer = load_tokenizer(checkpoint_dir)
    model = GPT2LMHeadModel.from_pretrained(checkpoint_dir)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(get_device())
    model.eval()
    return model, tokenizer


class TextBlockDataset(Dataset[dict[str, torch.Tensor]]):
    """Tokenized, padded blocks suitable for causal-language-model training."""

    def __init__(
        self,
        text: str,
        tokenizer: GPT2Tokenizer,
        block_size: int = MAX_LENGTH,
    ) -> None:
        if block_size < 8:
            raise ValueError("block_size must be at least 8.")
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < 8:
            raise ValueError("Dataset is too short after tokenization.")

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must have a padding token.")
        self.blocks: list[dict[str, torch.Tensor]] = []
        for start in range(0, len(token_ids), block_size):
            block = token_ids[start : start + block_size]
            if len(block) < 2:
                continue
            padding = block_size - len(block)
            input_ids = block + [pad_id] * padding
            attention_mask = [1] * len(block) + [0] * padding
            labels = block + [-100] * padding
            self.blocks.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )
        if not self.blocks:
            raise ValueError("Dataset did not contain enough tokens for training.")

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.blocks[index]


def prepare_training_dataset(
    text: str,
    tokenizer: GPT2Tokenizer,
    block_size: int = MAX_LENGTH,
) -> TextBlockDataset:
    """Validate text and prepare padded tokenized training examples."""
    return TextBlockDataset(validate_training_text(text), tokenizer, block_size)


def fine_tune_model(
    text: str,
    steps: int = 10,
    learning_rate: float = 5e-5,
    checkpoint_dir: str = CHECKPOINT_DIR,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> dict[str, Any]:
    """Fine-tune GPT-2 for exactly ``steps`` updates and save its checkpoint."""
    if int(steps) < 1:
        raise ValueError("Training steps must be at least 1.")
    if float(learning_rate) <= 0:
        raise ValueError("Learning rate must be greater than zero.")

    cleaned_text = validate_training_text(text)
    tokenizer = load_tokenizer()
    dataset = prepare_training_dataset(cleaned_text, tokenizer)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.pad_token_id
    device = get_device()
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))

    requested_steps = int(steps)
    history: list[dict[str, float | int]] = []
    completed_steps = 0
    while completed_steps < requested_steps:
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            if output.loss is None:
                raise RuntimeError("The model did not return a training loss.")
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            completed_steps += 1
            loss_value = float(output.loss.detach().cpu())
            history.append({"step": completed_steps, "loss": loss_value})
            if progress_callback is not None:
                progress_callback(completed_steps, requested_steps, loss_value)
            if completed_steps >= requested_steps:
                break

    checkpoint = Path(checkpoint_dir)
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    if not checkpoint_exists(str(checkpoint)):
        raise RuntimeError("Training finished, but the checkpoint could not be verified.")
    model.eval()
    return {"history": history, "checkpoint": str(checkpoint)}


@torch.no_grad()
def generate_text(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
) -> str:
    """Generate a safe sampled continuation for a non-empty prompt."""
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Enter a non-empty prompt.")
    if int(max_new_tokens) < 1:
        raise ValueError("Generation length must be at least 1.")

    device = next(model.parameters()).device
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    model.eval()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=min(int(max_new_tokens), 200),
        do_sample=True,
        temperature=0.8,
        top_p=0.92,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
