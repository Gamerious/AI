"""
NeuroSpark Training Loop

Features:
- Cosine learning rate schedule with warmup
- Gradient clipping
- Efficiency monitoring (FLOPs, active params, avg depth)
- Automatic mixed precision support
- Synthetic data generation for testing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
import math
from typing import Dict, Optional, List
from core.model import NeuroSparkModel, NeuroSparkConfig


class SyntheticDataset(Dataset):
    """
    Generates synthetic sequences for architecture testing.

    Tasks:
    1. Copy task: output = input (tests basic memory)
    2. Reverse task: output = reversed input (tests reasoning)
    3. Sort task: output = sorted input (tests complex reasoning)
    4. Arithmetic: simple addition patterns
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        seq_len: int = 128,
        n_samples: int = 10000,
        task: str = "mixed",
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.task = task
        self.data = self._generate()

    def _generate(self) -> List[Dict[str, torch.Tensor]]:
        data = []
        for i in range(self.n_samples):
            if self.task == "mixed":
                task = ["copy", "reverse", "pattern"][i % 3]
            else:
                task = self.task

            # Generate input (avoid 0 which is padding)
            tokens = torch.randint(1, self.vocab_size, (self.seq_len,))

            if task == "copy":
                labels = tokens.clone()
            elif task == "reverse":
                labels = tokens.flip(0)
            elif task == "pattern":
                # Pattern: each token predicts next token + 1
                labels = (tokens + 1) % self.vocab_size
            else:
                labels = tokens.clone()

            data.append({"input_ids": tokens, "labels": labels})
        return data

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


class NeuroSparkTrainer:
    """Training loop with efficiency monitoring."""

    def __init__(
        self,
        model: NeuroSparkModel,
        config: NeuroSparkConfig,
        lr: float = 3e-4,
        warmup_steps: int = 100,
        max_steps: int = 1000,
        grad_clip: float = 1.0,
        log_interval: int = 10,
    ):
        self.model = model
        self.config = config
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.device = next(model.parameters()).device

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95)
        )

        self.history: List[Dict[str, float]] = []

    def _get_lr(self, step: int) -> float:
        """Cosine schedule with linear warmup."""
        if step < self.warmup_steps:
            return self.lr * step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(
            1, self.max_steps - self.warmup_steps
        )
        return self.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def train(
        self,
        dataset: Dataset,
        batch_size: int = 16,
        num_epochs: int = 1,
    ) -> List[Dict[str, float]]:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        self.model.train()
        step = 0
        epoch_losses = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in dataloader:
                if step >= self.max_steps:
                    break

                # Update learning rate
                lr = self._get_lr(step)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                start_time = time.time()
                outputs = self.model(input_ids=input_ids, labels=labels)
                forward_time = time.time() - start_time

                loss = outputs["loss"]

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

                # Logging
                if step % self.log_interval == 0:
                    diag = self.model.get_diagnostics()
                    log_entry = {
                        "step": step,
                        "loss": loss.item(),
                        "ce_loss": outputs["ce_loss"].item(),
                        "aux_loss": outputs["aux_loss"].item(),
                        "lr": lr,
                        "grad_norm": grad_norm.item(),
                        "forward_ms": forward_time * 1000,
                        **diag,
                    }
                    self.history.append(log_entry)

                    if step % (self.log_interval * 5) == 0:
                        print(
                            f"Step {step:4d} | Loss: {loss.item():.4f} | "
                            f"CE: {outputs['ce_loss'].item():.4f} | "
                            f"LR: {lr:.2e} | "
                            f"Time: {forward_time*1000:.1f}ms | "
                            f"Depth: {diag.get('avg_depth', 'N/A'):.1f} | "
                            f"Reasoning: {diag.get('reasoning_steps', 'N/A')}"
                        )

                step += 1

            if n_batches > 0:
                avg_loss = epoch_loss / n_batches
                epoch_losses.append(avg_loss)
                print(f"\nEpoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

        return self.history
