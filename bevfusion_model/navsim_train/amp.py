"""AMP wrapper matching SparseDrive's fixed fp16 loss-scale recipe."""

from __future__ import annotations

from contextlib import nullcontext

import torch


class Fp16Wrapper:
    def __init__(self, enabled: bool, init_scale: float):
        self.enabled = bool(enabled)
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.enabled,
            init_scale=float(init_scale),
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
        )

    def autocast(self):
        if not self.enabled:
            return nullcontext()
        return torch.cuda.amp.autocast(enabled=True)

    def scale(self, loss):
        return self.scaler.scale(loss) if self.enabled else loss

    def step(self, optimizer):
        if self.enabled:
            return self.scaler.step(optimizer)
        return optimizer.step()

    def update(self):
        if self.enabled:
            self.scaler.update()

    def unscale_(self, optimizer):
        if self.enabled:
            self.scaler.unscale_(optimizer)

    def state_dict(self):
        return self.scaler.state_dict() if self.enabled else {"enabled": False}

    def load_state_dict(self, state_dict):
        if self.enabled and state_dict:
            self.scaler.load_state_dict(state_dict)
