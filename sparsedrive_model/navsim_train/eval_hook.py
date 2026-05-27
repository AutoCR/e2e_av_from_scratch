"""Lightweight eval hook for the pure-PyTorch NAVSIM runner."""

from __future__ import annotations

import json
from pathlib import Path

import torch

_MAX_SERIALIZED_ITEMS = 2048
_MAX_LOSS_BATCHES = 2


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_device(v, device) for v in value)
    return value


def _safe_json(value):
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.numel() > _MAX_SERIALIZED_ITEMS:
            return {"shape": list(tensor.shape), "truncated": tensor.flatten()[:_MAX_SERIALIZED_ITEMS].tolist()}
        return tensor.tolist()
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        seq = list(value)
        if len(seq) > _MAX_SERIALIZED_ITEMS:
            return {"length": len(seq), "truncated": [_safe_json(v) for v in seq[:_MAX_SERIALIZED_ITEMS]]}
        return [_safe_json(v) for v in seq]
    if hasattr(value, "tolist"):
        try:
            return _safe_json(value.tolist())
        except Exception:
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _tokens_from_batch(batch, count, offset):
    for key in ("token", "tokens", "sample_token", "sample_tokens", "scene_token"):
        if key in batch:
            value = batch[key]
            if torch.is_tensor(value):
                value = value.detach().cpu().tolist()
            if isinstance(value, (list, tuple)):
                return [str(v) for v in value[:count]]
            return [str(value)] + [f"sample_{offset + i:06d}" for i in range(1, count)]
    return [f"sample_{offset + i:06d}" for i in range(count)]


def _first_tensor_with_name(value, names):
    if torch.is_tensor(value):
        return None
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(name in lowered for name in names) and torch.is_tensor(item):
                return item
            found = _first_tensor_with_name(item, names)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor_with_name(item, names)
            if found is not None:
                return found
    return None


def _count_predictions(output):
    if not isinstance(output, dict):
        return 0
    payload = output.get("img_bbox", output)
    for key in ("labels_3d", "labels", "cls", "scores_3d", "scores", "boxes_3d", "boxes"):
        value = payload.get(key) if isinstance(payload, dict) else None
        if torch.is_tensor(value):
            return int(value.shape[0]) if value.dim() > 0 else int(value.numel())
        if isinstance(value, (list, tuple)):
            return len(value)
    return 0


def _maybe_val_loss(model, dataloader, device):
    losses = []
    was_training = model.training
    model.train()
    with torch.no_grad():
        for idx, raw_batch in enumerate(dataloader):
            if idx >= _MAX_LOSS_BATCHES:
                break
            if not isinstance(raw_batch, dict) or "img" not in raw_batch:
                continue
            if not any(str(k).startswith("gt_") for k in raw_batch):
                continue
            batch = _to_device(raw_batch, device)
            img = batch.pop("img")
            try:
                loss_dict = model(img, **batch)
                total = sum(v.detach().float() for v in loss_dict.values() if torch.is_tensor(v))
                if torch.is_tensor(total):
                    losses.append(float(total.cpu()))
            except Exception:
                continue
    model.train(was_training)
    return sum(losses) / len(losses) if losses else None


def run_eval(model, dataloader, device, output_dir, writer, global_iter, eval_mode, tag):
    output_path = Path(output_dir) / "eval" / str(tag) / f"iter_{global_iter}"
    output_path.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()
    total_predictions = 0
    token_count = 0
    planning_l2_values = []

    with torch.no_grad():
        for raw_batch in dataloader:
            if not isinstance(raw_batch, dict) or "img" not in raw_batch:
                continue
            batch_for_model = _to_device(raw_batch, device)
            img = batch_for_model.pop("img")
            outputs = model(img, **batch_for_model)
            if not isinstance(outputs, list):
                outputs = [outputs]
            tokens = _tokens_from_batch(raw_batch, len(outputs), token_count)
            gt_plan = batch_for_model.get("gt_ego_fut_trajs")

            for index, output in enumerate(outputs):
                token = tokens[index] if index < len(tokens) else f"sample_{token_count + index:06d}"
                with (output_path / f"{token}.json").open("w") as handle:
                    json.dump(_safe_json(output), handle)
                total_predictions += _count_predictions(output)

                if eval_mode.get("with_planning") and torch.is_tensor(gt_plan):
                    pred_plan = _first_tensor_with_name(output, ("plan", "ego_fut"))
                    if torch.is_tensor(pred_plan) and index < gt_plan.shape[0]:
                        pred = pred_plan.detach().float().reshape(-1, 2)
                        target = gt_plan[index].detach().float().reshape(-1, 2).to(pred.device)
                        length = min(pred.shape[0], target.shape[0])
                        if length > 0:
                            planning_l2_values.append(float(torch.linalg.norm(pred[:length] - target[:length], dim=-1).mean().cpu()))
            token_count += len(outputs)

    val_loss = _maybe_val_loss(model, dataloader, device)
    model.train(was_training)

    summary = {
        "tokens": token_count,
        "prediction_count": total_predictions,
        "prediction_count_per_token": total_predictions / max(1, token_count),
        "det_map_map_placeholder": 0.0,
    }
    if val_loss is not None:
        summary["val_loss"] = val_loss
    if planning_l2_values:
        summary["planning_l2"] = sum(planning_l2_values) / len(planning_l2_values)

    with (output_path / "summary.json").open("w") as handle:
        json.dump(_safe_json(summary), handle, indent=2)

    if writer is not None:
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"{tag}/{key}", value, global_iter)
    return summary
