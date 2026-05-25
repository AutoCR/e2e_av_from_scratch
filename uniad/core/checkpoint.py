from collections import OrderedDict

import torch


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, str):
        checkpoint = torch.load(checkpoint, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], checkpoint.get("meta", {})
    return checkpoint, {}


def normalize_state_dict_keys(state_dict):
    converted = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        key = key.replace(".self_attn.", ".attentions.0.")
        key = key.replace(".multihead_attn.", ".attentions.1.")
        key = key.replace(".ffn.", ".ffns.0.")
        key = key.replace(".decoder.norm.", ".decoder.post_norm.")
        converted[key] = value
    return converted


def load_state_dict(model, state_dict, strict=False):
    state_dict = normalize_state_dict_keys(state_dict)
    skipped = []
    if not strict:
        model_state = model.state_dict()
        filtered = OrderedDict()
        for key, value in state_dict.items():
            if key in model_state and tuple(model_state[key].shape) != tuple(value.shape):
                skipped.append((key, tuple(value.shape), tuple(model_state[key].shape)))
                continue
            filtered[key] = value
        state_dict = filtered
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if hasattr(model, "_checkpoint_tensors"):
        model._checkpoint_tensors.update({key: value.detach().clone() for key, value in state_dict.items()})
        unexpected = []
    for name, module in model.named_modules():
        if hasattr(module, "load_checkpoint_tensors"):
            module.load_checkpoint_tensors(name, state_dict)
    if strict and (missing or unexpected):
        raise RuntimeError(f"Error(s) in loading state_dict: missing={missing}, unexpected={unexpected}")
    return {"missing_keys": missing, "unexpected_keys": unexpected, "skipped_mismatched_keys": skipped}


def load_checkpoint(model, filename_or_checkpoint, map_location="cpu", strict=False):
    checkpoint = torch.load(filename_or_checkpoint, map_location=map_location) if isinstance(filename_or_checkpoint, str) else filename_or_checkpoint
    state_dict, meta = _extract_state_dict(checkpoint)
    result = load_state_dict(model, state_dict, strict=strict)
    result["meta"] = meta
    return result
