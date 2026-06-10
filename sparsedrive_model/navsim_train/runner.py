"""Iter-based pure-PyTorch training runner for SparseDrive on NAVSIM."""

from __future__ import annotations

import random
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import torch
import os
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

if __package__ in {None, ""}:
    _THIS_FILE = Path(__file__).resolve()
    _REPO_ROOT = _THIS_FILE.parents[2]
    _SPARSEDRIVE_ROOT = _REPO_ROOT / "sparsedrive_model"
    for _path in (str(_REPO_ROOT), str(_SPARSEDRIVE_ROOT)):
        if _path not in sys.path:
            sys.path.insert(0, _path)

from sparsedrive_model.configs.sparsedrive_hyperparams import get_camera_order, get_stage_hyperparams
from sparsedrive_model.navsim_train.eval_hook import run_eval
from sparsedrive_model.navsim_train.optim import CosineWithLinearWarmup, build_optimizer, clip_grad_norm, top_grad_norms
from sparsedrive_model.navsim_train.scene_filter_loader import load_log_names, load_scene_filter_fields
from sparsedrive_model.sparsedrive import SparseDrive
from sparsedrive_model.sparsedrive import nn_utils as _sparsedrive_nn_utils
from sparsedrive_model.sparsedrive import debug_probe


class TrainingStalledError(RuntimeError):
    """Raised when too many optimizer steps are skipped consecutively.

    A sustained run of skipped steps means the model has diverged into a regime
    where every batch produces a pathological (explosion-guard-tripping) gradient
    and no weight update ever lands -- training is frozen but still consuming
    compute. Aborting loudly turns ~1.5 days of wasted GPU time into a fast,
    actionable failure. Tuned via recipe['grad_skip_abort_after'].
    """


def _set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _choose_device(device_config):
    if device_config != "auto":
        return torch.device(device_config)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _patch_cuda_calls_for_local_device(device):
    if device.type == "cuda" or getattr(torch.Tensor, "_navsim_cuda_patched", False):
        return

    def _cuda_to_local(self, *args, **kwargs):
        non_blocking = bool(kwargs.get("non_blocking", False))
        return self.to(device, non_blocking=non_blocking)

    torch.Tensor.cuda = _cuda_to_local
    torch.Tensor._navsim_cuda_patched = True


def _patch_loss_weight_broadcasting():
    if getattr(_sparsedrive_nn_utils, "_navsim_weight_broadcast_patched", False):
        return

    def _weighted_loss(loss, weight=None, reduction="mean", avg_factor=None):
        if weight is not None:
            while weight.ndim < loss.ndim:
                weight = weight.unsqueeze(-1)
            loss = loss * weight
        if avg_factor is not None:
            return loss.sum() / max(float(avg_factor), 1.0)
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        if reduction == "none":
            return loss
        raise ValueError(f"unsupported reduction: {reduction}")

    _sparsedrive_nn_utils.weighted_loss = _weighted_loss
    _sparsedrive_nn_utils._navsim_weight_broadcast_patched = True


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(v, device) for v in value)
    return value


def _load_data_api():
    try:
        from sparsedrive_model.navsim_train.data import NavSimSparseDriveDataset, build_dataloader, collate_fn
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "sparsedrive_model.navsim_train.data is required for full training. "
            "It is owned by the data-targets task and must provide "
            "NavSimSparseDriveDataset, build_dataloader, and collate_fn."
        ) from exc
    return NavSimSparseDriveDataset, build_dataloader, collate_fn


def _resolve_split_config(split_config, repo_root=None):
    """Resolve a split config entry to (directory_name, log_names_or_None, tokens_or_None).

    Accepts:
    - str: directory name only, no filtering applied.
    - dict with keys:
        - "dir": directory name under navsim_logs/ (required)
        - "log_names_yaml": YAML path for log names (optional, relative to repo_root or absolute)
        - "log_names_key": key in log_names_yaml; defaults to "log_names" (optional)
        - "tokens_yaml": YAML path for scene tokens (optional, may differ from log_names_yaml)
        - "tokens_key": key in tokens_yaml; defaults to "tokens" (optional)
      Omitting or setting a yaml field to None disables that filter.
    """
    if isinstance(split_config, str):
        return split_config, None, None

    dir_name = split_config["dir"]

    def _resolve_path(yaml_path_str):
        if yaml_path_str is None:
            return None
        p = Path(yaml_path_str)
        if not p.is_absolute() and repo_root is not None:
            p = Path(repo_root) / p
        return p

    log_names = None
    log_names_yaml = _resolve_path(split_config.get("log_names_yaml"))
    if log_names_yaml is not None:
        log_names_key = split_config.get("log_names_key", "log_names")
        log_names, _ = load_scene_filter_fields(log_names_yaml, log_names_key=log_names_key, tokens_key=None)

    tokens = None
    tokens_yaml = _resolve_path(split_config.get("tokens_yaml"))
    if tokens_yaml is not None:
        tokens_key = split_config.get("tokens_key", "tokens")
        _, tokens = load_scene_filter_fields(tokens_yaml, log_names_key=None, tokens_key=tokens_key)

    return dir_name, log_names, tokens


def _build_dataset(dataset_cls, *, split, config, test_mode, max_scenes, log_names=None, tokens=None):
    kwargs = {
        "split": split,
        "openscene_data_root": config["openscene_data_root"],
        "nuplan_maps_root": config["nuplan_maps_root"],
        "image_hw": (256, 704),
        "test_mode": test_mode,
        "camera_order": get_camera_order(),
    }
    if max_scenes is not None:
        kwargs["max_scenes"] = max_scenes
    if log_names is not None:
        kwargs["log_names"] = log_names
    if tokens is not None:
        kwargs["tokens"] = tokens
    try:
        return dataset_cls(**kwargs)
    except TypeError:
        kwargs["mode"] = test_mode
        return dataset_cls(**kwargs)


def _build_loader(build_dataloader_fn, dataset, *, batch_size, num_workers, shuffle, collate_fn, sampler=None):
    effective_shuffle = shuffle if sampler is None else False
    try:
        return build_dataloader_fn(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=effective_shuffle,
            collate_fn=collate_fn,
            sampler=sampler,
        )
    except TypeError:
        # Fallback for older build_dataloader signatures that don't accept sampler
        return build_dataloader_fn(dataset, batch_size, num_workers, effective_shuffle, collate_fn)


def _prepare_hyperparams_for_local_build(hyperparams):
    prepared = dict(hyperparams)
    pretrained = prepared.get("backbone_pretrained")
    if pretrained and not Path(pretrained).exists():
        # Drop a missing pretrained-backbone path so a CPU/local smoke build can
        # still construct the model. But warn LOUDLY: a typo'd or wrong-CWD path
        # would otherwise silently train the backbone from RANDOM init instead of
        # ImageNet, degrading a full run with no error. Only rank 0 prints (the
        # message is identical across ranks).
        if _is_main_process():
            print(
                f"WARNING: backbone_pretrained={pretrained!r} does not exist "
                f"(cwd={Path.cwd()}); falling back to RANDOM backbone init. "
                f"For a real training run this is almost certainly wrong -- fix "
                f"the path so the ImageNet-pretrained ResNet actually loads.",
                file=sys.stderr,
                flush=True,
            )
        prepared["backbone_pretrained"] = None
    return prepared


def _extract_model_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def _load_weights(model, path, device, label):
    checkpoint = torch.load(path, map_location=device)
    incompatible = model.load_state_dict(_extract_model_state(checkpoint), strict=False)
    print(f"Loaded {label} from {path}: missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}")


def _save_checkpoint(path, model, optimizer, scheduler, iteration, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if isinstance(model, DDP) else model
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "iter": iteration,
            "config": dict(config),
        },
        path,
    )


def _checkpoint_if_needed(output_dir, model, optimizer, scheduler, iteration, config):
    ckpt_dir = Path(output_dir) / "ckpt"
    iter_path = ckpt_dir / f"iter_{iteration}.pth"
    last_path = ckpt_dir / "last.pth"
    _save_checkpoint(iter_path, model, optimizer, scheduler, iteration, config)
    shutil.copyfile(iter_path, last_path)
    print(f"Saved checkpoint {iter_path}")


# ---------------------------------------------------------------------------
# Distributed training helpers
# ---------------------------------------------------------------------------


def _init_distributed():
    """Initialize NCCL process group when launched via torchrun.

    Reads RANK / LOCAL_RANK / WORLD_SIZE from environment variables.
    Returns (local_rank, world_size) or (0, 1) for non-distributed runs.
    """
    rank = int(os.environ.get("RANK", -1))
    if rank == -1:
        return 0, 1  # non-DDP
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    # The default NCCL collective timeout is 10 min. Rank 0 periodically leaves
    # the collective path to run eval / write checkpoints, so the timeout must be
    # long enough to cover the longest such gap or the other ranks' watchdogs fire.
    timeout_min = int(os.environ.get("NCCL_TIMEOUT_MINUTES", "60"))
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout_min))
    return local_rank, world_size


def _get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _is_main_process() -> bool:
    return _get_rank() == 0


class _TeeStream:
    """Mirror a stream (stdout/stderr) to a log file while still writing through.

    Lets a bare ``torchrun ...`` (no shell ``| tee``) always produce a console
    log next to the checkpoints/TB data, capturing tqdm output, the [SD_DEBUG]
    probe lines, skip/grad dumps, and tracebacks. Line-buffered + flushed so the
    log is current if the run is killed mid-explosion.
    """

    def __init__(self, original, file_handle):
        self._original = original
        self._file = file_handle

    def write(self, data):
        self._original.write(data)
        try:
            self._file.write(data)
            self._file.flush()
        except Exception:
            pass
        return len(data)

    def flush(self):
        self._original.flush()
        try:
            self._file.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        # Delegate isatty(), fileno(), encoding, etc. to the real stream so
        # tqdm and libraries that introspect the terminal keep working.
        return getattr(self._original, name)


def _install_console_log(output_dir: Path):
    """Tee rank-0 stdout+stderr to ``output_dir/console_<timestamp>.log``.

    Returns the log path (or None on non-main ranks / failure). Idempotent-safe:
    only the main process writes, so DDP workers don't clobber each other.
    """
    if not _is_main_process():
        return None
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(output_dir) / f"console_{ts}.log"
        fh = open(log_path, "a", buffering=1, encoding="utf-8")
        sys.stdout = _TeeStream(sys.stdout, fh)
        sys.stderr = _TeeStream(sys.stderr, fh)
        print(f"[console-log] mirroring stdout/stderr to {log_path}", flush=True)
        return log_path
    except Exception as exc:  # never let logging setup break training
        print(f"[console-log] failed to set up file logging: {exc}", flush=True)
        return None


def _cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _build_sampler(dataset, *, shuffle: bool):
    """Return a DistributedSampler when DDP is active, else None."""
    if _get_world_size() > 1:
        return DistributedSampler(dataset, shuffle=shuffle)
    return None


def _log_nonfinite_grads(model, iteration, max_report=12):
    """Report which parameters received NaN/inf gradients.

    Called only on the rare step where clip_grad_norm reports a non-finite
    norm, so the overhead is negligible. Pinpoints the offending module(s) by
    parameter name, turning a generic 'NaN/inf gradients' message into an
    actionable location.
    """
    offenders = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        g = param.grad
        finite = torch.isfinite(g)
        if not bool(finite.all()):
            n_nan = int(torch.isnan(g).sum())
            n_inf = int(torch.isinf(g).sum())
            finite_vals = g[finite]
            max_abs = float(finite_vals.abs().max()) if finite_vals.numel() else float("nan")
            offenders.append((name, n_nan, n_inf, g.numel(), max_abs))
    if not offenders:
        tqdm.write(f"iter={iteration}: grad norm non-finite but no per-parameter NaN/inf found (possible overflow in norm reduction)")
        _log_large_grads(model, iteration)
        return
    offenders.sort(key=lambda x: (x[1] + x[2]), reverse=True)
    tqdm.write(f"iter={iteration}: {len(offenders)} parameter tensor(s) with non-finite grads. Top offenders:")
    for name, n_nan, n_inf, numel, max_abs in offenders[:max_report]:
        tqdm.write(
            f"    {name}: nan={n_nan} inf={n_inf} / {numel} elems; max|finite grad|={max_abs:.3e}"
        )


def _log_large_grads(model, iteration, max_report=8):
    """Report the parameters carrying the largest (finite) gradient norms.

    Used on a gradient-explosion skip (or fp32 norm-reduction overflow) to
    identify which module the explosion originates in. Cheap: one norm per
    parameter tensor, only on the rare bad step.
    """
    ranked = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        g = param.grad
        finite = g[torch.isfinite(g)]
        if finite.numel() == 0:
            continue
        ranked.append((float(finite.norm()), float(finite.abs().max()), name))
    if not ranked:
        return
    ranked.sort(reverse=True)
    tqdm.write(f"iter={iteration}: largest finite grad-norm parameters:")
    for gnorm, gmax, name in ranked[:max_report]:
        tqdm.write(f"    {name}: |grad|_2={gnorm:.3e} max|grad|={gmax:.3e}")


def _reset_temporal_state(model):
    """Reset all InstanceBank caches to break NaN propagation across iterations."""
    for module in model.modules():
        if hasattr(module, "reset") and callable(module.reset):
            try:
                module.reset()
            except Exception:
                pass


def run(config: dict):
    # Wire the config debug knobs into the env vars the probe reads. A value
    # already present in the environment wins, so a command-line `SD_DEBUG=2 ...`
    # override (e.g. to escalate to anomaly mode for one run) still takes effect
    # without editing the config.
    if "SD_DEBUG" not in os.environ and config.get("debug_level") is not None:
        os.environ["SD_DEBUG"] = str(int(config["debug_level"]))
    if "SD_DEBUG_ALL_RANKS" not in os.environ and config.get("debug_all_ranks"):
        os.environ["SD_DEBUG_ALL_RANKS"] = "1"

    local_rank, world_size = _init_distributed()
    _set_seed(int(config.get("seed", 0)))
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = _choose_device(config.get("device", "auto"))
    _patch_cuda_calls_for_local_device(device)
    _patch_loss_weight_broadcasting()
    stage = config.get("stage", "stage1")
    hyperparams, recipe = get_stage_hyperparams(stage)

    for key in ("total_batch_size", "num_epochs", "ckpt_epoch_interval", "eval_epoch_interval", "log_interval", "effective_batch_size", "grad_accum_steps"):
        if config.get(key) is not None:
            recipe[key] = config[key]
    if config.get("quick_smoke"):
        recipe.update(
            {
                "total_batch_size": 1,
                "num_epochs": 1,
                "warmup_iters": 2,
                "log_interval": 1,
                "ckpt_epoch_interval": 1,
                "eval_epoch_interval": 1,
                "grad_accum_steps": 1,
            }
        )

    total_batch_size = int(recipe["total_batch_size"])

    # --- Resolve gradient-accumulation steps ---
    # Effective batch = total_batch_size * world_size * grad_accum_steps.
    micro_global_batch = total_batch_size * max(1, world_size)
    grad_accum_steps = recipe.get("grad_accum_steps")
    if not grad_accum_steps or int(grad_accum_steps) < 1:
        target_eff = recipe.get("effective_batch_size")
        if target_eff:
            grad_accum_steps = max(1, round(int(target_eff) / micro_global_batch))
        else:
            grad_accum_steps = 1
    grad_accum_steps = int(grad_accum_steps)
    recipe["grad_accum_steps"] = grad_accum_steps
    effective_batch_size = micro_global_batch * grad_accum_steps
    if _is_main_process():
        print(
            f"[grad-accum] micro_batch={total_batch_size} x world_size={max(1, world_size)} "
            f"x accum={grad_accum_steps} -> effective_batch={effective_batch_size}"
        )
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    # Auto-mirror the console to output_dir/console_<timestamp>.log on rank 0, so
    # a bare `torchrun ...` always produces a log without a manual `| tee`. Uses
    # the config's output_dir (same place as checkpoints + TB).
    _install_console_log(output_dir)

    NavSimSparseDriveDataset, build_dataloader_fn, collate_fn_fn = _load_data_api()
    max_scenes = 2 if config.get("quick_smoke") else None
    splits = config["splits"]
    _REPO_ROOT_RUNNER = Path(__file__).resolve().parents[2]

    train_split_dir, train_log_names, train_tokens = _resolve_split_config(splits["train"], _REPO_ROOT_RUNNER)
    val_split_dir, val_log_names, val_tokens = _resolve_split_config(splits["val"], _REPO_ROOT_RUNNER)
    test_split_dir, test_log_names, test_tokens = _resolve_split_config(splits["test"], _REPO_ROOT_RUNNER)

    train_dataset = _build_dataset(
        NavSimSparseDriveDataset,
        split=train_split_dir,
        config=config,
        test_mode=False,
        max_scenes=max_scenes,
        log_names=train_log_names,
        tokens=train_tokens,
    )
    assert len(train_dataset.camera_order) == hyperparams["num_cams"], (
        f"Camera mismatch: dataset loaded {len(train_dataset.camera_order)} cameras "
        f"but model num_cams={hyperparams['num_cams']}. Both must derive from CAMERA_ORDER."
    )
    val_dataset = _build_dataset(
        NavSimSparseDriveDataset,
        split=val_split_dir,
        config=config,
        test_mode=True,
        max_scenes=max_scenes,
        log_names=val_log_names,
        tokens=val_tokens,
    )
    test_dataset = _build_dataset(
        NavSimSparseDriveDataset,
        split=test_split_dir,
        config=config,
        test_mode=True,
        max_scenes=max_scenes,
        log_names=test_log_names,
        tokens=test_tokens,
    )
    train_sampler = _build_sampler(train_dataset, shuffle=True)

    train_loader = _build_loader(
        build_dataloader_fn,
        train_dataset,
        batch_size=total_batch_size,
        num_workers=int(config.get("num_workers", 0)),
        shuffle=False,
        collate_fn=collate_fn_fn,
        sampler=train_sampler,
    )
    val_loader = _build_loader(
        build_dataloader_fn,
        val_dataset,
        batch_size=total_batch_size,
        num_workers=int(config.get("num_workers", 0)),
        shuffle=False,
        collate_fn=collate_fn_fn,
        sampler=None,
    )
    test_loader = _build_loader(
        build_dataloader_fn,
        test_dataset,
        batch_size=total_batch_size,
        num_workers=int(config.get("num_workers", 0)),
        shuffle=False,
        collate_fn=collate_fn_fn,
        sampler=None,
    )

    model = SparseDrive(_prepare_hyperparams_for_local_build(hyperparams))
    model.init_weights()
    model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if isinstance(model, DDP) else model

    optimizer = build_optimizer(raw_model, recipe["lr"], recipe["weight_decay"], recipe["backbone_lr_mult"])
    num_iters_per_epoch = max(1, len(train_dataset) // (total_batch_size * grad_accum_steps))
    max_iters = num_iters_per_epoch * int(recipe["num_epochs"])
    scheduler = CosineWithLinearWarmup(
        optimizer,
        max_iters=max_iters,
        warmup_iters=int(recipe["warmup_iters"]),
        warmup_ratio=recipe["warmup_ratio"],
        min_lr_ratio=recipe["min_lr_ratio"],
    )

    start_iter = 0
    if config.get("resume_from"):
        checkpoint = torch.load(config["resume_from"], map_location=device)
        raw_model.load_state_dict(_extract_model_state(checkpoint), strict=False)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_iter = int(checkpoint.get("iter", 0))
        print(f"Resumed full training state from {config['resume_from']} at iter {start_iter}")
    elif config.get("load_from"):
        _load_weights(raw_model, config["load_from"], device, "model weights")

    writer = None
    if _is_main_process():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(log_dir=str(output_dir / "tb" / f"{stage}_{timestamp}"))
    eval_cadence = max(1, num_iters_per_epoch * int(recipe["eval_epoch_interval"]))
    ckpt_cadence = max(1, num_iters_per_epoch * int(recipe["ckpt_epoch_interval"]))

    iteration = start_iter
    # Gradient-explosion guard: skip any step whose pre-clip norm exceeds the
    # absolute floor skip_norm. clip_grad_norm_ bounds step magnitude but not
    # direction; skipping pathologically large-norm steps avoids corrupting weights.
    skip_norm = recipe.get("grad_skip_norm")
    if skip_norm is None:
        skip_norm = float(recipe["grad_clip_max_norm"]) * 1000.0
    skip_norm = float(skip_norm)
    # Stall guard: abort if this many optimizer steps are skipped consecutively
    # (a frozen-but-running divergence). A successful step resets the counter, so
    # one-off spikes never trip it. None/<=0 disables. See grad_skip_abort_after.
    abort_after = recipe.get("grad_skip_abort_after")
    abort_after = int(abort_after) if abort_after else 0
    consecutive_skips = 0
    pbar = tqdm(
        total=max_iters,
        initial=start_iter,
        desc="Training",
        unit="iter",
        dynamic_ncols=True,
        disable=not _is_main_process(),
    )
    try:
        for _epoch in range(int(recipe["num_epochs"])):
            if train_sampler is not None:
                train_sampler.set_epoch(_epoch)
            optimizer.zero_grad(set_to_none=True)
            micro_step = 0
            window_ok = True
            window_loss_sum = 0.0
            window_loss_components: dict[str, float] = {}
            for raw_batch in train_loader:
                if iteration >= max_iters:
                    break
                model.train()
                if not isinstance(raw_batch, dict) or "img" not in raw_batch:
                    raise KeyError("Training batches must be dicts containing an 'img' tensor.")
                batch = _move_to_device(raw_batch, device)
                img = batch.pop("img")

                # Debug step counter for the forward probes (no-op when SD_DEBUG=0).
                debug_probe.set_step(iteration + 1)

                loss_dict = model(img=img, **batch)
                loss = sum(v for v in loss_dict.values() if torch.is_tensor(v))

                # Decide finiteness collectively so all DDP ranks take the same
                # branch (otherwise mismatched backward() calls deadlock).
                micro_finite = bool(torch.isfinite(loss))
                if world_size > 1:
                    flag = torch.tensor([1.0 if micro_finite else 0.0], device=device)
                    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                    micro_finite = flag.item() > 0.5

                if micro_finite:
                    # Scale by accum so the summed gradient equals the mean over
                    # the full effective batch. Under SD_DEBUG=2, run the backward
                    # inside autograd anomaly detection so the FIRST backward op
                    # that produces a NaN/Inf is named in the traceback -- the
                    # definitive locator for the x/z^2-style explosion.
                    if debug_probe.anomaly_enabled():
                        with torch.autograd.detect_anomaly():
                            (loss / grad_accum_steps).backward()
                    else:
                        (loss / grad_accum_steps).backward()
                    window_loss_sum += float(loss.detach().cpu())
                    for k, v in loss_dict.items():
                        if torch.is_tensor(v):
                            window_loss_components[k] = window_loss_components.get(k, 0.0) + float(v.detach().cpu())
                else:
                    bad = {k: float(v) for k, v in loss_dict.items() if torch.is_tensor(v) and not torch.isfinite(v)}
                    tqdm.write(
                        f"iter={iteration + 1} micro={micro_step + 1}/{grad_accum_steps}: "
                        f"non-finite loss, discarding accumulation window; bad_losses={bad}"
                    )
                    window_ok = False
                    _reset_temporal_state(raw_model)

                micro_step += 1
                if micro_step < grad_accum_steps:
                    continue

                # --- Accumulation window complete: take one optimizer step ---
                step_done = False
                grad_norm = None
                if window_ok:
                    # clip_grad_norm computes the total norm in fp64 (no fp32
                    # reduction overflow) and only clips when the norm is finite,
                    # leaving grads intact on a genuine NaN/inf so the diagnostic
                    # below can inspect the real (pre-clip-zeroing) gradients.
                    grad_norm = clip_grad_norm(raw_model, recipe["grad_clip_max_norm"], recipe["grad_clip_norm_type"])
                    gn = float(grad_norm)
                    if not torch.isfinite(grad_norm):
                        tqdm.write(f"iter={iteration + 1}: NaN/inf gradients detected, resetting temporal state")
                        _log_nonfinite_grads(raw_model, iteration + 1)
                        debug_probe.first_nonfinite_param(raw_model)
                        _reset_temporal_state(raw_model)
                    elif gn > skip_norm:
                        # Norm is finite but pathologically large: the batch's
                        # gradient direction is untrustworthy. Skip the step
                        # entirely to avoid corrupting the weights. clip_grad_norm
                        # already scaled the grads by max_norm/gn (~1/gn), so undo
                        # that scale before ranking offenders -- otherwise a true
                        # 5e5 grad reads back as ~0.86 and the offender looks
                        # innocent. The skip happens only on the rare bad step, so
                        # this extra pass costs nothing on the healthy path.
                        unclip = gn / float(recipe["grad_clip_max_norm"])
                        tqdm.write(
                            f"iter={iteration + 1}: grad norm {gn:.3e} exceeds skip "
                            f"threshold {skip_norm:.3e}; skipping step (gradient-explosion guard)"
                        )
                        tqdm.write(f"iter={iteration + 1}: largest PRE-CLIP grad-norm parameters:")
                        for name, gl2, gmax in top_grad_norms(raw_model, recipe["grad_clip_norm_type"]):
                            tqdm.write(f"    {name}: |grad|_2={gl2 * unclip:.3e} max|grad|={gmax * unclip:.3e}")
                        debug_probe.first_nonfinite_param(raw_model)
                        _reset_temporal_state(raw_model)
                    else:
                        optimizer.step()
                        step_done = True

                optimizer.zero_grad(set_to_none=True)
                global_iter = iteration + 1
                scheduler.step(global_iter)
                pbar.update(1)

                # Stall guard: a successful step resets the streak; any skipped
                # window (explosion-guard, NaN/inf grads, or non-finite loss)
                # extends it. A long streak means training is frozen in a
                # divergent regime -- abort instead of burning compute. The
                # decision is identical on every DDP rank (step_done derives from
                # the collectively-decided loss/grads), so all ranks raise
                # together and none hang in a later collective.
                if step_done:
                    consecutive_skips = 0
                else:
                    consecutive_skips += 1
                    if abort_after and consecutive_skips >= abort_after:
                        msg = (
                            f"Training stalled: {consecutive_skips} consecutive optimizer "
                            f"steps skipped (last grad_norm="
                            f"{'non-finite' if grad_norm is None or not torch.isfinite(grad_norm) else f'{float(grad_norm):.3e}'}"
                            f", skip_threshold={skip_norm:.3e}) at iter {global_iter}/{max_iters}. "
                            f"The model has diverged and no weight update is landing. Restart from "
                            f"the last pre-divergence checkpoint with a fix in place "
                            f"(grad_skip_abort_after={abort_after})."
                        )
                        tqdm.write(msg)
                        raise TrainingStalledError(msg)

                # Log on every completed window (not just successful steps) so the
                # displayed/recorded grad_norm reflects the CURRENT iteration. The
                # old `step_done`-gated path left tqdm showing a stale grad_norm
                # from the last good step whenever a window was skipped, hiding
                # ongoing explosions behind a healthy-looking number.
                if grad_norm is not None and global_iter % int(recipe["log_interval"]) == 0:
                    avg_loss = window_loss_sum / grad_accum_steps
                    losses_for_log = {k: v / grad_accum_steps for k, v in window_loss_components.items()}
                    lr = optimizer.param_groups[-1]["lr"]
                    gn_finite = bool(torch.isfinite(grad_norm))
                    gn_str = f"{float(grad_norm):.3f}" if gn_finite else "non-finite"
                    pbar.set_description(f"Epoch [{_epoch + 1}/{int(recipe['num_epochs'])}]")
                    pbar.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        lr=f"{lr:.2e}",
                        grad_norm=gn_str,
                        skipped="" if step_done else "1",
                    )
                    if _is_main_process():
                        writer.add_scalar("train/loss_total", avg_loss, global_iter)
                        writer.add_scalar("train/lr", lr, global_iter)
                        if gn_finite:
                            writer.add_scalar("train/grad_norm", float(grad_norm), global_iter)
                        writer.add_scalar("train/step_skipped", 0 if step_done else 1, global_iter)
                        for key, value in losses_for_log.items():
                            writer.add_scalar(f"train/{key}", value, global_iter)
                        tqdm.write(
                            f"iter={global_iter}/{max_iters} loss={avg_loss:.6f} "
                            f"lr={lr:.8f} grad_norm={gn_str} "
                            f"step={'ok' if step_done else 'SKIPPED'}"
                        )

                # In-loop validation is disabled: rank-0-only eval stalls the other
                # DDP ranks and risks NCCL watchdog timeouts. Re-enable by
                # uncommenting the block below (the barrier already covers it).
                ran_eval = False  # global_iter % eval_cadence == 0
                ran_ckpt = global_iter % ckpt_cadence == 0
                # if ran_eval and _is_main_process():
                #     summary = run_eval(model, val_loader, device, output_dir, writer, global_iter, recipe["eval_mode"], "val")
                #     print(f"val@{global_iter}: {summary}")
                if ran_ckpt and _is_main_process():
                    _checkpoint_if_needed(output_dir, model, optimizer, scheduler, global_iter, config)
                # Hold every rank here until rank 0 finishes eval/checkpoint.
                # Without this, the non-main ranks race into the next collective
                # (all_reduce / DDP backward) while rank 0 is busy, and their NCCL
                # watchdogs time out -> the whole job dies.
                if world_size > 1 and (ran_eval or ran_ckpt):
                    dist.barrier()

                iteration += 1
                micro_step = 0
                window_ok = True
                window_loss_sum = 0.0
                window_loss_components = {}
            if iteration >= max_iters:
                break

        final_iter = max_iters
        if world_size > 1:
            dist.barrier()
        if _is_main_process():
            print(f"final_val@{final_iter}: {run_eval(model, val_loader, device, output_dir, writer, final_iter, recipe['eval_mode'], 'val_final')}")
            print(f"final_test@{final_iter}: {run_eval(model, test_loader, device, output_dir, writer, final_iter, recipe['eval_mode'], 'test_final')}")
    finally:
        pbar.close()
        _cleanup_distributed()
        if writer is not None:
            writer.flush()
            writer.close()


def _standalone_smoke():
    hyperparams, recipe = get_stage_hyperparams("stage1")
    model = SparseDrive(_prepare_hyperparams_for_local_build(hyperparams))
    model.init_weights()
    optimizer = build_optimizer(model, recipe["lr"], recipe["weight_decay"], recipe["backbone_lr_mult"])
    scheduler = CosineWithLinearWarmup(
        optimizer,
        max_iters=10,
        warmup_iters=2,
        warmup_ratio=recipe["warmup_ratio"],
        min_lr_ratio=recipe["min_lr_ratio"],
    )
    print("lr_schedule:")
    for iteration in (0, 1, 2, 5, 9):
        lrs = scheduler.step(iteration)
        print(f"iter {iteration}: " + ", ".join(f"{lr:.10f}" for lr in lrs))
    tensor = torch.tensor(1.0, requires_grad=True, device="cuda" if torch.cuda.is_available() else "cpu")
    loss = tensor * 2.0
    loss.backward()
    print(f"dummy_grad={float(tensor.grad.detach().cpu()):.1f}")


if __name__ == "__main__":
    _standalone_smoke()
