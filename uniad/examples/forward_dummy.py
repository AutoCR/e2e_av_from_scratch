from pathlib import Path
import random
import sys
import warnings

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from uniad import build_uniad, load_checkpoint


SEED = 20240523
FAKE_BEV_SHAPE = (20, 20)
NUM_FAKE_MODES = 6


def _repo_root() -> Path:
    return _REPO_ROOT


def _seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _fake_planning_inputs(planning_head, generator: torch.Generator) -> dict[str, torch.Tensor]:
    bev_h, bev_w = FAKE_BEV_SHAPE
    batch_size = 1
    embed_dims = planning_head.navi_embed.embedding_dim
    num_decoder_layers = 3

    planning_head.bev_h = bev_h
    planning_head.bev_w = bev_w

    return {
        "bev_embed": torch.randn(bev_h * bev_w, batch_size, embed_dims, generator=generator),
        "occ_mask": torch.zeros(batch_size, 5, 1, bev_h, bev_w, dtype=torch.bool),
        "bev_pos": torch.randn(batch_size, embed_dims, bev_h, bev_w, generator=generator),
        "sdc_traj_query": torch.randn(
            num_decoder_layers,
            batch_size,
            NUM_FAKE_MODES,
            embed_dims,
            generator=generator,
        ),
        "sdc_track_query": torch.randn(batch_size, embed_dims, generator=generator),
        "command": torch.tensor([1], dtype=torch.long),
    }


def _shape_summary(outputs: dict[str, torch.Tensor]) -> str:
    return ", ".join(f"{name}={tuple(value.shape)}" for name, value in outputs.items())


def main() -> None:
    warnings.filterwarnings("ignore", message=r"The arguments `ffn_?.*`", category=UserWarning)
    warnings.filterwarnings("ignore", message=r"The arguments `feedforward_channels`.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=r"torch\.meshgrid:.*", category=UserWarning)

    repo_root = _repo_root()
    checkpoint_path = repo_root / "model_weights" / "uniad" / "uniad_base_e2e.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"UniAD checkpoint not found: {checkpoint_path}")

    generator = _seed_everything(SEED)

    model = build_uniad(dummy_motion_anchors=True)
    checkpoint_result = load_checkpoint(model, str(checkpoint_path), strict=False)
    model.eval()

    native_bev_shape = (model.planning_head.bev_h, model.planning_head.bev_w)
    fake_inputs = _fake_planning_inputs(model.planning_head, generator)

    with torch.no_grad():
        outputs = model.planning_head(
            fake_inputs["bev_embed"],
            fake_inputs["occ_mask"],
            fake_inputs["bev_pos"],
            fake_inputs["sdc_traj_query"],
            fake_inputs["sdc_track_query"],
            fake_inputs["command"],
        )

    for name, value in outputs.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Non-finite values in planning output: {name}")

    print("UniAD checkpoint dummy forward succeeded.")
    print(f"checkpoint: {checkpoint_path}")
    print(
        "checkpoint_load: "
        f"missing={len(checkpoint_result['missing_keys'])} "
        f"unexpected={len(checkpoint_result['unexpected_keys'])}"
    )
    print(
        "fake_path: planning_head random BEV/query tensors; "
        "full camera path skipped "
        f"({model.img_backbone.__class__.__name__}/{model.img_neck.__class__.__name__})"
    )
    print(f"bev_shape: native={native_bev_shape} fake={FAKE_BEV_SHAPE}")
    print(f"outputs: {_shape_summary(outputs)}")
    print(f"sdc_traj_sum: {outputs['sdc_traj'].sum().item():.6f}")


if __name__ == "__main__":
    main()
