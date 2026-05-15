import numpy as np
import torch
from configs.sparsedrive_small_stage1 import build

B = 1
N_CAM = 6
H, W = 704, 256

model = build()
model.init_weights()
model.eval()

data = dict(
    img=torch.randn(B, N_CAM, 3, H, W),
    projection_mat=torch.randn(B, N_CAM, 3, 4),
    image_wh=torch.full((B, N_CAM, 2), fill_value=1.0).float(),
    timestamp=torch.zeros(B),
    img_metas=[
        {"T_global": np.eye(4, dtype=np.float32), "T_global_inv": np.eye(4, dtype=np.float32)}
        for _ in range(B)
    ],
)

with torch.no_grad():
    out = model(**data)

print("smoke test passed, output keys:", list(out[0].keys()) if isinstance(out, list) else list(out.keys()))
