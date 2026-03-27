import torch
import math
from einops import rearrange

def image_to_patches(image: torch.Tensor, patch_size: int) -> torch.Tensor:
	channels, height, width = image.shape
	if height % patch_size != 0 or width % patch_size != 0:
		raise ValueError("Image height and width must be divisible by patch_size.")

	patches = rearrange(
		image,
		"c (h ph) (w pw) -> (h w) (c ph pw)",
		ph=patch_size,
		pw=patch_size,
	)
	return patches

def learning_rate_schedule(t, lr_max, lr_min, t_warm_up, total_steps):
    if total_steps <= 0:
        return lr_min

    t = min(t, total_steps)
    warmup_steps = max(0, min(t_warm_up, total_steps))

    if warmup_steps > 0 and t < warmup_steps:
        return t / warmup_steps * lr_max

    if t >= total_steps:
        return lr_min

    cosine_steps = max(1, total_steps - warmup_steps)
    progress = (t - warmup_steps) / cosine_steps
    progress = min(max(progress, 0.0), 1.0)
    return lr_min + 0.5 * (1 + math.cos(progress * math.pi)) * (lr_max - lr_min)
