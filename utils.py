import torch
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