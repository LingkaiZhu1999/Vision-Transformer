import torch
from einops import rearrange, repeat
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, dff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.dff = dff
        self.linear1 = nn.Linear(d_model, dff)
        self.linear2 = nn.Linear(dff, d_model)
        self.linear3 = nn.Linear(d_model, dff)

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = x1 * torch.sigmoid(x1)
        x2 = self.linear3(x)
        x3 = x1 * x2
        x3 = self.linear2(x3)
        return x3

class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.q_proj = nn.Linear(d_model, num_heads * self.d_k)
        self.k_proj = nn.Linear(d_model, num_heads * self.d_k)
        self.v_proj = nn.Linear(d_model, num_heads * self.d_v)
        self.o_proj = nn.Linear(d_model, num_heads * self.d_v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = rearrange(self.q_proj(x), "... seq (num_heads d_q) -> ... num_heads seq d_q", num_heads=self.num_heads)
        K = rearrange(self.k_proj(x), "... seq (num_heads d_k) -> ... num_heads seq d_k", num_heads=self.num_heads)
        V = rearrange(self.v_proj(x), "... seq (num_heads d_v) -> ... num_heads seq d_v", num_heads=self.num_heads)
        attention = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
        attention = rearrange(attention, "... num_heads seq d_v -> ... seq (num_heads d_v)")
        output = self.o_proj(attention)
        return output
    

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.rmsnorm1 = nn.RMSNorm(d_model)
        self.rmsnorm2 = nn.RMSNorm(d_model)
        self.multihead_self_att = MultiheadSelfAttention(d_model, num_heads)
        self.swiglu = SwiGLU(d_model, d_ff)

    def forward(self, x):
        x = x + self.multihead_self_att(self.rmsnorm1(x))
        x = x + self.swiglu(self.rmsnorm2(x))
        return x

class Transformer_VM(torch.nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        in_channels,
        d_model,
        num_heads,
        d_ff,
        num_classes,
        num_layers,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.patch_size = patch_size
        self.patch_dim = in_channels * patch_size * patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.proj = nn.Linear(self.patch_dim, d_model)
        self.positional_encoding = nn.Parameter(
            torch.nn.init.trunc_normal_(torch.empty(1, self.num_patches + 1, d_model))
        )
        self.class_token = nn.Parameter(
            torch.nn.init.trunc_normal_(torch.empty(1, 1, d_model))
        )
        self.transformer_blocks = torch.nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.rmsnorm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)


    def forward(self, x):
        if x.dim() == 4:
            x = rearrange(
                x,
                "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                ph=self.patch_size,
                pw=self.patch_size,
            )
        elif x.dim() != 3:
            raise ValueError("Input x must be [B, C, H, W] or [B, N, patch_dim]")

        x = self.proj(x)
        batch_size = x.shape[0]
        cls_tokens = repeat(self.class_token, "1 1 d -> b 1 d", b=batch_size)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.positional_encoding[:, :x.shape[1]]

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        x = self.rmsnorm(x)
        cls_rep = x[:, 0]
        logits = self.head(cls_rep)
        return logits

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(8, 3, 32, 32, device=device)
    transformer = Transformer_VM(
        image_size=32,
        patch_size=4,
        in_channels=3,
        d_model=128,
        num_heads=8,
        d_ff=256,
        num_classes=100,
        num_layers=4,
    ).to(device)
    y = transformer(x)
    print(y.shape)
    
