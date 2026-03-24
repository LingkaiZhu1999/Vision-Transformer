import torch
from einops import rearrange, repeat
import torch.nn as nn
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, d_model, dff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.linear2 = nn.Linear(dff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.activation(self.linear1(x)))
        x = self.dropout(self.linear2(x))
        return x


class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
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
        self.dropout_rate = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = rearrange(self.q_proj(x), "... seq (num_heads d_q) -> ... num_heads seq d_q", num_heads=self.num_heads)
        K = rearrange(self.k_proj(x), "... seq (num_heads d_k) -> ... num_heads seq d_k", num_heads=self.num_heads)
        V = rearrange(self.v_proj(x), "... seq (num_heads d_v) -> ... num_heads seq d_v", num_heads=self.num_heads)
        attention = F.scaled_dot_product_attention(Q, K, V, is_causal=False,
        dropout_p=self.dropout_rate if self.training else 0.0)
        attention = rearrange(attention, "... num_heads seq d_v -> ... seq (num_heads d_v)")
        output = self.o_proj(attention)
        return output
    

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.multihead_self_att = MultiheadSelfAttention(d_model, num_heads, dropout)
        self.mlp = MLP(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.multihead_self_att(self.layernorm1(x)))
        x = x + self.mlp(self.layernorm2(x))
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
        dropout=0.1,
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
            [TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.layernorm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(0.1)


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
        x = self.dropout(x)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        x = self.layernorm(x)
        cls_rep = x[:, 0]
        logits = self.head(cls_rep)
        return logits

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(8, 3, 224, 224, device=device)
    transformer = Transformer_VM(
        image_size=224,
        patch_size=16,
        in_channels=3,
        d_model=128,
        num_heads=8,
        d_ff=256,
        num_classes=100,
        num_layers=4,
    ).to(device)
    y = transformer(x)
    print(y.shape)
    
