import math
import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    """
    Learned attention pooling over sequence positions.

    For each head h, we learn a query vector q_h ∈ R^H.

    x    : [B, L, H]
    mask : [B, L] in {0,1}

    Returns pooled representation [B, H] that can focus on short driver
    motifs in long windows.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.query = nn.Parameter(torch.randn(num_heads, hidden_dim))
        self.out_proj = nn.Linear(hidden_dim * num_heads, hidden_dim)
        self.register_buffer(
            "_scale",
            torch.tensor(1.0 / math.sqrt(hidden_dim)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x    : [B, L, H]
        mask : [B, L] float {0,1}
        """
        B, L, H = x.shape
        k = self.key_proj(x)  # [B, L, H]

        # scores: [B, L, num_heads]
        scores = torch.matmul(k, self.query.t()) * self._scale

        # mask: 0 -> large negative
        mask_expanded = (mask == 0).unsqueeze(-1)  # [B, L, 1]
        very_neg = -1e4 if x.dtype in (torch.float16, torch.bfloat16) else -1e9
        scores = scores.masked_fill(mask_expanded, very_neg)

        attn = scores.softmax(dim=1)  # [B, L, num_heads]

        # pooled per head: [B, num_heads, H]
        pooled = torch.einsum("bln,blh->bnh", attn, x)
        pooled = pooled.reshape(B, self.num_heads * H)
        pooled = self.out_proj(pooled)  # [B, H]
        return pooled


class DepthwiseSeparableConvBlock(nn.Module):
    """
    1D depthwise-separable convolution with a residual connection.

    Expects input of shape [B, H, L]. Internally applies:
        depthwise Conv1d (groups=H) -> pointwise Conv1d (1x1) -> GELU -> Dropout
    and adds the input back (residual).
    """
    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
            groups=hidden_dim,
        )
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, L]
        residual = x
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.act(out)
        out = self.dropout(out)
        return out + residual
