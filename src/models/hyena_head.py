"""Hyena-based classifier head components and helper blocks."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.attention_pooling import AttentionPooling


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


class HyenaChrBinHead(nn.Module):
    """
    Multi-task head on top of Hyena hidden states:

      - chromosome classification
      - fine start-bin classification
      - fine end-bin classification
      - optional coarse start/end-bin classification

    Optimizations:
      - Depthwise separable conv branches.
      - Optional temporal downsampling before conv trunk
        (conv_downsample > 1) to reduce FLOPs on very long sequences.

    Bin losses:
      - Fine start/end use neighbor-soft targets over [i-1, i, i+1] to
        better match the ±1-bin success metric.
      - An additional distance-aware term penalizes the (normalized)
        expected bin index when far from the true bin.
    """

    def __init__(
        self,
        d_model: int,
        num_chromosomes: int,
        num_bins: int,
        hidden_dim: int = 768,
        conv_kernel_sizes: Tuple[int, ...] = (3, 7, 15),
        conv_dilations: Tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.1,
        label_smoothing: float = 0.0,
        *,
        num_coarse_bins: Optional[int] = None,
        w_chr: float = 0.0,
        w_start: float = 1.0,
        w_end: float = 1.0,
        w_coarse_start: float = 0.0,
        w_coarse_end: float = 0.0,
        w_start_dist: float = 0.0,
        w_end_dist: float = 0.0,
        conv_downsample: int = 4,
    ) -> None:
        super().__init__()

        assert len(conv_kernel_sizes) == len(conv_dilations), \
            "conv_kernel_sizes and conv_dilations must have same length"

        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_chromosomes = num_chromosomes
        self.num_bins = num_bins
        self.num_coarse_bins = num_coarse_bins
        self.label_smoothing = label_smoothing

        self.w_chr = float(w_chr)
        self.w_start = float(w_start)
        self.w_end = float(w_end)
        self.w_coarse_start = float(w_coarse_start)
        self.w_coarse_end = float(w_coarse_end)
        self.w_start_dist = float(w_start_dist)
        self.w_end_dist = float(w_end_dist)

        # Neighbor-soft label parameters for fine bins (±1-bin metric)
        self.neighbor_center_p = 0.8
        self.neighbor_neighbor_p = 0.1

        # Precompute bin indices [0 .. num_bins-1] for distance-aware loss,
        # and normalize by (num_bins-1) to keep the Huber term well-scaled.
        self.register_buffer(
            "bin_indices",
            torch.arange(num_bins, dtype=torch.float32).unsqueeze(0),
            persistent=False,
        )
        self.bin_index_scale = float(max(1, num_bins - 1))

        # Input projection
        self.in_ln = nn.LayerNorm(d_model)
        if hidden_dim == d_model:
            # Avoid an unnecessary projection when dimensions already match.
            self.in_proj = nn.Identity()
        else:
            self.in_proj = nn.Linear(d_model, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # -------------------------------------------------------
        # Efficient conv trunk
        # -------------------------------------------------------
        self.conv_downsample = max(1, int(conv_downsample))
        if self.conv_downsample > 1:
            self.downsample = nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=self.conv_downsample,
                padding=1,
                groups=hidden_dim,
            )
        else:
            self.downsample = None

        branches = []
        for ks, dil in zip(conv_kernel_sizes, conv_dilations):
            branches.append(
                DepthwiseSeparableConvBlock(
                    hidden_dim=hidden_dim,
                    kernel_size=ks,
                    dilation=dil,
                    dropout=dropout,
                )
            )
        self.branches = nn.ModuleList(branches)
        self.fuse_conv = nn.Conv1d(hidden_dim * len(branches), hidden_dim, kernel_size=1)
        self.fuse_ln = nn.LayerNorm(hidden_dim)

        # Attention pooling
        self.attn_pool = AttentionPooling(hidden_dim, num_heads=4)
        head_in_dim = hidden_dim

        def make_head(out_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(head_in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )

        # Fine heads
        self.chr_head = make_head(num_chromosomes)
        self.start_head = make_head(num_bins)
        self.end_head = make_head(num_bins)

        # Optional coarse heads
        if num_coarse_bins is not None and num_coarse_bins > 0:
            self.coarse_start_head = make_head(num_coarse_bins)
            self.coarse_end_head = make_head(num_coarse_bins)
        else:
            self.coarse_start_head = None
            self.coarse_end_head = None

    def forward(
        self,
        hidden_states: torch.Tensor,            # [B, L, D]
        attention_mask: Optional[torch.Tensor], # [B, L]
        chr_labels: Optional[torch.Tensor] = None,
        start_bin_labels: Optional[torch.Tensor] = None,
        end_bin_labels: Optional[torch.Tensor] = None,
        coarse_start_labels: Optional[torch.Tensor] = None,
        coarse_end_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        B, L, D = hidden_states.shape
        device = hidden_states.device

        x = self.in_ln(hidden_states)
        x = self.in_proj(x)  # [B, L, H]
        x = self.dropout(x)

        if attention_mask is None:
            mask = torch.ones(B, L, device=device, dtype=x.dtype)
        else:
            # assume device is already correct; just cast dtype when needed
            mask = attention_mask
            if mask.dtype != x.dtype:
                mask = mask.to(dtype=x.dtype)

        # Conv trunk
        x_c = x.transpose(1, 2).contiguous()  # [B, H, L]

        if self.conv_downsample > 1 and self.downsample is not None:
            x_c = self.downsample(x_c)  # [B, H, L']
            mask = F.max_pool1d(
                mask.unsqueeze(1),      # [B, 1, L]
                kernel_size=self.conv_downsample,
                stride=self.conv_downsample,
            ).squeeze(1)               # [B, L']

        branch_outs = [branch(x_c) for branch in self.branches]  # each [B, H, L']
        y_cat = torch.cat(branch_outs, dim=1)                    # [B, H * n_branches, L']
        y_fused = self.fuse_conv(y_cat)                          # [B, H, L']
        y_fused = y_fused + x_c                                  # residual
        y_fused = y_fused.transpose(1, 2).contiguous()           # [B, L', H]
        y_fused = self.fuse_ln(y_fused)
        y_fused = self.dropout(y_fused)

        # Attention pooling
        pooled = self.attn_pool(y_fused, mask)  # [B, H]

        logits_chr = self.chr_head(pooled)
        logits_start = self.start_head(pooled)
        logits_end = self.end_head(pooled)

        logits_coarse_start = logits_coarse_end = None
        if self.coarse_start_head is not None:
            logits_coarse_start = self.coarse_start_head(pooled)
            logits_coarse_end = self.coarse_end_head(pooled)

        out: Dict[str, Any] = {
            "logits_chr": logits_chr,
            "logits_start_bin": logits_start,
            "logits_end_bin": logits_end,
        }
        if logits_coarse_start is not None:
            out["logits_coarse_start_bin"] = logits_coarse_start
            out["logits_coarse_end_bin"] = logits_coarse_end

        # ---------------------------------------------------
        # Loss helpers
        # ---------------------------------------------------
        def soft_ce(
            logits: Optional[torch.Tensor],
            labels: Optional[torch.Tensor],
            weight: float,
            num_bins: int,
        ) -> Optional[torch.Tensor]:
            """
            Neighbor-soft cross-entropy using gather instead of building a
            dense [B, num_bins] target matrix.

            We give probability mass to [bin-1, bin, bin+1] to better align
            training with the ±1-bin evaluation metric.
            """
            if logits is None or labels is None or weight == 0.0:
                return None

            if logits.ndim != 2:
                raise ValueError(f"soft_ce expects [B, C] logits, got shape {tuple(logits.shape)}")

            B_local, C = logits.shape
            # Safety: keep indices within the actual logits size.
            num_bins_eff = min(num_bins, C)

            labels_long = labels.long()
            device_l = logits.device

            # Clamp the central index to a valid range
            center_idx = labels_long.clamp(min=0, max=num_bins_eff - 1)

            arange_b = torch.arange(B_local, device=device_l)

            logp = F.log_softmax(logits, dim=-1)

            # Central position
            center_logp = logp[arange_b, center_idx]

            # Left neighbor
            left_idx = labels_long - 1
            left_valid = (left_idx >= 0) & (left_idx < num_bins_eff)
            left_idx = left_idx.clamp(min=0, max=num_bins_eff - 1)
            left_logp = logp[arange_b, left_idx]

            # Right neighbor
            right_idx = labels_long + 1
            right_valid = (right_idx >= 0) & (right_idx < num_bins_eff)
            right_idx = right_idx.clamp(min=0, max=num_bins_eff - 1)
            right_logp = logp[arange_b, right_idx]

            # Neighbor-soft targets: [0.1, 0.8, 0.1]
            target_center = self.neighbor_center_p
            target_neighbor = self.neighbor_neighbor_p

            # label smoothing on the central bin: (1 - 2*neighbor - smooth)
            if self.label_smoothing > 0:
                target_center = target_center - self.label_smoothing
                target_neighbor = target_neighbor + self.label_smoothing / (num_bins_eff - 1)

            # Combine log-probabilities with weights, masking invalid neighbors
            loss_vec = torch.zeros_like(center_logp)
            loss_vec += target_center * center_logp
            loss_vec += torch.where(left_valid, target_neighbor * left_logp, torch.zeros_like(left_logp))
            loss_vec += torch.where(right_valid, target_neighbor * right_logp, torch.zeros_like(right_logp))

            loss = -(loss_vec.sum()) / B_local
            return weight * loss

        def ce_loss(
            logits: Optional[torch.Tensor],
            labels: Optional[torch.Tensor],
            weight: float,
            smoothing: Optional[float] = None,
        ) -> Optional[torch.Tensor]:
            if logits is None or labels is None or weight == 0.0:
                return None
            if smoothing is None:
                smoothing = self.label_smoothing
            if smoothing > 0:
                return weight * F.cross_entropy(logits, labels, label_smoothing=smoothing)
            return weight * F.cross_entropy(logits, labels)

        def distance_loss(
            logits: torch.Tensor,
            labels: Optional[torch.Tensor],
            weight: float,
        ) -> Optional[torch.Tensor]:
            if labels is None or weight == 0.0:
                return None

            if logits.ndim != 2:
                raise ValueError(f"distance_loss expects [B, C] logits, got shape {tuple(logits.shape)}")

            if logits.size(1) != self.num_bins:
                raise ValueError(
                    f"distance_loss expects logits dim1={self.num_bins}, "
                    f"got {logits.size(1)}"
                )

            B_local = logits.size(0)
            labels_long = labels.long().clamp(min=0, max=self.num_bins - 1)

            probs = logits.softmax(dim=-1)
            expected_idx = (probs * self.bin_indices.to(logits.device)).sum(dim=-1)
            expected_idx = expected_idx / self.bin_index_scale

            target = labels_long.to(dtype=expected_idx.dtype) / self.bin_index_scale

            loss = F.huber_loss(expected_idx, target, reduction="mean", delta=1.0)
            return weight * loss

        loss_chr = ce_loss(logits_chr, chr_labels, self.w_chr)
        loss_start = soft_ce(logits_start, start_bin_labels, self.w_start, self.num_bins)
        loss_end = soft_ce(logits_end, end_bin_labels, self.w_end, self.num_bins)

        loss_start_dist = distance_loss(
            logits_start,
            start_bin_labels,
            self.w_start_dist,
        )
        loss_end_dist = distance_loss(
            logits_end,
            end_bin_labels,
            self.w_end_dist,
        )

        # Keep coarse bins (if any) as hard CE
        loss_coarse_start = ce_loss(
            logits_coarse_start,
            coarse_start_labels,
            self.w_coarse_start,
        )
        loss_coarse_end = ce_loss(
            logits_coarse_end,
            coarse_end_labels,
            self.w_coarse_end,
        )

        losses = [
            l
            for l in (
                loss_chr,
                loss_start,
                loss_end,
                loss_coarse_start,
                loss_coarse_end,
                loss_start_dist,
                loss_end_dist,
            )
            if l is not None
        ]
        total_loss = torch.stack(losses).sum() if losses else None

        out.update(
            {
                "loss": total_loss,
                "loss_chr": loss_chr,
                "loss_start_bin": loss_start,
                "loss_end_bin": loss_end,
                "loss_coarse_start_bin": loss_coarse_start,
                "loss_coarse_end_bin": loss_coarse_end,
                "loss_start_bin_dist": loss_start_dist,
                "loss_end_bin_dist": loss_end_dist,
            }
        )
        return out
