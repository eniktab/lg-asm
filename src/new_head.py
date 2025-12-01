"""Shim module preserving legacy imports for Hyena head training components."""
from __future__ import annotations

from src.data.chr_bin_dataset import ChrBinDataset
from src.models.attention_pooling import AttentionPooling
from src.models.hyena_head import DepthwiseSeparableConvBlock, HyenaChrBinHead
from src.training.runner import (
    EpochMetrics,
    HyenaLayerController,
    _acc_from_bucket,
    _bucket_to_int,
    _new_bucket,
    _run_epoch,
    _summarize_bin_distribution,
    _update_bin_bucket,
    build_hyena_backend,
    tokenize_batch,
    train_from_pt,
)

__all__ = [
    "ChrBinDataset",
    "AttentionPooling",
    "DepthwiseSeparableConvBlock",
    "HyenaChrBinHead",
    "HyenaLayerController",
    "EpochMetrics",
    "_acc_from_bucket",
    "_bucket_to_int",
    "_new_bucket",
    "_run_epoch",
    "_summarize_bin_distribution",
    "_update_bin_bucket",
    "build_hyena_backend",
    "tokenize_batch",
    "train_from_pt",
]
