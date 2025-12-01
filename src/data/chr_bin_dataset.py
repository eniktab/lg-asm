"""Dataset utilities for chromosome/bin training data."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset


class ChrBinDataset(Dataset):
    """
    Simple dataset for:

      - seq : raw DNA sequence (string)
      - chr_idx
      - start_bin, end_bin
      - optional coarse_start_bin, coarse_end_bin
    """

    def __init__(
        self,
        seqs: List[str],
        chr_idxs: torch.Tensor,
        start_bins: torch.Tensor,
        end_bins: torch.Tensor,
        coarse_start_bins: Optional[torch.Tensor] = None,
        coarse_end_bins: Optional[torch.Tensor] = None,
    ) -> None:
        assert len(seqs) == chr_idxs.numel() == start_bins.numel() == end_bins.numel()
        if coarse_start_bins is not None:
            assert coarse_end_bins is not None
            assert coarse_start_bins.numel() == len(seqs)
            assert coarse_end_bins.numel() == len(seqs)
        self.seqs = seqs
        self.chr_idxs = chr_idxs.long()
        self.start_bins = start_bins.long()
        self.end_bins = end_bins.long()
        self.coarse_start_bins = coarse_start_bins.long() if coarse_start_bins is not None else None
        self.coarse_end_bins = coarse_end_bins.long() if coarse_end_bins is not None else None

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "seq": self.seqs[idx],
            "chr_idx": self.chr_idxs[idx],
            "start_bin": self.start_bins[idx],
            "end_bin": self.end_bins[idx],
        }
        if self.coarse_start_bins is not None and self.coarse_end_bins is not None:
            item["coarse_start_bin"] = self.coarse_start_bins[idx]
            item["coarse_end_bin"] = self.coarse_end_bins[idx]
        return item
