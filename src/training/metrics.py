from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import torch

@dataclass(slots=True)
class EpochMetrics:
    loss: float
    start_acc1: float
    start_acc1_adj: float
    end_acc1: float
    end_acc1_adj: float
    coarse_start_acc1: Optional[float] = None
    coarse_start_acc1_adj: Optional[float] = None
    coarse_end_acc1: Optional[float] = None
    coarse_end_acc1_adj: Optional[float] = None

def _new_bucket(device: torch.device) -> Dict[str, torch.Tensor]:
    """On-device counter bucket to avoid per-step host syncs."""
    return {
        "ok": torch.zeros((), device=device, dtype=torch.long),
        "adj": torch.zeros((), device=device, dtype=torch.long),
        "wrong": torch.zeros((), device=device, dtype=torch.long),
    }

def _bucket_to_int(bucket_t: Dict[str, torch.Tensor]) -> Dict[str, int]:
    return {k: int(v.item()) for k, v in bucket_t.items()}

def _acc_from_bucket(bucket: Dict[str, int], n: int) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    acc1 = bucket["ok"] / n
    acc1_adj = (bucket["ok"] + bucket["adj"]) / n
    return acc1, acc1_adj

def _update_bin_bucket(
    bucket: Dict[str, torch.Tensor],
    true_bins: torch.Tensor,
    pred_bins: torch.Tensor,
) -> None:
    """Vectorized, on-device accuracy bucket update."""
    diff = (pred_bins - true_bins).abs()
    bucket["ok"] += (diff == 0).sum()
    bucket["adj"] += (diff == 1).sum()
    bucket["wrong"] += (diff > 1).sum()

def _summarize_bin_distribution(
    name: str,
    bins: torch.Tensor,
    num_bins_global: int,
    top_k: int = 10,
) -> None:
    bins = bins.view(-1)
    print(f"\n[DEBUG] Bin distribution summary ({name})")
    print(f"  num_samples           : {bins.numel()}")
    print(f"  num_bins_global       : {num_bins_global}")
    print(f"  min(bin) / max(bin)   : {int(bins.min())} / {int(bins.max())}")

    bad_low = (bins < 0).nonzero(as_tuple=False)
    bad_high = (bins >= num_bins_global).nonzero(as_tuple=False)
    print(f"  out_of_range <0        : {bad_low.size(0)}")
    print(f"  out_of_range >=num_bins: {bad_high.size(0)}")

    unique, counts = torch.unique(bins, return_counts=True)
    coverage = unique.numel() / num_bins_global if num_bins_global > 0 else float('nan')
    print(f"  unique_bins           : {unique.numel()}")
    print(f"  coverage_fraction     : {coverage:.4f}")

    if counts.numel() > 0:
        k = min(top_k, counts.numel())
        top_counts, idx = torch.topk(counts, k=k, largest=True)
        top_bins = unique[idx]
        print(f"  Top {k} bins by count (bin_idx: count):")
        for b, c in zip(top_bins.tolist(), top_counts.tolist()):
            print(f"    bin {b:6d}: {c}")
    print("")
