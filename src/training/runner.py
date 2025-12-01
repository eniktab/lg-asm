"""Training utilities and loops for Hyena-based chromosome/bin models."""
from __future__ import annotations

import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/g/data/te53/en9803/workspace/sync/ANU/graph_genomic_data/'])

import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
_HAS_CUDAGRAPH_MARK = hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.HyenaBackend import HyenaBackend
from src.dna_tokenizer import DNATok
from src.data.chr_bin_dataset import ChrBinDataset
from src.models.hyena_head import HyenaChrBinHead

# Optional TensorBoard logging
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    SummaryWriter = None  # type: ignore

# Optional profiler
try:
    from torch.profiler import (
        ProfilerActivity,
        profile,
        record_function,
        schedule,
        tensorboard_trace_handler,
    )
    _HAVE_PROFILER = True
except Exception:  # pragma: no cover
    profile = record_function = schedule = tensorboard_trace_handler = None  # type: ignore
    ProfilerActivity = None  # type: ignore
    _HAVE_PROFILER = False


# ================================================================
# Metrics helpers
# ================================================================
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


# ================================================================
# Hyena utilities
# ================================================================
def build_hyena_backend(
    model_dir: Path,
    model_name: str = "hyenadna-large-1m-seqlen-hf",
) -> HyenaBackend:
    backend = HyenaBackend(
        model_name=model_name,
        model_dir=str(model_dir),
        pooling="none",
        normalize=False,
        offline=True,
        prefer_cuda=True,
    )
    return backend


def tokenize_batch(
    backend: HyenaBackend,
    seqs: List[str],
    fast_tokenizer_fn: Optional[Callable[[List[str]], Dict[str, torch.Tensor]]] = None,
    *,
    max_tokens: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Tokenization + truncation.

    If max_tokens is not None, sequences are truncated on the Python side
    *before* calling DNATok or the HF/Hyena tokenizer.

    If fast_tokenizer_fn is provided (e.g. DNATok / KevTok batch tokenizer),
    it must return:

        {
          "input_ids": LongTensor [B, T],
          "attention_mask": Long/Bool Tensor [B, T]
        }
    """
    # Hard cap on raw sequence length
    if max_tokens is not None:
        seqs = [s[:max_tokens] for s in seqs]

    if fast_tokenizer_fn is not None:
        enc = fast_tokenizer_fn(seqs)
        if "input_ids" not in enc or "attention_mask" not in enc:
            raise ValueError("fast_tokenizer_fn must return 'input_ids' and 'attention_mask'.")
        return enc

    # HF / Hyena tokenizer path
    if backend.max_length is not None:
        max_len = backend.max_length
        if max_tokens is not None:
            max_len = min(max_len, max_tokens)
    else:
        max_len = max_tokens

    pad_to_multiple = HyenaBackend._choose_pad_multiple(max_len)
    enc = backend.tokenizer(
        seqs,
        padding="longest",
        truncation=True,
        max_length=max_len,
        pad_to_multiple_of=pad_to_multiple,
        add_special_tokens=False,
        return_tensors="pt",
    )
    return enc


# ================================================================
# Gradual Hyena unfreezing controller
# ================================================================
class HyenaLayerController:
    """
    Controls gradual unfreezing of Hyena backbone layers.

    Assumes that `model` is the Hyena *backbone* (without the classifier head).
    If you pass a full model (backbone + head), the head parameters will also
    be affected unless you manage them separately.

    Behavior:

    - If freeze_backbone is False:
        * All model parameters are trainable from epoch 0.
        * model.train() is called.

    - If freeze_backbone is True:
        * epochs < start_epoch:
            - All backbone parameters are frozen.
            - Backbone is put in eval() mode (no BN/Dropout updates).
        * epochs in [start_epoch, end_epoch]:
            - Linearly increase the number of unfrozen *top* layers
              up to max_unfrozen_layers.
        * epochs > end_epoch:
            - Keep max_unfrozen_layers unfrozen.

    Implementation details:

    - We try a list of common attribute paths first
      ("encoder.layers", "backbone.layers", "backbone.blocks", "layers",
       "blocks", "h") and use the first one that looks like a sequence of
      transformer blocks.

    - If that fails, we auto-detect a suitable ModuleList/Sequential by
      scanning named_modules(), scoring candidates by name and length.

    - When partially frozen:
        * Entire backbone is put in eval() mode and frozen.
        * The last N layers (N = n_unfrozen) are switched back to train()
          and have requires_grad=True, so only they update weights and
          BatchNorm / Dropout behave in training mode there.

    NOTE:
        To keep classifier heads fully trainable, either:
        (a) pass only the backbone into this controller, or
        (b) re-enable head modules' train() / requires_grad=True outside
            this controller after calling apply().
    """

    def __init__(
        self,
        model: nn.Module,
        freeze_backbone: bool,
        start_epoch: int,
        end_epoch: int,
        max_unfrozen_layers: int,
    ) -> None:
        self.model = model
        self.freeze_backbone = freeze_backbone
        self.start_epoch = start_epoch
        # Ensure end_epoch is not earlier than start_epoch
        self.end_epoch = max(start_epoch, end_epoch)
        # Negative values make no sense; clamp to >= 0
        self.max_unfrozen_layers = max(0, max_unfrozen_layers)

        self.layers: Optional[List[nn.Module]] = None

        # ------------------------------------------------------------
        # 1) Try explicit attribute paths first (most robust if they exist)
        # ------------------------------------------------------------
        candidates = (
            "encoder.layers",
            "backbone.layers",
            "backbone.blocks",
            "layers",
            "blocks",
            "h",
        )

        for attr_path in candidates:
            mod: Any = model
            ok = True
            for sub in attr_path.split("."):
                if not hasattr(mod, sub):
                    ok = False
                    break
                mod = getattr(mod, sub)
            if not ok:
                continue

            if isinstance(mod, (nn.ModuleList, nn.Sequential, list, tuple)):
                self.layers = list(mod)
                print(
                    f"[INFO] HyenaLayerController using explicit path '{attr_path}' "
                    f"with {len(self.layers)} layers."
                )
                break

        # ------------------------------------------------------------
        # 2) Fallback: auto-detect a good container of layers
        # ------------------------------------------------------------
        if self.layers is None:
            best_candidate: Optional[Tuple[str, nn.Module, int, float]] = None
            best_score = -1.0

            for name, module in self.model.named_modules():
                if isinstance(module, (nn.ModuleList, nn.Sequential)):
                    length = len(module)
                    if length < 2:
                        continue

                    lname = name.lower()
                    score = 0.0

                    # Prefer names that look like layer / block / stage containers
                    if any(key in lname for key in ("layers", "layer", "blocks", "block", "stage", "h")):
                        score += 10.0
                    # Prefer encoder/backbone/transformer-style containers
                    if any(key in lname for key in ("encoder", "backbone", "transformer")):
                        score += 5.0

                    # Slight bonus for more layers (capped)
                    score += min(float(length), 100.0)

                    if score > best_score:
                        best_score = score
                        best_candidate = (name, module, length, score)

            if best_candidate is not None:
                name, module, length, score = best_candidate
                self.layers = list(module)
                print(
                    f"[INFO] HyenaLayerController auto-detected layer container '{name}' "
                    f"with {length} layers (score={score:.1f})."
                )

        if self.layers is None:
            print(
                "[WARN] HyenaLayerController could not find a transformer-layer container; "
                "will treat the entire backbone as a single unit for freezing/unfreezing."
            )

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def apply(self, epoch: int) -> None:
        """
        Apply the gradual unfreezing policy for a given epoch.

        This should typically be called once per epoch before running the
        training loop for that epoch. If your training code calls model.train()
        / model.eval() inside the epoch loop, call `apply()` *after* setting
        the global mode to ensure the partial train/eval flags on layers
        are not overwritten.
        """
        # --------------------------------------------------------
        # Case 0: no freezing strategy requested; ensure everything trains
        # --------------------------------------------------------
        if not self.freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = True
            self.model.train()
            return

        # --------------------------------------------------------
        # Compute how many layers should be unfrozen this epoch
        # --------------------------------------------------------
        if epoch < self.start_epoch:
            n_unfrozen = 0
        elif epoch >= self.end_epoch:
            n_unfrozen = self.max_unfrozen_layers
        else:
            # Linear schedule from start_epoch to end_epoch (inclusive)
            total_steps = max(1, self.end_epoch - self.start_epoch + 1)
            progress = (epoch - self.start_epoch + 1) / total_steps
            n_unfrozen = int(round(progress * self.max_unfrozen_layers))

        # Handle edge cases and clamp to valid range if we know the layers
        if self.layers is not None:
            n_unfrozen = max(0, min(n_unfrozen, len(self.layers)))

        # --------------------------------------------------------
        # No layer container found; fallback to freezing everything then
        # unfreezing all at once when n_unfrozen > 0
        # --------------------------------------------------------
        if self.layers is None:
            if n_unfrozen <= 0:
                for p in self.model.parameters():
                    p.requires_grad = False
                self.model.eval()
                print(
                    f"[INFO] Epoch {epoch}: backbone fully frozen "
                    f"(no layer container detected)."
                )
            else:
                # Typically you'd use max_unfrozen_layers = total layers, but
                # since we don't know, interpret any n_unfrozen > 0 as "ok,
                # then unfreeze everything".
                for p in self.model.parameters():
                    p.requires_grad = True
                self.model.train()
                print(
                    f"[INFO] Epoch {epoch}: backbone fully unfrozen "
                    f"(no layer container detected; treating as single block)."
                )
            return

        # --------------------------------------------------------
        # Normal case: we have a list of backbone layers.
        # Strategy:
        #   1. Put the entire backbone in eval() and freeze all params.
        #   2. Unfreeze last `n_unfrozen` layers and set them to train().
        # --------------------------------------------------------
        # 1) Base state: everything frozen, eval mode to avoid BN drift
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # 2) Unfreeze specific layers
        if n_unfrozen > 0:
            # Unfreeze last n_unfrozen layers (top of the stack)
            for layer in self.layers[-n_unfrozen:]:
                layer.train()  # enable Dropout/BN updates here
                for p in layer.parameters():
                    p.requires_grad = True

        # Note: if `model` is backbone-only, classifier heads are unaffected.
        # If `model` includes the head, remember to re-enable the head
        # explicitly in your training loop if needed.

        print(
            f"[INFO] Epoch {epoch}: unfreezing last {n_unfrozen} Hyena layers "
            f"(max_unfrozen_layers={self.max_unfrozen_layers}, "
            f"total_layers={len(self.layers)})."
        )


# ================================================================
# Epoch loops
# ================================================================
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


def _run_epoch(
    *,
    epoch: int,
    model_hyena: nn.Module,
    head: HyenaChrBinHead,
    backend_hyena: HyenaBackend,
    dataloader: DataLoader,
    device: torch.device,
    hyena_hidden_layer_idx: int,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.amp.GradScaler],
    fast_tokenizer_fn: Optional[Callable[[List[str]], Dict[str, torch.Tensor]]],
    accum_steps: int,
    is_train: bool,
    use_amp: bool,
    amp_dtype: torch.dtype,
    profiler_ctx: Any = None,
    max_tokens: Optional[int] = None,
    use_compile: bool = False,
    distributed: bool = False,
) -> Tuple[EpochMetrics, int]:
    if is_train and optimizer is None:
        raise ValueError("Optimizer must be provided for training epoch.")
    if accum_steps < 1:
        raise ValueError(f"accum_steps must be >= 1, got {accum_steps}")

    # If the head is wrapped in DistributedDataParallel, unwrap it
    # for attribute inspection (coarse_start_head, coarse_end_head, etc.).
    # We still call the *wrapped* head for the actual forward pass.
    head_for_attrs = head.module if hasattr(head, "module") else head

    # Hyena backbone train/eval + freezing is controlled externally by
    # HyenaLayerController.apply(epoch=...), which is called in the outer loop.
    # Here we only toggle the head, and for evaluation we put the whole model
    # in eval mode to disable dropout/BN updates.

    if is_train:
        head.train()
    else:
        model_hyena.eval()
        head.eval()

    epoch_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    epoch_loss_count = 0
    global_step_inc = 0

    bucket_start = _new_bucket(device)
    bucket_end = _new_bucket(device)
    bucket_coarse_start = (
        _new_bucket(device)
        if getattr(head_for_attrs, "coarse_start_head", None) is not None
        else None
    )
    bucket_coarse_end = (
        _new_bucket(device)
        if getattr(head_for_attrs, "coarse_end_head", None) is not None
        else None
    )
    n_samples = 0

    if device.type == "cuda" and use_amp:
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        autocast_ctx = nullcontext()

    profiler = profiler_ctx

    # Only clone hidden states when actually using torch.compile on CUDA.
    clone_hidden = bool(use_compile and device.type == "cuda")

    # Prefer inference_mode for eval; no-op for training.
    outer_ctx = nullcontext() if is_train else torch.inference_mode()

    with outer_ctx:
        step = -1
        for step, batch in enumerate(dataloader):
            # Mark the step boundary before any model invocations for compiled modules
            if is_train and _HAS_CUDAGRAPH_MARK and device.type == "cuda":
                torch.compiler.cudagraph_mark_step_begin()

            seqs: List[str] = batch["seq"]
            chr_idx = batch["chr_idx"].to(device, non_blocking=True)
            start_bin = batch["start_bin"].to(device, non_blocking=True)
            end_bin = batch["end_bin"].to(device, non_blocking=True)
            coarse_start_bin = batch.get("coarse_start_bin", None)
            coarse_end_bin = batch.get("coarse_end_bin", None)
            if coarse_start_bin is not None:
                coarse_start_bin = coarse_start_bin.to(device, non_blocking=True)
                coarse_end_bin = coarse_end_bin.to(device, non_blocking=True)

            B = chr_idx.size(0)
            n_samples += B

            enc = tokenize_batch(
                backend_hyena,
                seqs,
                fast_tokenizer_fn=fast_tokenizer_fn,
                max_tokens=max_tokens,
            )
            input_ids = enc["input_ids"].to(device, non_blocking=True)
            attention_mask = enc["attention_mask"].to(device, non_blocking=True)

            need_hidden_states = (hyena_hidden_layer_idx != -1)

            with autocast_ctx:
                out = model_hyena(
                    input_ids=input_ids,
                    return_dict=True,
                    output_hidden_states=need_hidden_states,
                )

                if hyena_hidden_layer_idx == -1:
                    hidden = out.last_hidden_state
                else:
                    if not hasattr(out, "hidden_states") or out.hidden_states is None:
                        raise RuntimeError(
                            "Hyena model did not return hidden_states; "
                            "set output_hidden_states=True."
                        )
                    try:
                        hidden = out.hidden_states[hyena_hidden_layer_idx]
                    except IndexError as e:
                        raise IndexError(
                            f"Requested hyena_hidden_layer_idx={hyena_hidden_layer_idx}, "
                            f"but model returned {len(out.hidden_states)} hidden states."
                        ) from e

                # IMPORTANT: only clone hidden when actually compiled on CUDA,
                # to avoid aliasing CUDAGraph-managed storage. Otherwise skip.
                if clone_hidden:
                    hidden = hidden.clone()

                head_out = head(
                    hidden_states=hidden,
                    attention_mask=attention_mask,
                    chr_labels=chr_idx,
                    start_bin_labels=start_bin,
                    end_bin_labels=end_bin,
                    coarse_start_labels=coarse_start_bin,
                    coarse_end_labels=coarse_end_bin,
                )

                loss = head_out["loss"]
                if loss is None:
                    raise RuntimeError("Head returned None loss during epoch run.")

            epoch_loss_sum += loss.detach().float() * B
            epoch_loss_count += B

            # Metrics (no grad)
            with torch.no_grad():
                logits_start = head_out["logits_start_bin"]
                logits_end = head_out["logits_end_bin"]
                pred_start = torch.argmax(logits_start, dim=-1)
                pred_end = torch.argmax(logits_end, dim=-1)
                _update_bin_bucket(bucket_start, start_bin, pred_start)
                _update_bin_bucket(bucket_end, end_bin, pred_end)

                if bucket_coarse_start is not None and "logits_coarse_start_bin" in head_out:
                    logits_c_start = head_out["logits_coarse_start_bin"]
                    logits_c_end = head_out["logits_coarse_end_bin"]
                    if coarse_start_bin is not None and coarse_end_bin is not None:
                        pred_c_start = torch.argmax(logits_c_start, dim=-1)
                        pred_c_end = torch.argmax(logits_c_end, dim=-1)
                        _update_bin_bucket(bucket_coarse_start, coarse_start_bin, pred_c_start)
                        _update_bin_bucket(bucket_coarse_end, coarse_end_bin, pred_c_end)

            # Backprop / optimizer step
            if is_train:
                assert optimizer is not None
                scaled_loss = loss / accum_steps
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                if (step + 1) % accum_steps == 0:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step_inc += 1

            if profiler is not None:
                profiler.step()

        # Handle a final partial microbatch that accumulated gradients
        # but did not trigger an optimizer.step() inside the loop.
        if is_train and optimizer is not None and epoch_loss_count > 0:
            if (step + 1) % accum_steps != 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step_inc += 1

    # --------------------------------------------------------
    # Distributed aggregation (sum over all ranks)
    # --------------------------------------------------------
    if distributed and dist.is_available() and dist.is_initialized():
        loss_sum_tensor = epoch_loss_sum.clone()
        count_tensor = torch.tensor(
            epoch_loss_count,
            device=device,
            dtype=torch.long,
        )
        n_samples_tensor = torch.tensor(
            n_samples,
            device=device,
            dtype=torch.long,
        )

        dist.all_reduce(loss_sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_samples_tensor, op=dist.ReduceOp.SUM)

        epoch_loss_sum = loss_sum_tensor
        epoch_loss_count = int(count_tensor.item())
        n_samples = int(n_samples_tensor.item())

        for bucket in (bucket_start, bucket_end):
            for key in ("ok", "adj", "wrong"):
                dist.all_reduce(bucket[key], op=dist.ReduceOp.SUM)

        if bucket_coarse_start is not None and bucket_coarse_end is not None:
            for bucket in (bucket_coarse_start, bucket_coarse_end):
                for key in ("ok", "adj", "wrong"):
                    dist.all_reduce(bucket[key], op=dist.ReduceOp.SUM)

    if epoch_loss_count > 0:
        epoch_loss = (epoch_loss_sum / epoch_loss_count).item()
    else:
        epoch_loss = float("nan")

    start_bucket_int = _bucket_to_int(bucket_start)
    end_bucket_int = _bucket_to_int(bucket_end)
    start_acc1, start_acc1_adj = _acc_from_bucket(start_bucket_int, n_samples)
    end_acc1, end_acc1_adj = _acc_from_bucket(end_bucket_int, n_samples)

    coarse_start_acc1 = coarse_start_acc1_adj = None
    coarse_end_acc1 = coarse_end_acc1_adj = None
    if bucket_coarse_start is not None and bucket_coarse_end is not None:
        coarse_start_bucket_int = _bucket_to_int(bucket_coarse_start)
        coarse_end_bucket_int = _bucket_to_int(bucket_coarse_end)
        coarse_start_acc1, coarse_start_acc1_adj = _acc_from_bucket(
            coarse_start_bucket_int, n_samples
        )
        coarse_end_acc1, coarse_end_acc1_adj = _acc_from_bucket(
            coarse_end_bucket_int, n_samples
        )

    metrics = EpochMetrics(
        loss=epoch_loss,
        start_acc1=start_acc1,
        start_acc1_adj=start_acc1_adj,
        end_acc1=end_acc1,
        end_acc1_adj=end_acc1_adj,
        coarse_start_acc1=coarse_start_acc1,
        coarse_start_acc1_adj=coarse_start_acc1_adj,
        coarse_end_acc1=coarse_end_acc1,
        coarse_end_acc1_adj=coarse_end_acc1_adj,
    )
    return metrics, global_step_inc


# ================================================================
# Main training function
# ================================================================
def train_from_pt(
    pt_path: Union[str, Path],
    model_dir: Union[str, Path],
    *,
    save_dir: Union[str, Path],
    batch_size: int = 16,
    num_epochs: int = 20,
    lr: float = 1e-4,
    lr_backbone: Optional[float] = None,
    lr_head: Optional[float] = None,
    val_fraction: float = 0.1,
    accum_steps: int = 4,
    hyena_hidden_layer_idx: int = -1,
    hyena_model_name: str = "hyenadna-large-1m-seqlen-hf",
    freeze_hyena: bool = True,
    hyena_unfreeze_start_epoch: int = 5,
    hyena_unfreeze_end_epoch: int = 10,
    hyena_max_unfrozen_layers: int = 8,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    use_profiler: bool = True,
    profiler_max_steps: int = 200,
    dataloader_num_workers: int = 0,
    fast_tokenizer_fn: Optional[Callable[[List[str]], Dict[str, torch.Tensor]]] = None,
    early_stop_patience: int = 5,
    early_stop_use_acc: bool = False,
    min_delta: float = 0.0,
    coarse_factor: Optional[int] = None,
    coarse_loss_weight: float = 0.0,
    resume_ckpt: Optional[Union[str, Path]] = None,
    use_compile: bool = True,
    compile_mode: str = "max-autotune",
    hyena_max_tokens: int = 10_000,
) -> None:
    # ------------------------------------------------------------------
    # Distributed setup (single-node multi-GPU via torchrun)
    # ------------------------------------------------------------------
    if torch.cuda.is_available() and ("RANK" in os.environ and "WORLD_SIZE" in os.environ):
        distributed = True
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
        print(f"[DIST] Initialized DDP: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    else:
        distributed = False
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DIST] Running in single-process mode on device={device}")

    is_main_process = (rank == 0)

    pt_path = Path(pt_path)
    model_dir = Path(model_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if is_main_process:
        print(f"Using device: {device}")
        print(f"Loading dataset from: {pt_path}")
    data = torch.load(pt_path, map_location="cpu")

    seqs: List[str] = list(data["seqs"])
    chr_idxs: torch.Tensor = data["chr_idxs"].long()
    start_bins: torch.Tensor = data["start_bins"].long()
    end_bins: torch.Tensor = data["end_bins"].long()
    chr_to_idx = data["chr_to_idx"]
    num_bins_global: int = int(data["num_bins_global"])

    N = len(seqs)
    num_chromosomes = len(chr_to_idx)

    if is_main_process:
        print(f"Loaded {N} reads.")
        print(f"num_chromosomes={num_chromosomes}, num_bins_global={num_bins_global}")

    if is_main_process:
        _summarize_bin_distribution("ALL (start_bins)", start_bins, num_bins_global)
        _summarize_bin_distribution("ALL (end_bins)", end_bins, num_bins_global)

    if (start_bins < 0).any() or (start_bins >= num_bins_global).any():
        raise ValueError("Found out-of-range values in start_bins.")
    if (end_bins < 0).any() or (end_bins >= num_bins_global).any():
        raise ValueError("Found out-of-range values in end_bins.")

    # ----------------------------------------------------
    # Resume checkpoint (metadata & state dicts)
    # ----------------------------------------------------
    start_epoch = 0
    global_step = 0
    best_metric: Optional[float] = None
    best_epoch: Optional[int] = None
    best_metric_label: Optional[str] = None
    epochs_no_improve = 0

    loaded_head_state = None
    loaded_opt_state = None
    loaded_sched_state = None

    if resume_ckpt is not None:
        ckpt_path = Path(resume_ckpt)
        if ckpt_path.is_file():
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                start_epoch = int(ckpt.get("epoch", 0)) + 1
                global_step = int(ckpt.get("global_step", 0))
                best_metric = ckpt.get("best_metric", None)
                best_epoch = ckpt.get("best_epoch", None)
                best_metric_label = ckpt.get("best_metric_label", None)

                loaded_head_state = ckpt.get("head_state_dict", None)
                loaded_opt_state = ckpt.get("optimizer_state_dict", None)
                loaded_sched_state = ckpt.get("scheduler_state_dict", None)

                if is_main_process:
                    print(
                        f"[INFO] Resuming from checkpoint {ckpt_path}; "
                        f"start_epoch={start_epoch}, global_step={global_step}"
                    )
            except Exception as e:
                if is_main_process:
                    print(f"[WARN] Failed to load checkpoint {ckpt_path}: {e}")

    # ----------------------------------------------------
    # Split into train/val
    # ----------------------------------------------------
    if val_fraction > 0:
        N_train = int((1.0 - val_fraction) * N)
        perm = torch.randperm(N)
        train_idx = perm[:N_train]
        val_idx = perm[N_train:]
    else:
        train_idx = torch.arange(N)
        val_idx = torch.tensor([], dtype=torch.long)

    coarse_start_bins: Optional[torch.Tensor] = None
    coarse_end_bins: Optional[torch.Tensor] = None
    num_coarse_bins: Optional[int] = None
    if coarse_factor is not None:
        if coarse_factor <= 0:
            raise ValueError("coarse_factor must be positive if provided")
        if num_bins_global % coarse_factor != 0:
            raise ValueError("coarse_factor must divide num_bins_global")
        num_coarse_bins = num_bins_global // coarse_factor
        coarse_start_bins = torch.div(start_bins, coarse_factor, rounding_mode='floor')
        coarse_end_bins = torch.div(end_bins, coarse_factor, rounding_mode='floor')

    if is_main_process:
        print(f"Split: train={train_idx.numel()}, val={val_idx.numel()} (val_fraction={val_fraction})")

        # Sanity checks on bins
        if train_idx.numel() > 0:
            _summarize_bin_distribution("TRAIN (start_bins)", start_bins[train_idx], num_bins_global)
            _summarize_bin_distribution("TRAIN (end_bins)", end_bins[train_idx], num_bins_global)
            if num_coarse_bins is not None:
                assert coarse_start_bins is not None and coarse_end_bins is not None
                _summarize_bin_distribution(
                    "TRAIN (coarse_start_bins)", coarse_start_bins[train_idx], num_coarse_bins
                )
                _summarize_bin_distribution(
                    "TRAIN (coarse_end_bins)", coarse_end_bins[train_idx], num_coarse_bins
                )
        if val_idx.numel() > 0:
            _summarize_bin_distribution("VAL (start_bins)", start_bins[val_idx], num_bins_global)
            _summarize_bin_distribution("VAL (end_bins)", end_bins[val_idx], num_bins_global)
            if num_coarse_bins is not None:
                _summarize_bin_distribution(
                    "VAL (coarse_start_bins)", coarse_start_bins[val_idx], num_coarse_bins
                )
                _summarize_bin_distribution(
                    "VAL (coarse_end_bins)", coarse_end_bins[val_idx], num_coarse_bins
                )

    # ----------------------------------------------------
    # Datasets & loaders (DDP-aware)
    # ----------------------------------------------------
    train_dataset = ChrBinDataset(
        [seqs[i] for i in train_idx.tolist()],
        chr_idxs[train_idx],
        start_bins[train_idx],
        end_bins[train_idx],
        None if coarse_start_bins is None else coarse_start_bins[train_idx],
        None if coarse_end_bins is None else coarse_end_bins[train_idx],
    )
    if val_idx.numel() > 0:
        val_dataset: Optional[ChrBinDataset] = ChrBinDataset(
            [seqs[i] for i in val_idx.tolist()],
            chr_idxs[val_idx],
            start_bins[val_idx],
            end_bins[val_idx],
            None if coarse_start_bins is None else coarse_start_bins[val_idx],
            None if coarse_end_bins is None else coarse_end_bins[val_idx],
        )
    else:
        val_dataset = None

    if distributed:
        train_sampler: Optional[DistributedSampler[ChrBinDataset]] = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        train_shuffle = False
    else:
        train_sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=dataloader_num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(dataloader_num_workers > 0),
    )
    val_loader = None
    val_sampler: Optional[DistributedSampler[ChrBinDataset]] = None
    if val_dataset is not None:
        if distributed:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=dataloader_num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            persistent_workers=(dataloader_num_workers > 0),
        )

    # ----------------------------------------------------
    # TensorBoard
    # ----------------------------------------------------
    writer = None    # type: ignore[assignment]
    if SummaryWriter is not None and is_main_process:
        tb_log_dir = save_dir / "tb_logs"
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_log_dir))
        print(f"TensorBoard logs at: {tb_log_dir}")
    elif SummaryWriter is None and is_main_process:
        print("TensorBoard SummaryWriter not available; skipping TB logging.")

    # ----------------------------------------------------
    # Hyena backend & DNATok
    # ----------------------------------------------------
    backend_hy = build_hyena_backend(model_dir=model_dir, model_name=hyena_model_name)
    hyena_model = backend_hy.model.to(device)

    dna_tok = DNATok(embedder=backend_hy)
    dna_tok.discover()
    if not dna_tok.use_ids_path or dna_tok.ascii_lut is None:
        if is_main_process:
            print("[WARN] DNATok IDs path not available; falling back to Hyena tokenizer.")
        dnatok_batch_fn: Optional[Callable[[List[str]], Dict[str, torch.Tensor]]] = None
    else:
        if is_main_process:
            print("[INFO] DNATok ASCII→ID path enabled.")
        _dnatok_core_buffer: Dict[int, torch.Tensor] = {}
        def dnatok_batch_fn(seqs_batch: List[str]) -> Dict[str, torch.Tensor]:
            """
            Fast ASCII→ID path using DNATok for a batch of DNA strings.

            - Right-pads sequences with 'N' to batch max length.
            - DNATok may additionally left-pad (offset_left) for its internal alignment.
            - attention_mask marks only real bases.
            """
            if not seqs_batch:
                raise ValueError("dnatok_batch_fn: empty batch")

            max_len = max(len(s) for s in seqs_batch)
            if max_len <= 0:
                raise ValueError("dnatok_batch_fn: all sequences are empty")

            if hasattr(backend_hy, "max_length") and backend_hy.max_length is not None:
                if max_len > backend_hy.max_length:
                    raise ValueError(
                        f"Batch contains sequence of length {max_len}, "
                        f"but HyenaBackend.max_length={backend_hy.max_length}."
                    )

            padded_seqs = [s.ljust(max_len, "N") for s in seqs_batch]

            ids_cpu = dna_tok.encode_batch_to_ids_staging(
                padded_seqs,
                dtype=torch.long,
            )  # [B, T_out]

            B_local, T_out = ids_cpu.shape
            L_max = max_len
            offset_left = T_out - L_max

            lens = torch.tensor([len(s) for s in seqs_batch], dtype=torch.long)  # [B]

            # Reuse a 1 x L_max arange buffer on CPU to cut per-batch allocations
            core_range = _dnatok_core_buffer.get(L_max)
            if core_range is None:
                core_range = torch.arange(L_max, dtype=torch.long).unsqueeze(0)  # [1, L_max]
                _dnatok_core_buffer[L_max] = core_range
            mask_core = core_range < lens.unsqueeze(1)  # [B, L_max]

            attention_mask = torch.zeros(B_local, T_out, dtype=torch.long)
            attention_mask[:, offset_left:] = mask_core.to(attention_mask.dtype)

            return {
                "input_ids": ids_cpu,
                "attention_mask": attention_mask,
            }

    fast_tokenizer_fn_local = fast_tokenizer_fn if fast_tokenizer_fn is not None else dnatok_batch_fn

    # ----------------------------------------------------
    # Sanity check: raw lengths vs token lengths under hyena_max_tokens
    # ----------------------------------------------------
    if is_main_process:
        print(f"\n[SANITY] hyena_max_tokens={hyena_max_tokens}, backend_max_length={backend_hy.max_length}")
        subset_size = min(32, len(seqs))
        if subset_size > 0:
            seq_subset = seqs[:subset_size]
            raw_lens = torch.tensor([len(s) for s in seq_subset], dtype=torch.long)
            print(
                "[SANITY] Raw read lengths on subset "
                f"(before cropping): min={int(raw_lens.min())}, "
                f"mean={float(raw_lens.float().mean()):.1f}, "
                f"max={int(raw_lens.max())}"
            )

            enc_subset = tokenize_batch(
                backend_hy,
                seq_subset,
                fast_tokenizer_fn=fast_tokenizer_fn_local,
                max_tokens=hyena_max_tokens,
            )
            ids_subset = enc_subset["input_ids"]
            B_sub, T_sub = ids_subset.shape
            print(
                f"[SANITY] Tokenized subset: batch_size={B_sub}, token_length={T_sub}"
            )

            expected_max_raw_after = min(int(raw_lens.max()), hyena_max_tokens)
            print(
                f"[SANITY] Expected max raw length after cap={expected_max_raw_after} "
                "(min(len, hyena_max_tokens))."
            )

            if fast_tokenizer_fn_local is None:
                # HF tokenizer path: token length should be <= hyena_max_tokens (and backend max_length)
                if T_sub > hyena_max_tokens:
                    raise RuntimeError(
                        f"[SANITY] HF tokenizer produced token length {T_sub} "
                        f"> hyena_max_tokens={hyena_max_tokens}."
                    )
                if backend_hy.max_length is not None and T_sub > backend_hy.max_length:
                    raise RuntimeError(
                        f"[SANITY] HF tokenizer produced token length {T_sub} "
                        f"> backend_hy.max_length={backend_hy.max_length}."
                    )
            else:
                # DNATok path: allow some left-padding but ensure we never exceed backend's max_length.
                if backend_hy.max_length is not None and T_sub > backend_hy.max_length:
                    raise RuntimeError(
                        f"[SANITY] DNATok produced token length {T_sub} "
                        f"> backend_hy.max_length={backend_hy.max_length}."
                    )

            print("[SANITY] Token capping under hyena_max_tokens appears consistent.\n")

    # ----------------------------------------------------
    # Dummy forward to infer d_model (respecting hyena_max_tokens)
    # ----------------------------------------------------
    dummy_seq = [seqs[0]]
    dummy_enc = tokenize_batch(
        backend_hy,
        dummy_seq,
        fast_tokenizer_fn=fast_tokenizer_fn_local,
        max_tokens=hyena_max_tokens,
    )
    dummy_ids = dummy_enc["input_ids"].to(device)
    with torch.no_grad():
        dummy_out = hyena_model(
            input_ids=dummy_ids,
            return_dict=True,
            output_hidden_states=(hyena_hidden_layer_idx != -1),
        )
        if hyena_hidden_layer_idx == -1:
            dummy_hidden = dummy_out.last_hidden_state
        else:
            if not hasattr(dummy_out, "hidden_states") or dummy_out.hidden_states is None:
                raise RuntimeError("Hyena model did not return hidden_states in dummy forward.")
            dummy_hidden = dummy_out.hidden_states[hyena_hidden_layer_idx]
    d_model = dummy_hidden.size(-1)
    if is_main_process:
        print(f"[DEBUG] Inferred d_model={d_model} from Hyena hidden states.")

    # ----------------------------------------------------
    # Head
    # ----------------------------------------------------
    head = HyenaChrBinHead(
        d_model=d_model,
        num_chromosomes=num_chromosomes,
        num_bins=num_bins_global,
        hidden_dim=768,
        conv_kernel_sizes=(3, 7, 15),
        conv_dilations=(1, 2, 4),
        dropout=0.1,
        label_smoothing=0.05,
        num_coarse_bins=num_coarse_bins,
        w_chr=0.1,
        w_start=1.0,
        w_end=1.0,
        w_coarse_start=coarse_loss_weight,
        w_coarse_end=coarse_loss_weight,
        w_start_dist=0.1,
        w_end_dist=0.1,
        conv_downsample=4,
    ).to(device)

    if loaded_head_state is not None:
        try:
            head.load_state_dict(loaded_head_state)
            if is_main_process:
                print("[INFO] Loaded head state from checkpoint.")
        except Exception as e:
            if is_main_process:
                print(f"[WARN] Failed to load head state from checkpoint: {e}")

    # DDP wrappers
    if distributed:
        hyena_model = DDP(
            hyena_model,
            device_ids=[device],
            output_device=device,
            find_unused_parameters=False,
        )
        head = DDP(head, device_ids=[device], output_device=device, find_unused_parameters=False)

    # Compile only the head; Hyena backbone likely not torch.compile-friendly
    compiled_used = False
    if use_compile and torch.cuda.is_available():
        try:
            head = torch.compile(head, mode=compile_mode)
            compiled_used = True
            if is_main_process:
                print(f"[INFO] torch.compile enabled with mode='{compile_mode}' for head.")
        except Exception as e:
            if is_main_process:
                print(f"[WARN] torch.compile failed: {e}; continuing without compilation.")

    # Hyena layer controller
    hyena_controller = HyenaLayerController(
        model=hyena_model if not isinstance(hyena_model, DDP) else hyena_model.module,
        freeze_backbone=freeze_hyena,
        start_epoch=hyena_unfreeze_start_epoch,
        end_epoch=hyena_unfreeze_end_epoch,
        max_unfrozen_layers=hyena_max_unfrozen_layers,
    )

    if not freeze_hyena and is_main_process:
        print("[INFO] Hyena backbone is fully trainable from the start.")

    # ----------------------------------------------------
    # Optimizer + schedulers
    # ----------------------------------------------------
    lr_head_eff = lr_head if lr_head is not None else lr
    lr_backbone_eff = lr_backbone if lr_backbone is not None else lr * 0.1

    if is_main_process:
        print(
            f"[INFO] Using separate learning rates - "
            f"backbone_lr={lr_backbone_eff:.3g}, head_lr={lr_head_eff:.3g}"
        )

    if distributed:
        backbone_params = hyena_model.module.parameters()
        head_params = head.module.parameters()
    else:
        backbone_params = hyena_model.parameters()
        head_params = head.parameters()

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone_eff},
            {"params": head_params, "lr": lr_head_eff},
        ],
        lr=lr_head_eff,
    )

    # Cosine LR schedule over epochs (simple, metric-agnostic)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, num_epochs),
        eta_min=lr * 0.1,
    )

    if use_amp and device.type == "cuda" and amp_dtype == torch.float16:
        scaler: Optional[torch.amp.GradScaler] = torch.amp.GradScaler("cuda")
        if is_main_process:
            print(f"[INFO] Using AMP with dtype={amp_dtype} and GradScaler.")
    else:
        scaler = None
        if use_amp and device.type == "cuda" and is_main_process:
            print(f"[INFO] Using AMP with dtype={amp_dtype} (no GradScaler needed).")
        elif use_amp and is_main_process:
            print("[INFO] AMP requested but no CUDA device; running without AMP.")

    if loaded_opt_state is not None:
        try:
            optimizer.load_state_dict(loaded_opt_state)
            if is_main_process:
                print("[INFO] Loaded optimizer state from checkpoint.")
        except Exception as e:
            if is_main_process:
                print(f"[WARN] Failed to load optimizer state from checkpoint: {e}")

    if loaded_sched_state is not None:
        try:
            scheduler.load_state_dict(loaded_sched_state)
            if is_main_process:
                print("[INFO] Loaded scheduler state from checkpoint.")
        except Exception as e:
            if is_main_process:
                print(f"[WARN] Failed to load scheduler state from checkpoint: {e}")

    hyena_controller.apply(epoch=start_epoch)

    # ----------------------------------------------------
    # Early stopping state
    # ----------------------------------------------------
    epochs_no_improve = 0

    # ----------------------------------------------------
    # Profiler setup
    # ----------------------------------------------------
    use_profiler = use_profiler and _HAVE_PROFILER and device.type == "cuda"
    profiler_dir = save_dir / "tb_profiler"

    # ----------------------------------------------------
    # Training loop
    # ----------------------------------------------------
    for epoch in range(start_epoch, num_epochs):
        if is_main_process:
            print(f"\n========== Epoch {epoch} / {num_epochs - 1} ==========")

        hyena_controller.apply(epoch=epoch)

        # Ensure deterministic shuffling across ranks
        if distributed:
            train_sampler = getattr(train_loader, "sampler", None)
            if isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(epoch)

        # Training epoch
        if use_profiler and epoch == 0 and is_main_process:
            profiler_dir.mkdir(parents=True, exist_ok=True)

            # How many steps to actively record with the profiler.
            # We clamp to at least 1 and at most len(train_loader) so the
            # schedule is sensible even for tiny datasets.
            num_train_steps = len(train_loader)
            prof_active = max(1, min(int(profiler_max_steps), num_train_steps))

            prof_schedule = schedule(
                wait=0,
                warmup=1,                # 1 warmup step (unrecorded)
                active=prof_active,      # <= profiler_max_steps recorded steps
                repeat=1,
            )

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=prof_schedule,
                on_trace_ready=tensorboard_trace_handler(str(profiler_dir)),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                train_metrics, step_inc = _run_epoch(
                    epoch=epoch,
                    model_hyena=hyena_model,
                    head=head,
                    backend_hyena=backend_hy,
                    dataloader=train_loader,
                    device=device,
                    hyena_hidden_layer_idx=hyena_hidden_layer_idx,
                    optimizer=optimizer,
                    scaler=scaler,
                    fast_tokenizer_fn=fast_tokenizer_fn_local,
                    accum_steps=accum_steps,
                    is_train=True,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    profiler_ctx=prof,
                    max_tokens=hyena_max_tokens,
                    use_compile=compiled_used,
                    distributed=distributed,
                )
                global_step += step_inc
        else:
            train_metrics, step_inc = _run_epoch(
                epoch=epoch,
                model_hyena=hyena_model,
                head=head,
                backend_hyena=backend_hy,
                dataloader=train_loader,
                device=device,
                hyena_hidden_layer_idx=hyena_hidden_layer_idx,
                optimizer=optimizer,
                scaler=scaler,
                fast_tokenizer_fn=fast_tokenizer_fn_local,
                accum_steps=accum_steps,
                is_train=True,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                profiler_ctx=None,
                max_tokens=hyena_max_tokens,
                use_compile=compiled_used,
                distributed=distributed,
            )
            global_step += step_inc

        # Validation epoch
        if val_loader is not None:
            val_metrics, _ = _run_epoch(
                epoch=epoch,
                model_hyena=hyena_model,
                head=head,
                backend_hyena=backend_hy,
                dataloader=val_loader,
                device=device,
                hyena_hidden_layer_idx=hyena_hidden_layer_idx,
                optimizer=None,
                scaler=None,
                fast_tokenizer_fn=fast_tokenizer_fn_local,
                accum_steps=1,
                is_train=False,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                profiler_ctx=None,
                max_tokens=hyena_max_tokens,
                use_compile=compiled_used,
                distributed=distributed,
            )
        else:
            val_metrics = None

        # -------- Logging --------
        log_msg = (
            f"[epoch {epoch}] "
            f"train_loss={train_metrics.loss:.4f} "
            f"train_start_acc@1={train_metrics.start_acc1:.4f} "
            f"train_start_acc@1+adj={train_metrics.start_acc1_adj:.4f} "
            f"train_end_acc@1={train_metrics.end_acc1:.4f} "
            f"train_end_acc@1+adj={train_metrics.end_acc1_adj:.4f} | "
        )
        if val_metrics is not None:
            log_msg += (
                f"val_loss={val_metrics.loss:.4f} "
                f"val_start_acc@1={val_metrics.start_acc1:.4f} "
                f"val_start_acc@1+adj={val_metrics.start_acc1_adj:.4f} "
                f"val_end_acc@1={val_metrics.end_acc1:.4f} "
                f"val_end_acc@1+adj={val_metrics.end_acc1_adj:.4f}"
            )
        else:
            log_msg += "no validation set"
        if is_main_process:
            print(log_msg)

        # -------- LR schedule step + LR logging --------
        scheduler.step()
        current_lr_backbone = optimizer.param_groups[0]["lr"]
        current_lr_head = optimizer.param_groups[1]["lr"]
        if is_main_process:
            print(
                f"[INFO] Learning rate after epoch {epoch}: "
                f"backbone={current_lr_backbone:.6g}, head={current_lr_head:.6g}"
            )

        # -------- TensorBoard logging --------
        if writer is not None and is_main_process:
            writer.add_scalar("epoch/train_loss", train_metrics.loss, epoch)
            writer.add_scalar("epoch/train_start_acc@1", train_metrics.start_acc1, epoch)
            writer.add_scalar("epoch/train_start_acc@1+adj", train_metrics.start_acc1_adj, epoch)
            writer.add_scalar("epoch/train_end_acc@1", train_metrics.end_acc1, epoch)
            writer.add_scalar("epoch/train_end_acc@1+adj", train_metrics.end_acc1_adj, epoch)
            writer.add_scalar("epoch/lr_backbone", current_lr_backbone, epoch)
            writer.add_scalar("epoch/lr_head", current_lr_head, epoch)

            if val_metrics is not None:
                writer.add_scalar("epoch/val_loss", val_metrics.loss, epoch)
                writer.add_scalar("epoch/val_start_acc@1", val_metrics.start_acc1, epoch)
                writer.add_scalar("epoch/val_start_acc@1+adj", val_metrics.start_acc1_adj, epoch)
                writer.add_scalar("epoch/val_end_acc@1", val_metrics.end_acc1, epoch)
                writer.add_scalar("epoch/val_end_acc@1+adj", val_metrics.end_acc1_adj, epoch)

        # -------- Early stopping metric selection --------
        if val_metrics is not None:
            monitor_metrics = val_metrics
            prefix = "val"
        else:
            monitor_metrics = train_metrics
            prefix = "train"

        if early_stop_use_acc:
            metric = -monitor_metrics.start_acc1_adj
            cur_metric_label = f"neg_{prefix}_start_acc@1+adj"
        else:
            metric = monitor_metrics.loss
            cur_metric_label = f"{prefix}_loss"

        # -------- Early stopping + checkpointing --------
        stop_training = False

        if is_main_process:
            def _make_checkpoint_state() -> dict:
                return {
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_metric": best_metric,
                    "best_epoch": best_epoch,
                    "best_metric_label": best_metric_label,
                    "head_state_dict": head.module.state_dict() if isinstance(head, DDP) else head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }

            if epoch == start_epoch and best_metric is None:
                best_metric = metric
                best_epoch = epoch
                best_metric_label = cur_metric_label
                epochs_no_improve = 0
                best_path = save_dir / "best_head.pt"
                torch.save(_make_checkpoint_state(), best_path)
                print(
                    f"[INFO] Initial best checkpoint saved at epoch {epoch} "
                    f"({best_metric_label}={best_metric:.4f}) to {best_path}"
                )
            else:
                # Update best metric / early stopping
                if best_metric is None or metric < best_metric - min_delta:
                    best_metric = metric
                    best_epoch = epoch
                    best_metric_label = cur_metric_label
                    epochs_no_improve = 0
                    best_path = save_dir / "best_head.pt"
                    torch.save(_make_checkpoint_state(), best_path)
                    print(
                        f"[INFO] Saved improved checkpoint at epoch {epoch} "
                        f"({best_metric_label}={best_metric:.4f}) to {best_path}"
                    )
                else:
                    epochs_no_improve += 1
                    if best_metric is not None and best_metric_label is not None:
                        print(
                            f"No improvement this epoch (best {best_metric_label}={best_metric:.4f}, "
                            f"current {cur_metric_label}={metric:.4f}); "
                            f"epochs_no_improve={epochs_no_improve}"
                        )
                    else:
                        print(
                            f"No improvement this epoch (no previous best); "
                            f"epochs_no_improve={epochs_no_improve}"
                        )

                    if early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
                        print(
                            f"[INFO] Early stopping triggered at epoch {epoch} "
                            f"(no improvement for {epochs_no_improve} epochs)."
                        )
                        last_path = save_dir / "last.pt"
                        torch.save(_make_checkpoint_state(), last_path)
                        print(f"[INFO] Saved last checkpoint to {last_path}")
                        stop_training = True

            # -------- Always save 'last' checkpoint for this epoch --------
            last_path = save_dir / "last.pt"
            torch.save(_make_checkpoint_state(), last_path)
            print(f"[INFO] Saved last checkpoint to {last_path}")

        # Broadcast stop flag from rank 0 to all ranks
        if distributed and dist.is_available() and dist.is_initialized():
            stop_tensor = torch.tensor(1 if stop_training else 0, device=device, dtype=torch.int)
            dist.broadcast(stop_tensor, src=0)
            stop_training = bool(stop_tensor.item())

        if stop_training:
            break

    # -------- After training loop --------
    if writer is not None and is_main_process:
        writer.flush()
        writer.close()

    if is_main_process:
        print("Training finished.")
        if best_metric is not None and best_epoch is not None and best_metric_label is not None:
            print(f"Best epoch: {best_epoch} with metric={best_metric:.4f} ({best_metric_label})")
        else:
            print("No best epoch recorded (metric never finite?).")

    if distributed and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
