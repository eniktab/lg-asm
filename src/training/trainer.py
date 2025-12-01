import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple, Union
from contextlib import nullcontext

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

from ..config import TrainingConfig, ModelConfig, DataConfig, LossConfig
from ..data.dataset import ChrBinDataset, compute_coarse_bins
from ..data.tokenizer import tokenize_batch
from ..models.head import HyenaChrBinHead
from ..models.utils import build_hyena_backend, HyenaLayerController
from .metrics import (
    EpochMetrics,
    _new_bucket,
    _bucket_to_int,
    _acc_from_bucket,
    _update_bin_bucket,
    _summarize_bin_distribution,
)

_HAS_CUDAGRAPH_MARK = hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin")

class Trainer:
    def __init__(
        self,
        train_config: TrainingConfig,
        model_config: ModelConfig,
        data_config: DataConfig,
        loss_config: LossConfig,
        fast_tokenizer_fn: Optional[Callable[[List[str]], Dict[str, torch.Tensor]]] = None,
    ):
        self.train_config = train_config
        self.model_config = model_config
        self.data_config = data_config
        self.loss_config = loss_config
        self.fast_tokenizer_fn = fast_tokenizer_fn

        self.setup_distributed()
        
        self.save_dir = Path(train_config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        if self.is_main_process:
            print(f"Using device: {self.device}")

    def setup_distributed(self):
        if torch.cuda.is_available() and ("RANK" in os.environ and "WORLD_SIZE" in os.environ):
            self.distributed = True
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank % torch.cuda.device_count()))
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl", init_method="env://")
            self.device = torch.device("cuda", self.local_rank)
            print(f"[DIST] Initialized DDP: rank={self.rank}, world_size={self.world_size}, local_rank={self.local_rank}")
        else:
            self.distributed = False
            self.rank = 0
            self.world_size = 1
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[DIST] Running in single-process mode on device={self.device}")
        
        self.is_main_process = (self.rank == 0)

    def _run_epoch(
        self,
        *,
        epoch: int,
        model_hyena: nn.Module,
        head: HyenaChrBinHead,
        backend_hyena: Any, # HyenaBackend
        dataloader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer],
        scaler: Optional[torch.amp.GradScaler],
        is_train: bool,
        profiler_ctx: Any = None,
        compiled_used: bool = False,
    ) -> Tuple[EpochMetrics, int]:
        
        if is_train and optimizer is None:
            raise ValueError("Optimizer must be provided for training epoch.")
        
        accum_steps = self.train_config.accum_steps
        if accum_steps < 1:
            raise ValueError(f"accum_steps must be >= 1, got {accum_steps}")

        head_for_attrs = head.module if hasattr(head, "module") else head

        if is_train:
            head.train()
        else:
            model_hyena.eval()
            head.eval()

        epoch_loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        epoch_loss_count = 0
        global_step_inc = 0

        bucket_start = _new_bucket(self.device)
        bucket_end = _new_bucket(self.device)
        bucket_coarse_start = (
            _new_bucket(self.device)
            if getattr(head_for_attrs, "coarse_start_head", None) is not None
            else None
        )
        bucket_coarse_end = (
            _new_bucket(self.device)
            if getattr(head_for_attrs, "coarse_end_head", None) is not None
            else None
        )
        n_samples = 0

        if self.device.type == "cuda" and self.train_config.use_amp:
            autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=self.train_config.amp_dtype)
        else:
            autocast_ctx = nullcontext()

        profiler = profiler_ctx
        clone_hidden = bool(compiled_used and self.device.type == "cuda")
        outer_ctx = nullcontext() if is_train else torch.inference_mode()

        with outer_ctx:
            step = -1
            for step, batch in enumerate(dataloader):
                if is_train and _HAS_CUDAGRAPH_MARK and self.device.type == "cuda":
                    torch.compiler.cudagraph_mark_step_begin()

                seqs: List[str] = batch["seq"]
                chr_idx = batch["chr_idx"].to(self.device, non_blocking=True)
                start_bin = batch["start_bin"].to(self.device, non_blocking=True)
                end_bin = batch["end_bin"].to(self.device, non_blocking=True)
                coarse_start_bin = batch.get("coarse_start_bin", None)
                coarse_end_bin = batch.get("coarse_end_bin", None)
                if coarse_start_bin is not None:
                    coarse_start_bin = coarse_start_bin.to(self.device, non_blocking=True)
                    coarse_end_bin = coarse_end_bin.to(self.device, non_blocking=True)

                B = chr_idx.size(0)
                n_samples += B

                enc = tokenize_batch(
                    backend_hyena,
                    seqs,
                    fast_tokenizer_fn=self.fast_tokenizer_fn,
                    max_tokens=self.data_config.max_tokens,
                )
                input_ids = enc["input_ids"].to(self.device, non_blocking=True)
                attention_mask = enc["attention_mask"].to(self.device, non_blocking=True)

                need_hidden_states = (self.model_config.hyena_hidden_layer_idx != -1)

                with autocast_ctx:
                    out = model_hyena(
                        input_ids=input_ids,
                        return_dict=True,
                        output_hidden_states=need_hidden_states,
                    )

                    if self.model_config.hyena_hidden_layer_idx == -1:
                        hidden = out.last_hidden_state
                    else:
                        if not hasattr(out, "hidden_states") or out.hidden_states is None:
                            raise RuntimeError(
                                "Hyena model did not return hidden_states; "
                                "set output_hidden_states=True."
                            )
                        try:
                            hidden = out.hidden_states[self.model_config.hyena_hidden_layer_idx]
                        except IndexError as e:
                            raise IndexError(
                                f"Requested hyena_hidden_layer_idx={self.model_config.hyena_hidden_layer_idx}, "
                                f"but model returned {len(out.hidden_states)} hidden states."
                            ) from e

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

            if is_train and optimizer is not None and epoch_loss_count > 0:
                if (step + 1) % accum_steps != 0:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step_inc += 1

        if self.distributed and dist.is_available() and dist.is_initialized():
            loss_sum_tensor = epoch_loss_sum.clone()
            count_tensor = torch.tensor(epoch_loss_count, device=self.device, dtype=torch.long)
            n_samples_tensor = torch.tensor(n_samples, device=self.device, dtype=torch.long)

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

    def train(self):
        if self.is_main_process:
            print(f"Loading dataset from: {self.data_config.pt_path}")
        data = torch.load(self.data_config.pt_path, map_location="cpu")

        seqs: List[str] = list(data["seqs"])
        chr_idxs: torch.Tensor = data["chr_idxs"].long()
        start_bins: torch.Tensor = data["start_bins"].long()
        end_bins: torch.Tensor = data["end_bins"].long()
        chr_to_idx = data["chr_to_idx"]
        num_bins_global: int = int(data["num_bins_global"])

        N = len(seqs)
        num_chromosomes = len(chr_to_idx)

        if self.is_main_process:
            print(f"Loaded {N} reads.")
            print(f"num_chromosomes={num_chromosomes}, num_bins_global={num_bins_global}")
            _summarize_bin_distribution("ALL (start_bins)", start_bins, num_bins_global)
            _summarize_bin_distribution("ALL (end_bins)", end_bins, num_bins_global)

        if (start_bins < 0).any() or (start_bins >= num_bins_global).any():
            raise ValueError("Found out-of-range values in start_bins.")
        if (end_bins < 0).any() or (end_bins >= num_bins_global).any():
            raise ValueError("Found out-of-range values in end_bins.")

        start_epoch = 0
        global_step = 0
        best_metric: Optional[float] = None
        best_epoch: Optional[int] = None
        best_metric_label: Optional[str] = None

        loaded_hyena_state: Optional[Dict[str, Any]] = None
        loaded_head_state: Optional[Dict[str, Any]] = None
        loaded_opt_state: Optional[Dict[str, Any]] = None
        loaded_sched_state: Optional[Dict[str, Any]] = None
        ckpt_num_coarse_bins: Optional[int] = None

        if self.train_config.resume_ckpt is not None:
            resume_ckpt = Path(self.train_config.resume_ckpt)
            if not resume_ckpt.is_file():
                raise FileNotFoundError(f"resume_ckpt not found: {resume_ckpt}")
            if self.is_main_process:
                print(f"[INFO] Resuming from checkpoint: {resume_ckpt}")
            ckpt = torch.load(resume_ckpt, map_location="cpu")

            ckpt_num_chromosomes = int(ckpt["num_chromosomes"])
            ckpt_num_bins_global = int(ckpt["num_bins_global"])
            if ckpt_num_chromosomes != num_chromosomes:
                raise ValueError(
                    f"Checkpoint num_chromosomes={ckpt_num_chromosomes} "
                    f"but dataset has num_chromosomes={num_chromosomes}"
                )
            if ckpt_num_bins_global != num_bins_global:
                raise ValueError(
                    f"Checkpoint num_bins_global={ckpt_num_bins_global} "
                    f"but dataset has num_bins_global={num_bins_global}"
                )

            self.model_config.hyena_model_name = ckpt.get("hyena_model_name", self.model_config.hyena_model_name)
            self.model_config.hyena_hidden_layer_idx = ckpt.get("hyena_hidden_layer_idx", self.model_config.hyena_hidden_layer_idx)
            self.data_config.coarse_factor = ckpt.get("coarse_factor", self.data_config.coarse_factor)
            ckpt_num_coarse_bins = ckpt.get("num_coarse_bins", None)

            start_epoch = int(ckpt.get("epoch", -1)) + 1
            global_step = int(ckpt.get("global_step", 0))
            best_metric = ckpt.get("best_metric", None)
            best_epoch = ckpt.get("best_epoch", None)
            best_metric_label = ckpt.get("metric_label", None)

            loaded_hyena_state = ckpt.get("state_dict_hyena") or ckpt.get("hyena_state_dict")
            loaded_head_state = ckpt.get("state_dict_head") or ckpt.get("head_state_dict")
            loaded_opt_state = ckpt.get("state_dict_optimizer") or ckpt.get("optimizer_state_dict")
            loaded_sched_state = ckpt.get("state_dict_scheduler")

            if self.is_main_process:
                print(
                    f"[INFO] Resume metadata: start_epoch={start_epoch}, global_step={global_step}, "
                    f"best_metric={best_metric}, best_epoch={best_epoch}, "
                    f"hyena_model_name={self.model_config.hyena_model_name}, "
                    f"hyena_hidden_layer_idx={self.model_config.hyena_hidden_layer_idx}, "
                    f"coarse_factor={self.data_config.coarse_factor}"
                )

        coarse_start_bins = coarse_end_bins = None
        num_coarse_bins: Optional[int] = None
        if self.data_config.coarse_factor is not None and self.data_config.coarse_factor > 1:
            coarse_start_bins, coarse_end_bins, num_coarse_bins = compute_coarse_bins(
                start_bins,
                end_bins,
                num_bins_global,
                self.data_config.coarse_factor,
            )
            c = int(self.data_config.coarse_factor)
            if ckpt_num_coarse_bins is not None and num_coarse_bins != ckpt_num_coarse_bins:
                raise ValueError(
                    f"Checkpoint num_coarse_bins={ckpt_num_coarse_bins}, "
                    f"but recomputed num_coarse_bins={num_coarse_bins} "
                    f"from coarse_factor={c} and num_bins_global={num_bins_global}"
                )
            if self.is_main_process:
                print(
                    f"Using coarse_factor={c}: num_coarse_bins={num_coarse_bins} "
                    "(e.g. 1Mb windows if fine bins are 10kb)."
                )
                _summarize_bin_distribution("ALL (coarse_start_bins)", coarse_start_bins, num_coarse_bins)
                _summarize_bin_distribution("ALL (coarse_end_bins)", coarse_end_bins, num_coarse_bins)
        else:
            if self.is_main_process:
                print("No coarse binning.")

        all_idx = torch.arange(N, dtype=torch.long)
        if self.data_config.val_fraction > 0.0 and N > 1:
            n_val = max(1, int(N * self.data_config.val_fraction))
            n_val = min(n_val, N - 1)
            perm = torch.randperm(N)
            val_idx = perm[:n_val]
            train_idx = perm[n_val:]
            if self.is_main_process:
                print(
                    f"Train/val split: {train_idx.numel()} train, "
                    f"{val_idx.numel()} val (val_fraction={self.data_config.val_fraction:.3f})"
                )
        else:
            train_idx = all_idx
            val_idx = torch.zeros(0, dtype=torch.long)
            if self.is_main_process:
                print("No validation split; early stopping will use training loss only.")

        if self.is_main_process:
            _summarize_bin_distribution("TRAIN (start_bins)", start_bins[train_idx], num_bins_global)
            _summarize_bin_distribution("TRAIN (end_bins)", end_bins[train_idx], num_bins_global)
            if num_coarse_bins is not None:
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

        if self.distributed:
            train_sampler: Optional[DistributedSampler[ChrBinDataset]] = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=False,
            )
            train_shuffle = False
        else:
            train_sampler = None
            train_shuffle = True

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=self.data_config.dataloader_num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
            persistent_workers=(self.data_config.dataloader_num_workers > 0),
        )
        val_loader = None
        val_sampler: Optional[DistributedSampler[ChrBinDataset]] = None
        if val_dataset is not None:
            if self.distributed:
                val_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False,
                    drop_last=False,
                )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.train_config.batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=self.data_config.dataloader_num_workers,
                pin_memory=(self.device.type == "cuda"),
                drop_last=False,
                persistent_workers=(self.data_config.dataloader_num_workers > 0),
            )

        writer = None
        if SummaryWriter is not None and self.is_main_process:
            tb_log_dir = self.save_dir / "tb_logs"
            tb_log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(tb_log_dir))
            print(f"TensorBoard logs at: {tb_log_dir}")
        elif SummaryWriter is None and self.is_main_process:
            print("TensorBoard SummaryWriter not available; skipping TB logging.")

        backend_hy = build_hyena_backend(model_dir=self.train_config.model_dir, model_name=self.model_config.hyena_model_name)
        hyena_model = backend_hy.model.to(self.device)

        # Tokenizer setup code omitted for brevity as it relies on external DNATok logic
        # which is handled by self.fast_tokenizer_fn or fallback.
        # Assuming fast_tokenizer_fn is properly set up by caller or we use default.
        if self.fast_tokenizer_fn is None:
             # Try to discover DNATok if not provided
            from src.utils.dna_tokenizer import DNATok
            from ..data.tokenizer import get_dnatok_batch_fn
            
            dna_tok = DNATok(embedder=backend_hy)
            dna_tok.discover()
            if not dna_tok.use_ids_path or dna_tok.ascii_lut is None:
                if self.is_main_process:
                    print("[WARN] DNATok IDs path not available; falling back to Hyena tokenizer.")
                fast_tokenizer_fn_local = None
            else:
                if self.is_main_process:
                    print("[INFO] DNATok ASCII→ID path enabled.")
                fast_tokenizer_fn_local = get_dnatok_batch_fn(backend_hy, dna_tok)
        else:
            fast_tokenizer_fn_local = self.fast_tokenizer_fn
        
        self.fast_tokenizer_fn = fast_tokenizer_fn_local # Update instance var

        # Dummy forward to infer d_model
        dummy_seq = [seqs[0]]
        dummy_enc = tokenize_batch(
            backend_hy,
            dummy_seq,
            fast_tokenizer_fn=self.fast_tokenizer_fn,
            max_tokens=self.data_config.max_tokens,
        )
        dummy_ids = dummy_enc["input_ids"].to(self.device)
        with torch.no_grad():
            dummy_out = hyena_model(
                input_ids=dummy_ids,
                return_dict=True,
                output_hidden_states=(self.model_config.hyena_hidden_layer_idx != -1),
            )
            if self.model_config.hyena_hidden_layer_idx == -1:
                dummy_hidden = dummy_out.last_hidden_state
            else:
                if not hasattr(dummy_out, "hidden_states") or dummy_out.hidden_states is None:
                    raise RuntimeError("Hyena model did not return hidden_states in dummy forward.")
                dummy_hidden = dummy_out.hidden_states[self.model_config.hyena_hidden_layer_idx]
        d_model = dummy_hidden.size(-1)
        if self.is_main_process:
            print(f"[DEBUG] Inferred d_model={d_model} from Hyena hidden states.")

        head = HyenaChrBinHead(
            d_model=d_model,
            num_chromosomes=num_chromosomes,
            num_bins=num_bins_global,
            hidden_dim=self.model_config.hidden_dim,
            conv_kernel_sizes=self.model_config.conv_kernel_sizes,
            conv_dilations=self.model_config.conv_dilations,
            conv_downsample=self.model_config.conv_downsample,
            dropout=self.model_config.dropout,
            label_smoothing=self.model_config.label_smoothing,
            num_coarse_bins=num_coarse_bins,
            w_chr=self.loss_config.w_chr,
            w_start=self.loss_config.w_start,
            w_end=self.loss_config.w_end,
            w_coarse_start=self.loss_config.w_coarse_start,
            w_coarse_end=self.loss_config.w_coarse_end,
            w_start_dist=self.loss_config.w_start_dist,
            w_end_dist=self.loss_config.w_end_dist,
        ).to(self.device)

        if loaded_hyena_state is not None:
            missing, unexpected = hyena_model.load_state_dict(loaded_hyena_state, strict=False)
            if self.is_main_process:
                print("[INFO] Loaded Hyena backbone from checkpoint.")
                if missing:
                    print(f"  [resume] missing Hyena keys: {missing}")
                if unexpected:
                    print(f"  [resume] unexpected Hyena keys: {unexpected}")

        if loaded_head_state is not None:
            missing, unexpected = head.load_state_dict(loaded_head_state, strict=False)
            if self.is_main_process:
                print("[INFO] Loaded head from checkpoint.")
                if missing:
                    print(f"  [resume] missing head keys: {missing}")
                if unexpected:
                    print(f"  [resume] unexpected head keys: {unexpected}")

        compiled_used = False
        if self.model_config.use_compile and hasattr(torch, "compile") and self.device.type == "cuda":
            triton_ok = True
            try:
                import torch.utils._triton as _triton
                _ = _triton.triton_hash_with_backend()
            except Exception as e:
                if self.is_main_process:
                    print(
                        "[WARN] Triton backend unavailable or misconfigured for torch.compile "
                        f"(likely libcuda.so issue): {e}\n"
                        "       Disabling torch.compile and falling back to eager."
                    )
                triton_ok = False

            if triton_ok:
                try:
                    compile_mode_local = self.model_config.compile_mode
                    if compile_mode_local == "max-autotune":
                        compile_mode_local = "max-autotune-no-cudagraphs"
                        if self.is_main_process:
                            print(
                                "[INFO] Switching torch.compile mode from 'max-autotune' "
                                "to 'max-autotune-no-cudagraphs' to avoid internal CUDA "
                                "graph state reuse issues."
                            )

                    head = torch.compile(
                        head,
                        mode=compile_mode_local,
                    )
                    compiled_used = True
                    if self.is_main_process:
                        print(
                            f"[INFO] torch.compile enabled for HyenaChrBinHead "
                            f"(mode='{compile_mode_local}', triton.cudagraphs=False)."
                        )
                        print(
                            "[INFO] Hyena backbone left in eager mode (no torch.compile) "
                            "to avoid CUDAGraphs output reuse errors."
                        )
                except Exception as e:
                    if self.is_main_process:
                        print(f"[WARN] torch.compile failed at compile() call: {e}. Continuing without compilation.")
        elif self.model_config.use_compile and self.device.type != "cuda":
            if self.is_main_process:
                print("[WARN] torch.compile requested but device is not CUDA; skipping compilation.")

        if self.distributed:
            ddp_device_id = self.device.index if self.device.type == "cuda" else None
            hyena_model = DDP(
                hyena_model,
                device_ids=[ddp_device_id] if ddp_device_id is not None else None,
                output_device=ddp_device_id,
                find_unused_parameters=True,
            )
            head = DDP(
                head,
                device_ids=[ddp_device_id] if ddp_device_id is not None else None,
                output_device=ddp_device_id,
                find_unused_parameters=True,
            )
            if self.is_main_process:
                print(f"[DIST] Wrapped Hyena backbone and head with DDP on device {ddp_device_id}")

        hyena_controller = HyenaLayerController(
            model=hyena_model.module if self.distributed else hyena_model,
            freeze_backbone=self.model_config.freeze_hyena,
            start_epoch=self.model_config.hyena_unfreeze_start_epoch,
            end_epoch=self.model_config.hyena_unfreeze_end_epoch,
            max_unfrozen_layers=self.model_config.hyena_max_unfrozen_layers,
        )

        lr = self.train_config.lr
        lr_head = self.train_config.lr_head
        lr_backbone = self.train_config.lr_backbone
        
        if lr_backbone is None and lr_head is None:
            lr_head_eff = lr
            lr_backbone_eff = lr * 0.1
        else:
            lr_head_eff = lr_head if lr_head is not None else lr
            lr_backbone_eff = lr_backbone if lr_backbone is not None else lr

        if self.is_main_process:
            print(
                f"[INFO] Using separate learning rates - "
                f"backbone_lr={lr_backbone_eff:.3g}, head_lr={lr_head_eff:.3g}"
            )

        if self.distributed:
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

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.train_config.num_epochs),
            eta_min=lr * 0.1,
        )

        if self.train_config.use_amp and self.device.type == "cuda" and self.train_config.amp_dtype == torch.float16:
            scaler: Optional[torch.amp.GradScaler] = torch.amp.GradScaler("cuda")
            if self.is_main_process:
                print(f"[INFO] Using AMP with dtype={self.train_config.amp_dtype} and GradScaler.")
        else:
            scaler = None
            if self.train_config.use_amp and self.device.type == "cuda" and self.is_main_process:
                print(f"[INFO] Using AMP with dtype={self.train_config.amp_dtype} (no GradScaler needed).")
            elif self.train_config.use_amp and self.is_main_process:
                print("[INFO] AMP requested but no CUDA device; running without AMP.")

        if loaded_opt_state is not None:
            try:
                optimizer.load_state_dict(loaded_opt_state)
                if self.is_main_process:
                    print("[INFO] Loaded optimizer state from checkpoint.")
            except Exception as e:
                if self.is_main_process:
                    print(f"[WARN] Failed to load optimizer state from checkpoint: {e}")

        if loaded_sched_state is not None:
            try:
                scheduler.load_state_dict(loaded_sched_state)
                if self.is_main_process:
                    print("[INFO] Loaded scheduler state from checkpoint.")
            except Exception as e:
                if self.is_main_process:
                    print(f"[WARN] Failed to load scheduler state from checkpoint: {e}")

        hyena_controller.apply(epoch=start_epoch)

        epochs_no_improve = 0
        use_profiler = self.train_config.use_profiler and _HAVE_PROFILER and self.device.type == "cuda"
        profiler_dir = self.save_dir / "tb_profiler"

        for epoch in range(start_epoch, self.train_config.num_epochs):
            if self.is_main_process:
                print(f"\n========== Epoch {epoch} / {self.train_config.num_epochs - 1} ==========")

            hyena_controller.apply(epoch=epoch)

            if self.distributed:
                train_sampler = getattr(train_loader, "sampler", None)
                if isinstance(train_sampler, DistributedSampler):
                    train_sampler.set_epoch(epoch)

            if use_profiler and epoch == 0 and self.is_main_process:
                profiler_dir.mkdir(parents=True, exist_ok=True)
                num_train_steps = len(train_loader)
                prof_active = max(1, min(int(self.train_config.profiler_max_steps), num_train_steps))

                prof_schedule = schedule(
                    wait=0,
                    warmup=1,
                    active=prof_active,
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
                    train_metrics, step_inc = self._run_epoch(
                        epoch=epoch,
                        model_hyena=hyena_model,
                        head=head,
                        backend_hyena=backend_hy,
                        dataloader=train_loader,
                        optimizer=optimizer,
                        scaler=scaler,
                        is_train=True,
                        profiler_ctx=prof,
                        compiled_used=compiled_used,
                    )
                    global_step += step_inc
            else:
                train_metrics, step_inc = self._run_epoch(
                    epoch=epoch,
                    model_hyena=hyena_model,
                    head=head,
                    backend_hyena=backend_hy,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    scaler=scaler,
                    is_train=True,
                    profiler_ctx=None,
                    compiled_used=compiled_used,
                )
                global_step += step_inc

            if val_loader is not None:
                val_metrics, _ = self._run_epoch(
                    epoch=epoch,
                    model_hyena=hyena_model,
                    head=head,
                    backend_hyena=backend_hy,
                    dataloader=val_loader,
                    optimizer=None,
                    scaler=None,
                    is_train=False,
                    profiler_ctx=None,
                    compiled_used=compiled_used,
                )
            else:
                val_metrics = None

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
            if self.is_main_process:
                print(log_msg)

            scheduler.step()
            current_lr_backbone = optimizer.param_groups[0]["lr"]
            current_lr_head = optimizer.param_groups[1]["lr"]
            if self.is_main_process:
                print(
                    f"[INFO] Learning rate after epoch {epoch}: "
                    f"backbone={current_lr_backbone:.6g}, head={current_lr_head:.6g}"
                )

            if writer is not None and self.is_main_process:
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

            if val_metrics is not None:
                monitor_metrics = val_metrics
                prefix = "val"
            else:
                monitor_metrics = train_metrics
                prefix = "train"

            if self.train_config.early_stop_use_acc:
                metric = -monitor_metrics.start_acc1_adj
                cur_metric_label = f"neg_{prefix}_start_acc@1+adj"
            else:
                metric = monitor_metrics.loss
                cur_metric_label = f"{prefix}_loss"

            stop_training = False

            if self.is_main_process:
                def _make_checkpoint_state() -> dict:
                    model_to_save = hyena_model.module if isinstance(hyena_model, DDP) else hyena_model
                    head_to_save = head.module if isinstance(head, DDP) else head
                    return {
                        "epoch": epoch,
                        "global_step": global_step,
                        "hyena_model_name": self.model_config.hyena_model_name,
                        "hyena_hidden_layer_idx": self.model_config.hyena_hidden_layer_idx,
                        "num_chromosomes": num_chromosomes,
                        "num_bins_global": num_bins_global,
                        "coarse_factor": self.data_config.coarse_factor,
                        "num_coarse_bins": num_coarse_bins,
                        "state_dict_hyena": model_to_save.state_dict(),
                        "state_dict_head": head_to_save.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_metric": best_metric,
                        "best_epoch": best_epoch,
                        "metric_label": best_metric_label,
                    }

                if epoch == start_epoch and best_metric is None:
                    best_metric = metric
                    best_epoch = epoch
                    best_metric_label = cur_metric_label
                    epochs_no_improve = 0
                    best_path = self.save_dir / "best_head.pt"
                    torch.save(_make_checkpoint_state(), best_path)
                    print(
                        f"[INFO] Initial best checkpoint saved at epoch {epoch} "
                        f"({best_metric_label}={best_metric:.4f}) to {best_path}"
                    )
                else:
                    if best_metric is None or metric < best_metric - self.train_config.min_delta:
                        best_metric = metric
                        best_epoch = epoch
                        best_metric_label = cur_metric_label
                        epochs_no_improve = 0
                        best_path = self.save_dir / "best_head.pt"
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

                        if self.train_config.early_stop_patience > 0 and epochs_no_improve >= self.train_config.early_stop_patience:
                            print(
                                f"[INFO] Early stopping triggered at epoch {epoch} "
                                f"(no improvement for {epochs_no_improve} epochs)."
                            )
                            last_path = self.save_dir / "last.pt"
                            torch.save(_make_checkpoint_state(), last_path)
                            print(f"[INFO] Saved last checkpoint to {last_path}")
                            stop_training = True

                last_path = self.save_dir / "last.pt"
                torch.save(_make_checkpoint_state(), last_path)
                print(f"[INFO] Saved last checkpoint to {last_path}")

            if self.distributed and dist.is_available() and dist.is_initialized():
                stop_tensor = torch.tensor(1 if stop_training else 0, device=self.device, dtype=torch.int)
                dist.broadcast(stop_tensor, src=0)
                stop_training = bool(stop_tensor.item())

            if stop_training:
                break

        if writer is not None and self.is_main_process:
            writer.flush()
            writer.close()

        if self.is_main_process:
            print("Training finished.")
            if best_metric is not None and best_epoch is not None and best_metric_label is not None:
                print(f"Best epoch: {best_epoch} with metric={best_metric:.4f} ({best_metric_label})")
            else:
                print("No best epoch recorded (metric never finite?).")

        if self.distributed and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
