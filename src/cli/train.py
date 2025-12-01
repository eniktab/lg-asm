"""Command-line entrypoint for running Hyena chromosome/bin training."""
from __future__ import annotations

from typing import Optional

import torch

from src.training.runner import train_from_pt


if __name__ == "__main__":
    PT_PATH = "/g/data/te53/en9803/sandpit/graph_genomics/chr22/simulated_v2/pbsim3_chr22_depth35_acc85_train_demo.pt"
    MODEL_DIR = "/g/data/te53/en9803/data/scratch/hf-cache/models/"
    SAVE_DIR = "/g/data/te53/en9803/sandpit/graph_genomics/chr22/simulated_v2/tests"

    RESUME_CKPT: Optional[str] = None
    # RESUME_CKPT = "/g/data/te53/en9803/sandpit/graph_genomics/chr22/simulated_v2/chr22_modern_v3_ckpts/best_head.pt"

    train_from_pt(
        pt_path=PT_PATH,
        model_dir=MODEL_DIR,
        save_dir=SAVE_DIR,
        batch_size=256,
        num_epochs=45,
        lr=1e-4,
        # lr_backbone / lr_head can be overridden; by default:
        #   lr_head = lr
        #   lr_backbone = lr * 0.1
        val_fraction=0.1,
        accum_steps=2,
        hyena_hidden_layer_idx=-1,
        hyena_model_name="hyenadna-large-1m-seqlen-hf",
        freeze_hyena=True,
        hyena_unfreeze_start_epoch=40,
        hyena_unfreeze_end_epoch=45,
        hyena_max_unfrozen_layers=1,
        use_amp=True,
        amp_dtype=torch.bfloat16,     # bfloat16 AMP, no scaler
        use_profiler=False,
        profiler_max_steps=50,
        dataloader_num_workers=8,
        fast_tokenizer_fn=None,       # DNATok is used by default if available
        early_stop_patience=10,
        early_stop_use_acc=True,
        min_delta=0.0,
        coarse_factor=100,            # e.g. 100 for 1Mb bins if fine bins are 10kb
        coarse_loss_weight=0.2,
        resume_ckpt=RESUME_CKPT,
        use_compile=True,             # will auto-disable if Triton/libcuda is broken
        compile_mode="max-autotune-no-cudagraphs",
        hyena_max_tokens=10_000,      # hard cap on read length
    )
