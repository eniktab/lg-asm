#!/usr/bin/env python3
from __future__ import annotations

import sys; print('Python %s on %s' % (sys.version, sys.platform))
# Preserve original path extension
sys.path.extend(['/g/data/te53/en9803/workspace/sync/ANU/graph_genomic_data/'])

import torch
# Set global flags
_HAS_CUDAGRAPH_MARK = hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

from src.config import DataConfig, ModelConfig, LossConfig, TrainingConfig
from src.training.trainer import Trainer

def main():
    # Hardcoded configuration matching the original script's __main__ block
    PT_PATH = "/g/data/te53/en9803/sandpit/graph_genomics/chr22/simulated_v2/pbsim3_chr22_depth35_acc85_train_demo.pt"
    MODEL_DIR = "/g/data/te53/en9803/data/scratch/hf-cache/models/"
    SAVE_DIR = "/g/data/te53/en9803/sandpit/graph_genomics/chr22/simulated_v2/tests"
    RESUME_CKPT = None 

    data_config = DataConfig(
        pt_path=PT_PATH,
        val_fraction=0.1,
        dataloader_num_workers=8,
        max_tokens=10_000,
        coarse_factor=100,
    )

    model_config = ModelConfig(
        hyena_model_name="hyenadna-large-1m-seqlen-hf",
        hyena_hidden_layer_idx=-1,
        hidden_dim=1024,
        conv_kernel_sizes=(3, 7, 15),
        conv_dilations=(1, 2, 4),
        conv_downsample=4,
        dropout=0.1,
        label_smoothing=0.05,
        freeze_hyena=True,
        hyena_unfreeze_start_epoch=40,
        hyena_unfreeze_end_epoch=45,
        hyena_max_unfrozen_layers=1,
        use_compile=False,
        compile_mode="max-autotune-no-cudagraphs",
    )

    loss_config = LossConfig(
        w_chr=0.2,
        w_start=1.0,
        w_end=1.0,
        w_coarse_start=0.2,
        w_coarse_end=0.2,
        w_start_dist=0.5,
        w_end_dist=0.5,
    )

    train_config = TrainingConfig(
        save_dir=SAVE_DIR,
        model_dir=MODEL_DIR,
        batch_size=256,
        num_epochs=45,
        lr=1e-4,
        lr_backbone=None, # Will default to lr * 0.1 inside Trainer
        lr_head=None,     # Will default to lr inside Trainer
        accum_steps=2,
        use_amp=True,
        amp_dtype=torch.bfloat16,
        use_profiler=False,
        profiler_max_steps=50,
        early_stop_patience=10,
        early_stop_use_acc=True,
        min_delta=0.0,
        resume_ckpt=RESUME_CKPT,
    )

    trainer = Trainer(
        train_config=train_config,
        model_config=model_config,
        data_config=data_config,
        loss_config=loss_config,
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
