from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Tuple
import torch

@dataclass
class DataConfig:
    pt_path: Union[str, Path]
    val_fraction: float = 0.1
    dataloader_num_workers: int = 0
    max_tokens: int = 10_000
    coarse_factor: Optional[int] = None
    
@dataclass
class ModelConfig:
    hyena_model_name: str = "hyenadna-large-1m-seqlen-hf"
    hyena_hidden_layer_idx: int = -1
    d_model: Optional[int] = None # Inferred at runtime if None
    hidden_dim: int = 1024
    conv_kernel_sizes: Tuple[int, ...] = (3, 7, 15)
    conv_dilations: Tuple[int, ...] = (1, 2, 4)
    conv_downsample: int = 4
    dropout: float = 0.1
    label_smoothing: float = 0.05
    freeze_hyena: bool = True
    hyena_unfreeze_start_epoch: int = 5
    hyena_unfreeze_end_epoch: int = 10
    hyena_max_unfrozen_layers: int = 8
    use_compile: bool = True
    compile_mode: str = "max-autotune-no-cudagraphs"

@dataclass
class LossConfig:
    w_chr: float = 0.0
    w_start: float = 1.0
    w_end: float = 1.0
    w_coarse_start: float = 0.0
    w_coarse_end: float = 0.0
    w_start_dist: float = 0.0
    w_end_dist: float = 0.0

@dataclass
class TrainingConfig:
    save_dir: Union[str, Path]
    model_dir: Union[str, Path]
    batch_size: int = 256
    num_epochs: int = 45
    lr: float = 1e-4
    lr_backbone: Optional[float] = None
    lr_head: Optional[float] = None
    accum_steps: int = 4
    use_amp: bool = True
    amp_dtype: torch.dtype = torch.bfloat16
    use_profiler: bool = True
    profiler_max_steps: int = 50
    early_stop_patience: int = 7
    early_stop_use_acc: bool = True
    min_delta: float = 0.0
    resume_ckpt: Optional[Union[str, Path]] = None
