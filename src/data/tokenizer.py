from typing import List, Dict, Optional, Callable
import torch
from src.utils.HyenaBackend import HyenaBackend
from src.utils.dna_tokenizer import DNATok

# Global buffer for DNATok
_dnatok_core_buffer: Dict[int, torch.Tensor] = {}

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

def get_dnatok_batch_fn(backend_hy: HyenaBackend, dna_tok: DNATok) -> Callable[[List[str]], Dict[str, torch.Tensor]]:
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
    return dnatok_batch_fn
