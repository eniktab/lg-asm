from __future__ import annotations

import inspect
import logging
from collections import deque
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch


def _as_torch_device(device: object) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return torch.device("cuda", device)
    if isinstance(device, str):
        return torch.device(device)
    # Fallback: assume default CUDA
    return torch.device("cuda")


def _is_cuda_device(device: object) -> bool:
    try:
        return torch.device(device).type == "cuda"
    except Exception:
        return False


class DNATok:
    """
    KevTok IDs-path helper.

    Encapsulates the "IDs path" for KevTok:
      - Discovering ASCII→token ID lookup table (LUT) from an embedder/tokenizer
      - Efficient vectorized encoding of equal-length DNA strings to IDs
      - Optional left-padding to a fixed token length (if model requires it)
      - Safe micro-batching of embed_tokens() to avoid 32-bit index math / OOM
      - Persistent pinned CPU staging buffers and host→device (H2D) / compute
        overlap with ping–pong device buffers, with optional int32 H2D to halve
        bandwidth.

    Requirements for the embedder:
      - An `embed_tokens(LongTensor[B,T]) -> Tensor[B,D]` method. If absent,
        `use_ids_path` will be False and callers should use a string-based path.
      - (Recommended) `embedder.tokenizer` compatible with Hugging Face
        Tokenizers / Transformers / Vortex tokenizers. When missing, a
        conservative fixed DNA mapping {A:1,C:2,G:3,T:4,N:0} is used.
    """

    # Default raised from 256k → 1M tokens per call now that 32-bit index
    # math failures are handled with on-the-fly shrinking.
    DEFAULT_IDS_MAX_TOKENS_PER_CALL = 1_048_576

    def __init__(
        self,
        embedder: object,
        ids_max_tokens_per_call: int = DEFAULT_IDS_MAX_TOKENS_PER_CALL,
        logger: Optional[logging.Logger] = None,
        *,
        prefer_int32_h2d: bool = True,
        overlap_h2d_compute: bool = True,
        force_fp32_outputs: bool = True,  # keep legacy behavior by default
    ) -> None:
        self.embedder = embedder
        self.log = logger or logging.getLogger("KevTokIDsPathHelper")
        if not self.log.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            self.log.addHandler(ch)

        # discovered at runtime by discover()
        self.use_ids_path: bool = False
        self.ascii_lut: Optional[np.ndarray] = None  # shape [256] int64
        self.id_pad: int = 0
        self.id_N: int = 0
        self.token_len: Optional[int] = None

        # runtime safety cap
        self.ids_max_tokens_per_call: int = int(ids_max_tokens_per_call)

        # performance knobs
        self.prefer_int32_h2d: bool = bool(prefer_int32_h2d)
        self.overlap_h2d_compute: bool = bool(overlap_h2d_compute)
        self.force_fp32_outputs: bool = bool(force_fp32_outputs)

        # persistent staging (CPU) and ping–pong (CUDA)
        self._staging_ids_cpu: Optional[torch.Tensor] = None  # int32 or int64
        self._staging_bytes_cpu: Optional[torch.Tensor] = None  # uint8
        self._dev_ping_i: Optional[torch.Tensor] = None  # int32 or int64
        self._dev_pong_i: Optional[torch.Tensor] = None
        self._dev_ping_l: Optional[torch.Tensor] = None  # int64
        self._dev_pong_l: Optional[torch.Tensor] = None
        self._lut_cuda: Optional[torch.Tensor] = None  # cached 256 LUT on device

    # ---------------------------------------------------------------------
    # Tokenizer utilities (guarded; never assume .vocab is safe)
    # ---------------------------------------------------------------------

    def _maybe_unwrap_tokenizer(self, tok: object) -> object:
        inner = getattr(tok, "tokenizer", None)
        return inner if inner is not None else tok

    def _safe_get_vocab_dict(self, tok: object) -> Optional[Dict[str, int]]:
        get_vocab = getattr(tok, "get_vocab", None)
        if callable(get_vocab):
            try:
                v = get_vocab()
                if isinstance(v, dict):
                    return v
            except Exception:
                pass
        try:
            v = getattr(tok, "vocab")
            if isinstance(v, dict):
                return v
        except Exception:
            pass
        return None

    def _safe_token_to_id(self, tok: object, token_str: str) -> Optional[int]:
        fn = getattr(tok, "token_to_id", None)  # HF tokenizers
        if callable(fn):
            try:
                out = fn(token_str)
                if isinstance(out, int) and out >= 0:
                    return out
            except Exception:
                pass
        cti = getattr(tok, "convert_tokens_to_ids", None)  # Transformers
        if callable(cti):
            try:
                out = cti(token_str)
                if isinstance(out, int) and out >= 0:
                    return out
            except Exception:
                pass
        vocab = self._safe_get_vocab_dict(tok)
        if vocab and token_str in vocab and isinstance(vocab[token_str], int):
            return int(vocab[token_str])
        return None

    def _encode_char_to_single_id(self, tok: object, ch: str) -> Optional[int]:
        enc = getattr(tok, "encode", None)
        if not callable(enc):
            return None
        try:
            try:
                out = enc(ch, add_special_tokens=False)
            except TypeError:
                out = enc(ch)
            ids = out.ids if hasattr(out, "ids") else out
            if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], int):
                return ids[0]
        except Exception:
            pass
        return None

    def _discover_char_id(self, tok: object, ch: str) -> Optional[int]:
        tid = self._safe_token_to_id(tok, ch)
        if isinstance(tid, int):
            return tid
        code = ord(ch)
        for key in (f"<0x{code:02X}>", f"<0x{code:02x}>"):
            tid = self._safe_token_to_id(tok, key)
            if isinstance(tid, int):
                return tid
        tid = self._encode_char_to_single_id(tok, ch)
        if isinstance(tid, int):
            return tid
        return None

    def _discover_pad_id(self, tok_or_embedder: object) -> Optional[int]:
        # Fixed: actually use the provided object, not self.embedder unconditionally.
        src = tok_or_embedder
        for name in ("pad_id", "pad_token_id"):
            v = getattr(src, name, None)
            if isinstance(v, int):
                return v
        tok = getattr(src, "tokenizer", None)
        if tok is not None:
            tok = self._maybe_unwrap_tokenizer(tok)
            for token_str in ("<pad>", "[PAD]", "PAD", "pad"):
                tid = self._safe_token_to_id(tok, token_str)
                if isinstance(tid, int):
                    return tid
        return None

    def _resolve_base_ids_for_acgtn_safe(self, tok: object, pad_id: int) -> Dict[str, int]:
        char_to_id: Dict[str, int] = {}
        for ch in ("A", "C", "G", "T", "N", "a", "c", "g", "t", "n"):
            tid = self._discover_char_id(tok, ch)
            if isinstance(tid, int):
                char_to_id[ch] = tid
        for ch in ("A", "C", "G", "T", "N"):
            if ch in char_to_id and ch.lower() not in char_to_id:
                char_to_id[ch.lower()] = char_to_id[ch]
        if "N" not in char_to_id:
            char_to_id["N"] = char_to_id.get("n", pad_id)
        if "n" not in char_to_id:
            char_to_id["n"] = char_to_id["N"]
        return char_to_id

    def _build_ascii_lut_probe_all(
        self, tok: object, default_id: int, dna_overrides: Dict[str, int]
    ) -> np.ndarray:
        lut = np.full(256, int(default_id), dtype=np.int64)
        for b in range(256):
            tid = self._safe_token_to_id(tok, f"<0x{b:02X}>")
            if tid is None:
                tid = self._safe_token_to_id(tok, f"<0x{b:02x}>")
            if tid is None:
                ch = chr(b)
                if b in (9, 10, 13) or 32 <= b <= 126:
                    tid = self._encode_char_to_single_id(tok, ch)
            if isinstance(tid, int):
                lut[b] = int(tid)
        for ch, tid in dna_overrides.items():
            if len(ch) == 1:
                lut[ord(ch)] = int(tid)
        return lut

    # --------------------------- Discovery ---------------------------------
    def discover(self) -> None:
        embed_tokens = getattr(self.embedder, "embed_tokens", None)
        if not callable(embed_tokens):
            self.use_ids_path = False
            return
        pad_id = self._discover_pad_id(self.embedder)
        if pad_id is None:
            pad_id = 0
        tok = getattr(self.embedder, "tokenizer", None)
        if tok is None:
            char_to_id = {
                "A": 1,
                "C": 2,
                "G": 3,
                "T": 4,
                "N": 0,
                "a": 1,
                "c": 2,
                "g": 3,
                "t": 4,
                "n": 0,
            }
            n_id = char_to_id["N"]
            self.ascii_lut = np.full(256, n_id, dtype=np.int64)
            for ch, tid in char_to_id.items():
                self.ascii_lut[ord(ch)] = tid
            self.id_pad = int(pad_id)
            self.id_N = int(n_id)
            self.use_ids_path = True
            self.log.warning(
                "IDs path: tokenizer missing; using fixed DNA vocab {A:1,C:2,G:3,T:4,N:0}."
            )
            return
        tok = self._maybe_unwrap_tokenizer(tok)
        dna_ids = self._resolve_base_ids_for_acgtn_safe(tok, pad_id)
        n_id = int(dna_ids.get("N", pad_id))
        self.ascii_lut = self._build_ascii_lut_probe_all(
            tok, default_id=n_id, dna_overrides=dna_ids
        )
        self.id_pad = int(pad_id)
        self.id_N = int(n_id)
        # Optional fixed token length hint
        for name in ("model_max_length", "max_position_embeddings", "max_seq_len"):
            v = getattr(self.embedder, name, None)
            if isinstance(v, int) and v > 0:
                self.token_len = v
                break
        self.use_ids_path = True
        self.log.info(
            "KevTok IDs path enabled (embed_tokens): PAD=%d, N=%d",
            self.id_pad,
            self.id_N,
        )
        # Sanity: verify ACGTN mapping is self-consistent using both LUT and tokenizer.
        try:
            self._sanity_verify_mapping(tok)
        except Exception as e:
            self.log.warning("IDs path mapping verification raised: %s", e)

    def _sanity_verify_mapping(self, tok: object) -> None:
        """Ensure LUT-encoded ACGTN matches tokenizer-derived single-char ids.
        Raises on egregious inconsistencies; logs otherwise.
        """
        if self.ascii_lut is None:
            return
        test = "ACGTNacgtn"
        # Tokenizer single-char discovery
        tok_ids = []
        for ch in test:
            tid = self._discover_char_id(tok, ch)
            tok_ids.append(self.id_N if tid is None else int(tid))
        tok_ids = np.asarray(tok_ids, dtype=np.int64).reshape(1, -1)
        # LUT encoding
        lut_ids = self._encode_batch_ascii_lut_numpy([test], self.ascii_lut)
        if not np.array_equal(tok_ids, lut_ids):
            diff = (tok_ids != lut_ids).sum()
            self.log.warning(
                "IDs path: %d/%d LUT bytes differ from tokenizer single-char ids.",
                int(diff),
                tok_ids.size,
            )

    # ------------------------------ Encoding -------------------------------
    @staticmethod
    def _encode_batch_ascii_lut_numpy(seqs: List[str], lut: np.ndarray) -> np.ndarray:
        if lut is None:
            raise RuntimeError("DNATok.discover() must be called before encoding.")
        assert len(seqs) > 0
        T = len(seqs[0])
        for s in seqs:
            if len(s) != T:
                raise ValueError("All sequences in a batch must have equal length.")
        buf = ("".join(seqs)).encode("ascii", errors="ignore")
        arr = np.frombuffer(buf, dtype=np.uint8)
        if arr.size != len(seqs) * T:
            out = np.empty((len(seqs), T), dtype=np.uint8)
            for i, s in enumerate(seqs):
                out[i, :] = np.frombuffer(
                    s.encode("ascii", errors="replace"), dtype=np.uint8
                )[:T]
            arr = out
        else:
            arr = arr.reshape(len(seqs), T)
        return lut[arr]  # int64 [B,T]

    def _maybe_left_pad(self, ids_np: np.ndarray) -> np.ndarray:
        if self.token_len is not None and self.token_len > ids_np.shape[1]:
            pad = self.token_len - ids_np.shape[1]
            ids_np = np.pad(ids_np, ((0, 0), (pad, 0)), constant_values=self.id_pad)
        return ids_np

    def encode_batch_to_ids(self, seqs: List[str]) -> torch.Tensor:
        """Backward-compatible path: returns CPU pinned int64 [B,T]."""
        if not self.use_ids_path or self.ascii_lut is None:
            raise RuntimeError("IDs path not available; call discover() first.")
        ids_np = self._maybe_left_pad(
            self._encode_batch_ascii_lut_numpy(seqs, self.ascii_lut)
        )
        ids_cpu = torch.as_tensor(ids_np, dtype=torch.long)
        try:
            ids_cpu = ids_cpu.pin_memory()
        except Exception:
            pass
        return ids_cpu

    # New: persistent pinned staging (int32 or int64)
    def encode_batch_to_ids_staging(
        self, seqs: List[str], *, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Return persistent pinned CPU tensor (int32 by default for H2D), reused across calls."""
        if not self.use_ids_path or self.ascii_lut is None:
            raise RuntimeError("IDs path not available; call discover() first.")
        if dtype is None:
            dtype = torch.int32 if self.prefer_int32_h2d else torch.long
        ids_np = self._maybe_left_pad(
            self._encode_batch_ascii_lut_numpy(seqs, self.ascii_lut)
        )
        want_shape = (ids_np.shape[0], ids_np.shape[1])
        if (
            self._staging_ids_cpu is None
            or self._staging_ids_cpu.shape != want_shape
            or self._staging_ids_cpu.dtype != dtype
        ):
            # (Re)allocate pinned staging
            self._staging_ids_cpu = torch.empty(want_shape, dtype=dtype)
            try:
                self._staging_ids_cpu = self._staging_ids_cpu.pin_memory()
            except Exception:
                pass
        if dtype == torch.int32:
            self._staging_ids_cpu.copy_(
                torch.as_tensor(ids_np, dtype=torch.int32), non_blocking=False
            )
        else:
            self._staging_ids_cpu.copy_(
                torch.as_tensor(ids_np, dtype=torch.long), non_blocking=False
            )
        return self._staging_ids_cpu

    # -------------------------- H2D/Compute overlap ------------------------
    def _ensure_device_pingpong(self, micro_bs: int, T: int, device: object, use_i32: bool) -> None:
        dev = _as_torch_device(device)
        # If a fixed token length is known and larger than T, size to token_len once.
        alloc_T = max(int(T), int(self.token_len) if self.token_len else int(T))
        dtype_i = torch.int32 if use_i32 else torch.long
        # Upload buffers for copy stream
        for name in ("_dev_ping_i", "_dev_pong_i"):
            buf = getattr(self, name)
            if (
                buf is None
                or buf.shape != (micro_bs, alloc_T)
                or buf.dtype != dtype_i
                or buf.device != dev
            ):
                setattr(self, name, torch.empty((micro_bs, alloc_T), dtype=dtype_i, device=dev))
        # Long views for embedder call
        for name in ("_dev_ping_l", "_dev_pong_l"):
            buf = getattr(self, name)
            if (
                buf is None
                or buf.shape != (micro_bs, alloc_T)
                or buf.dtype != torch.long
                or buf.device != dev
            ):
                setattr(
                    self,
                    name,
                    torch.empty((micro_bs, alloc_T), dtype=torch.long, device=dev),
                )

    def _ids_micro_bs_for_T(self, T: int, emb_batch: int) -> int:
        tokens_budget = max(1, int(self.ids_max_tokens_per_call))
        per_call = max(1, tokens_budget // max(1, T))
        return max(1, min(int(emb_batch), per_call))

    def iter_embed_tokens_in_slices(
        self,
        ids_cpu: torch.Tensor,
        emb_batch: int,
        device: object = "cuda",
    ) -> Iterator[torch.Tensor]:
        """
        Baseline safe streaming without overlap (kept for backwards-compatibility).
        Yields CUDA activations [cur_bs, D]. Shrinks on index-math or OOM.
        """
        if not self.use_ids_path:
            raise RuntimeError("IDs path not available.")
        if _is_cuda_device(device) and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")
        B, T = int(ids_cpu.shape[0]), int(ids_cpu.shape[1])
        start_idx = 0
        micro_bs = self._ids_micro_bs_for_T(T, emb_batch)
        kwargs_template = {}
        try:
            sig = inspect.signature(self.embedder.embed_tokens)
            if "rc_invariant" in sig.parameters:
                kwargs_template["rc_invariant"] = False
        except Exception:
            pass
        dev = _as_torch_device(device)
        while start_idx < B:
            end_idx = min(B, start_idx + micro_bs)
            sub_cpu = ids_cpu[start_idx:end_idx]
            cur_bs = int(sub_cpu.shape[0])
            while True:
                try:
                    sub_dev = sub_cpu.to(device=dev, non_blocking=True)
                    with torch.amp.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                        enabled=_is_cuda_device(dev),
                    ):
                        out = self.embedder.embed_tokens(sub_dev, **kwargs_template)
                    if out.device != dev:
                        out = out.to(device=dev, non_blocking=True)
                    if self.force_fp32_outputs and out.dtype != torch.float32:
                        out = out.float()
                    yield out
                    break
                except RuntimeError as e:
                    msg = str(e).lower()
                    triggers = (
                        "canuse32bitindexmath" in msg
                        or "32-bit index" in msg
                        or ("conv1d" in msg and "index" in msg)
                        or "out of memory" in msg
                    )
                    if triggers and cur_bs > 1:
                        new_bs = max(1, cur_bs // 2)
                        if new_bs == cur_bs:
                            raise
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        cur_bs = new_bs
                        sub_cpu = ids_cpu[start_idx : start_idx + cur_bs]
                        continue
                    else:
                        raise
            start_idx += cur_bs

    def iter_embed_tokens_pipelined(
        self,
        ids_cpu: torch.Tensor,
        emb_batch: int,
        device: object = "cuda",
        *,
        use_int32_h2d: Optional[bool] = None,
    ) -> Iterator[torch.Tensor]:
        """
        Overlapping streamer: copies the next micro-batch on a separate CUDA
        stream while the current micro-batch computes on the default stream. If
        `use_int32_h2d` is True, host→device uses int32 then casts to int64 on
        device just before `embed_tokens`.

        Falls back to iter_embed_tokens_in_slices for non-CUDA devices or on
        failure mid-stream (without duplicating already-emitted outputs).
        """
        if not self.use_ids_path:
            raise RuntimeError("IDs path not available.")
        if not _is_cuda_device(device) or not torch.cuda.is_available():
            # Nothing to overlap; just stream baseline.
            yield from self.iter_embed_tokens_in_slices(ids_cpu, emb_batch, device=device)
            return

        dev = _as_torch_device(device)
        if use_int32_h2d is None:
            use_int32_h2d = self.prefer_int32_h2d

        B, T = int(ids_cpu.shape[0]), int(ids_cpu.shape[1])
        if B == 0:
            return

        micro_bs = self._ids_micro_bs_for_T(T, emb_batch)
        self._ensure_device_pingpong(micro_bs, T, dev, use_i32=use_int32_h2d)

        copy_stream = torch.cuda.Stream(device=dev)
        ready_ping = torch.cuda.Event()
        ready_pong = torch.cuda.Event()

        kwargs_template = {}
        try:
            sig = inspect.signature(self.embedder.embed_tokens)
            if "rc_invariant" in sig.parameters:
                kwargs_template["rc_invariant"] = False
        except Exception:
            pass

        # queue of (lo, hi, bs, use_ping) — now a deque for O(1) pops
        scheduled: deque[Tuple[int, int, int, bool]] = deque()

        def schedule_h2d(lo: int, hi: int, into_ping: bool) -> None:
            cur_bs = max(0, hi - lo)
            if cur_bs <= 0:
                return
            cur = ids_cpu[lo:hi]
            # write targets (consider token_len)
            alloc_T = max(T, int(self.token_len) if self.token_len else T)
            with torch.cuda.stream(copy_stream):
                if use_int32_h2d:
                    if cur.dtype != torch.int32:
                        cur = cur.to(torch.int32, copy=True)
                    dev_i = self._dev_ping_i if into_ping else self._dev_pong_i
                    dev_l = self._dev_ping_l if into_ping else self._dev_pong_l
                    # right-align if we have larger alloc_T due to token_len
                    dev_i[:cur_bs, -T:].copy_(cur, non_blocking=True)
                    # cast to long on device (no extra H2D)
                    dev_l[:cur_bs, -T:].copy_(dev_i[:cur_bs, -T:].to(torch.long))
                    (ready_ping if into_ping else ready_pong).record(copy_stream)
                else:
                    if cur.dtype != torch.long:
                        cur = cur.to(torch.long, copy=True)
                    dev_l = self._dev_ping_l if into_ping else self._dev_pong_l
                    dev_l[:cur_bs, -T:].copy_(cur, non_blocking=True)
                    (ready_ping if into_ping else ready_pong).record(copy_stream)
            scheduled.append((lo, hi, cur_bs, into_ping))

        # Prime up to two micro-batches (ping then pong)
        next_lo = 0
        use_ping = True
        while next_lo < B and len(scheduled) < 2:
            next_hi = min(B, next_lo + micro_bs)
            schedule_h2d(next_lo, next_hi, into_ping=use_ping)
            use_ping = not use_ping
            next_lo = next_hi

        # Main pipeline loop
        while scheduled:
            lo, hi, cur_bs, use_ping = scheduled.popleft()
            if cur_bs <= 0:
                continue
            ready_ev = ready_ping if use_ping else ready_pong
            dev_l_full = self._dev_ping_l if use_ping else self._dev_pong_l
            # effective slice accounts for potential left-padding area
            dev_slice = dev_l_full[:cur_bs]

            # Ensure H2D copy for this batch is visible to default stream
            torch.cuda.current_stream(device=dev).wait_event(ready_ev)

            # While we compute on this buffer, schedule another batch (if any)
            if next_lo < B:
                next_hi = min(B, next_lo + micro_bs)
                # Reuse the buffer we are *not* currently using.
                schedule_h2d(next_lo, next_hi, into_ping=not use_ping)
                next_lo = next_hi

            # Compute on default stream
            try:
                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=True,
                ):
                    out = self.embedder.embed_tokens(dev_slice, **kwargs_template)
                if out.device != dev:
                    out = out.to(device=dev, non_blocking=True)
                if self.force_fp32_outputs and out.dtype != torch.float32:
                    out = out.float()
                yield out
            except RuntimeError as e:
                # Fallback: if we trip index-math / OOM, stream the remainder
                # (from `lo` onwards) using the baseline helper. Already-emitted
                # batches are not recomputed.
                self.log.warning(
                    "KevTok pipelined path failed (%s); falling back to baseline from index %d.",
                    str(e),
                    lo,
                )
                remaining = ids_cpu[lo:]
                if remaining.numel() > 0:
                    yield from self.iter_embed_tokens_in_slices(
                        remaining, emb_batch, device=dev
                    )
                return

        torch.cuda.synchronize(dev)

    # -------------------- ASCII bytes → ids (device) -----------------------
    def encode_batch_to_ascii_bytes(self, seqs: List[str]) -> torch.Tensor:
        if not seqs:
            raise ValueError("No sequences provided.")
        T = len(seqs[0])
        for s in seqs:
            if len(s) != T:
                raise ValueError("All sequences must have equal length.")

        # Use bytearray so the NumPy view is writable (avoids torch warning).
        buf = bytearray(("".join(seqs)).encode("ascii", errors="replace"))
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(len(seqs), T)  # writable

        if self._staging_bytes_cpu is None or self._staging_bytes_cpu.shape != arr.shape:
            self._staging_bytes_cpu = torch.empty(arr.shape, dtype=torch.uint8)
            try:
                self._staging_bytes_cpu = self._staging_bytes_cpu.pin_memory()
            except Exception:
                pass

        # Source is now writable → torch.from_numpy(arr) is safe and quiet.
        self._staging_bytes_cpu.copy_(torch.from_numpy(arr))
        return self._staging_bytes_cpu

    def ids_from_ascii_bytes_cuda(
        self, ascii_bytes_cpu: torch.Tensor, device: object = "cuda:0"
    ) -> torch.Tensor:
        if self.ascii_lut is None:
            raise RuntimeError(
                "discover() must be called before mapping bytes→ids."
            )
        if ascii_bytes_cpu.dtype != torch.uint8 or ascii_bytes_cpu.device.type != "cpu":
            raise TypeError("ascii_bytes_cpu must be CPU uint8 tensor.")
        dev = _as_torch_device(device)
        if self._lut_cuda is None or self._lut_cuda.device != dev:
            self._lut_cuda = torch.as_tensor(self.ascii_lut, dtype=torch.long, device=dev)
        ascii_dev = ascii_bytes_cpu.to(dev, non_blocking=True)
        ids_dev = self._lut_cuda[ascii_dev.long()]
        return ids_dev

    # -------------------- Device-side left padding -------------------------
    def _left_pad_device(self, ids_dev: torch.Tensor, T: int, dev: torch.device) -> torch.Tensor:
        """Left-pad on device to self.token_len if needed (right-align original)."""
        if not self.token_len or self.token_len <= T:
            return ids_dev
        B = ids_dev.shape[0]
        out = torch.empty((B, self.token_len), dtype=ids_dev.dtype, device=dev)
        out.fill_(self.id_pad)
        out[:, -T:].copy_(ids_dev)
        return out

    # Convenience: end-to-end from strings with overlap or bytes path
    def embed_from_strings(
        self,
        seqs: List[str],
        emb_batch: int,
        device: object = "cuda",
        *,
        path: str = "ids",  # "ids" (legacy default), "bytes", or "auto"
    ) -> Iterator[torch.Tensor]:
        """
        End-to-end convenience wrapper.
        - path="ids"  : legacy behavior (CPU ids staging, optional overlap)
        - path="bytes": CPU ascii bytes → device LUT map → device left-pad
        - path="auto" : choose "bytes" on CUDA else "ids"
        """
        if path not in ("ids", "bytes", "auto"):
            raise ValueError("path must be one of {'ids','bytes','auto'}")
        if path == "auto":
            path = "bytes" if (_is_cuda_device(device) and torch.cuda.is_available()) else "ids"

        if path == "bytes":
            if not self.use_ids_path:
                raise RuntimeError("IDs path not available; call discover() first.")
            dev = _as_torch_device(device)
            ascii_cpu = self.encode_batch_to_ascii_bytes(seqs)  # pinned uint8 [B,T]
            B, T = int(ascii_cpu.shape[0]), int(ascii_cpu.shape[1])
            if B == 0:
                return
            # Copy bytes and map to ids on device
            if self._lut_cuda is None or self._lut_cuda.device != dev:
                self._lut_cuda = torch.as_tensor(self.ascii_lut, dtype=torch.long, device=dev)
            ascii_dev = ascii_cpu.to(dev, non_blocking=True)
            ids_dev = self._lut_cuda[ascii_dev.long()]  # long [B,T]
            # Optional left-pad on device
            ids_dev = self._left_pad_device(ids_dev, T, dev)
            # Micro-batch purely on device
            micro_bs = self._ids_micro_bs_for_T(int(ids_dev.shape[1]), emb_batch)
            start = 0
            kwargs_template = {}
            try:
                sig = inspect.signature(self.embedder.embed_tokens)
                if "rc_invariant" in sig.parameters:
                    kwargs_template["rc_invariant"] = False
            except Exception:
                pass
            while start < B:
                end = min(B, start + micro_bs)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=_is_cuda_device(dev)):
                    out = self.embedder.embed_tokens(ids_dev[start:end], **kwargs_template)
                if out.device != dev:
                    out = out.to(device=dev, non_blocking=True)
                if self.force_fp32_outputs and out.dtype != torch.float32:
                    out = out.float()
                yield out
                start = end
            return

        # Legacy "ids" path (unchanged defaults)
        ids_cpu_staging = self.encode_batch_to_ids_staging(seqs)
        if self.overlap_h2d_compute:
            yield from self.iter_embed_tokens_pipelined(
                ids_cpu_staging, emb_batch, device=device
            )
        else:
            yield from self.iter_embed_tokens_in_slices(
                ids_cpu_staging.to(torch.long), emb_batch, device=device
            )
