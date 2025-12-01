# ============================================================
# HyenaDNA backend (offline-first, masked pooling)
# ============================================================
import os
import re
import json
import math
from pathlib import Path
from functools import partial
from types import SimpleNamespace
from typing import *
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.modeling_outputs import BaseModelOutput


# Reverse-complement util (same mapping as in the dataset helper)
RC_MAP = str.maketrans({'A':'T','C':'G','G':'C','T':'A','a':'t','c':'g','g':'c','t':'a'})
def reverse_complement(seq: str) -> str:
    return seq.translate(RC_MAP)[::-1]

class CharacterTokenizer(PreTrainedTokenizer):
    def __init__(self, characters, model_max_length, padding_side='left',
                 default_add_special_tokens=True, **kwargs):
        """Character tokenizer for Hugging Face transformers.
        Args:
            characters (Sequence[str]): List of desired characters. Any character which
                is not included in this list will be replaced by a special token called
                [UNK] with id=6. Following are list of all of the special tokens with
                their corresponding ids:
                    "[CLS]": 0
                    "[SEP]": 1
                    "[BOS]": 2
                    "[MASK]": 3
                    "[PAD]": 4
                    "[RESERVED]": 5
                    "[UNK]": 6
                an id (starting at 7) will be assigned to each character.
            model_max_length (int): Model maximum sequence length.
        """
        self.characters = list(characters)
        self.model_max_length = model_max_length
        self._default_add_special_tokens = default_add_special_tokens

        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "[RESERVED]": 5,
            "[UNK]": 6,
            **{ch: i + 7 for i, ch in enumerate(self.characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

        # 2) Define special tokens
        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)
        mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)

        # 3) Call super().__init__ AFTER vocab is ready
        super().__init__(
            bos_token=bos_token,
            eos_token=sep_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs,
        )

        # ---- REQUIRED by HF base classes ----

    def __call__(self, *args, add_special_tokens=None, **kwargs):
        if add_special_tokens is None:
            add_special_tokens = self._default_add_special_tokens
        return super().__call__(*args, add_special_tokens=add_special_tokens, **kwargs)

    def get_vocab(self) -> Dict[str, int]:
        # Return a copy to avoid accidental external mutation
        return dict(self._vocab_str_to_int)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        result = cls + token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )
        result = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        result = len(cls + token_ids_0 + sep) * [0]
        if token_ids_1 is not None:
            result += len(token_ids_1 + sep) * [1]
        return result

    def get_config(self) -> Dict:
        return {
            "char_ords": [ord(ch) for ch in self.characters],
            "model_max_length": self.model_max_length,
        }

    @classmethod
    def from_config(cls, config: Dict) -> "CharacterTokenizer":
        return cls(
            characters=[chr(i) for i in config["char_ords"]],
            model_max_length=config["model_max_length"],
        )

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        # Save your minimal config, then let HF save the usual tokenizer files
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        with open(cfg_file, "w") as f:
            json.dump(self.get_config(), f, indent=4)
        # Optionally also store special tokens map for HF tooling
        super().save_pretrained(save_directory, **kwargs)

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        with open(cfg_file) as f:
            cfg = json.load(f)
        return cls.from_config(cfg)

class GPT2Embeddings(nn.Module):

    def __init__(self, embed_dim, vocab_size, max_position_embeddings, padding_idx=None,
                 word_embed_proj_dim=None, device=None, dtype=None):
        """
            If max_position_embeddings <= 0, there's no position embeddings
            If word_embe_proj_dim is not None (e.g., OPT-350m), we embed to that dimension
                the project up to embed_dim
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if word_embed_proj_dim is None:
            self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx,
                                                **factory_kwargs)
            self.project_in = None
        else:
            self.word_embeddings = nn.Embedding(vocab_size, word_embed_proj_dim,
                                                padding_idx=padding_idx, **factory_kwargs)
            self.project_in = nn.Linear(word_embed_proj_dim, embed_dim, bias=False,
                                        **factory_kwargs)
        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(max_position_embeddings, embed_dim,
                                                    **factory_kwargs)

    def forward(self, input_ids, position_ids=None):
        """
            input_ids: (batch, seqlen)
            position_ids: (batch, seqlen)
        """
        batch_size, seqlen = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
        if self.project_in is not None:
            embeddings = self.project_in(embeddings)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        return embeddings

class LMBackbone(nn.Module):

    def __init__(self, d_model: int, n_layer: int, d_inner: int, vocab_size: int,
                 process_group=None, layer=None,
                 attn_layer_idx=None, attn_cfg=None, max_position_embeddings=0,
                 resid_dropout: float = 0.0, embed_dropout: float = 0.1,
                 layer_norm_epsilon: float = 1e-5, initializer_cfg=None, residual_in_fp32=False,
                 device=None, dtype=None, **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.process_group = process_group
        self.residual_in_fp32 = residual_in_fp32
        # note max_position_embeddings is 0 for Hyena, and therefore isn't used
        self.embeddings = GPT2Embeddings(d_model, vocab_size, max_position_embeddings,
                                         **factory_kwargs)

        self.layers = nn.ModuleList([create_block(
            d_model, d_inner=d_inner,
            layer=layer, attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg, layer_norm_epsilon=layer_norm_epsilon,
            resid_dropout1=embed_dropout if i == 0 else resid_dropout,
            resid_dropout2=resid_dropout, residual_in_fp32=residual_in_fp32, layer_idx=i,
            **factory_kwargs,
        ) for i in range(n_layer)])

        self.drop_f = nn.Dropout(resid_dropout)
        self.ln_f = nn.LayerNorm(d_model, eps=layer_norm_epsilon, **factory_kwargs)

        self.apply(partial(_init_weights, n_layer=n_layer,
                           **(initializer_cfg if initializer_cfg is not None else {})))

    def forward(
            self,
            input_ids,
            position_ids=None,
            *,
            output_hidden_states: bool = False,
            return_dict: bool = False,
    ):
        hidden_states = self.embeddings(input_ids, position_ids=position_ids)
        residual = None

        all_hidden = () if output_hidden_states else None

        # Append BEFORE each block (captures embeddings for i=0, then inputs to blocks 1..N-1)
        for layer in self.layers:
            if output_hidden_states:
                all_hidden = all_hidden + (hidden_states,)
            hidden_states, residual = layer(hidden_states, residual)

        dropped = self.drop_f(hidden_states)
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))

        if output_hidden_states:
            all_hidden = all_hidden + (hidden_states,)

        if return_dict:
            return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden)
        return (hidden_states, all_hidden) if output_hidden_states else hidden_states

class SequenceDecoder(nn.Module):
    def __init__(
        self, d_model, d_output=None, l_output=None, use_lengths=False, mode="last"
    ):
        super().__init__()

        self.output_transform = nn.Identity() if d_output is None else nn.Linear(d_model, d_output)

        if l_output is None:
            self.l_output = None
            self.squeeze = False
        elif l_output == 0:
            # Equivalent to getting an output of length 1 and then squeezing
            self.l_output = 1
            self.squeeze = True
        else:
            assert l_output > 0
            self.l_output = l_output
            self.squeeze = False

        self.use_lengths = use_lengths
        self.mode = mode

        if mode == 'ragged':
            assert not use_lengths

    def forward(self, x, state=None, lengths=None, l_output=None):
        """
        x: (n_batch, l_seq, d_model)
        Returns: (n_batch, l_output, d_output)
        """

        if self.l_output is None:
            if l_output is not None:
                assert isinstance(l_output, int)  # Override by pass in
            else:
                # Grab entire output
                l_output = x.size(-2)
            squeeze = False
        else:
            l_output = self.l_output
            squeeze = self.squeeze

        if self.mode == "last":
            restrict = lambda x: x[..., -l_output:, :]
        elif self.mode == "first":
            restrict = lambda x: x[..., :l_output, :]
        elif self.mode == "pool":
            restrict = lambda x: (
                torch.cumsum(x, dim=-2)
                / torch.arange(
                    1, 1 + x.size(-2), device=x.device, dtype=x.dtype
                ).unsqueeze(-1)
            )[..., -l_output:, :]

            def restrict(x):
                L = x.size(-2)
                s = x.sum(dim=-2, keepdim=True)
                if l_output > 1:
                    c = torch.cumsum(x[..., -(l_output - 1) :, :].flip(-2), dim=-2)
                    c = F.pad(c, (0, 0, 1, 0))
                    s = s - c  # (B, l_output, D)
                    s = s.flip(-2)
                denom = torch.arange(
                    L - l_output + 1, L + 1, dtype=x.dtype, device=x.device
                )
                s = s / denom
                return s

        elif self.mode == "sum":
            restrict = lambda x: torch.cumsum(x, dim=-2)[..., -l_output:, :]
            # TODO use same restrict function as pool case
        elif self.mode == 'ragged':
            assert lengths is not None, "lengths must be provided for ragged mode"
            # remove any additional padding (beyond max length of any sequence in the batch)
            restrict = lambda x: x[..., : max(lengths), :]
        else:
            raise NotImplementedError(
                "Mode must be ['last' | 'first' | 'pool' | 'sum']"
            )

        # Restrict to actual length of sequence
        if self.use_lengths:
            assert lengths is not None
            x = torch.stack(
                [
                    restrict(out[..., :length, :])
                    for out, length in zip(torch.unbind(x, dim=0), lengths)
                ],
                dim=0,
            )
        else:
            x = restrict(x)

        if squeeze:
            assert x.size(-2) == 1
            x = x.squeeze(-2)

        x = self.output_transform(x)

        return x

    def step(self, x, state=None):
        # Ignore all length logic
        return self.output_transform(x)

class LinearResidual(nn.Linear):
    """Wrap nn.Linear to return the residual as well. For compatibility with FusedDense.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input), input

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, activation=F.gelu,
                 return_residual=False, device=None, dtype=None):
        """
        From https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/modules/mlp.py
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)

class SelfAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, causal=None, key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            causal: if passed, will override self.causal
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, S)
        """
        batch_size, seqlen = qkv.shape[0], qkv.shape[1]
        causal = self.causal if causal is None else causal
        q, k, v = qkv.unbind(dim=2)
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum('bthd,bshd->bhts', q, k * softmax_scale)
        if key_padding_mask is not None:
            padding_mask = torch.full((batch_size, seqlen), -10000.0, dtype=scores.dtype,
                                      device=scores.device)
            padding_mask.masked_fill_(key_padding_mask, 0.0)
            scores = scores + rearrange(padding_mask, 'b s -> b 1 1 s')
        if causal:
            # "triu_tril_cuda_template" not implemented for 'BFloat16'
            # So we have to construct the mask in float
            causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
            scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = F.dropout(attention, self.dropout_p if self.training else 0.0)
        output = torch.einsum('bhts,bshd->bthd', attention_drop, v)
        return output


class MHA(nn.Module):
    """Multi-head self-attention and cross-attention
    """

    def __init__(self, embed_dim, num_heads, bias=True, dropout=0.0,
                 softmax_scale=None, causal=False, layer_idx=None, dwconv=False,return_residual=False,device=None, dtype=None) -> None:
        """
            return_residual: whether to return the input x along with the output. This is for
                performance reason: for post-norm architecture, returning the input allows us
                to fuse the backward of nn.Linear with the residual connection.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.layer_idx = layer_idx
        self.dwconv = dwconv
        self.return_residual = return_residual

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads

        linear_cls = nn.Linear
        linear_resid_cls = LinearResidual
        inner_attn_cls =  SelfAttention

        if not self.return_residual:
            self.Wqkv = linear_cls(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        else:
            self.Wqkv = linear_resid_cls(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        if self.dwconv:
            self.dwconv_qkv = nn.Conv1d(3 * embed_dim, 3 * embed_dim, kernel_size=3, padding=2,
                                        groups=3 * embed_dim)

        self.inner_attn = inner_attn_cls(causal=causal, softmax_scale=softmax_scale,
                                         attention_dropout=dropout)

        # output projection always have the bias (for now)
        self.out_proj = linear_cls(embed_dim, embed_dim, **factory_kwargs)

    def forward(self, x, key_padding_mask=None, **kwargs):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
                cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
                is the is the sum of the sequence lengths in the batch.
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into x. Only applicable when using
                FlashAttention.
            max_seqlen: int. Maximum sequence length in the batch.
            key_padding_mask: boolean mask, True means to keep, False means to mask out.
                (batch, seqlen). Only applicable when not using FlashAttention.
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """

        kwargs = ({'key_padding_mask': key_padding_mask, **kwargs})

        if not self.return_residual:
            qkv = self.Wqkv(x)
        else:
            qkv, x = self.Wqkv(x)
        if self.dwconv:
            qkv = rearrange(self.dwconv_qkv(rearrange(qkv, 'b s d -> b d s'))[..., :-2],
                            'b d s -> b s d').contiguous()
        qkv = rearrange(qkv, '... (three h d) -> ... three h d', three=3, d=self.head_dim)

        context = self.inner_attn(qkv, **kwargs)

        out = self.out_proj(rearrange(context, '... h d -> ... (h d)'))
        return out if not self.return_residual else (out, x)

class Block(nn.Module):

    def __init__(self, dim, mixer_cls=None, mlp_cls=None, norm_cls=nn.LayerNorm,
                 dropout_cls=nn.Dropout, prenorm=True, resid_dropout1=0., resid_dropout2=0.,
                 drop_path1=0., drop_path2=0.,
                 return_residual=False,
                 residual_in_fp32=False):
        """
        From https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/modules/block.py
        For prenorm=True, this Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Dropout -> Add -> LN -> MHA -> Dropout -> Add -> LN -> MLP, returning both
        the hidden_states (output of the MLP) and the residual.
        This is for performance reasons, as we can fuse the dropout, add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        For prenorm=False, this Block has the same structure as a regular postnorm Transformer
        block: MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add -> LN.
        return_residual: whether each of the sub-layers (mixer and mlp) will return the residual.
        This is for performance reason: for post-norm architecture, returning the input allows us
        to fuse the backward of nn.Linear with the residual connection.
        """
        super().__init__()
        self.prenorm = prenorm
        self.return_residual = return_residual
        self.residual_in_fp32 = residual_in_fp32
        if self.residual_in_fp32:
            assert self.prenorm, 'residual_in_fp32 is only compatible with prenorm=True'
        if mixer_cls is None:
            mixer_cls = partial(MHA, num_heads=dim // 64)
        if mlp_cls is None:
            mlp_cls = partial(Mlp, hidden_features=4 * dim)
        self.mixer = mixer_cls()
        self.dropout1 = dropout_cls(resid_dropout1)
        self.drop_path1 = StochasticDepth(drop_path1, mode='row')
        self.norm1 = norm_cls(dim)
        self.mlp = mlp_cls(dim)
        if not isinstance(self.mlp, nn.Identity):
            self.dropout2 = dropout_cls(resid_dropout2)
            self.drop_path2 = StochasticDepth(drop_path2, mode='row')
            self.norm2 = norm_cls(dim)

    def forward(self, hidden_states, residual = None,
                mixer_subset=None, mixer_kwargs=None):
        r"""Pass the input through the encoder layer.
        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: if postnorm, residual=None, If prenorm, hidden_states = Attn/MLP(LN(residual))
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
        """
        if self.prenorm:
            dropped = self.drop_path1(self.dropout1(hidden_states))
            residual = (dropped + residual) if residual is not None else dropped
            hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
            if mixer_kwargs is None:
                mixer_kwargs = {}
            if mixer_subset is not None:
                mixer_kwargs['mixer_subset'] = mixer_subset
            hidden_states = self.mixer(hidden_states, **mixer_kwargs)
            if mixer_subset is not None:
                residual = residual[:, mixer_subset]
            if not isinstance(self.mlp, nn.Identity):
                dropped = self.drop_path2(self.dropout2(hidden_states))
                residual = (dropped + residual) if residual is not None else dropped
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)

                hidden_states = self.mlp(hidden_states)
            return hidden_states, residual
        else:
            assert residual is None
            mixer_out = self.mixer(
                hidden_states, **(mixer_kwargs if mixer_kwargs is not None else {})
            )
            if self.return_residual:  # mixer out is actually a pair here
                mixer_out, hidden_states = mixer_out

            hidden_states = self.norm1((self.drop_path1(self.dropout1(mixer_out))
                                        + hidden_states).to(dtype=self.norm1.weight.dtype))

            if not isinstance(self.mlp, nn.Identity):
                mlp_out = self.mlp(hidden_states)
                if self.return_residual:  # mlp out is actually a pair here
                    mlp_out, hidden_states = mlp_out

                hidden_states = self.norm2((self.drop_path2(self.dropout2(mlp_out))
                                            + hidden_states).to(dtype=self.norm2.weight.dtype))

            return hidden_states

class OptimModule(nn.Module):
    """ Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters """

    def register(self, name, tensor, lr=None, wd=0.0):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {}
            if lr is not None: optim["lr"] = lr
            if wd is not None: optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)

class PositionalEmbedding(OptimModule):
    def __init__(self, emb_dim: int, seq_len: int, lr_pos_emb: float=1e-5, **kwargs):
        """Complex exponential positional embeddings for Hyena filters."""
        super().__init__()

        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None] # 1, L, 1

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.register("z", z, lr=lr_pos_emb)
        self.register("t", t, lr=0.0)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]

class Sin(nn.Module):
    """The Sin activation function for the Hyena Filter function."""
    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim)) if train_freq else w * torch.ones(1, dim)

    def forward(self, x):
        return torch.sin(self.freq * x)

class ExponentialModulation(OptimModule):
    """The window function applied to the output of the (MLP) filter function."""
    def __init__(
        self,
        d_model,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        modulation_lr=0.0,
        modulate: bool=True,
        shift: float = 0.05,
        **kwargs
    ):
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register("deltas", deltas, lr=modulation_lr)

    def forward(self, t, x):
        if self.modulate:
            decay = torch.exp(-t * self.deltas.abs())
            x = x * (decay + self.shift)
        return x

def fftconv(u, k, D):
    """
    We apply a convolution through the fourier domain (from the Convolution Theorem)

    """
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen

    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    if len(u.shape) > 3: k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=u.dtype)

class HyenaFilter(OptimModule):
    def __init__(
            self,
            d_model,
            emb_dim=3,  # dim of input to MLP, augments with positional encoding
            order=16,  # width of the implicit MLP
            fused_fft_conv=False,
            seq_len=1024,
            lr=1e-3,
            lr_pos_emb=1e-5,
            dropout=0.0,
            w=1,  # frequency of periodic activations
            wd=0,  # weight decay of kernel parameters
            bias=True,
            num_inner_mlps=2,
            normalized=False,
            train_freq=True,
            **kwargs
    ):
        """
        Implicit long filter with modulation.

        Args:
            d_model: number of channels in the input
            emb_dim: dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP

        Note:
            filter_dropout is not implemented
        """
        super().__init__()

        self.d_model = d_model
        self.use_bias = bias
        self.fused_fft_conv = fused_fft_conv
        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(dropout)

        act = Sin(dim=order, w=w, train_freq=train_freq)
        self.emb_dim = emb_dim
        assert emb_dim % 2 != 0 and emb_dim >= 3, "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
        self.seq_len = seq_len

        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr_pos_emb)

        self.implicit_filter = nn.Sequential(
            nn.Linear(emb_dim, order),
            act,
        )
        for i in range(num_inner_mlps):
            self.implicit_filter.append(nn.Linear(order, order))
            self.implicit_filter.append(act)

        self.implicit_filter.append(nn.Linear(order, d_model, bias=False))

        self.modulation = ExponentialModulation(d_model, **kwargs)

        self.normalized = normalized
        for c in self.implicit_filter.children():
            for name, v in c.state_dict().items():
                optim = {"weight_decay": wd, "lr": lr}
                setattr(getattr(c, name), "_optim", optim)

    def filter(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        h = self.modulation(t, h)
        return h

    def forward(self, x, L, k=None, bias=None, *args, **kwargs):
        if k is None: k = self.filter(L)

        # Ensure compatibility with filters that return a tuple
        k = k[0] if type(k) is tuple else k

        y = fftconv(x, k, bias)
        return y

class HyenaOperator(nn.Module):
    def __init__(
            self,
            d_model,
            l_max,
            order=2,
            filter_order=64,
            dropout=0.0,
            filter_dropout=0.0,
            **filter_args,
        ):
        r"""
        Hyena operator described in the paper https://arxiv.org/pdf/2302.10866.pdf

        Args:
            d_model (int): Dimension of the input and output embeddings (width of the layer)
            l_max: (int): Maximum input sequence length. Defaults to None
            order: (int): Depth of the Hyena recurrence. Defaults to 2
            dropout: (float): Dropout probability. Defaults to 0.0
            filter_dropout: (float): Dropout probability for the filter. Defaults to 0.0
        """
        super().__init__()

        self.d_model = d_model
        self.l_max = l_max
        self.order = order
        inner_width = d_model * (order + 1)
        self.dropout = nn.Dropout(dropout)
        self.in_proj = nn.Linear(d_model, inner_width)
        self.out_proj = nn.Linear(d_model, d_model)

        self.short_filter = nn.Conv1d(
            inner_width,
            inner_width,
            3,
            padding=2,
            groups=inner_width
        )
        self.filter_fn = HyenaFilter(
            d_model * (order - 1),
            order=filter_order,
            seq_len=l_max,
            channels=1,
            dropout=filter_dropout,
            **filter_args
        )

    def forward(self, u, *args, **kwargs):
        l = u.size(-2)
        l_filter = min(l, self.l_max)
        u = self.in_proj(u)
        u = rearrange(u, 'b l d -> b d l')

        uc = self.short_filter(u)[...,:l_filter]
        *x, v = uc.split(self.d_model, dim=1)

        k = self.filter_fn.filter(l_filter)[0]
        k = rearrange(k, 'l (o d) -> o d l', o=self.order - 1)
        bias = rearrange(self.filter_fn.bias, '(o d) -> o d', o=self.order - 1)

        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)
            v = self.filter_fn(v, l_filter, k=k[o], bias=bias[o])

        y = rearrange(v * x[0], 'b d l -> b l d')

        y = self.out_proj(y)
        return y

def create_mixer_cls(layer=None,
                     attn_layer_idx=None, attn_cfg=None, layer_idx=None,
                     device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    if attn_layer_idx is not None and layer_idx in attn_layer_idx:
        causal = True if attn_cfg is None else attn_cfg.pop('causal', True)

        mha_cls = MHA

        mixer_cls = partial(mha_cls, causal=causal, layer_idx=layer_idx,
                            **(attn_cfg if attn_cfg is not None else {}),**factory_kwargs)
    else:
        # mixer_cls = instantiate(registry.layer, layer, partial=True, layer_idx=layer_idx, **factory_kwargs)

        mixer_cls = partial(HyenaOperator, **layer)

    return mixer_cls

def create_mlp_cls(d_model, d_inner=None, device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    inner_dim = d_inner if d_inner is not None else 4 * d_model

    mlp_cls = partial(Mlp, hidden_features=inner_dim,
                          activation=partial(F.gelu, approximate='tanh'), **factory_kwargs)

    return mlp_cls


def create_block(d_model, d_inner=None,
                 layer=None, attn_layer_idx=None,
                 attn_cfg=None, layer_norm_epsilon=1e-5,
                 resid_dropout1=0.0, resid_dropout2=0.0, residual_in_fp32=False,
                 layer_idx=None,
                 device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    mixer_cls = create_mixer_cls(layer=layer,
                                 attn_layer_idx=attn_layer_idx,
                                 attn_cfg=attn_cfg, layer_idx=layer_idx,
                                 **factory_kwargs)
    mlp_cls = create_mlp_cls(d_model, d_inner=d_inner,
                             **factory_kwargs)
    norm_cls = partial(nn.LayerNorm, eps=layer_norm_epsilon, **factory_kwargs)
    block = Block(d_model, mixer_cls, mlp_cls, norm_cls=norm_cls,
                  prenorm=True, resid_dropout1=resid_dropout1, resid_dropout2=resid_dropout2,residual_in_fp32=residual_in_fp32)
    block.layer_idx = layer_idx
    return block

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True,
                  glu_act=False):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))
            # If using GLU activation for now, we scale the std by 2
            elif name in ["output_linear.0.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                if not glu_act:
                    nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))
                else:
                    out_features = p.shape[0]
                    # Multiplying the first half of the matrix by 2 since sigmoid scales it down by 0.5
                    # on average.
                    nn.init.normal_(p[:out_features // 2], mean=0.0, std=initializer_range / math.sqrt(2 * n_layer) * 2)


class HyenaDNAModel(nn.Module):

    def __init__(self, d_model: int, n_layer: int, d_inner: int, vocab_size: int,
                 layer=None, attn_layer_idx=None, attn_cfg=None, max_position_embeddings=0,
                 resid_dropout: float = 0.0, embed_dropout: float = 0.1,
                 layer_norm_epsilon: float = 1e-5, initializer_cfg=None, residual_in_fp32=False,
                 pad_vocab_size_multiple: int = 1, use_head=False, n_classes: int = 2,
                 device=None, dtype=None, **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)

        self.use_head = use_head

        # check if layer (config) has d_model (HF code differs from main Safari code)
        if 'd_model' not in layer:
            layer['d_model'] = d_model

        self.backbone = LMBackbone(
            d_model=d_model, n_layer=n_layer, d_inner=d_inner, vocab_size=vocab_size,
            layer=layer, attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg,
            max_position_embeddings=max_position_embeddings,
            resid_dropout=resid_dropout, embed_dropout=embed_dropout,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_cfg=initializer_cfg, residual_in_fp32=residual_in_fp32,
            **factory_kwargs, **kwargs
        )

        # we only need a head if doing classification, otherwise we'll use the
        # hidden states as embeddings
        if self.use_head:
            self.head = SequenceDecoder(d_model=d_model, d_output=n_classes, l_output=0, mode='pool')

        # Initialize weights and apply final processing
        self.apply(partial(_init_weights, n_layer=n_layer,
                           **(initializer_cfg if initializer_cfg is not None else {})))

        # if self.use_head:
        #     self.tie_weights()

    # def tie_weights(self):
    #     self.head.weight = self.backbone.embeddings.word_embeddings.weight

    def forward(
        self,
        input_ids,
        position_ids=None,
        state=None,
        *,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ):
        out = self.backbone(
            input_ids,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # If you’re using the classification head, keep HF-style output envelope
        if self.use_head:
            if return_dict:
                # out is BaseModelOutput
                logits = self.head(out.last_hidden_state)
                return BaseModelOutput(
                    last_hidden_state=logits,
                    hidden_states=out.hidden_states if output_hidden_states else None,
                )
            else:
                # out could be tensor or (last, hidden_states)
                if output_hidden_states:
                    last, all_hidden = out
                    return self.head(last), all_hidden
                return self.head(out)

        # Embedding mode (no head): just pass through
        return out


class HyenaDNAHF(nn.Module):
    """Unified HyenaDNA loader that prefers local files and optionally downloads.
    - **Flexible weights**: supports `model.safetensors`, `pytorch_model.bin`, or
    Lightning-style `weights.ckpt` (with top-level `state_dict`).
    - **Key mapping**: handles gradient-checkpointing diffs (`.mixer/.mlp → .mixer.layer/.mlp.layer`).

    This class wraps a user-provided `HyenaDNAModel`. It does not depend on
    `transformers` objects and is safe for HPC/offline environments.

    Usage
    -----
    ```python
    from hyenadna_hf_wrapper import HyenaDNAHF

    # Strictly offline (assumes you already downloaded to path/model_name)
    model = HyenaDNAHF.from_pretrained(
        path="/data/scratch/hf-cache/models",
        model_name="hyenadna-small-32k-seqlen-hf",
        device="cuda",
        use_head=False,
    )

    # Optional: allow a one-time download if files are missing
    # (keeps offline behavior otherwise)
    model = HyenaDNAHF.from_pretrained(
        path="/data/scratch/hf-cache/models",
        model_name="hyenadna-small-32k-seqlen-hf",
        device="cuda",
        use_head=False,
        allow_download=True,  # ← only used if local files are absent
        repo_id="LongSafari/hyenadna-small-32k-seqlen-hf",
    )
    ```
    Note: You must have a real `HyenaDNAModel` implementation importable in your env.
    Pass it via `model_ctor` if it is not registered in the current module scope.
        """

    base_model_prefix = "hyenadna"

    # -------------------------
    # Public API
    # -------------------------
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        # HF-like knobs your code can tweak later
        self.config = SimpleNamespace(
            output_hidden_states=False,
            use_return_dict=True,
        )


    def forward(self, input_ids: torch.LongTensor, **kwargs):
        kwargs.setdefault("output_hidden_states", getattr(self.config, "output_hidden_states", False))
        kwargs.setdefault("return_dict", getattr(self.config, "use_return_dict", True))
        return self.model(input_ids, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        *,
        path: str,
        model_name: str,
        device: str | torch.device = "cpu",
        use_head: bool = False,
        n_classes: int = 2,
        model_ctor: Optional[Callable[..., nn.Module]] = None,
        # Offline-first behavior; download is optional if files are missing
        allow_download: bool = False,
        repo_id: Optional[str] = None,
        force_redownload: bool = False,
        strict_backbone: bool = False,
        verbose: bool = True,
    ) -> "HyenaDNAHF":
        """Instantiate from local (preferred) or optional HF snapshot.

        Parameters
        ----------
        path : str
            Root directory under which `model_name/` lives (or will be created).
        model_name : str
            Subdirectory name for the model (e.g., 'hyenadna-small-32k-seqlen-hf').
        device : str | torch.device
            Target device for the instantiated model.
        use_head : bool
            If True, initialize classification head.
        n_classes : int
            Output classes when `use_head=True`.
        model_ctor : Callable
            Constructor for your HyenaDNAModel if not available in globals().
        allow_download : bool
            If True and files are missing, fetch from HF `repo_id` once.
        repo_id : str | None
            HF repo to pull from (e.g., 'LongSafari/hyenadna-small-32k-seqlen-hf').
        force_redownload : bool
            If True, re-fetch snapshot even if files are present (rarely needed).
        strict_backbone : bool
            If True, error on any missing backbone params during grafting.
        verbose : bool
            Print diagnostics.
        """
        model_dir = cls._ensure_local_repo(
            path=path,
            model_name=model_name,
            allow_download=allow_download,
            repo_id=repo_id,
            force_redownload=force_redownload,
            verbose=verbose,
        )

        config_path, weight_path = cls._find_model_files(model_dir)
        if not config_path or not weight_path:
            raise FileNotFoundError(
                f"Missing config/weights in {model_dir}. Found: config={config_path}, weights={weight_path}"
            )

        cfg = cls._load_config(config_path)

        # --- CONFIG SANITIZATION (handles layer=None and missing l_max) ---
        # 1) Ensure `layer` is a dict
        if cfg.get('layer') is None:
            cfg['layer'] = {}
        layer = cfg['layer']

        # 2) Ensure d_model is available inside layer for Safari variants that expect it there
        if 'd_model' in cfg and isinstance(layer, dict) and 'd_model' not in layer:
            layer['d_model'] = cfg['d_model']

        # 3) Ensure l_max is present where HyenaOperator expects it
        if 'l_max' not in layer:
            # Prefer max_position_embeddings if present, else try common aliases
            candidates = [
                cfg.get('max_position_embeddings'),
                cfg.get('l_max'),
                cfg.get('max_seq_len'),
                cfg.get('sequence_length'),
            ]
            l_max = next((int(x) for x in candidates if isinstance(x, int) and x), None)
            # Some configs serialize numbers as strings — handle that too
            if l_max is None:
                for x in candidates:
                    if isinstance(x, str) and x.isdigit():
                        l_max = int(x)
                        break
            if l_max is None:
                raise ValueError(
                    "Config is missing 'l_max'/'max_position_embeddings'. Please add one or pass via cfg before construction."
                )
            layer['l_max'] = l_max

        # --- INFER HYENA FILTER HYPERPARAMS FROM CHECKPOINT (emb_dim, filter_order) ---
        # Load once here so we can peek at shapes; we reuse it for grafting below.
        pretrained_sd = cls._load_pretrained_dict(weight_path, map_location=device)

        # Find the first-layer implicit MLP input weight; its shape is [order, emb_dim].
        suffix = "backbone.layers.0.mixer.filter_fn.implicit_filter.0.weight"
        roots = ["", "model.", "hyena.", "hyena.model.", "hyenadna.", "hyenadna.model."]

        found_key = None
        for r in roots:
            k = f"{r}{suffix}"
            if k in pretrained_sd:
                found_key = k
                break

        if found_key is not None:
            w0 = pretrained_sd[found_key]
            # For Linear(out_features, in_features): weight.shape == (out_features, in_features)
            filter_order_inferred = int(w0.shape[0])   # 'order' (width) of the implicit MLP
            emb_dim_inferred     = int(w0.shape[1])    # positional-embedding channel dimension

            # Only set if missing in the config
            layer.setdefault("filter_order", filter_order_inferred)
            layer.setdefault("emb_dim", emb_dim_inferred)
        else:
            # If a snapshot uses an unusual prefix, proceed with cfg values.
            pass

        checkpointing = cls._detect_checkpointing_from_config(cfg)

        # Build user scratch model
        if model_ctor is None:
            try:
                model_ctor = globals()["HyenaDNAModel"]  # must be defined/importable by the user
            except KeyError as e:
                raise RuntimeError(
                    "HyenaDNAModel is not defined in scope. Pass `model_ctor=` explicitly."
                ) from e

        scratch = model_ctor(**cfg, use_head=use_head, n_classes=n_classes)
        scratch.to(device)

        # Load and graft pretrained tensors
        # Load and graft pretrained tensors (we already loaded pretrained_sd above)
        merged = cls._graft_backbone(
            scratch.state_dict(),
            pretrained_sd,
            checkpointing=checkpointing,
            strict=strict_backbone,
            verbose=verbose,
        )

        scratch.load_state_dict(merged, strict=False)
        scratch.to(device)
        scratch.eval()
        if verbose:
            print(f"[hyena] Loaded weights from {weight_path.name} (dir={model_dir})")

        return cls(scratch)

    # -------------------------
    # Private helpers (encapsulated)
    # -------------------------
    @staticmethod
    def _inject_substring(k: str) -> str:
        # Handle gradient-checkpointed naming deltas
        s = re.sub(r"\.mixer", ".mixer.layer", k)
        s = re.sub(r"\.mlp", ".mlp.layer", s)
        return s

    @staticmethod
    def _maybe_unwrap_state_dict(obj: Dict) -> Dict[str, torch.Tensor]:
        if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        return obj

    @staticmethod
    def _detect_checkpointing_from_config(cfg: Dict) -> bool:
        for k in ("checkpoint_mixer", "gradient_checkpointing", "grad_checkpointing"):
            v = cfg.get(k, False)
            if isinstance(v, bool) and v:
                return True
        return False

    @classmethod
    def _find_model_files(cls, model_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
        cfg = None
        for name in ("config.json", "hf_config.json"):
            p = model_dir / name
            if p.is_file():
                cfg = p
                break

        weights = None
        for name in ("model.safetensors", "pytorch_model.bin", "weights.ckpt"):
            p = model_dir / name
            if p.is_file():
                weights = p
                break

        return cfg, weights

    @staticmethod
    def _load_config(config_path: Path) -> Dict:
        with open(config_path, "r") as f:
            return json.load(f)

    @classmethod
    def _load_pretrained_dict(cls, weight_path: Path, map_location: str | torch.device = "cpu") -> Dict[str, torch.Tensor]:
        suffix = weight_path.suffix.lower()
        if suffix == ".safetensors":
            try:
                from safetensors.torch import load_file as load_safetensors  # local import
            except Exception as e:
                raise RuntimeError("Install safetensors to load .safetensors weights: pip install safetensors") from e
            return load_safetensors(str(weight_path))

        obj = torch.load(str(weight_path), map_location=map_location)
        sd = cls._maybe_unwrap_state_dict(obj)
        if not isinstance(sd, dict):
            raise ValueError(f"Unexpected weight object at {weight_path}: {type(sd)}")
        return sd

    @classmethod
    def _graft_backbone(
            cls,
            scratch_dict: dict[str, torch.Tensor],
            pretrained_dict: dict[str, torch.Tensor],
            *,
            checkpointing: bool,
            strict: bool,
            verbose: bool,
    ) -> dict[str, torch.Tensor]:
        """
        Copy matching tensors from `pretrained_dict` into `scratch_dict` for backbone params,
        being forgiving about common prefix/name variants seen in HyenaDNA snapshots.
        """

        def inject_layer(k: str) -> str:
            # gradient-checkpointed naming sometimes inserts ".layer" after mixer/mlp
            return k.replace(".mixer.", ".mixer.layer.").replace(".mlp.", ".mlp.layer.")

        # Try these root prefixes in this order
        ROOTS = ["", "model.", "hyena.", "hyena.model.", "hyenadna.", "hyenadna.model."]

        hits, misses = 0, []
        for skey in list(scratch_dict.keys()):
            if "backbone" not in skey:
                continue
            # Only graft if shapes are compatible
            target_shape = scratch_dict[skey].shape

            # Build candidate key variants to probe in the checkpoint
            bases = {skey}
            if skey.startswith("backbone."):
                tail = skey[len("backbone."):]  # e.g. 'layers.0.mixer.in_proj.weight'
                bases |= {tail, f"backbone.{tail}"}  # allow snapshots that dropped/repeated 'backbone.'
            if checkpointing:
                bases |= {inject_layer(b) for b in list(bases)}

            m = re.search(r"implicit_filter\.(\d+)\.freq$", skey)
            if m:
                idx = m.group(1)
                cluster = ("1", "3", "5")
                if idx in cluster:
                    for alt in cluster:
                        bases.add(
                            skey.replace(
                                f".implicit_filter.{idx}.freq",
                                f".implicit_filter.{alt}.freq"
                            )
                        )

            # Embedding + final norm synonyms (kept minimal; adjust if you meet other snapshots)
            if skey.endswith("embeddings.word_embeddings.weight"):
                bases |= {"embed_tokens.weight", "wte.weight",
                          "backbone.embed_tokens.weight", "backbone.wte.weight"}

            if skey.endswith("ln_f.weight"):
                bases |= {"final_layer_norm.weight", "final_layernorm.weight"}
            if skey.endswith("ln_f.bias"):
                bases |= {"final_layer_norm.bias", "final_layernorm.bias"}

            # Try every root prefix × base
            chosen = None
            for base in bases:
                for root in ROOTS:
                    cand = f"{root}{base}"
                    t = pretrained_dict.get(cand)
                    if t is not None and tuple(t.shape) == tuple(target_shape):
                        chosen = cand
                        break
                if chosen is not None:
                    break

            if chosen is not None:
                scratch_dict[skey] = pretrained_dict[chosen]
                hits += 1
            else:
                misses.append(skey)

        if verbose:
            print(f"[hyena] grafted backbone: {hits} params; missing: {len(misses)}")
            if strict and misses:
                for k in misses:
                    print("  miss:", k)

        if strict and misses:
            raise RuntimeError(f"Missing {len(misses)} backbone weights (strict mode)")
        return scratch_dict

    @classmethod
    def _ensure_local_repo(
        cls,
        *,
        path: str,
        model_name: str,
        allow_download: bool,
        repo_id: Optional[str],
        force_redownload: bool,
        verbose: bool,
    ) -> Path:
        """Return the folder with model files. Offline-first; only downloads if allowed and needed."""
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        root = Path(path).expanduser().resolve()
        model_dir = root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # If files are present, use them (offline path)
        cfg, w = cls._find_model_files(model_dir)
        if cfg and w and not force_redownload:
            if verbose:
                print(f"[hyena] using local files at {model_dir}")
            return model_dir

        # If no files and download not allowed → fail fast (offline-first default)
        if not allow_download:
            raise FileNotFoundError(
                f"Local files not found at '{model_dir}'. Set allow_download=True and provide repo_id to fetch once."
            )

        # Perform a one-time snapshot download into model_dir
        if repo_id is None:
            raise ValueError("repo_id is required to download from Hugging Face.")

        if verbose:
            print(f"[hyena] downloading snapshot from '{repo_id}' → {model_dir}")
        try:
            from huggingface_hub import snapshot_download
        except Exception as e:
            raise RuntimeError("huggingface_hub is required to download: pip install huggingface_hub") from e

        snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            allow_patterns=("*.json", "*.bin", "*.safetensors", "*.ckpt", "*.txt"),
            ignore_patterns=("*.msgpack", "*.h5", "*.onnx"),
            resume_download=True,
        )
        return model_dir

class HyenaDNAPooler:
    """
    HyenaDNAPooler — Position-aware pooling for DNA embeddings
    ==========================================================

    Overview
    --------
    `HyenaDNAPooler` is a configurable, position-aware pooling head on top of a HyenaDNA
    backend (`HyenaBackend`) that produces **one embedding per DNA sequence**. It treats
    positions as base-pair indices (not “time”) and lets you control:

      • Directional position weighting:
          {'mean','linear_left','linear_right','exp_left','exp_right','head_k','tail_k'}
      • Pooling axis:
          {'position','layers→position','position→layers','channels→position','position→channels'}
      • Hidden-layer tap:
          an int (e.g. -7) or ('last_k', k) for shallow layer mixing
      • Optional reverse-complement (RC) averaging:
          averages forward and RC-aligned features
      • Optional channel grouping (GeM) for dimensionality reduction

    Recommended config (10 kb windows; clustering focus)
    ----------------------------------------------------
    A strong, empirically good choice for **maximizing between-sequence separation**:

      direction='exp_left', tau=56.0, pooling_axis='position', layer_spec=-7, rc_average=False

    Use it via the preset:
      • `HyenaDNAPooler.from_preset(backend, "cluster_max_sep")`
    or explicitly via:
      • `backend.build_pooler(direction="exp_left", tau=56.0, pooling_axis="position", layer_spec=-7)`

    Auto-selection (optional)
    -------------------------
    Set `auto_select=True` and pass a small sample `auto_seqs` (≥3). The pooler sweeps a compact,
    high-yield set of configs (τ ∈ {56, 64, 72} + light 'last_k' mixes), and picks the one that
    maximizes worst-case separation, ranking by:
      max(min_L2) → max(min_cos_dist) → max(mean_L2) → max(mean_cos_dist) → max(rc_mean).
    Gate with `auto_rc_floor` (default 0.85) to avoid configs with poor forward↔RC agreement.

    API summary
    -----------
    HyenaDNAPooler(
        backend,
        *,
        direction='exp_left',
        tau=64.0,
        head_k=None,
        tail_k=None,
        pooling_axis='position',
        layer_spec=-7,                 # or ('last_k', k)
        channel_groups=None,           # required for channel pooling axes
        rc_average=False,
        gem_p=3.0,
        auto_select=False,
        auto_seqs=None,
        auto_max_seqs=10,
        auto_rc_floor=0.85,
        auto_verbose=True
    )

    Methods:
      .embed(seqs) -> torch.FloatTensor [N, D]   # L2-normalized embeddings per sequence (float32)
      .set_config(**kwargs)                      # update config at runtime
      .get_config() -> dict                      # inspect current config
      .from_preset(backend, preset=...)          # 'cluster_max_sep' | 'balanced_rc' | 'rc_invariant'

    Backend helpers:
      backend.build_pooler(**overrides) -> HyenaDNAPooler
      backend.embed_best(data, **overrides) -> torch.FloatTensor
        - data: List[str] | LongTensor[B,T] | Iterable[LongTensor[B,T]]
        - ids paths default to pooling='mean', rc_invariant=True

    Notes
    -----
    - Keep `HyenaBackend(pooling='none')` if you want full control via the Pooler.
      (Backend `embed_list(..., pooling='mean'/'gem')` uses the backend’s original pooled modes.)
    - RC averaging improves strand invariance but often reduces separation (L2/cos distances).
    - Channel-pooling options reduce dimensionality (e.g., to 16D) using GeM.
    - Examples assume local/offline model files and a left-padded causal model.
    """

    def __init__(
        self,
        backend: Any,                         # HyenaBackend instance
        *,
        # ---- config (defaults = clustering winner family) ----
        direction: str = "exp_left",
        tau: float = 40.0,
        head_k: Optional[int] = None,
        tail_k: Optional[int] = None,
        pooling_axis: str = "position→layers",
        layer_spec: Union[int, Tuple[str, int]] = -7,
        channel_groups: Optional[int] = None, # required for channel pooling axes
        rc_average: bool = False,
        gem_p: float = 3.0,                   # GeM exponent for channel pooling
        # ---- optional auto-selection on a small sample ----
        auto_select: bool = False,
        auto_seqs: Optional[List[str]] = None,
        auto_max_seqs: int = 10,
        auto_rc_floor: float = 0.85,
        auto_verbose: bool = True
    ):
        self.backend = backend
        self.direction = direction
        self.tau = float(tau)
        self.head_k = head_k
        self.tail_k = tail_k
        self.pooling_axis = pooling_axis
        self.layer_spec = layer_spec
        self.channel_groups = channel_groups
        self.rc_average = bool(rc_average)
        self.gem_p = float(gem_p)

        self._validate()

        # ---- optional: auto-select best config on a small sample ----
        if auto_select and auto_seqs is not None and len(auto_seqs) >= 3:
            sample = auto_seqs[:max(3, min(auto_max_seqs, len(auto_seqs)))]
            best = self._auto_select_config(sample, rc_floor=auto_rc_floor, verbose=auto_verbose)
            # apply winner
            self.direction = best["direction"]
            self.tau = best.get("tau", self.tau)
            self.head_k = best.get("head_k", self.head_k)
            self.tail_k = best.get("tail_k", self.tail_k)
            self.pooling_axis = best["pooling_axis"]
            self.layer_spec = best["layer_spec"]
            self.channel_groups = best.get("channel_groups", self.channel_groups)
            self.rc_average = best.get("rc_average", self.rc_average)
            if auto_verbose:
                print("[HyenaDNAPooler] Auto-selected config:", best)
            self._validate()

    # --------- public API ---------
    @torch.inference_mode()
    def embed(self, seqs: List[str]) -> torch.Tensor:
        """One embedding per sequence with current config. Returns [N, D] float32, L2-normalized rows."""
        vecs = [self._embed_one(s) for s in seqs]
        return torch.stack(vecs, dim=0)

    def set_config(self, **kwargs):
        """Update config at runtime and re-validate."""
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f"Unknown config field '{k}'")
            setattr(self, k, v)
        self._validate()

    def get_config(self) -> Dict[str, Any]:
        return {
            "direction": self.direction,
            "tau": self.tau,
            "head_k": self.head_k,
            "tail_k": self.tail_k,
            "pooling_axis": self.pooling_axis,
            "layer_spec": self.layer_spec,
            "channel_groups": self.channel_groups,
            "rc_average": self.rc_average,
            "gem_p": self.gem_p,
        }

    @classmethod
    def from_preset(cls, backend, preset: str = "cluster_max_sep"):
        """
        Presets:
          - 'cluster_max_sep' : maximize between-sequence distances (default winner)
          - 'balanced_rc'     : small nudge toward RC with decent separation
          - 'rc_invariant'    : strong RC invariance (expect tighter clusters)
        """
        preset = preset.lower()
        if preset == "cluster_max_sep":
            return cls(backend,
                       direction="exp_right", tau=40.0,
                       pooling_axis="position→layers", layer_spec=-7,
                       rc_average=False)
        elif preset == "balanced_rc":
            return cls(backend,
                       direction="exp_left", tau=64.0,
                       pooling_axis="layers→position", layer_spec=("last_k", 2),
                       rc_average=False)
        elif preset == "rc_invariant":
            return cls(backend,
                       direction="exp_left", tau=64.0,
                       pooling_axis="position", layer_spec=-7,
                       rc_average=True)
        else:
            raise ValueError("Unknown preset: " + preset)

    # --------- internals ---------
    def _validate(self):
        dirs = {"mean","linear_left","linear_right","exp_left","exp_right","head_k","tail_k"}
        if self.direction not in dirs:
            raise ValueError(f"direction must be one of {dirs}")
        axes = {"position","layers→position","position→layers","channels→position","position→channels"}
        if self.pooling_axis not in axes:
            raise ValueError(f"pooling_axis must be one of {axes}")
        if self.pooling_axis in {"channels→position","position→channels"} and (self.channel_groups is None or self.channel_groups < 2):
            raise ValueError("channel_groups (>=2) is required for channel pooling axes")
        if self.direction in {"exp_left","exp_right"} and not (self.tau > 0):
            raise ValueError("tau must be > 0 for exponential modes")
        if self.direction == "head_k" and (not self.head_k or self.head_k <= 0):
            raise ValueError("head_k>0 is required for 'head_k' mode")
        if self.direction == "tail_k" and (not self.tail_k or self.tail_k <= 0):
            raise ValueError("tail_k>0 is required for 'tail_k' mode")

    @torch.no_grad()
    def _tokenize(self, seq: str) -> Tuple[torch.Tensor, torch.Tensor]:
        tok = self.backend.tokenizer(
            [seq],
            padding="longest",
            truncation=True,
            max_length=self.backend.max_length,
            pad_to_multiple_of=self.backend._choose_pad_multiple(self.backend.max_length),
            return_tensors="pt",
        )
        ids = tok["input_ids"].to(self.backend.device, non_blocking=True)
        pad_id = getattr(self.backend.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            mask = (ids != int(pad_id))
        else:
            mask = torch.ones_like(ids, dtype=torch.bool)
        return ids, mask

    @torch.no_grad()
    def _forward_all_layers(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        # Use backend's safe forward (never passes attention_mask)
        fwd = getattr(self.backend, "_forward", None)
        out = fwd(input_ids) if callable(fwd) else self.backend.model(input_ids)
        if hasattr(out, "hidden_states") and out.hidden_states:
            return list(out.hidden_states)
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            return [out.last_hidden_state]
        raise RuntimeError("Model didn't return hidden states.")

    @staticmethod
    def _pos_indices(mask_bt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = mask_bt.shape
        c = torch.cumsum(mask_bt.to(torch.int64), dim=1)
        pos_bt = (c - 1).masked_fill(mask_bt == 0, -1)
        len_b = c[:, -1]
        return pos_bt, len_b

    def _make_weights(self, mask_bt: torch.Tensor) -> torch.Tensor:
        """Direction-aware position weights [B,T], pads = 0."""
        mode = self.direction
        B, T = mask_bt.shape
        pos_bt, len_b = self._pos_indices(mask_bt)
        w = torch.zeros_like(mask_bt, dtype=torch.float32)

        if mode == "mean":
            w = mask_bt.to(torch.float32)

        elif mode in {"exp_left","exp_right"}:
            tau = float(self.tau)
            Lbt = torch.gather(len_b[:, None].expand(B, T), 1, (pos_bt.clamp_min(0)))
            if mode == "exp_left":
                expo = -pos_bt.to(torch.float32) / tau
            else:
                expo = -(Lbt.to(torch.float32) - 1 - pos_bt.to(torch.float32)) / tau
            expo = torch.where(pos_bt >= 0, expo, torch.full_like(expo, -1e9))
            w = torch.exp(expo)

        elif mode == "head_k":
            k = int(self.head_k)
            w = ((pos_bt >= 0) & (pos_bt < k)).to(torch.float32)

        elif mode == "tail_k":
            k = int(self.tail_k)
            Lbt = torch.gather(len_b[:, None].expand(B, T), 1, (pos_bt.clamp_min(0)))
            w = ((pos_bt >= (Lbt - k)) & (pos_bt >= 0)).to(torch.float32)

        elif mode == "linear_left":
            Lbt = torch.gather(len_b[:, None].expand(B, T), 1, (pos_bt.clamp_min(0))).to(torch.float32)
            w = torch.where(pos_bt >= 0, (Lbt - pos_bt.to(torch.float32)), torch.zeros_like(Lbt))

        elif mode == "linear_right":
            w = torch.where(pos_bt >= 0, (pos_bt.to(torch.float32) + 1.0), torch.zeros_like(pos_bt, dtype=torch.float32))

        else:
            raise ValueError(f"Unknown direction mode: {mode}")

        return w * mask_bt.to(torch.float32)

    @staticmethod
    def _pool_position(hs_bth: torch.Tensor, mask_bt: torch.Tensor, weights_bt: torch.Tensor) -> torch.Tensor:
        """Pool over **position**: [B,T,H] -> [B,H], L2-normalized."""
        w = weights_bt.unsqueeze(-1)
        num = (hs_bth.float() * w).sum(dim=1)
        den = weights_bt.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return F.normalize(num / den, p=2, dim=1).to(torch.float32)

    @staticmethod
    def _mean_layers_token(hs_l_bth: List[torch.Tensor], which: Iterable[int]) -> torch.Tensor:
        """Uniform mean of selected layers at token level: [B,T,H]."""
        acc = None; c = 0
        for li in which:
            x = hs_l_bth[li].float()
            acc = x if acc is None else (acc + x)
            c += 1
        return acc / max(c, 1)

    def _position_then_layers(self, hs_l_bth: List[torch.Tensor], mask_bt: torch.Tensor, wpos_bt: torch.Tensor, which: Iterable[int]) -> torch.Tensor:
        """Pool each layer over position, then mean across layers -> [B,H]."""
        pooled = [self._pool_position(hs_l_bth[li], mask_bt, wpos_bt) for li in which]
        out = torch.stack(pooled, dim=0).mean(dim=0)
        return F.normalize(out, p=2, dim=1).to(torch.float32)

    def _assert_channel_groups(self, hidden_size: int, K: int):
        if hidden_size % K != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by channel_groups ({K})")

    def _chunk_channels(self, x_bth: torch.Tensor, K: int) -> List[torch.Tensor]:
        self._assert_channel_groups(x_bth.shape[-1], K)
        return torch.chunk(x_bth, chunks=K, dim=-1)

    def _pool_channels_groupwise(self, hs_bth: torch.Tensor, K: int) -> torch.Tensor:
        """Groupwise GeM over channels: [B,T,H] -> [B,T,K]."""
        groups = self._chunk_channels(hs_bth, K)
        eps = 1e-6
        outs = []
        p = self.gem_p
        for g in groups:
            g = g.float().clamp_min(eps)
            out = (g ** p).mean(dim=-1).clamp_min(eps) ** (1.0 / p)  # [B,T]
            outs.append(out.unsqueeze(-1))
        return torch.cat(outs, dim=-1)  # [B,T,K]

    def _pool_channels_after_position(self, vec_bh: torch.Tensor, K: int) -> torch.Tensor:
        """Position pool -> [B,H], then group channels to K via GeM: [B,K]."""
        self._assert_channel_groups(vec_bh.shape[-1], K)
        chunks = torch.chunk(vec_bh, chunks=K, dim=-1)
        eps = 1e-6
        outs = []
        p = self.gem_p
        for c in chunks:
            c = c.float().clamp_min(eps)
            out = (c ** p).mean(dim=-1, keepdim=True).clamp_min(eps) ** (1.0 / p)  # [B,1]
            outs.append(out)
        out = torch.cat(outs, dim=-1)  # [B,K]
        return F.normalize(out, p=2, dim=1).to(torch.float32)

    # --------- core embedding ops ---------
    @torch.no_grad()
    def _embed_forward_or_rc(self, seq: str, use_rc: bool) -> torch.Tensor:
        """Return [D] for forward (use_rc=False) or RC-aligned (use_rc=True) under current config."""
        if not use_rc:
            ids, m = self._tokenize(seq)
            hs_list = self._forward_all_layers(ids)
            flip = False
        else:
            tbl = str.maketrans("ACGTNacgtn","TGCANtgcan")
            rc = seq.translate(tbl)[::-1]
            ids, m = self._tokenize(rc)
            hs_list = self._forward_all_layers(ids)
            flip = True

        L_total = len(hs_list)
        if isinstance(self.layer_spec, int):
            layers = [self.layer_spec % L_total]
        else:
            mode, k = self.layer_spec
            assert mode == "last_k" and k > 0
            layers = list(range(max(0, L_total - int(k)), L_total))

        wpos = self._make_weights(m)
        pa = self.pooling_axis
        K = self.channel_groups

        def _flip(x):  # flip along position axis if RC (left padding preserved)
            return torch.flip(x, dims=[1]) if flip else x

        if pa == "position":
            li = layers[0]
            e = self._pool_position(_flip(hs_list[li]), _flip(m), _flip(wpos))

        elif pa == "layers→position":
            hf = self._mean_layers_token(hs_list, layers)
            e = self._pool_position(_flip(hf), _flip(m), _flip(wpos))

        elif pa == "position→layers":
            pooled = [self._pool_position(_flip(hs_list[li]), _flip(m), _flip(wpos)) for li in layers]
            e = F.normalize(torch.stack(pooled, dim=0).mean(dim=0), p=2, dim=1).to(torch.float32)

        elif pa == "channels→position":
            li = layers[0] if len(layers) == 1 else layers[-1]
            hK = self._pool_channels_groupwise(hs_list[li], int(K))
            e = self._pool_position(_flip(hK), _flip(m), _flip(wpos))

        elif pa == "position→channels":
            li = layers[0] if len(layers) == 1 else layers[-1]
            eH = self._pool_position(_flip(hs_list[li]), _flip(m), _flip(wpos))
            e = self._pool_channels_after_position(eH, int(K))
        else:
            raise ValueError(f"Unknown pooling_axis: {pa}")

        return e.squeeze(0)

    @torch.no_grad()
    def _embed_one(self, seq: str) -> torch.Tensor:
        """Embed one sequence with current config (optionally RC-averaged). Returns [D]."""
        ef = self._embed_forward_or_rc(seq, use_rc=False)
        if not self.rc_average:
            return ef
        er = self._embed_forward_or_rc(seq, use_rc=True)
        return F.normalize(0.5 * (ef + er), p=2, dim=0)

    # --------- auto-selection helpers ---------
    @staticmethod
    @torch.no_grad()
    def _pairwise_separation(X: torch.Tensor) -> Dict[str, float]:
        """
        X: [N, D]; returns separation stats (i != j):
          mean/min cosine distance, mean/min L2, and average nearest-nonself (cos & L2).
        """
        N = X.shape[0]
        if N < 2:
            return dict(mean_cos_dist=float("nan"), min_cos_dist=float("nan"),
                        mean_L2=float("nan"), min_L2=float("nan"),
                        nn_cos_sim=float("nan"), nn_L2=float("nan"))
        Xn = F.normalize(X, p=2, dim=1)
        cos_sim = Xn @ Xn.T
        cos_dist = 1.0 - cos_sim
        eye = torch.eye(N, dtype=torch.bool, device=X.device)
        cos_dist_ns = cos_dist.masked_fill(eye, float("inf"))
        nn_cos_sim = (1.0 - cos_dist_ns.min(dim=1).values).mean().item()

        xx = (X**2).sum(dim=1, keepdim=True)
        L2sq = (xx + xx.T - 2.0 * (X @ X.T)).clamp_min(0.0)
        L2 = torch.sqrt(L2sq + 1e-12)
        L2_ns = L2.masked_fill(eye, float("inf"))
        nn_L2 = L2_ns.min(dim=1).values.mean().item()

        return dict(
            mean_cos_dist=float(cos_dist[~eye].mean().item()),
            min_cos_dist=float(cos_dist_ns.min().item()),
            mean_L2=float(L2[~eye].mean().item()),
            min_L2=float(L2_ns.min().item()),
            nn_cos_sim=float(nn_cos_sim),
            nn_L2=float(nn_L2),
        )

    @torch.no_grad()
    def _eval_config(self, seqs: List[str], cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate one config on a sample: returns separation stats + rc_mean."""
        tmp = HyenaDNAPooler(
            self.backend,
            direction=cfg["direction"],
            tau=cfg.get("tau", self.tau),
            head_k=cfg.get("head_k"),
            tail_k=cfg.get("tail_k"),
            pooling_axis=cfg["pooling_axis"],
            layer_spec=cfg["layer_spec"],
            channel_groups=cfg.get("channel_groups"),
            rc_average=False,
            gem_p=self.gem_p
        )
        Ef = torch.stack([tmp._embed_forward_or_rc(s, use_rc=False) for s in seqs], dim=0)
        sep = self._pairwise_separation(Ef)

        Er = torch.stack([tmp._embed_forward_or_rc(s, use_rc=True) for s in seqs], dim=0)
        Ef_n = F.normalize(Ef, p=2, dim=1)
        Er_n = F.normalize(Er, p=2, dim=1)
        rc_mean = float((Ef_n * Er_n).sum(dim=1).mean().item())

        return {**cfg, **sep, "rc_mean": rc_mean}

    @torch.no_grad()
    def _auto_select_config(self, seqs: List[str], *, rc_floor: float = 0.85, verbose: bool = True) -> Dict[str, Any]:
        """
        Try a small, high-yield candidate set and pick the best for clustering.
        Ranking: max(min_L2) → max(min_cos_dist) → max(mean_L2) → max(mean_cos_dist) → max(rc_mean).
        """
        C: List[Dict[str, Any]] = []

        # Strong single-layer path (winner family): exp_left, position, layer=-7
        for t in (56.0, 64.0, 72.0):
            C.append(dict(direction="exp_left", tau=t, pooling_axis="position", layer_spec=-7))

        # Light last-k mixes (can nudge RC up slightly)
        for k in (2, 3):
            C.append(dict(direction="exp_left", tau=64.0, pooling_axis="layers→position", layer_spec=("last_k", k)))

        rows = [self._eval_config(seqs, cfg) for cfg in C]

        # RC gating
        rows = [r for r in rows if (r["rc_mean"] >= rc_floor)] or rows  # if all fail, keep all

        # Rank for clustering
        rows.sort(key=lambda r: (r["min_L2"], r["min_cos_dist"], r["mean_L2"], r["mean_cos_dist"], r["rc_mean"]),
                  reverse=True)

        if verbose:
            print("\n[HyenaDNAPooler] Auto-select candidates (top→bottom):")
            for r in rows:
                desc = f"{r['direction']} τ={r.get('tau','-')} axis={r['pooling_axis']} layer={r['layer_spec']}"
                print(f"  {desc:55s}  minL2={r['min_L2']:.4f}  minCos={r['min_cos_dist']:.4f}  "
                      f"meanL2={r['mean_L2']:.4f}  meanCos={r['mean_cos_dist']:.4f}  rc_mean={r['rc_mean']:.4f}")

        # Fill in defaults for optional keys and return winner
        win = rows[0]
        win.setdefault("head_k", None)
        win.setdefault("tail_k", None)
        win.setdefault("channel_groups", None)
        win["rc_average"] = False
        return win


class HyenaBackend:
    """
    Embedding backend for HyenaDNA that:
      - Loads locally (HF offline-friendly)
      - Uses LEFT padding (causal) and masks pads manually
      - Never passes `attention_mask` to model.forward
      - Pools per-token states with masked mean/GeM, or returns token-level

    Extended:
      - Flat token inventory for LUT-based tokenization on GPU
      - Direct ids path: embed_tokens(ids: LongTensor[B,T]) -> pooled or token-level
      - Streaming path: embed_tokens_streaming(iter(LongTensor[B,T])) -> [N,D]
      - embed_best works with sequences, a single ids batch, or an iterator of ids batches

    New fast-path helpers (added without removing any functionality):
      - both_strands_batch(ids): build [2B,T] batch with forward + RC ids (pads stay LEFT)
      - iter_both_strands(batch_iter): wrap an iterator to yield [2B,T] for strand-aware indexing
      - to_cupy(t): zero-copy conversion of float32 pooled torch tensors to CuPy via DLPack
    """

    _MODEL_MAX_BP = {
        "hyenadna-tiny-1k-seqlen-hf":     1_024,
        "hyenadna-small-32k-seqlen-hf":   32_768,
        "hyenadna-medium-160k-seqlen-hf": 160_000,
        "hyenadna-medium-450k-seqlen-hf": 450_000,
        "hyenadna-large-1m-seqlen-hf":  1_000_000,
    }

    def __init__(
        self,
        model_name: str = "hyenadna-small-32k-seqlen-hf",
        model_dir : str = "hf-cache/models/",
        max_lengths: Optional[dict] = None,
        pooling: str = "mean",                # "mean" | "gem" | "none"
        normalize: bool = True,
        offline: bool = True,
        prefer_cuda: bool = True,
        gem_p: float = 3.0,
        rc_invariant: bool = True,
    ):
        max_lengths = self._MODEL_MAX_BP if max_lengths is None else dict(max_lengths)

        # Offline env (no downloads)
        if offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        # Device & dtype
        if prefer_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        else:
            self.device = torch.device("cpu")
            self.torch_dtype = torch.float32

        # Config
        if model_name not in max_lengths:
            raise ValueError(f"Unknown model_name '{model_name}'. Provide max_lengths for it.")
        self.pretrained_model_name = model_name
        self.max_length = int(max_lengths[model_name])

        self.pooling = pooling.lower()
        if self.pooling not in {"mean", "gem", "none"}:
            raise ValueError("pooling must be one of {'mean','gem','none'}")
        self.normalize = bool(normalize)
        self.gem_p = float(gem_p)
        self.rc_invariant = bool(rc_invariant)

        # Tokenizer (local)
        # NOTE: CharacterTokenizer is expected to exist as in your setup.
        self.tokenizer = CharacterTokenizer(
            characters=['A', 'C', 'G', 'T', 'N'],
            model_max_length=max(self.max_length, 10_000),
            padding_side='left',
            default_add_special_tokens=False,
        )

        # Model (local)
        # NOTE: HyenaDNAHF is expected to exist as in your setup.
        self.model = HyenaDNAHF.from_pretrained(
            path=model_dir,
            model_name=model_name,
            device=self.device.type,
            use_head=False,
        )
        self.model.eval()

        # Ensure hidden states are returned so we can pool robustly
        if hasattr(self.model, "config"):
            try:
                self.model.config.output_hidden_states = True
            except Exception:
                pass

        # Cached RC LUT for ids path (built lazily)
        self._rc_id_lut_cuda: Optional[torch.Tensor] = None

    # --------------------
    # Helpers
    # --------------------
    @staticmethod
    def _reverse_complement(seqs: List[str]) -> List[str]:
        tbl = str.maketrans("ACGTNacgtn", "TGCANtgcan")
        return [s.translate(tbl)[::-1] for s in seqs]

    @staticmethod
    def _choose_pad_multiple(L: int) -> Optional[int]:
        # Prefer larger multiples when they divide L (GPU kernel-friendly)
        for m in (128, 64, 32, 16, 8, 4):
            if L % m == 0:
                return m
        return None

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask_bt: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, H]; mask_bt: [B, T] in {0,1}; returns [B, H]
        """
        x = x.float()
        m = mask_bt.to(dtype=torch.float32).unsqueeze(-1)  # [B,T,1]
        summed = (x * m).sum(dim=1)                        # [B,H]
        lengths = m.sum(dim=1).clamp_min(1.0)              # [B,1]
        return summed / lengths

    @staticmethod
    def _masked_gem(x: torch.Tensor, mask_bt: torch.Tensor, p: float) -> torch.Tensor:
        """
        GeM pooling with mask. x: [B,T,H]; mask_bt: [B,T]; returns [B,H]
        """
        x = x.float()
        eps = 1e-6
        m = mask_bt.to(dtype=torch.float32).unsqueeze(-1)  # [B,T,1]
        xc = x.clamp_min(eps)
        s = ((xc ** p) * m).sum(dim=1)                     # [B,H]
        z = m.sum(dim=1).clamp_min(1.0)                    # [B,1]
        return (s / z).clamp_min(eps) ** (1.0 / p)

    def _pool(self, hs: torch.Tensor, mask: torch.Tensor, kind: str) -> torch.Tensor:
        if kind == "mean":
            pooled = self._masked_mean(hs, mask)
        elif kind == "gem":
            pooled = self._masked_gem(hs, mask, self.gem_p)
        else:
            raise ValueError("pool kind must be 'mean' or 'gem'")
        if self.normalize:
            pooled = F.normalize(pooled, p=2, dim=1)
        return pooled.to(torch.float32)

    @staticmethod
    def _last_hidden_from_output(out) -> Optional[torch.Tensor]:
        if hasattr(out, "hidden_states") and out.hidden_states:
            return out.hidden_states[-1]
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            return out.last_hidden_state
        if isinstance(out, torch.Tensor) and out.ndim == 3:
            return out
        return None

    def _get_last_hidden(self, out) -> torch.Tensor:
        hs = self._last_hidden_from_output(out)
        if hs is None or hs.ndim != 3:
            raise RuntimeError(
                "HyenaDNA model did not return hidden states [B,T,H]. "
                "Ensure output_hidden_states=True and pool from hidden states, not logits."
            )
        return hs

    def _forward(self, ids: torch.Tensor):
        """Call the model without passing attention_mask; tolerate both calling styles."""
        try:
            return self.model(ids)
        except TypeError:
            return self.model(input_ids=ids)

    # --------------------
    # Token id exposure
    # --------------------
    def token_inventory_details(self) -> dict:
        """
        Returns:
          {
            'token_to_id': dict[str,int],
            'id_to_token': dict[int,str],
            'specials':   {name+'_token', name+'_id', ...},
            'dna_ids':    {'A','C','G','T','N','a','c','g','t','n' -> id}
          }
        """
        tok = self.tokenizer

        # Robust token->id function
        to_id = getattr(tok, "convert_tokens_to_ids", None) or getattr(tok, "_convert_token_to_id", None)
        if to_id is None:
            raise RuntimeError("Tokenizer lacks convert_tokens_to_ids/_convert_token_to_id")

        # 1) token_to_id (full vocab)
        vocab: Dict[str, int] = {}
        get_vocab = getattr(tok, "get_vocab", None)
        if get_vocab is not None:
            raw = get_vocab()  # token->id
            for k, v in raw.items():
                ks = getattr(k, "content", k)
                vocab[str(ks)] = int(v)

        # 2) specials
        special_names = ("pad", "unk", "bos", "eos", "cls", "sep", "mask")
        specials: Dict[str, Optional[Union[str, int]]] = {}
        for name in special_names:
            tok_attr = getattr(tok, f"{name}_token", None)
            id_attr  = getattr(tok, f"{name}_token_id", None)
            if tok_attr is not None:
                specials[f"{name}_token"] = str(getattr(tok_attr, "content", tok_attr))
            if id_attr is not None:
                specials[f"{name}_id"] = int(id_attr)
            if tok_attr is not None and id_attr is not None:
                t_str = str(getattr(tok_attr, "content", tok_attr))
                if t_str not in vocab:
                    vocab[t_str] = int(id_attr)

        # 3) id_to_token
        id_to_token = {int(v): str(k) for k, v in vocab.items()}

        # 4) DNA base ids (upper+lower). If lowercase missing, mirror uppercase.
        dna_ids: Dict[str, int] = {}
        for b in ("A", "C", "G", "T", "N", "a", "c", "g", "t", "n"):
            idx = to_id(b)
            if idx is not None:
                dna_ids[b] = int(idx)
        for lo, up in zip("acgtn", "ACGTN"):
            if lo not in dna_ids and up in dna_ids:
                dna_ids[lo] = dna_ids[up]

        return {
            "token_to_id": vocab,
            "id_to_token": id_to_token,
            "specials": specials,
            "dna_ids": dna_ids,
        }

    def full_token_inventory(self) -> Dict[str, int]:
        """Flat view: token->id mapping."""
        return self.token_inventory_details()["token_to_id"]

    def vocab_ids(self) -> dict:
        """Uppercase DNA base ids only ({'A','C','G','T','N'})."""
        det = self.token_inventory_details()
        need = {}
        for b in ("A", "C", "G", "T", "N"):
            if b not in det["dna_ids"]:
                raise ValueError(f"Tokenizer lacks a concrete id for '{b}'")
            need[b] = det["dna_ids"][b]
        return need

    def token_special_ids(self) -> dict:
        """pad/unk/bos/eos/cls/sep/mask ids (may be None)."""
        return self.token_inventory_details()["specials"]

    # --------------------
    # Direct-ids embedding path
    # --------------------
    def _ensure_rc_lut_cuda(self) -> torch.Tensor:
        """
        Build (or return cached) id->id reverse-complement LUT on device.
        Unknown/pad map to themselves.
        """
        if self._rc_id_lut_cuda is not None:
            return self._rc_id_lut_cuda

        inv = self.full_token_inventory()         # token->id
        specials = self.token_special_ids()       # may include pad/unk ids

        vocab_n = 1 + max(int(v) for v in inv.values())
        rc = torch.arange(vocab_n, dtype=torch.long)

        def gid(token: str) -> Optional[int]:
            return inv.get(token, inv.get(token.lower(), inv.get(token.upper(), None)))

        A = gid("A"); C = gid("C"); G = gid("G"); T = gid("T"); N = gid("N")
        if A is not None and T is not None:
            rc[A] = T; rc[T] = A
        if C is not None and G is not None:
            rc[C] = G; rc[G] = C
        if N is not None:
            rc[N] = N

        pad_id = specials.get("pad_id", None)
        unk_id = specials.get("unk_id", None)
        if pad_id is not None:
            rc[int(pad_id)] = int(pad_id)
        if unk_id is not None:
            rc[int(unk_id)] = int(unk_id)

        self._rc_id_lut_cuda = rc.to(self.device, non_blocking=True)
        return self._rc_id_lut_cuda

    @torch.inference_mode()
    def embed_tokens(
        self,
        ids: torch.Tensor,  # LongTensor [B, T]
        *,
        pooling: Optional[str] = None,         # "mean" | "gem" | "none"
        rc_invariant: Optional[bool] = None,   # if True, compute RC-averaged embeddings
        attention_mask: Optional[torch.Tensor] = None,  # optional; if not provided, inferred from pad_id
        max_length: Optional[int] = None,               # truncate if longer than model max
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Embed *pre-tokenized* ids (no HF tokenizer). Does not pass attention_mask to the model.
        - For pooling='none': returns (token_states [B,T,H], mask [B,T])
        - For pooled modes: returns [B,H] (float32)
        """
        if not torch.is_tensor(ids) or ids.ndim != 2 or ids.dtype not in (torch.long, torch.int64):
            raise ValueError("ids must be a LongTensor of shape [B, T]")

        B, T = ids.shape
        target_len = int(self.max_length if max_length is None else min(max_length, self.max_length))
        if T > target_len:
            ids = ids[:, :target_len].contiguous()
            T = target_len

        # Device move
        ids = ids.to(self.device, non_blocking=True)

        # Mask (infer from pad_id if not provided)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if attention_mask is None:
            if pad_id is not None:
                mask = (ids != int(pad_id))
            else:
                mask = torch.ones_like(ids, dtype=torch.bool)
        else:
            mask = attention_mask.to(device=ids.device, dtype=torch.bool)

        pl = (pooling or self.pooling).lower()
        if pl not in {"mean", "gem", "none"}:
            raise ValueError("pooling must be one of {'mean','gem','none'}")
        rc_flag = self.rc_invariant if rc_invariant is None else bool(rc_invariant)

        # Forward pass (+ optional RC) without attention_mask
        with torch.autocast(device_type=self.device.type, dtype=self.torch_dtype, enabled=(self.device.type == "cuda")):
            out_f = self._forward(ids)
            hs_f = self._get_last_hidden(out_f)  # [B,T,H]

            hs_r_aligned = None
            mask_r_aligned = None
            if rc_flag:
                # Reverse-complement ids such that pads remain on the LEFT
                rc_lut = self._ensure_rc_lut_cuda()
                comp = rc_lut[ids]                              # [B,T]

                lengths = mask.sum(dim=1)                       # [B]
                starts = (T - lengths).unsqueeze(1)             # [B,1]
                J = torch.arange(T, device=ids.device).view(1, T)  # [1,T] 0..T-1
                idx = torch.where(J >= starts, starts + (T - 1 - J), J)  # [B,T]

                ids_rc = comp.gather(dim=1, index=idx)          # [B,T], pads still on LEFT

                out_r = self._forward(ids_rc)
                hs_r = self._get_last_hidden(out_r)             # [B,T,H]

                # Align RC time dimension to forward orientation
                hs_r_aligned = torch.flip(hs_r, dims=[1])
                mask_r_aligned = torch.flip(mask, dims=[1])

        if pl == "none":
            if rc_flag:
                m_f = mask.to(torch.float32).unsqueeze(-1)
                m_r = mask_r_aligned.to(torch.float32).unsqueeze(-1)
                denom = (m_f + m_r).clamp_min(1.0)
                hs_avg = ((hs_f.float() * m_f) + (hs_r_aligned.float() * m_r)) / denom
                mask_any = (mask | mask_r_aligned).to(mask.dtype)
                return hs_avg.to(torch.float32), mask_any
            return hs_f, mask.to(mask.dtype)

        # pooled path
        if rc_flag:
            pooled_f = self._pool(hs_f, mask, pl)                      # [B,H]
            pooled_r = self._pool(hs_r_aligned, mask_r_aligned, pl)    # [B,H]
            pooled = 0.5 * (pooled_f + pooled_r)
            if self.normalize:
                pooled = F.normalize(pooled, p=2, dim=1)
            return pooled.to(torch.float32)
        return self._pool(hs_f, mask, pl)

    @torch.inference_mode()
    def embed_tokens_streaming(
        self,
        batch_iter: Iterable[torch.Tensor],
        *,
        pooling: Optional[str] = None,        # "mean" | "gem" | "none"  (pooled modes recommended)
        rc_invariant: Optional[bool] = None,  # True: RC-avg per batch; False: assume caller provided both strands
        out_device: Optional[Union[str, torch.device]] = None,  # "cuda" | "cpu" | device | None
        cat: bool = True,                     # True: return a single [N,D]; False: list of [B,D]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Stream pooled embeddings from an iterator of token-id batches.
        Each item from batch_iter must be LongTensor[B,T]. No attention_mask is used.
        Returns float32 pooled embeddings. For token-level work, use embed_tokens(..., pooling='none').
        """
        pl = (pooling or self.pooling).lower()
        if pl not in {"mean", "gem", "none"}:
            raise ValueError("pooling must be one of {'mean','gem','none'}")
        if pl == "none":
            raise ValueError("embed_tokens_streaming is intended for pooled modes ('mean' or 'gem').")

        outs: List[torch.Tensor] = []
        target_device = self.device if out_device is None else torch.device(out_device)

        for ids in batch_iter:
            vec = self.embed_tokens(ids, pooling=pl, rc_invariant=rc_invariant)  # [B,D]
            if vec.device != target_device:
                vec = vec.to(target_device, non_blocking=True)
            outs.append(vec)

        if not cat:
            return outs
        if not outs:
            # Unknown feature dim; return empty on target device
            return torch.empty((0, 0), dtype=torch.float32, device=target_device)
        return torch.cat(outs, dim=0)

    def embed_token_batches_pooled(
        self,
        batch_iter: Iterable[torch.Tensor],
        *,
        rc_invariant: bool = False,     # usually pre-generate + and - windows, so don't RC-average again
        pooling: str = "mean",
        out_device: Union[str, torch.device] = "cuda",
    ) -> torch.Tensor:
        """Convenience wrapper around embed_tokens_streaming for pooled usage."""
        return self.embed_tokens_streaming(
            batch_iter,
            pooling=pooling,
            rc_invariant=rc_invariant,
            out_device=out_device,
            cat=True,
        )

    # --------------------
    # Fast-path helpers for strand-aware indexing (NEW)
    # --------------------
    @torch.inference_mode()
    def both_strands_batch(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Build a [2B, T] batch containing forward ids (first B) and reverse-complement
        ids (next B), *preserving LEFT padding*.

        This is the exact RC transform used internally by embed_tokens(..., rc_invariant=True),
        exposed as a utility to create strand-aware indices without RC-averaging.

        Args:
          ids: LongTensor[B,T], left-padded, on CPU or CUDA.

        Returns:
          LongTensor[2B, T] on self.device (CUDA if available).
        """
        if not torch.is_tensor(ids) or ids.ndim != 2 or ids.dtype not in (torch.long, torch.int64):
            raise ValueError("ids must be LongTensor[B,T]")

        ids = ids.to(self.device, non_blocking=True)
        B, T = ids.shape

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            mask = (ids != int(pad_id))
        else:
            mask = torch.ones_like(ids, dtype=torch.bool)

        rc_lut = self._ensure_rc_lut_cuda()
        comp = rc_lut[ids]  # [B,T]

        lengths = mask.sum(dim=1)                # [B]
        starts = (T - lengths).unsqueeze(1)      # [B,1]
        J = torch.arange(T, device=ids.device).view(1, T)
        idx = torch.where(J >= starts, starts + (T - 1 - J), J)  # [B,T]

        ids_rc = comp.gather(dim=1, index=idx)   # [B,T], pads remain on LEFT

        return torch.cat([ids, ids_rc], dim=0)   # [2B, T]

    def iter_both_strands(self, batch_iter: Iterable[torch.Tensor]) -> Iterable[torch.Tensor]:
        """
        Wrap an iterator of [B,T] batches to yield [2B,T] (forward + RC).
        Use with embed_tokens_streaming(..., rc_invariant=False) for strand-aware indexing.
        """
        for ids in batch_iter:
            yield self.both_strands_batch(ids)

    @staticmethod
    def to_cupy(t: torch.Tensor):
        """
        Zero-copy convert a contiguous float32 torch tensor to CuPy using DLPack.
        Useful for passing pooled embeddings directly to a GPU ANN index.
        """
        import cupy as cp
        from torch.utils.dlpack import to_dlpack
        if not (t.is_floating_point() and t.dtype == torch.float32):
            t = t.to(torch.float32, copy=False)
        if not t.is_contiguous():
            t = t.contiguous()
        return cp.fromDlpack(to_dlpack(t))

    # --------------------
    # Original string path
    # --------------------
    @torch.inference_mode()
    def embed_list(
        self,
        seqs: List[str],
        *,
        pooling: Optional[str] = None,  # "mean" | "gem" | "none"
        max_length: Optional[int] = None,
        rc_invariant: Optional[bool] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Embed DNA sequences. RC-invariant by default:
        - For pooled outputs, averages forward and reverse-complement embeddings.
        - For token-level outputs ("none"), returns a mask-weighted average of forward
          states and time-reversed RC states; also returns a combined mask.
        """
        if not seqs:
            raise ValueError("No sequences provided.")
        seqs = [s.strip().upper() for s in seqs]

        pl = (pooling or self.pooling).lower()
        if pl not in {"mean", "gem", "none"}:
            raise ValueError("pooling must be one of {'mean','gem','none'}")

        rc_flag = self.rc_invariant if rc_invariant is None else bool(rc_invariant)

        # Effective max length and optional pad multiple
        target_len = int(self.max_length if max_length is None else min(max_length, self.max_length))
        pad_mult = self._choose_pad_multiple(target_len)

        # Tokenize forward and RC identically (do not pass attention_mask to model)
        enc_f = self.tokenizer(
            seqs,
            padding="longest",
            truncation=True,
            max_length=target_len,
            pad_to_multiple_of=pad_mult,
            return_tensors="pt",
        )
        rc_seqs = self._reverse_complement(seqs)
        enc_r = self.tokenizer(
            rc_seqs,
            padding="longest",
            truncation=True,
            max_length=target_len,
            pad_to_multiple_of=pad_mult,
            return_tensors="pt",
        )

        input_ids_f = enc_f["input_ids"].to(self.device, non_blocking=True)  # [B,T]
        input_ids_r = enc_r["input_ids"].to(self.device, non_blocking=True)  # [B,T]

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            pad_id = int(pad_id)
            mask_f = (input_ids_f != pad_id)  # [B,T]
            mask_r = (input_ids_r != pad_id)  # [B,T]
        else:
            mask_f = torch.ones_like(input_ids_f, dtype=torch.bool)
            mask_r = torch.ones_like(input_ids_r, dtype=torch.bool)

        with torch.autocast(device_type=self.device.type, dtype=self.torch_dtype, enabled=(self.device.type == "cuda")):
            out_f = self._forward(input_ids_f)
            out_r = self._forward(input_ids_r)

        hs_f = self._get_last_hidden(out_f)  # [B,T,H]
        hs_r = self._get_last_hidden(out_r)  # [B,T,H]

        # Align RC time dimension and its mask
        hs_r_aligned = torch.flip(hs_r, dims=[1])      # reverse time
        mask_r_aligned = torch.flip(mask_r, dims=[1])  # reverse mask

        if pl == "none":
            if rc_flag:
                m_f = mask_f.to(dtype=torch.float32).unsqueeze(-1)
                m_r = mask_r_aligned.to(dtype=torch.float32).unsqueeze(-1)
                denom = (m_f + m_r).clamp_min(1.0)
                hs_avg = ((hs_f.float() * m_f) + (hs_r_aligned.float() * m_r)) / denom
                mask_any = (mask_f | mask_r_aligned).to(mask_f.dtype)
                return hs_avg.to(torch.float32), mask_any
            return hs_f, mask_f.to(mask_f.dtype)

        # Pooled outputs
        if rc_flag:
            pooled_f = self._pool(hs_f, mask_f, pl)              # [B,H]
            pooled_r = self._pool(hs_r_aligned, mask_r_aligned, pl)
            pooled = 0.5 * (pooled_f + pooled_r)
            if self.normalize:
                pooled = F.normalize(pooled, p=2, dim=1)
            return pooled.to(torch.float32)
        return self._pool(hs_f, mask_f, pl)

    # --------------------
    # Pooler builder (unchanged)
    # --------------------
    def build_pooler(self, **kwargs) -> "HyenaDNAPooler":
        """
        Construct a HyenaDNAPooler tied to this backend.
        Defaults = best from your benchmark:
          direction='exp_left', tau=64.0, pooling_axis='position', layer_spec=-7, rc_average=False
        """
        defaults = dict(direction="exp_right", tau=40.0, pooling_axis="position→layers", layer_spec=-7, rc_average=False)
        defaults.update(kwargs or {})
        return HyenaDNAPooler(self, **defaults)

    # --------------------
    # Fast-path router (unchanged behavior, documented)
    # --------------------
    @torch.inference_mode()
    def embed_best(self, data, **overrides) -> torch.Tensor:
        """
        Accepts:
          - List[str]             -> routed to HyenaDNAPooler (original behavior)
          - torch.LongTensor[B,T] -> direct ids path with pooled mean (or overridden)
          - Iterable[LongTensor]  -> streaming ids batches concatenated to [N,D]

        For ids paths, defaults mirror best pooled usage:
          pooling='mean', rc_invariant=True (override via kwargs).

        Fast-path usage patterns:
          # RC-collapsed (one vector per window):
          E.embed_best(ids_iter, pooling='mean', rc_invariant=True, out_device='cuda')

          # Strand-aware (two entries per window):
          E.embed_best(E.iter_both_strands(ids_iter), pooling='mean', rc_invariant=False, out_device='cuda')
        """
        # Case 1: sequences -> original Pooler path
        if isinstance(data, list) and (len(data) == 0 or isinstance(data[0], str)):
            pooler = self.build_pooler(**overrides)
            return pooler.embed(data)

        # Case 2: a single ids batch -> pooled direct path
        if torch.is_tensor(data):
            if data.ndim != 2 or data.dtype not in (torch.long, torch.int64):
                raise ValueError("For ids path, pass LongTensor[B,T].")
            pooling = overrides.pop("pooling", "mean")
            rc_invariant = overrides.pop("rc_invariant", True)
            return self.embed_tokens(
                data,
                pooling=pooling,
                rc_invariant=rc_invariant,
            )

        # Case 3: an iterator of ids batches -> streaming pooled path
        if hasattr(data, "__iter__"):
            pooling = overrides.pop("pooling", "mean")
            rc_invariant = overrides.pop("rc_invariant", True)
            out_device = overrides.pop("out_device", self.device)
            return self.embed_tokens_streaming(
                data,
                pooling=pooling,
                rc_invariant=rc_invariant,
                out_device=out_device,
                cat=True,
            )

        raise ValueError("embed_best expects List[str], LongTensor[B,T], or an Iterable of LongTensor[B,T].")





"""
# ---------------------------------------------------------------------------
# Usage examples
# ---------------------------------------------------------------------------

# 0) Build a backend (local/offline) and prepare some sequences
from pathlib import Path
import torch

backend_hy = HyenaBackend(
    model_name="hyenadna-small-32k-seqlen-hf",
    model_dir="hf-cache/models/",
    pooling="none",        # Pooler will handle pooling
    normalize=True,
    offline=True,
    prefer_cuda=True,
)

# Example toy sequences (replace with your own windows)
seqs = [
    "ACGT" * 2500,                           # 10 kb
    "TTTT" * 2000 + "AC",                    # ~8 kb + tail
    "NNNNACGT" * 1000,                       # includes 'N'
]

# 1) Recommended clustering config (preset) → one embedding per sequence
pooler = HyenaDNAPooler.from_preset(backend_hy, "cluster_max_sep")  # exp_left, τ=56, layer=-7
X = pooler.embed(seqs)   # [N, D], float32, rows L2-normalized
print("Preset embeddings:", X.shape, X.dtype, X.device)

# 2) Same idea via backend helper (explicit args)
pooler2 = backend_hy.build_pooler(direction="exp_left", tau=56.0, pooling_axis="position", layer_spec=-7)
X2 = pooler2.embed(seqs)

# 3) Flip on RC averaging (strand-invariant, typically tighter clusters)
pooler.set_config(rc_average=True)
X_rc = pooler.embed(seqs)

# 4) Try light last-k layer blending
pooler.set_config(pooling_axis="layers→position", layer_spec=("last_k", 2), tau=64.0, rc_average=False)
X_lastk = pooler.embed(seqs)

# 5) Channel-group pooling (dimension reduction). Requires hidden_size % K == 0.
pooler.set_config(pooling_axis="position→channels", channel_groups=16, layer_spec=-7)
X_16d = pooler.embed(seqs)   # [N, 16]

# 6) Auto-select a config from a small sample, then embed all
auto_pooler = HyenaDNAPooler(
    backend_hy,
    auto_select=True,
    auto_seqs=seqs[:3],      # ≥3 sequences recommended
    auto_max_seqs=10,
    auto_rc_floor=0.85,
    auto_verbose=True,
)
X_auto = auto_pooler.embed(seqs)
print("Auto-selected config:", auto_pooler.get_config())

# 7) Backend one-liner (no Pooler): best pooled usage for lists of sequences
#    (Uses backend’s internal mean/GeM pooling; RC invariant by default on ids paths)
X_best = backend_hy.embed_best(seqs)  # [N, D]

# 8) Inspect raw hidden states manually (advanced)
enc = backend_hy.tokenizer(
    seqs,
    padding="longest",
    truncation=True,
    max_length=backend_hy.max_length,
    return_tensors="pt",
)
input_ids = enc["input_ids"].to(backend_hy.device, non_blocking=True)  # [B, T] long
pad_id = backend_hy.tokenizer.pad_token_id

with torch.inference_mode(), torch.autocast(
    device_type=backend_hy.device.type,
    dtype=backend_hy.torch_dtype,
    enabled=(backend_hy.device.type == "cuda"),
):
    out = backend_hy.model(input_ids)   # backend already configures output_hidden_states=True
    hs_list = out.hidden_states if hasattr(out, "hidden_states") else [out.last_hidden_state]
    print(f"#layers={len(hs_list)}, shapes: first={hs_list[0].shape}, last={hs_list[-1].shape}")

    # Example: masked mean over positions on the last layer
    mask = (input_ids != int(pad_id)) if pad_id is not None else torch.ones_like(input_ids, dtype=torch.bool)
    m = mask.unsqueeze(-1).float()
    pooled_masked = (hs_list[-1].float() * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
    print("Pooled (masked) shape:", pooled_masked.shape)

"""




