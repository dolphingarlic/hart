"""This file contains code for basic blocks in HART Transformer.

This file is adopted and modified from https://github.com/FoundationVision/VAR/blob/main/models/basic_var.py
"""

import functools
import math

import mlx.core as mx
import mlx.nn as nn


# from visualizer import get_local

# this file only provides the 3 blocks used in VAR transformer
__all__ = [
    "TimestepEmbedder",
    "LlamaRMSNormFused",
    "LlamaMLP",
    "AdaLNSelfAttn",
    "AdaLNBeforeHead",
]


# automatically import fused operators
dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = (
    flash_attn_func
) = None


@functools.cache
def get_position_ids_1d(batch_size, L):
    # [batch_size, L]
    return mx.tile(mx.expand_dims(mx.arange(stop=L), 0), (batch_size, 1))


@functools.cache
def get_position_ids(batch_size, patch_nums, si=-1, m_maskgit=None):
    # [batch_size, L]
    all_position_ids = []
    largest_patch_num = patch_nums[-1]
    if si == -1:
        pns = patch_nums
    else:
        pns = patch_nums[si: si + 1]
    for level_idx in range(len(pns)):
        patch_num = pns[level_idx]
        _x = mx.arange(stop=patch_num)
        _y = mx.arange(stop=patch_num)
        # [pn, pn, 2]
        cartesian = mx.stack(mx.meshgrid(_x, _y, indexing="ij"), -1)
        # normalize to the size in the largest feature map
        coords = cartesian / patch_num * largest_patch_num
        # [pn * pn, 2]
        coords = coords.reshape(-1, 2)
        all_position_ids.append(coords)
    # [batch_size, L, 2]
    pos_ids = mx.tile(
        mx.expand_dims(mx.concatenate(all_position_ids, 0), 0),
        (batch_size, 1, 1)
    )
    if m_maskgit is None:
        return pos_ids
    pos_ids = pos_ids[m_maskgit]
    return pos_ids.reshape(batch_size, -1, pos_ids.shape[-1])


# from hf transformers:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
def rotate_half(x: mx.array):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return mx.concatenate((-x2, x1), -1)


# from hf transformers:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
# unsqueeze_dim=2 because by default our qk has shape [batch_size, seq_len, heads, head_dim]
def apply_rotary_pos_emb(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array, position_ids=None, unsqueeze_dim=2):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`mx.array`): The query tensor.
        k (`mx.array`): The key tensor.
        cos (`mx.array`): The cosine part of the rotary embedding.
        sin (`mx.array`): The sine part of the rotary embedding.
        position_ids (`mx.array`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(mx.array)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = mx.expand_dims(cos, unsqueeze_dim)
    sin = mx.expand_dims(sin, unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def context_pooling(context_tokens: mx.array, context_mask=None, mode="avg") -> mx.array:
    # context_tokens: [batch, context_tokens, embed_dim]
    # context_mask: [batch, context_tokens]
    if len(context_tokens.shape) == 2:
        # C2I
        return context_tokens
    assert len(context_tokens.shape) == 3 and context_tokens.shape[1] > 1
    if mode == "avg":
        c_mask = mx.expand_dims(context_mask, -1)
        # [batch, context_tokens, embed_dim]
        condition = context_tokens * c_mask.astype(context_tokens.dtype)
        # [batch, 1, embed_dim] => averaging
        condition = condition / mx.expand_dims(mx.maximum(c_mask.sum(1), 1), 1)
        # [batch, 1, embed_dim]
        condition = condition.sum(1)
    elif mode == "max":
        # [batch, 1, embed_dim]
        condition = context_tokens.max(1, keepdims=False).values
    else:
        raise NotImplementedError
    return condition


# From Junsong and Enze's EfficientDiT codebase.
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: mx.array, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = mx.exp(-math.log(max_period) *
                       mx.arange(start=0, stop=half, dtype=mx.float32) / half)
        args = t[:, None].astype(mx.float32) * freqs[None]
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], -1)
        if dim % 2:
            embedding = mx.concatenate(
                [embedding, mx.zeros_like(embedding[:, :1])], -1
            )
        return embedding

    def __call__(self, t: mx.array):
        flag = False
        if len(t.shape) == 2:
            flag = True
            t = t[0]
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        if not flag:
            return t_emb
        else:
            return mx.expand_dims(t_emb, 0)


class LlamaRMSNormFused(nn.Module):
    # Shang: kwargs for elementwise_affine
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self, hidden_size: int, eps: float = 1e-6, use_quant: bool = False, **kwargs
    ) -> None:
        super().__init__()
        self.weight = mx.array(mx.ones(hidden_size))
        self.variance_epsilon = eps
        self.use_quant = use_quant

    def __call__(self, x: mx.array) -> mx.array:
        norm = mx.fast.rms_norm(x, self.weight, self.variance_epsilon)
        if self.use_quant:
            return norm.astype(mx.int8)
        return norm


# from hf transformers:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
class LlamaRotaryEmbedding1D(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    def __call__(self, x: mx.array, position_ids: mx.array):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # position_ids: [bs, seq_len]
        # inv_freq: [head_size // 2]
        inv_freq = 1.0 / (
            self.base ** (mx.arange(0, self.dim, 2,
                          dtype=mx.float32) / self.dim)
        )
        # inv_freq_expanded: [bs, head_size // 2, 1]
        inv_freq_expanded = mx.repeat(
            inv_freq[None, :, None].astype(mx.float32),
            position_ids.shape[0],
            axis=0
        )
        # position_ids_expanded: [bs, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].astype(mx.float32)

        freqs = (inv_freq_expanded.astype(mx.float32) @
                 position_ids_expanded.astype(mx.float32)).transpose(0, 2, 1)
        emb = mx.concatenate((freqs, freqs), -1)
        return emb.cos().astype(x.dtype), emb.sin().astype(x.dtype)


# from hf transformers:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    def __call__(self, x: mx.array, position_ids: mx.array):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # position_ids: [bs, seq_len]
        # inv_freq: [head_size // 4]
        inv_freq = 1.0 / (
            self.base ** (mx.arange(0, self.dim, 4,
                          dtype=mx.float32) / self.dim)
        )
        # inv_freq_expanded: [bs, head_size // 4, 1, 1]
        inv_freq_expanded = mx.tile(
            mx.repeat(
                inv_freq[None, :, None, None].astype(mx.float32),
                position_ids.shape[0],
                axis=0
            ),
            (1, 1, 1, 2)
        )
        # position_ids_expanded: [bs, 1, seq_len, 2]
        position_ids_expanded = position_ids[:, None, :].astype(mx.float32)
        inv_freq_expanded = inv_freq_expanded.transpose(0, 3, 1, 2)
        position_ids_expanded = position_ids_expanded.transpose(0, 3, 1, 2)

        # freqs: [bs, 2, seq_len, head_size // 4]
        freqs = (inv_freq_expanded.astype(mx.float32) @
                 position_ids_expanded.astype(mx.float32)).transpose(0, 1, 3, 2)
        emb = mx.concatenate((freqs, freqs), -1)
        # cos, sin: [bs, seq_len, 2, head_size // 2]
        cos = emb.cos().transpose(0, 2, 1, 3)
        sin = emb.sin().transpose(0, 2, 1, 3)
        # [bs, seq_len, head_size]
        cos = cos.reshape(cos.shape[0], cos.shape[1], -1)
        sin = sin.reshape(sin.shape[0], sin.shape[1], -1)
        return cos.astype(x.dtype), sin.astype(x.dtype)


class LlamaMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features or in_features
        self.gate_proj = nn.Linear(
            self.in_features, self.hidden_features, bias=False)
        self.up_proj = nn.Linear(
            self.in_features, self.hidden_features, bias=False)
        self.down_proj = nn.Linear(
            self.hidden_features, self.out_features, bias=False)
        self.act_fn = nn.SiLU()
        self.fused_mlp_func = None

    def __call__(self, x: mx.array):
        down_proj = self.down_proj(self.act_fn(
            self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LlamaAttention(nn.Module):
    def __init__(
        self,
        block_idx,
        patch_nums,
        embed_dim=768,
        num_heads=12,
        max_position_embeddings=4096,
        rope_theta=10000,
        context_token=0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        assert patch_nums is not None
        self.context_token = context_token
        self.patch_nums = patch_nums
        self.block_idx, self.num_heads, self.head_dim = (
            block_idx,
            num_heads,
            embed_dim // num_heads,
        )  # =64

        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta

        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        if context_token != 0:
            self.context_rotary_emb = LlamaRotaryEmbedding1D(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )

        self.scale = 0.25 / math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias = mx.array(mx.zeros(embed_dim))
        self.v_bias = mx.array(mx.zeros(embed_dim))
        self.zero_k_bias = mx.array(mx.zeros(embed_dim))

        self.proj = nn.Linear(embed_dim, embed_dim)

        # only used during inference
        self.caching, self.cached_k, self.cached_v = False, None, None

    def kv_caching(self, enable: bool):
        self.caching, self.cached_k, self.cached_v = enable, None, None

    # NOTE: attn_bias is None during inference because kv cache is enabled
    # @get_local('attn')
    def __call__(
        self,
        x: mx.array,
        si=-1,
        context_position_ids=None,
        m_maskgit=None,
    ):
        B, L, C = x.shape
        # [B, L, 2]
        if self.context_token == 0:
            position_ids = get_position_ids(
                B, self.patch_nums, si=si, m_maskgit=m_maskgit
            )
        else:
            # text to image
            # level 0 does not appear in the position_ids
            # since it is included in context tokens
            # should be 679 tokens for 16x16 latent w/ default 10-stage VAR
            if si > 0:
                _position_ids = get_position_ids(
                    B, self.patch_nums[1:], si=si - 1, m_maskgit=m_maskgit
                )
                # largest position_id
                position_ids = _position_ids + mx.expand_dims(
                    mx.expand_dims(context_position_ids[:, -1], -1),
                    -1
                )
        # [B, context, 2]
        # if self.context_token > 0 and si <= 0:
        #     context_position_ids = get_position_ids_1d(B, self.context_token, x.device)

        qkv = (
            x @ self.qkv_proj.weight.T +
            mx.concatenate((self.q_bias, self.zero_k_bias, self.v_bias))
        ).reshape(B, L, 3, self.num_heads, self.head_dim)
        main_type = qkv.dtype
        # qkv: BL3Hc

        q, k, v = qkv.transpose(2, 0, 3, 1, 4).split(3)
        q = q.squeeze(0)
        k = k.squeeze(0)
        v = v.squeeze(0)
        dim_cat = 2  # q or k or v: BHLc
        dim_unsqueeze = 1

        ################## Use naive rotary embedding ##################
        # apply position embedding to visual tokens
        if self.context_token == 0:
            # position_ids exist for c2i
            # or t2i when stage id != 0
            # or t2i training phase (stage id = -1)
            cos, sin = self.rotary_emb(v, position_ids)
        elif self.context_token > 0:
            if si == 0:
                # inference step 1, only context tokens
                # TODO: This is 1D RoPE (should work with MLX)
                cos_c, sin_c = self.context_rotary_emb(v, context_position_ids)
                cos, sin = cos_c, sin_c
            else:
                # si > 0, no need to add rotary emb for context
                # inference step > 1, only new tokens
                # TODO: This is 2D RoPE (will not work with MLX)
                cos, sin = self.rotary_emb(v, position_ids)
        else:
            print("Context token cannot be negative", self.context_token)
            raise NotImplementedError
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=dim_unsqueeze)
        ################## Use naive rotary embedding ##################

        if self.caching:
            if self.cached_k is None:
                self.cached_k = k
                self.cached_v = v
            else:
                k = self.cached_k = mx.concatenate((self.cached_k, k), dim_cat)
                v = self.cached_v = mx.concatenate((self.cached_v, v), dim_cat)

        oup = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale
        ).transpose(0, 2, 1, 3).reshape(B, L, C)

        return self.proj(oup)


class AdaLNSelfAttn(nn.Module):
    def __init__(
        self,
        block_idx,
        embed_dim,
        cond_dim,
        norm_layer,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        max_position_embeddings=4096,
        patch_nums=None,
        context_token=0,
        sep_aln_pooling_mode="max",
    ):
        super().__init__()
        self.block_idx, self.C = block_idx, embed_dim
        self.C, self.D = embed_dim, cond_dim
        self.sep_aln_pooling_mode = sep_aln_pooling_mode

        self.attn = LlamaAttention(
            block_idx=block_idx,
            patch_nums=patch_nums,
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=10000,
            context_token=context_token,
        )
        # MLP ratio = 4: mul 8 / 3
        self.ffn = LlamaMLP(
            in_features=embed_dim,
            hidden_features=int(
                (embed_dim * mlp_ratio * 2) / 3 + 255) // 256 * 256,
            out_features=embed_dim,
        )

        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)

    def forward_function(
        self,
        x_BLC,
        cond_BD_or_gss,
        mask,
        context_position_ids=None,
        context_mask=None,
    ):
        return self(
            x=x_BLC,
            cond_BD=cond_BD_or_gss,
            m_maskgit=mask,
            context_position_ids=context_position_ids,
            context_mask=context_mask,
        )

    # NOTE: attn_bias is None during inference because kv cache is enabled
    def __call__(
        self,
        x,
        si=-1,
        context_position_ids=None,
        m_maskgit=None,
    ):  # C: embed_dim, D: cond_dim
        # We achieve multi-token conditioning through LLM attention mask.
        x = x + self.attn(
            self.ln_wo_grad(x),
            context_position_ids=context_position_ids,
            si=si,
            m_maskgit=m_maskgit,
        )
        x = x + self.ffn(self.ln_wo_grad(x))
        return x


class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, norm_layer):  # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.ada_lin = nn.Sequential(
            nn.SiLU(), nn.Linear(D, 2 * C))

    def __call__(self, x_BLC: mx.array, cond_BD: mx.array):
        # We achieve multi-token conditioning through LLM attention mask.
        if len(cond_BD.shape) == 3 and cond_BD.shape[1] > 1:
            cond_BD = cond_BD.max(1, keepdims=True)

        scale, shift = self.ada_lin(cond_BD).reshape(-1, 1, 2, self.C).split(2, axis=2)
        scale = scale.squeeze(2)
        shift = shift.squeeze(2)
        return (self.ln_wo_grad(x_BLC) * (scale + 1)) + shift
