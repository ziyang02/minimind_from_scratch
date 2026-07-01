from transformers import PretrainedConfig


class NinjaMindConfig(PretrainedConfig):
    model_type = "ninjamind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


import torch
import torch.nn as nn
import math


class RMSNorm(torch.nn.Module):
    def __init__(self, dim:int, eps:float=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _Norm(self,x):
        return torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
    
    def forward(self,x):
        return self.weight * self._Norm(x.float()).type_as(x)
    


def precompute_freqs_cis(dim:int, end:int=int(32*1024),rope_base:float=1e6,rope_scaling:dict = None):
    freqs,attn_factor = 1.0/rope_base**(torch.arrange(0,dim,2)[:dim//2].float() / dim), 1.0

    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get(original_max_position_embeddings,2048), 
            rope_scaling.get(factor,16),
            rope_scaling.get(beta_fast,32),
            rope_scaling.get(beta_slow,1),
            rope_scaling.get(attn_factor,1.0)
        )

        if end > orig_max:
            inv_dim = lambda b: (dim * math.log(orig_max / (2 * b * math.pi))) / (2 * math.log(rope_base))

            low,high = max(math.floor(inv_dim(beta_fast)),0), min(math.ceil(inv_dim(beta_slow)),dim)

            ramp = torch.clamp((torch.arrange(dim // 2, device = freqs.device).float() - low) / max(high - low, 0.01), 0, 1)

            freqs = freqs * (1 - ramp + ramp * attn_factor)
        

        t = torch.arrange(end, device = freqs.device)

        freqs = torch.outer(freqs, t).float()

        freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim = -1) * attn_factor

        freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim = -1) * attn_factor

        return freqs_cos, freqs_sin
    

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim = 1):
    def rotary_half(x): return torch.cat((-x[...,x.shape[-1] // 2: ], x[..., : x.shape[-1] // 2]), dim = -1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotary_half(q) * sin.unsqueeze(unsqueeze_dim)).to(q.dtype)

    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotary_half(k) * sin.unsqueeze(unsqueeze_dim)).to(k.dtype)

    return q_embed, k_embed

def repeat_kv(x:torch.Tensor, n_rep:int) -> torch.Tensor:
    bs, slen, num_key_value_head, head_dim = x.shape()
    if n_rep == 1:
        return x
    
    return (
        x[:,:,:,None,:]
        .expand(bs, slen, num_key_value_head, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_head * n_rep, head_dim)
        )
        

class Attention(nn.Module):
    def __init__(self, config:NinjaMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads

        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.q_proj = nn.Linear(config.hidden_size, self.head_dim * config.num_attention_heads, bias = False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias = False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias = False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias = False)

        self.q_norm = RMSNorm(self.head_dim, eps = config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps = config.rms_norm_eps)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_value_key = None, use_cache = False, attention_mask = None):
        







