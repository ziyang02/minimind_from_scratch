from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.activations import ACT2FN
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
import math, torch, torch.nn.functional as F

class NinjaMindConfig(PretrainedConfig):
    model_type = "ninjamind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)



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
    freqs, attention_factor = 1.0/rope_base**(torch.arange(0,dim,2)[:dim//2].float() / dim), 1.0

    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attention_factor = (
            rope_scaling.get("original_max_position_embeddings",2048), 
            rope_scaling.get("factor",16),
            rope_scaling.get("beta_fast",32),
            rope_scaling.get("beta_slow",1),
            rope_scaling.get("attention_factor",1.0)
        )

        if end > orig_max:
            inv_dim = lambda b: (dim * math.log(orig_max / (2 * b * math.pi))) / (2 * math.log(rope_base))

            low,high = max(math.floor(inv_dim(beta_fast)),0), min(math.ceil(inv_dim(beta_slow)),dim)

            ramp = torch.clamp((torch.arange(dim // 2, device = freqs.device).float() - low) / max(high - low, 0.01), 0, 1)

            freqs = freqs * (1 - ramp + ramp * attention_factor)

    t = torch.arange(end, device = freqs.device)

    freqs = torch.outer(t, freqs).float()

    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim = -1) * attention_factor

    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim = -1) * attention_factor

    return freqs_cos, freqs_sin
    

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim = 1):
    def rotary_half(x): return torch.cat((-x[...,x.shape[-1] // 2: ], x[..., : x.shape[-1] // 2]), dim = -1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotary_half(q) * sin.unsqueeze(unsqueeze_dim)).to(q.dtype)

    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotary_half(k) * sin.unsqueeze(unsqueeze_dim)).to(k.dtype)

    return q_embed, k_embed

def repeat_kv(x:torch.Tensor, n_rep:int) -> torch.Tensor:
    bs, slen, num_key_value_head, head_dim = x.shape
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
        self.is_causal = True

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

    def forward(self, x, position_embeddings, past_key_value = None, use_cache = False, attention_mask = None):
        bsz, seq_len, _ = x.shape
        cos, sin = position_embeddings

        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        ### KV Cache

        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim = 1)
            xv = torch.cat([past_key_value[1], xv], dim = 1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
                xq.transpose(1,2),
                repeat_kv(xk, self.n_rep).transpose(1,2),
                repeat_kv(xv, self.n_rep).transpose(1,2)
        )


        ## Compute attention
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv

        ## Concate heads
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForwrd(nn.Module):
    def __init__(self, config: NinjaMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size

        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias = False)

        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias = False)

        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias = False)

        self.act_fn = ACT2FN[config.hidden_act]

    
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    
class MoEFeedForward(nn.Module):
    def __init__(self, config: NinjaMindConfig):
        super().__init__()

        self.config = config

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias = False)

        self.experts = nn.ModuleList([FeedForwrd(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])

        self.act_fn = ACT2FN[config.hidden_act]
    
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)
        score = F.softmax(self.gate(x_flat), dim = -1)
        topk_weight, topk_idx = torch.topk(score, k = self.config.num_experts_per_tok, dim = -1, sorted = False)

        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim = -1, keepdim = True) + 1e-20)

        y = torch.zeros_like(x_flat)
        
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1,1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            
            elif self.training:
                y[0,0] += 0 * sum(p.sum() for p in expert.parameters())

        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * score.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = score.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_size)
    
class NinjaMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: NinjaMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.mlp = FeedForwrd(config) if not config.use_moe else MoEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class NinjaMindModel(nn.Module):
    def __init__(self, config: NinjaMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([NinjaMindBlock(l, config) for l in range(self.num_hidden_layers)])

        self.norm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MoEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss
    
class NinjaMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = NinjaMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: NinjaMindConfig = None):
        self.config = config or NinjaMindConfig()
        super().__init__(self.config)
        self.model = NinjaMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)




