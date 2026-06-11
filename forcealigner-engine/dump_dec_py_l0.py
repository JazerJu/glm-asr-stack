#!/usr/bin/env python3
import sys, os, struct, numpy as np, torch, wave
import warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
sys.path.insert(0, '/data/ASR模型/Qwen3-ASR')
from qwen_asr import Qwen3ForcedAligner

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

sub = {}

def patched(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
    bsz, q_len, _ = hidden_states.size()
    cfg = self.self_attn
    cfg_heads = cfg.config.num_attention_heads
    cfg_kvh = cfg.config.num_key_value_heads
    hd = cfg.head_dim
    
    normed = self.input_layernorm(hidden_states)
    sub['dec_l0_norm_in'] = normed[0].detach()
    
    q = cfg.q_proj(normed)
    sub['dec_l0_q_proj'] = q[0].detach()
    k = cfg.k_proj(normed)
    sub['dec_l0_k_proj'] = k[0].detach()
    
    q_rs = q.view(bsz, q_len, cfg_heads, hd)
    k_rs = k.view(bsz, q_len, cfg_kvh, hd)
    
    q_n = cfg.q_norm(q_rs).transpose(1, 2)
    k_n = cfg.k_norm(k_rs).transpose(1, 2)
    
    cos, sin = position_embeddings if position_embeddings is not None else (None, None)
    if cos is not None:
        q_r = (q_n * cos) + (rotate_half(q_n) * sin)
        k_r = (k_n * cos) + (rotate_half(k_n) * sin)
    else:
        q_r, k_r = q_n, k_n
    
    sub['dec_l0_q_rope'] = q_r.transpose(1,2).reshape(bsz, q_len, -1)[0].detach()
    sub['dec_l0_k_rope'] = k_r.transpose(1,2).reshape(bsz, q_len, -1)[0].detach()
    
    v_rs = cfg.v_proj(normed).view(bsz, q_len, cfg_kvh, hd).transpose(1, 2)
    sub['dec_l0_v_proj'] = v_rs.transpose(1,2).reshape(bsz, q_len, -1)[0].detach()
    
    attn_out = torch.nn.functional.scaled_dot_product_attention(
        q_r, k_r, v_rs, attn_mask=None, dropout_p=0.0, is_causal=True, scale=cfg.scaling)
    sub['dec_l0_attn_out'] = attn_out.transpose(1,2).reshape(bsz, q_len, -1)[0].detach()
    
    o = cfg.o_proj(attn_out.transpose(1, 2).reshape(bsz, q_len, -1))
    sub['dec_l0_o_proj'] = o[0].detach()
    
    post = hidden_states + o
    sub['dec_l0_post_attn_res'] = post[0].detach()
    
    n2 = self.post_attention_layernorm(post)
    g = self.mlp.gate_proj(n2)
    u = self.mlp.up_proj(n2)
    m = self.mlp.act_fn(g) * u
    d = self.mlp.down_proj(m)
    result = post + d
    sub['dec_l0_post_mlp_res'] = result[0].detach()
    
    return result

model.model.thinker.model.layers[0].forward = patched.__get__(model.model.thinker.model.layers[0])

with torch.no_grad():
    result = model.align(audio=AUDIO_PATH, text=TEXT, language="English")

print("Dumping...")
for name in sorted(sub.keys()):
    dump_tensor(sub[name], name)
print(f"Done -> {OUT_DIR}/")
