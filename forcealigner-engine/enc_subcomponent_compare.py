#!/usr/bin/env python3

import os
import struct
import sys
import types
import wave
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/ASR模型/Qwen3-ASR")

from qwen_asr import Qwen3ForcedAligner


MODEL_PATH = "/data/.cache/huggingface/hub/models--Qwen--Qwen3-ForcedAligner-0.6B/snapshots/c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
AUDIO_PATH = "/data/fwsr/glm-asr/GLM-ASR/cut.wav"
PY_DIR = "/tmp/enc_dump/py"
CPP_DIR = "/tmp/enc_dump/cpp_wmma"


@dataclass
class CaptureState:
    tensors: dict
    qkv_parts: dict


def choose_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_waveform(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        num_frames = wf.getnframes()
        raw = wf.readframes(num_frames)

    if channels != 1:
        raise ValueError(f"Expected mono audio, got {channels} channels")
    if sample_rate != 16000:
        raise ValueError(f"Expected 16kHz audio, got {sample_rate}Hz")
    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM audio, got sample width {sample_width}")

    return np.array(struct.unpack(f"<{num_frames}h", raw), dtype=np.float32) / 32768.0


def compute_mel(feature_extractor, waveform: np.ndarray, device: str) -> torch.Tensor:
    log_mel = feature_extractor._torch_extract_fbank_features(waveform, device=device)
    if log_mel.ndim != 2:
        raise ValueError(f"Unexpected mel shape from WhisperFeatureExtractor: {log_mel.shape}")

    cpp_frames = waveform.shape[0] // feature_extractor.hop_length
    log_mel = log_mel[:, :cpp_frames]
    return torch.from_numpy(log_mel).to(device=device, dtype=torch.bfloat16)


def dump_tensor_bf16(tensor: torch.Tensor, path: str) -> None:
    tensor.detach().cpu().to(torch.bfloat16).view(torch.uint16).numpy().tofile(path)


def load_bf16_file(path: str, shape: tuple[int, ...]) -> torch.Tensor:
    raw = np.fromfile(path, dtype=np.uint16)
    expected = int(np.prod(shape))
    if raw.size != expected:
        raise ValueError(f"{path} has {raw.size} elements, expected {expected} for shape {shape}")
    tensor = torch.from_numpy(raw.copy()).view(torch.bfloat16).reshape(shape)
    return tensor.float()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    if a_flat.numel() == 0:
        return float("nan")
    if torch.all(a_flat == 0) or torch.all(b_flat == 0):
        return 0.0
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0), dim=1).item()


def get_feat_extract_output_lengths(lengths: torch.Tensor) -> torch.Tensor:
    out = lengths
    for _ in range(3):
        out = (out + 2 * 1 - 3) // 2 + 1
    return out


def build_encoder_input(tower, mel: torch.Tensor, device: str):
    feature_len = torch.tensor([mel.shape[1]], dtype=torch.long, device=device)
    chunk_num = torch.ceil(feature_len / (tower.n_window * 2)).long()
    chunk_lengths = torch.tensor([tower.n_window * 2] * chunk_num.sum().item(), dtype=torch.long, device=device)
    tail_idx = torch.nn.functional.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_idx] = feature_len % (tower.n_window * 2)
    chunk_lengths[chunk_lengths == 0] = tower.n_window * 2

    chunk_list = mel.T.split(chunk_lengths.tolist(), dim=0)
    padded_feature = torch.nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
    padded_feature = padded_feature.unsqueeze(1)

    padded_embeds = []
    for chunk in padded_feature.split(tower.conv_chunksize, dim=0):
        conv_embed = F.gelu(tower.conv2d1(chunk))
        conv_embed = F.gelu(tower.conv2d2(conv_embed))
        conv_embed = F.gelu(tower.conv2d3(conv_embed))
        padded_embeds.append(conv_embed)
    padded_embed = torch.cat(padded_embeds, dim=0)

    batch, channels, freq, time = padded_embed.size()
    conv_out = tower.conv_out(
        padded_embed.permute(0, 3, 1, 2).contiguous().view(batch, time, channels * freq)
    )

    pos_emb = tower.positional_embedding.positional_embedding[:conv_out.shape[1], :].unsqueeze(0).to(conv_out.dtype)
    hidden_states = conv_out + pos_emb

    feature_lens_after_cnn = get_feat_extract_output_lengths(chunk_lengths)
    padded_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.ones(length, dtype=torch.bool, device=device) for length in feature_lens_after_cnn],
        batch_first=True,
    )
    hidden_states = hidden_states[padded_mask].view(-1, conv_out.shape[-1])

    window_aftercnn = padded_mask.shape[-1] * (tower.n_window_infer // (tower.n_window * 2))
    cu_chunk_lens = [0]
    for cnn_len in feature_lens_after_cnn.tolist():
        cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
        remainder = cnn_len % window_aftercnn
        if remainder != 0:
            cu_chunk_lens.append(remainder)
    cu_seqlens = torch.tensor(cu_chunk_lens, device=device, dtype=torch.int32).cumsum(-1)

    first_chunk_len = int((cu_seqlens[1] - cu_seqlens[0]).item())
    return hidden_states, cu_seqlens, first_chunk_len


def install_layer0_capture(layer0):
    state = CaptureState(tensors={}, qkv_parts={})
    handles = []

    def ln1_hook(_module, _inputs, output):
        state.tensors["layer00_post_ln1"] = output.detach().clone()

    def qkv_hook(name):
        def _hook(_module, _inputs, output):
            state.qkv_parts[name] = output.detach().clone()
        return _hook

    def out_proj_pre_hook(_module, inputs):
        state.tensors["layer00_post_attn"] = inputs[0].detach().clone()

    def ln2_hook(_module, _inputs, output):
        state.tensors["layer00_post_ln2"] = output.detach().clone()

    def fc2_pre_hook(_module, inputs):
        state.tensors["layer00_post_fc1_gelu"] = inputs[0].detach().clone()

    handles.append(layer0.self_attn_layer_norm.register_forward_hook(ln1_hook))
    handles.append(layer0.self_attn.q_proj.register_forward_hook(qkv_hook("q")))
    handles.append(layer0.self_attn.k_proj.register_forward_hook(qkv_hook("k")))
    handles.append(layer0.self_attn.v_proj.register_forward_hook(qkv_hook("v")))
    handles.append(layer0.self_attn.out_proj.register_forward_pre_hook(out_proj_pre_hook))
    handles.append(layer0.final_layer_norm.register_forward_hook(ln2_hook))
    handles.append(layer0.fc2.register_forward_pre_hook(fc2_pre_hook))

    original_forward = layer0.forward

    def patched_forward(self, hidden_states, cu_seqlens, attention_mask=None, **kwargs):
        state.qkv_parts.clear()
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            **kwargs,
        )
        state.tensors["layer00_post_qkv"] = torch.cat(
            [state.qkv_parts["q"], state.qkv_parts["k"], state.qkv_parts["v"]],
            dim=-1,
        )
        hidden_states = residual + hidden_states
        state.tensors["layer00_post_outproj"] = hidden_states.detach().clone()
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        state.tensors["layer00_post_fc2_res"] = hidden_states.detach().clone()

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return (hidden_states,)

    layer0.forward = types.MethodType(patched_forward, layer0)

    def restore():
        layer0.forward = original_forward
        for handle in handles:
            handle.remove()

    return state, restore


def dump_python_references(device: str):
    ensure_dir(PY_DIR)

    model = Qwen3ForcedAligner.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map=device)
    model.model.to(device)
    model.model.eval()
    tower = model.model.thinker.audio_tower

    waveform = read_waveform(AUDIO_PATH)
    mel = compute_mel(model.processor.feature_extractor, waveform, device)
    dump_tensor_bf16(mel, os.path.join(PY_DIR, "mel_input.bin"))

    hidden_states, cu_seqlens_chunked, first_chunk_len = build_encoder_input(tower, mel, device)
    dump_tensor_bf16(hidden_states, os.path.join(PY_DIR, "after_conv.bin"))

    full_seq_len = int(hidden_states.shape[0])

    # Use a SINGLE cu_seqlens for the full sequence to match C++ single-chunk mode
    # The encoder attention is non-causal, so chunking doesn't affect correctness,
    # but it affects which tokens attend to which. Using a single chunk means
    # all 182 tokens attend to all 182 tokens, matching C++ window_aftercnn > seq_len.
    cu_seqlens_single = torch.tensor([0, full_seq_len], dtype=torch.int32, device=device)

    layer0 = tower.layers[0]
    capture_state, restore = install_layer0_capture(layer0)

    try:
        with torch.no_grad():
            layer0(hidden_states, cu_seqlens=cu_seqlens_single)
    finally:
        restore()

    # Dump FULL sequence (not just first chunk) to match C++ which has single chunk = all tokens
    sliced = {}
    for name, tensor in capture_state.tensors.items():
        sliced_tensor = tensor[:full_seq_len].contiguous()
        sliced[name] = sliced_tensor
        dump_tensor_bf16(sliced_tensor, os.path.join(PY_DIR, f"{name}.bin"))

    return {
        "first_chunk_len": full_seq_len,
        "d_model": tower.config.d_model,
        "ffn_dim": tower.config.encoder_ffn_dim,
        "device": device,
        "mel_shape": tuple(mel.shape),
        "full_seq_len": full_seq_len,
        "captures": {name: tuple(t.shape) for name, t in sliced.items()},
    }


def compare_one(name: str, shape: tuple[int, ...]):
    py_path = os.path.join(PY_DIR, f"{name}.bin")
    cpp_path = os.path.join(CPP_DIR, f"{name}.bin")

    if not os.path.exists(py_path):
        return (name, "PY_MISSING", None, None, None)
    if not os.path.exists(cpp_path):
        return (name, "CPP_MISSING", None, None, None)

    py_tensor = load_bf16_file(py_path, shape)
    cpp_tensor = load_bf16_file(cpp_path, shape)
    diff = (cpp_tensor - py_tensor).abs()
    mean_abs = diff.mean().item()
    max_abs = diff.max().item()
    cos = cosine_similarity(cpp_tensor, py_tensor)
    return (name, "OK", mean_abs, max_abs, cos)


def run_comparison(first_chunk_len: int, d_model: int, ffn_dim: int):
    shapes = {
        "layer00_post_ln1": (first_chunk_len, d_model),
        "layer00_post_qkv": (first_chunk_len, d_model * 3),
        "layer00_post_attn": (first_chunk_len, d_model),
        "layer00_post_outproj": (first_chunk_len, d_model),
        "layer00_post_ln2": (first_chunk_len, d_model),
        "layer00_post_fc1_gelu": (first_chunk_len, ffn_dim),
        "layer00_post_fc2_res": (first_chunk_len, d_model),
    }

    print("\nComparison against C++ dumps")
    print("=" * 92)
    print(f"{'Tensor':24s} {'Shape':18s} {'MeanAbsDiff':>14s} {'MaxDiff':>14s} {'CosSim':>12s} {'Status':>10s}")
    print("=" * 92)
    for name, shape in shapes.items():
        row = compare_one(name, shape)
        if row[1] != "OK":
            print(f"{name:24s} {str(shape):18s} {'-':>14s} {'-':>14s} {'-':>12s} {row[1]:>10s}")
            continue
        print(
            f"{name:24s} {str(shape):18s} {row[2]:14.8f} {row[3]:14.8f} {row[4]:12.8f} {row[1]:>10s}"
        )
    print("=" * 92)


def main():
    device = choose_device()
    print(f"Loading model from {MODEL_PATH}")
    print(f"Using device: {device}")

    info = dump_python_references(device)
    print(f"Mel shape: {info['mel_shape']}")
    print(f"Full encoder input seq len: {info['full_seq_len']}")
    print(f"First encoder chunk len: {info['first_chunk_len']}")
    print(f"Python dumps written to {PY_DIR}")

    for name, shape in info["captures"].items():
        print(f"  dumped {name}: {shape}")

    run_comparison(info["full_seq_len"], info["d_model"], info["ffn_dim"])


if __name__ == "__main__":
    main()
