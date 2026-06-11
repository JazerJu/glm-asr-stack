#!/usr/bin/env python3
"""
Layer-by-layer encoder output comparison between C++ and Python.
Uses pure_torch model to match the C++ implementation exactly.

Usage:
  PYTHONPATH=/data/ASR模型/Qwen3-ASR python encoder_dump_compare.py [--cpp-dir DIR]

Outputs:
  /tmp/enc_dump/python/  - Python layer dumps (bf16 .bin files + stats)
  /tmp/enc_dump/cpp/     - C++ layer dumps (bf16 .bin files, from force_aligner --dump-encoder)
"""
import sys
import os
import argparse
import struct
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/data/ASR模型/Qwen3-ASR')
from qwen_asr.pure_torch.model import Qwen3ForcedAligner, AudioEncoder

DUMP_DIR = '/tmp/enc_dump'
PYTHON_DIR = os.path.join(DUMP_DIR, 'python')
CPP_DIR = os.path.join(DUMP_DIR, 'cpp')

MODEL_PATH = '/data/.cache/huggingface/hub/models--Qwen--Qwen3-ForcedAligner-0.6B/snapshots/c7cbfc2048c462b0d63a45797104fc9db3ad62b7'
AUDIO_PATH = '/data/fwsr/glm-asr/GLM-ASR/cut.wav'
MEL_FILTERS_PATH = '/data/fwsr/glm-asr/forcealigner-engine/mel_filterbank.bin'

def load_mel_filters(path):
    """Load mel filterbank from C++ binary format (128x201 float32)."""
    data = np.fromfile(path, dtype=np.float32)
    return torch.from_numpy(data.reshape(128, 201))

def compute_mel(audio_path, mel_filters):
    """Compute mel spectrogram matching C++ implementation."""
    import wave
    import struct
    
    with wave.open(audio_path, 'rb') as wf:
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        sr = wf.getframerate()
    
    samples = np.array(struct.unpack(f'<{n_frames}h', raw), dtype=np.float32) / 32768.0
    wav = torch.from_numpy(samples)
    
    n_fft = 400
    hop_length = 160
    window = torch.hann_window(n_fft)
    
    stft_out = torch.stft(wav, n_fft=n_fft, hop_length=hop_length, window=window, center=True, return_complex=True)
    magnitudes = stft_out.abs() ** 2
    
    mel_spec = torch.matmul(mel_filters, magnitudes)
    
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    
    mel = log_spec.to(torch.bfloat16)
    
    # C++ uses N/160 frames (1400 for 224026 samples), Python STFT gives N//160+1 (1401)
    # Truncate to match C++ frame count
    cpp_frames = len(samples) // 160
    if mel.shape[1] > cpp_frames:
        mel = mel[:, :cpp_frames]
    
    return mel.cuda()

def dump_tensor(tensor, path):
    """Save tensor as bf16 binary file."""
    bf16 = tensor.detach().cpu().to(torch.bfloat16)
    bf16.view(torch.uint16).numpy().tofile(path)

def load_bf16_bin(path, shape):
    """Load bf16 binary file as numpy array."""
    data = np.fromfile(path, dtype=np.float16)  # bf16 stored as uint16, numpy reads as raw bytes
    # bf16 is 2 bytes per element
    raw = np.fromfile(path, dtype=np.uint16)
    # Convert bf16 to float32 via torch
    t = torch.from_numpy(raw.view(np.uint16)).view(torch.bfloat16).float()
    return t.numpy().reshape(shape)

def compare_dump(name, cpp_path, py_tensor):
    """Compare C++ dump with Python tensor."""
    shape = py_tensor.shape
    total = 1
    for s in shape:
        total *= s
    
    if not os.path.exists(cpp_path):
        print(f"  {name}: C++ dump NOT FOUND at {cpp_path}")
        return None
    
    cpp_data = load_bf16_bin(cpp_path, shape)
    py_data = py_tensor.detach().cpu().float().numpy()
    
    diff = np.abs(cpp_data - py_data)
    mean_diff = diff.mean()
    max_diff = diff.max()
    rel_diff = mean_diff / (np.abs(py_data).mean() + 1e-10)
    
    # Correlation
    cpp_flat = cpp_data.flatten()
    py_flat = py_data.flatten()
    if cpp_flat.std() > 0 and py_flat.std() > 0:
        corr = np.corrcoef(cpp_flat, py_flat)[0, 1]
    else:
        corr = 0.0
    
    print(f"  {name}: shape={shape} mean_diff={mean_diff:.6f} max_diff={max_diff:.6f} rel_diff={rel_diff:.6f} corr={corr:.6f}")
    return {'name': name, 'mean_diff': mean_diff, 'max_diff': max_diff, 'corr': corr, 'shape': shape}

def run_python_dump():
    """Run Python model with hooks to dump intermediate outputs."""
    os.makedirs(PYTHON_DIR, exist_ok=True)
    
    print("=" * 60)
    print("Python Encoder Layer Dump")
    print("=" * 60)
    
    # Load model
    model = Qwen3ForcedAligner.from_pretrained(MODEL_PATH, device='cuda', dtype=torch.bfloat16)
    model.eval()
    audio_enc = model.audio_encoder
    
    # Compute mel
    mel_filters = load_mel_filters(MEL_FILTERS_PATH)
    mel = compute_mel(AUDIO_PATH, mel_filters)  # [128, T] bf16
    print(f"Mel shape: {mel.shape}")
    dump_tensor(mel, os.path.join(PYTHON_DIR, 'mel_input.bin'))
    
    # Forward through conv frontend manually (matching C++ chunked processing)
    with torch.no_grad():
        x = mel.unsqueeze(0).transpose(1, 2).unsqueeze(1)  # [1, 1, T, 128]
        
        x = F.gelu(audio_enc.conv2d1(x))
        x = F.gelu(audio_enc.conv2d2(x))
        x = F.gelu(audio_enc.conv2d3(x))
        
        b, c, f, t = x.size()
        conv_out = audio_enc.conv_out(x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
        
        pos_emb = audio_enc.positional_embedding(conv_out.shape[1]).unsqueeze(0).to(conv_out.dtype)
        x = conv_out + pos_emb
        
        # Remove batch dim (C++ doesn't use batch)
        x = x.squeeze(0)  # [seq_len, d_model]
        seq_len = x.shape[0]
        print(f"After conv frontend: shape={x.shape}")
        dump_tensor(x, os.path.join(PYTHON_DIR, 'after_conv.bin'))
        
        # Layer by layer
        for layer_idx, layer in enumerate(audio_enc.layers):
            x = layer(x)
            dump_tensor(x, os.path.join(PYTHON_DIR, f'layer{layer_idx:02d}.bin'))
            if layer_idx < 3 or layer_idx >= 21:
                print(f"  Layer {layer_idx}: shape={x.shape} mean={x.float().mean():.4f} std={x.float().std():.4f}")
            elif layer_idx == 3:
                print(f"  ... (layers 3-20 omitted for brevity) ...")
        
        # Final projections
        x = audio_enc.ln_post(x)
        dump_tensor(x, os.path.join(PYTHON_DIR, 'after_ln_post.bin'))
        
        x = audio_enc.proj1(x)
        x = audio_enc.act(x)
        dump_tensor(x, os.path.join(PYTHON_DIR, 'after_proj1_gelu.bin'))
        
        x = audio_enc.proj2(x)
        dump_tensor(x, os.path.join(PYTHON_DIR, 'encoder_output.bin'))
        print(f"Final encoder output: shape={x.shape}")
    
    print(f"\nPython dumps written to {PYTHON_DIR}/")
    return seq_len

def run_comparison(seq_len, d_model=1024):
    """Compare C++ dumps with Python dumps."""
    print("\n" + "=" * 60)
    print("C++ vs Python Comparison")
    print("=" * 60)
    
    results = []
    shape = (seq_len, d_model)
    
    # Compare each stage
    stages = [
        ('after_conv', shape),
        ('layer00', shape),
        ('layer01', shape),
        ('layer02', shape),
        ('layer03', shape),
        ('layer04', shape),
        ('layer05', shape),
        ('layer10', shape),
        ('layer15', shape),
        ('layer20', shape),
        ('layer21', shape),
        ('layer22', shape),
        ('layer23', shape),
        ('after_ln_post', shape),
        ('after_proj1_gelu', shape),
        ('encoder_output', (seq_len, 1024)),  # output_dim = 1024
    ]
    
    for name, s in stages:
        py_path = os.path.join(PYTHON_DIR, f'{name}.bin')
        cpp_path = os.path.join(CPP_DIR, f'{name}.bin')
        
        if not os.path.exists(py_path):
            print(f"  {name}: Python dump not found")
            continue
        
        py_data = torch.from_file(py_path, dtype=torch.bfloat16, shared=False).reshape(s)
        r = compare_dump(name, cpp_path, py_data)
        if r:
            results.append(r)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary: Layer where divergence starts")
    print("=" * 60)
    for r in results:
        flag = " ⚠️" if r['mean_diff'] > 0.01 else " ✅"
        print(f"  {r['name']:25s}: mean_diff={r['mean_diff']:.6f} corr={r['corr']:.6f}{flag}")
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpp-dir', default=None, help='C++ dump directory')
    parser.add_argument('--compare-only', action='store_true', help='Only run comparison (skip Python dump)')
    parser.add_argument('--seq-len', type=int, default=182, help='Sequence length for comparison')
    args = parser.parse_args()
    
    if args.cpp_dir:
        CPP_DIR = args.cpp_dir
    
    if not args.compare_only:
        seq_len = run_python_dump()
        print(f"\nSequence length: {seq_len}")
        print(f"To compare: run C++ force_aligner with --dump-encoder {CPP_DIR}")
        print(f"Then re-run: python {sys.argv[0]} --compare-only --seq-len {seq_len}")
    else:
        run_comparison(args.seq_len)
