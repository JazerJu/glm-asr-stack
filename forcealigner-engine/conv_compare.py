#!/usr/bin/env python3
"""Compare C++ vs Python conv frontend outputs chunk-by-chunk.

Processes only the first chunk (100 mel frames) through each conv stage
and compares with C++ dumps.
"""
import sys, os, struct, numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, '/data/ASR模型/Qwen3-ASR')
from qwen_asr import Qwen3ForcedAligner

CPP_DIR = '/tmp/enc_dump/cpp_conv'
PY_DIR = '/tmp/enc_dump/py_conv'
os.makedirs(PY_DIR, exist_ok=True)

MODEL_PATH = 'Qwen/Qwen3-ForcedAligner-0.6B'
AUDIO_PATH = '/data/fwsr/glm-asr/GLM-ASR/cut.wav'
MEL_FILTERS_PATH = '/data/fwsr/glm-asr/forcealigner-engine/mel_filterbank.bin'
CHUNK_SIZE = 100  # n_window * 2

def load_bf16_bin(path, shape):
    raw = np.fromfile(path, dtype=np.uint16)
    total = 1
    for s in shape: total *= s
    return torch.from_numpy(raw[:total].copy()).view(torch.bfloat16).float().reshape(shape)

def dump_tensor(t, name):
    path = os.path.join(PY_DIR, f'{name}.bin')
    t.detach().cpu().to(torch.bfloat16).view(torch.uint16).numpy().tofile(path)

def compute_mel():
    mel_filters = np.fromfile(MEL_FILTERS_PATH, dtype=np.float32).reshape(128, 201)
    mel_filters_t = torch.from_numpy(mel_filters)
    import wave
    with wave.open(AUDIO_PATH, 'rb') as wf:
        n = wf.getnframes(); raw = wf.readframes(n)
    wav = np.array(struct.unpack(f'<{n}h', raw), dtype=np.float32) / 32768.0
    wav_t = torch.from_numpy(wav)
    stft = torch.stft(wav_t, n_fft=400, hop_length=160, window=torch.hann_window(400), center=True, return_complex=True)
    mag = stft.abs() ** 2
    mel = torch.matmul(mel_filters_t, mag)
    log_mel = torch.clamp(mel, min=1e-10).log10()
    log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0
    return log_mel[:, :len(wav)//160].to(torch.bfloat16).cuda()

def compare(name, cpp_shape, py_tensor):
    cpp_path = os.path.join(CPP_DIR, f'{name}.bin')
    dump_tensor(py_tensor, name)
    py_flat = py_tensor.detach().cpu().float().flatten()
    if not os.path.exists(cpp_path):
        print(f"  {name:30s} CPP MISSING")
        return
    cpp = load_bf16_bin(cpp_path, cpp_shape).flatten()
    rows = min(len(cpp), len(py_flat))
    diff = np.abs(cpp[:rows].numpy() - py_flat[:rows].numpy())
    mean_d = diff.mean()
    max_d = diff.max()
    c = np.corrcoef(cpp[:rows].numpy(), py_flat[:rows].numpy())[0, 1] if cpp[:rows].std() > 0 else 0
    flag = "OK" if mean_d < 0.005 else ("~" if mean_d < 0.05 else ("!" if mean_d < 0.1 else "FAIL"))
    print(f"  {name:30s} mean={mean_d:.6f} max={max_d:.6f} corr={c:.6f} {flag}")
    return mean_d

def run():
    print("Loading model...")
    model = Qwen3ForcedAligner.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map='cuda:0')
    tower = model.model.thinker.audio_tower

    mel = compute_mel()
    print(f"Mel: {mel.shape}")

    # Take first chunk (100 frames), matching C++ chunked processing
    chunk0 = mel[:, :CHUNK_SIZE]  # [128, 100]
    # C++ zero-fills chunk buffer to chunk_size, then copies actual data
    # For first chunk of 1400 frames, chunk_len=100 so no padding needed
    print(f"Chunk 0 mel: {chunk0.shape}")

    with torch.no_grad():
        # Python transformers_backend uses [batch, 1, freq, time] = [1, 1, 128, 100]
        # but the conv weights are the same regardless of layout
        # C++ stores mel as [num_mel, max_frames] = [128, 100] in row-major
        # and copies into d_conv_chunk as [num_mel, chunk_size] = [128, 100]
        # then conv2d_forward treats input as CHW with C=1, H=num_mel, W=chunk_size
        # So the layout is [1, 128, 100] which matches [batch, channels=freq, time]

        # Python's tower.forward does: pad_sequence -> [batch, padded_time, freq]
        #   -> transpose -> [batch, freq, padded_time] -> unsqueeze(1) -> [batch, 1, freq, padded_time]
        # For chunk 0 (no padding): [1, 1, 128, 100]

        x = chunk0.unsqueeze(0).unsqueeze(0)  # [1, 1, 128, 100]
        print(f"Input to conv1: {x.shape}")

        # Stage 1: conv2d1 + GELU
        # C++ conv2d_forward(input=chunk, output=conv_a, ..., in_channels=1, out_channels=480,
        #                     in_h=128, in_w=100, stride=2, padding=1)
        # Output: [1, 480, 64, 50]
        x1 = F.gelu(tower.conv2d1(x))
        print(f"After conv1+gelu: {x1.shape}")
        compare("chunk0_conv1_gelu", (480, 64, 50), x1.squeeze(0))

        # Stage 2: conv2d2 + GELU
        # C++: in_h=64, in_w=50 -> out: [1, 480, 32, 25]
        x2 = F.gelu(tower.conv2d2(x1))
        print(f"After conv2+gelu: {x2.shape}")
        compare("chunk0_conv2_gelu", (480, 32, 25), x2.squeeze(0))

        # Stage 3: conv2d3 + GELU
        # C++: in_h=32, in_w=25 -> out: [1, 480, 16, 13]
        x3 = F.gelu(tower.conv2d3(x2))
        print(f"After conv3+gelu: {x3.shape}")
        compare("chunk0_conv3_gelu", (480, 16, 13), x3.squeeze(0))

        # Stage 4: Flatten - C++ flatten_conv3_kernel
        # C++ takes [downsample_hidden, conv3_h, conv3_w] = [480, 16, 13] CHW
        # flattens to [conv3_w, conv_out_dim] = [13, 480*16=7680]
        # where each time step t gets: [channel * freq for all channels and freq positions]
        # flatten_conv3 reads: for each (t, d): out[t * conv_out_dim + d] = input[d * conv3_h * conv3_w + h * conv3_w + t]
        # where d ranges 0..conv_out_dim-1, but conv_out_dim = downsample_hidden * conv3_h
        # So for output row t: concatenate all channels and freq positions at time t
        # = input[0, :, :, t] flatten, input[1, :, :, t] flatten, ...

        # Python: padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        # x3 is [1, 480, 16, 13]. permute(0,3,1,2) -> [1, 13, 480, 16]. view(1, 13, 7680)
        flat = x3.permute(0, 3, 1, 2).contiguous().view(1, 13, 480 * 16)
        print(f"After flatten: {flat.shape}")
        compare("chunk0_flattened", (13, 7680), flat.squeeze(0))

        # Stage 5: conv_out (linear projection)
        conv_out = tower.conv_out(flat)
        print(f"After conv_out: {conv_out.shape}")
        compare("chunk0_conv_out", (13, 1024), conv_out.squeeze(0))

        # Stage 6: Add positional embeddings
        pos_emb = tower.positional_embedding.positional_embedding[:13, :].unsqueeze(0).to(conv_out.dtype)
        after_pos = conv_out + pos_emb
        print(f"After pos_emb: {after_pos.shape}")
        compare("chunk0_after_posemb", (13, 1024), after_pos.squeeze(0))

    print("\nDone.")

if __name__ == '__main__':
    run()
