#!/usr/bin/env python3
import sys, os, struct, numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, '/data/ASR模型/Qwen3-ASR')
from qwen_asr.pure_torch.model import Qwen3ForcedAligner

DUMP_DIR = '/tmp/enc_dump'
PY_DIR = os.path.join(DUMP_DIR, 'python')
CPP_DIR = os.path.join(DUMP_DIR, 'cpp_wmma')

MODEL_PATH = '/data/.cache/huggingface/hub/models--Qwen--Qwen3-ForcedAligner-0.6B/snapshots/c7cbfc2048c462b0d63a45797104fc9db3ad62b7'
AUDIO_PATH = '/data/fwsr/glm-asr/GLM-ASR/cut.wav'
MEL_FILTERS_PATH = '/data/fwsr/glm-asr/forcealigner-engine/mel_filterbank.bin'

def load_mel_filters():
    data = np.fromfile(MEL_FILTERS_PATH, dtype=np.float32)
    return torch.from_numpy(data.reshape(128, 201))

def load_audio():
    import wave
    with wave.open(AUDIO_PATH, 'rb') as wf:
        n = wf.getnframes()
        raw = wf.readframes(n)
    return np.array(struct.unpack(f'<{n}h', raw), dtype=np.float32) / 32768.0

def compute_mel(wav_np, mel_filters):
    wav = torch.from_numpy(wav_np)
    stft = torch.stft(wav, n_fft=400, hop_length=160, window=torch.hann_window(400), center=True, return_complex=True)
    mag = stft.abs() ** 2
    mel = torch.matmul(mel_filters, mag)
    log_mel = torch.clamp(mel, min=1e-10).log10()
    log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0
    cpp_frames = len(wav_np) // 160
    return log_mel[:, :cpp_frames].to(torch.bfloat16).cuda()

def dump_tensor(t, path):
    t.detach().cpu().to(torch.bfloat16).view(torch.uint16).numpy().tofile(path)

def load_bf16_bin(path, cols):
    raw = np.fromfile(path, dtype=np.uint16)
    total = len(raw)
    rows = total // cols
    return torch.from_numpy(raw[:rows*cols].copy()).view(torch.bfloat16).float().numpy().reshape(rows, cols)

def run():
    os.makedirs(PY_DIR, exist_ok=True)

    print("Loading model...")
    model = Qwen3ForcedAligner.from_pretrained(MODEL_PATH, device='cuda', dtype=torch.bfloat16)
    model.eval()
    enc = model.audio_encoder

    mel_filters = load_mel_filters()
    wav = load_audio()
    mel = compute_mel(wav, mel_filters)
    print(f"Mel: {mel.shape}")

    with torch.no_grad():
        # pure_torch uses [1, 1, T, 128] layout (time, freq)
        # but conv_out expects c*f=7680=480*16 (freq_dim*channels after conv)
        # which means the model was trained with [1, 1, 128, T] layout (freq, time)
        # matching the transformers_backend convention
        input_features = mel.unsqueeze(0)  # [1, 128, T]
        x = input_features.unsqueeze(1)     # [1, 1, 128, T] - freq first
        
        x = F.gelu(enc.conv2d1(x))
        x = F.gelu(enc.conv2d2(x))
        x = F.gelu(enc.conv2d3(x))
        
        b, c, f, t = x.size()
        x = enc.conv_out(x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
        pos_emb = enc.positional_embedding(x.shape[1]).unsqueeze(0).to(x.dtype)
        x = (x + pos_emb).squeeze(0)

        seq_len = x.shape[0]
        d_model = x.shape[1]
        print(f"After conv: {x.shape} (seq_len={seq_len}, d_model={d_model})")
        dump_tensor(x, os.path.join(PY_DIR, 'after_conv.bin'))

        for i, layer in enumerate(enc.layers):
            x = layer(x)
            dump_tensor(x, os.path.join(PY_DIR, f'layer{i:02d}.bin'))

        x = enc.ln_post(x)
        dump_tensor(x, os.path.join(PY_DIR, 'after_ln_post.bin'))

        x = enc.proj1(x)
        x = enc.act(x)
        dump_tensor(x, os.path.join(PY_DIR, 'after_proj1_gelu.bin'))

        x = enc.proj2(x)
        dump_tensor(x, os.path.join(PY_DIR, 'encoder_output.bin'))
        print(f"Encoder output: {x.shape}")

    # Compare with C++ dumps
    print("\n" + "=" * 70)
    print(f"{'Stage':25s} {'MeanDiff':>10s} {'MaxDiff':>10s} {'Corr':>10s} {'Status':>6s}")
    print("=" * 70)

    stages = ['after_conv'] + [f'layer{i:02d}' for i in range(24)] + \
             ['after_ln_post', 'after_proj1_gelu', 'encoder_output']

    cpp_seq = 182
    py_seq = seq_len
    compare_rows = min(cpp_seq, py_seq)

    if cpp_seq != py_seq:
        print(f"\n*** C++ seq_len={cpp_seq}, Python seq_len={py_seq}, comparing first {compare_rows} rows ***\n")

    for stage in stages:
        py_path = os.path.join(PY_DIR, f'{stage}.bin')
        cpp_path = os.path.join(CPP_DIR, f'{stage}.bin')

        if not os.path.exists(py_path) or not os.path.exists(cpp_path):
            print(f"{stage:25s} {'MISSING':>10s}")
            continue

        cols = d_model
        if stage in ('after_proj1_gelu', 'encoder_output'):
            cols = 1024

        py_data = load_bf16_bin(py_path, cols)[:compare_rows, :]
        cpp_data = load_bf16_bin(cpp_path, cols)[:compare_rows, :]

        diff = np.abs(cpp_data - py_data)
        mean_d = diff.mean()
        max_d = diff.max()

        c_f = cpp_data.flatten()
        p_f = py_data.flatten()
        corr = np.corrcoef(c_f, p_f)[0, 1] if c_f.std() > 0 and p_f.std() > 0 else 0

        flag = "OK" if mean_d < 0.01 else ("!!" if mean_d < 0.1 else "FAIL")
        print(f"{stage:25s} {mean_d:10.6f} {max_d:10.6f} {corr:10.6f} {flag:>6s}")

    print("=" * 70)

if __name__ == '__main__':
    run()
