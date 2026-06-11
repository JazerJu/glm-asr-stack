#!/usr/bin/env python3
"""Dump Python encoder output for a specific audio to compare with C++."""
import sys, os, struct, numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, '/data/ASR模型/Qwen3-ASR')
from qwen_asr import Qwen3ForcedAligner

AUDIO_PATH = sys.argv[1] if len(sys.argv) > 1 else '/data/fwsr/glm-asr/data/buckeye/audio/s0902b_011.wav'
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else '/tmp/enc_dump/py_s0902b'
MEL_FILTERS_PATH = '/data/fwsr/glm-asr/forcealigner-engine/mel_filterbank.bin'

os.makedirs(OUT_DIR, exist_ok=True)

# Compute mel (Whisper-style, matching C++ implementation)
mel_filters = np.fromfile(MEL_FILTERS_PATH, dtype=np.float32).reshape(128, 201)
mel_filters_t = torch.from_numpy(mel_filters)
import wave
with wave.open(AUDIO_PATH, 'rb') as wf:
    n = wf.getnframes()
    raw = wf.readframes(n)
wav = np.array(struct.unpack(f'<{n}h', raw), dtype=np.float32) / 32768.0

# Peak normalize (matching C++: only if peak > 1.0)
peak = np.abs(wav).max()
if peak > 1.0:
    wav = wav / peak
wav = np.clip(wav, -1.0, 1.0)

wav_t = torch.from_numpy(wav)
stft = torch.stft(wav_t, n_fft=400, hop_length=160, window=torch.hann_window(400), center=True, return_complex=True)
mag = stft.abs() ** 2
mel = torch.matmul(mel_filters_t, mag)
log_mel = torch.clamp(mel, min=1e-10).log10()
log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
log_mel = (log_mel + 4.0) / 4.0

mel_T = len(wav) // 160
log_mel = log_mel[:, :mel_T].to(torch.bfloat16).cuda()
print(f"Audio: {len(wav)} samples, Mel: {log_mel.shape} (T={mel_T})")

def dump_tensor(t, name):
    path = os.path.join(OUT_DIR, f'{name}.bin')
    t.detach().cpu().view(torch.uint16).numpy().tofile(path)
    print(f"  Dumped {name}: {t.shape} -> {path}")

dump_tensor(log_mel, 'mel_input')

print("Loading model...")
model = Qwen3ForcedAligner.from_pretrained(
    '/data/.cache/huggingface/hub/models--Qwen--Qwen3-ForcedAligner-0.6B/snapshots/c7cbfc2048c462b0d63a45797104fc9db3ad62b7',
    dtype=torch.bfloat16, device_map='cuda:0',
)
tower = model.model.thinker.audio_tower

feature_len = torch.tensor([log_mel.shape[1]], device='cuda')

with torch.no_grad():
    # Chunk mel into n_window*2 chunks (matching C++ chunked conv + chunked infer attention)
    chunk_size = tower.n_window * 2
    chunk_num = (log_mel.shape[1] + chunk_size - 1) // chunk_size
    remainder = log_mel.shape[1] % chunk_size
    if remainder == 0:
        remainder = chunk_size
    chunk_lengths = [chunk_size] * (chunk_num - 1) + [remainder]
    chunk_lengths = [c for c in chunk_lengths if c > 0]
    
    mel_chunks = list(log_mel.T.split(chunk_lengths, dim=0))
    padded = torch.nn.utils.rnn.pad_sequence(mel_chunks, batch_first=True).transpose(1, 2).unsqueeze(1)
    
    padded_embeds = []
    for chunk in padded.split(tower.conv_chunksize, dim=0):
        ce = F.gelu(tower.conv2d1(chunk))
        ce = F.gelu(tower.conv2d2(ce))
        ce = F.gelu(tower.conv2d3(ce))
        padded_embeds.append(ce)
    padded_embed = torch.cat(padded_embeds, dim=0)
    
    b, c, f, t = padded_embed.size()
    conv_out = tower.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
    
    pos_emb = tower.positional_embedding.positional_embedding[:conv_out.shape[1], :].unsqueeze(0).to(conv_out.dtype)
    x = conv_out + pos_emb
    
    # Flatten chunks back to sequence (remove padding)
    feature_lens_after_cnn = [(l + 2*1 - 3)//2 + 1 for l in chunk_lengths]
    for i, cnn_len in enumerate(feature_lens_after_cnn):
        feature_lens_after_cnn[i] = (cnn_len + 2*1 - 3)//2 + 1
    for i, cnn_len in enumerate(feature_lens_after_cnn):
        feature_lens_after_cnn[i] = (cnn_len + 2*1 - 3)//2 + 1
    
    padded_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.ones(l, dtype=torch.bool, device='cuda') for l in feature_lens_after_cnn],
        batch_first=True,
    )
    x = x[padded_mask]
    
    seq_len = x.shape[0]
    d_model = x.shape[1]
    print(f"After conv: {x.shape} seq_len={seq_len} d_model={d_model}")
    dump_tensor(x, 'after_conv')
    
    # Build cu_seqlens for chunked attention (matching C++ window_aftercnn)
    window_aftercnn = padded_mask.shape[-1] * (tower.n_window_infer // (tower.n_window * 2))
    cu_chunk_lens = [0]
    for cnn_len in feature_lens_after_cnn:
        cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
        rem = cnn_len % window_aftercnn
        if rem != 0:
            cu_chunk_lens += [rem]
    cu_seqlens = torch.tensor(cu_chunk_lens, device='cuda').cumsum(-1, dtype=torch.int32)
    print(f"window_aftercnn={window_aftercnn}, cu_seqlens={cu_seqlens.tolist()}")
    
    for i, layer in enumerate(tower.layers):
        out = layer(x, cu_seqlens=cu_seqlens)
        x = out[0] if isinstance(out, tuple) else out
    
    x = tower.ln_post(x)
    x = tower.proj1(x)
    x = tower.act(x)
    x = tower.proj2(x)
    
    print(f"Encoder output: {x.shape}")
    dump_tensor(x, 'encoder_output')

print(f"\nPython encoder dumps written to {OUT_DIR}/")
