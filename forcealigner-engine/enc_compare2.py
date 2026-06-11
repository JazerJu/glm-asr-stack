#!/usr/bin/env python3
import sys, os, struct, numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, '/data/ASR模型/Qwen3-ASR')
from qwen_asr import Qwen3ForcedAligner

PY_DIR = '/tmp/enc_dump/python'
CPP_DIR = '/tmp/enc_dump/cpp_wmma'
MODEL_PATH = 'Qwen/Qwen3-ForcedAligner-0.6B'
AUDIO_PATH = '/data/fwsr/glm-asr/GLM-ASR/cut.wav'
MEL_FILTERS_PATH = '/data/fwsr/glm-asr/forcealigner-engine/mel_filterbank.bin'

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

def dump_tensor(t, path):
    t.detach().cpu().to(torch.bfloat16).view(torch.uint16).numpy().tofile(path)

def load_bf16_bin(path, cols):
    raw = np.fromfile(path, dtype=np.uint16)
    rows = len(raw) // cols
    return torch.from_numpy(raw[:rows*cols].copy()).view(torch.bfloat16).float().numpy().reshape(rows, cols)

def _get_feat_extract_output_lengths(lengths):
    out = lengths
    for _ in range(3):
        out = (out + 2 * 1 - 3) // 2 + 1
    return out

def run():
    os.makedirs(PY_DIR, exist_ok=True)

    print("Loading model...")
    model = Qwen3ForcedAligner.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map='cuda:0')
    tower = model.model.thinker.audio_tower

    mel = compute_mel()
    print(f"Mel: {mel.shape}")
    dump_tensor(mel, os.path.join(PY_DIR, 'mel_input.bin'))

    feature_len = torch.tensor([mel.shape[1]], device='cuda')

    with torch.no_grad():
        aftercnn_lens = _get_feat_extract_output_lengths(feature_len)
        chunk_num = torch.ceil(feature_len / (tower.n_window * 2)).long()
        chunk_lengths = torch.tensor([tower.n_window * 2] * chunk_num.sum().item(), dtype=torch.long, device='cuda')
        tail_idx = torch.nn.functional.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_idx] = feature_len % (tower.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = tower.n_window * 2

        chunk_list = mel.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = torch.nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
        padded_feature = padded_feature.unsqueeze(1)

        padded_embeds = []
        for chunk in padded_feature.split(tower.conv_chunksize, dim=0):
            ce = F.gelu(tower.conv2d1(chunk))
            ce = F.gelu(tower.conv2d2(ce))
            ce = F.gelu(tower.conv2d3(ce))
            padded_embeds.append(ce)
        padded_embed = torch.cat(padded_embeds, dim=0)

        b, c, f, t = padded_embed.size()
        conv_out = tower.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

        pos_emb = tower.positional_embedding.positional_embedding[:conv_out.shape[1], :].unsqueeze(0).to(conv_out.dtype)
        x = conv_out + pos_emb

        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
        padded_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.ones(l, dtype=torch.bool, device='cuda') for l in feature_lens_after_cnn],
            batch_first=True,
        )
        x = x[padded_mask]

        seq_len = x.shape[0]
        d_model = x.shape[1]
        print(f"After conv (chunked): {x.shape} seq_len={seq_len} d_model={d_model}")
        dump_tensor(x, os.path.join(PY_DIR, 'after_conv.bin'))

        # Build cu_seqlens matching Python's logic
        window_aftercnn = padded_mask.shape[-1] * (tower.n_window_infer // (tower.n_window * 2))
        cu_chunk_lens = [0]
        for cnn_len in feature_lens_after_cnn:
            cu_chunk_lens += [window_aftercnn] * (cnn_len.item() // window_aftercnn)
            remainder = cnn_len.item() % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        cu_seqlens = torch.tensor(cu_chunk_lens, device='cuda').cumsum(-1, dtype=torch.int32)
        print(f"cu_seqlens: {cu_seqlens.tolist()}")
        print(f"window_aftercnn: {window_aftercnn}")

        x = x.view(seq_len, d_model)

        for i, layer in enumerate(tower.layers):
            out = layer(x, cu_seqlens=cu_seqlens)
            x = out[0] if isinstance(out, tuple) else out
            dump_tensor(x, os.path.join(PY_DIR, f'layer{i:02d}.bin'))
            if i < 3 or i >= 22:
                print(f"  Layer {i}: mean={x.float().mean():.4f} std={x.float().std():.4f}")

        x = tower.ln_post(x)
        dump_tensor(x, os.path.join(PY_DIR, 'after_ln_post.bin'))
        x = tower.proj1(x)
        x = tower.act(x)
        dump_tensor(x, os.path.join(PY_DIR, 'after_proj1_gelu.bin'))
        x = tower.proj2(x)
        dump_tensor(x, os.path.join(PY_DIR, 'encoder_output.bin'))
        print(f"Encoder output: {x.shape}")

    print(f"\nPython dumps written to {PY_DIR}/")
    print(f"C++ dumps at {CPP_DIR}/")

    print("\n" + "=" * 70)
    print(f"{'Stage':25s} {'MeanDiff':>10s} {'MaxDiff':>10s} {'Corr':>10s} {'Status':>6s}")
    print("=" * 70)

    stages = ['after_conv'] + [f'layer{i:02d}' for i in range(24)] + \
             ['after_ln_post', 'after_proj1_gelu', 'encoder_output']

    for stage in stages:
        py_path = os.path.join(PY_DIR, f'{stage}.bin')
        cpp_path = os.path.join(CPP_DIR, f'{stage}.bin')

        if not os.path.exists(py_path) or not os.path.exists(cpp_path):
            print(f"{stage:25s} {'MISSING':>10s}")
            continue

        cols = d_model
        if stage in ('after_proj1_gelu', 'encoder_output'):
            cols = 1024

        py_data = load_bf16_bin(py_path, cols)
        cpp_data = load_bf16_bin(cpp_path, cols)
        rows = min(py_data.shape[0], cpp_data.shape[0])

        if py_data.shape[0] != cpp_data.shape[0]:
            print(f"{stage:25s} ROWS MISMATCH py={py_data.shape[0]} cpp={cpp_data.shape[0]}")

        diff = np.abs(cpp_data[:rows] - py_data[:rows])
        mean_d = diff.mean()
        max_d = diff.max()
        c_f = cpp_data[:rows].flatten()
        p_f = py_data[:rows].flatten()
        corr = np.corrcoef(c_f, p_f)[0, 1] if c_f.std() > 0 and p_f.std() > 0 else 0
        flag = "OK" if mean_d < 0.01 else ("~" if mean_d < 0.05 else ("!!" if mean_d < 0.1 else "FAIL"))
        print(f"{stage:25s} {mean_d:10.6f} {max_d:10.6f} {corr:10.6f} {flag:>6s}")

    print("=" * 70)

if __name__ == '__main__':
    run()
