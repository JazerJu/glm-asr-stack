#!/usr/bin/env python3
"""Export Python encoder output, decoder input, and decoder layer 0 for comparison with C++.
Usage: python export_py_states.py [output_dir]"""
import sys, os, struct, numpy as np, torch, wave, torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
sys.path.insert(0, '/data/ASR模型/Qwen3-ASR')
from qwen_asr import Qwen3ForcedAligner

MODEL_PATH = "/data/.cache/huggingface/hub/models--Qwen--Qwen3-ForcedAligner-0.6B/snapshots/c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else '/tmp/states_export/py'
os.makedirs(OUT_DIR, exist_ok=True)

def dump_bf16(t, name):
    path = os.path.join(OUT_DIR, name)
    t.detach().cpu().view(torch.uint16).numpy().tofile(path)
    print(f"  {name}: {list(t.shape)} -> {path}")

AUDIO_PATH = '/data/fwsr/glm-asr/GLM-ASR/cut.wav'
TEXT = "Let's get started. Welcome everyone to Claude Code Best Practices. In this talk, I am going to talk about kind of what Claude Code is at a high level. Then we'll peer under the hood a little."

print("=== Step 1: Compute mel spectrogram ===")
mel_filters = np.fromfile('/data/fwsr/glm-asr/forcealigner-engine/mel_filterbank.bin', dtype=np.float32).reshape(128, 201)
mel_filters_t = torch.from_numpy(mel_filters)
with wave.open(AUDIO_PATH, 'rb') as wf:
    n = wf.getnframes(); raw = wf.readframes(n)
wav = np.array(struct.unpack(f'<{n}h', raw), dtype=np.float32) / 32768.0
peak = np.abs(wav).max()
if peak > 1.0: wav = wav / peak
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
print(f"Mel shape: [{list(log_mel.shape)}], T={mel_T}")
dump_bf16(log_mel, 'mel_input.bin')

print("\n=== Step 2: Load model and compute encoder output ===")
model = Qwen3ForcedAligner.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map='cuda:0')
tower = model.model.thinker.audio_tower
chunk_size = tower.n_window * 2

with torch.no_grad():
    feature_len = torch.tensor([log_mel.shape[1]], device='cuda')
    chunk_num = (log_mel.shape[1] + chunk_size - 1) // chunk_size
    rem = log_mel.shape[1] % chunk_size
    if rem == 0:
        chunk_lengths = [chunk_size] * chunk_num
    else:
        chunk_lengths = [chunk_size] * (chunk_num - 1) + [rem]
    
    mel_chunks = list(log_mel.T.split(chunk_lengths, dim=0))
    padded = torch.nn.utils.rnn.pad_sequence(mel_chunks, batch_first=True).transpose(1,2).unsqueeze(1)
    
    padded_embeds = []
    for chunk in padded.split(tower.conv_chunksize, dim=0):
        ce = F.gelu(tower.conv2d1(chunk)); ce = F.gelu(tower.conv2d2(ce)); ce = F.gelu(tower.conv2d3(ce))
        padded_embeds.append(ce)
    padded_embed = torch.cat(padded_embeds, dim=0)
    
    b,c,f,t = padded_embed.size()
    conv_out = tower.conv_out(padded_embed.permute(0,3,1,2).contiguous().view(b,t,c*f))
    pos_emb = tower.positional_embedding.positional_embedding[:conv_out.shape[1],:].unsqueeze(0).to(conv_out.dtype)
    x = conv_out + pos_emb
    
    def cnn_len(l):
        for _ in range(3): l = (l+2*1-3)//2+1
        return l
    feature_lens = [cnn_len(l) for l in chunk_lengths]
    padded_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.ones(l, dtype=torch.bool, device='cuda') for l in feature_lens], batch_first=True)
    x = x[padded_mask]
    
    window_aftercnn = padded_mask.shape[-1] * (tower.n_window_infer // (tower.n_window*2))
    cu_lens = [0]
    for cnn_l in feature_lens:
        cu_lens += [window_aftercnn]*(cnn_l//window_aftercnn)
        r = cnn_l % window_aftercnn
        if r>0: cu_lens.append(r)
    cu_seqlens = torch.tensor(cu_lens, device='cuda').cumsum(-1, dtype=torch.int32)
    
    for layer in tower.layers:
        out = layer(x, cu_seqlens=cu_seqlens)
        x = out[0] if isinstance(out, tuple) else out
    x = tower.ln_post(x); x = tower.proj1(x); x = tower.act(x); x = tower.proj2(x)
    
    print(f"Encoder output shape: {list(x.shape)}")
    dump_bf16(x, 'encoder_output.bin')

print("\n=== Step 3: Get decoder input (token embeddings with audio injection) ===")
tokenizer = model.processor.tokenizer
word_list, aligner_input_text = model.aligner_processor.encode_timestamp(TEXT, 'English')

audio_start_id = 151669; audio_pad_id = 151676; audio_end_id = 151670; ts_id = 151705
n_audio = x.shape[0]

input_ids_list = [audio_start_id]
input_ids_list += [audio_pad_id] * n_audio
input_ids_list.append(audio_end_id)
for word in word_list:
    bpe_ids = tokenizer.encode(word, add_special_tokens=False)
    input_ids_list += bpe_ids
    input_ids_list += [ts_id, ts_id]
seq_len = len(input_ids_list)
input_ids_t = torch.tensor([input_ids_list], device='cuda')
print(f"input_ids: {seq_len} tokens, {n_audio} audio pads")

with torch.no_grad():
    embeds = model.model.thinker.get_input_embeddings()(input_ids_t)
    pad_positions = (input_ids_t[0] == audio_pad_id).nonzero(as_tuple=True)[0]
    audio_features = x.unsqueeze(0).to(embeds.dtype)
    for i, pos in enumerate(pad_positions):
        embeds[0, pos] = audio_features[0, i]
    
    dump_bf16(embeds[0], 'dec_input_embeds.bin')
    print(f"Decoder input embeddings shape: {list(embeds.shape)}")

print("\n=== Step 4: Get decoder layer 0 output via hook ===")
layer0_out = {}
def hook_l0(module, inp, outp):
    layer0_out['data'] = outp[0].detach() if isinstance(outp, tuple) else outp.detach()

handle = model.model.thinker.model.layers[0].register_forward_hook(hook_l0)

with torch.no_grad():
    result = model.align(audio=AUDIO_PATH, text=TEXT, language='English')

handle.remove()

if 'data' in layer0_out:
    dump_bf16(layer0_out['data'][0], 'dec_layer00_output.bin')
    print(f"Decoder layer 0 output shape: {list(layer0_out['data'].shape)}")

print(f"\n=== All states exported to {OUT_DIR}/ ===")
