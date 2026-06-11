#!/usr/bin/env python3
"""Dump Python decoder internal states for comparison with C++.
Runs the forced aligner and captures layer outputs via hooks."""
import sys, os, json, struct, numpy as np, torch
import warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

sys.path.insert(0, '/data/ASR模型/Qwen3-ASR')
from qwen_asr import Qwen3ForcedAligner

MODEL_PATH = "/data/.cache/huggingface/hub/models--Qwen--Qwen3-ForcedAligner-0.6B/snapshots/c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else '/tmp/dec_dump/py'
os.makedirs(OUT_DIR, exist_ok=True)

def dump_tensor(t, name):
    import os as _os
    path = _os.path.join(OUT_DIR, f'{name}.bin')
    t.detach().cpu().view(torch.uint16).numpy().tofile(path)
    print(f"  {name}: {list(t.shape)}")

print("Loading model...")
model = Qwen3ForcedAligner.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map='cuda:0')
thinker = model.model.thinker

# Collect encoder output using the full forcualigner pipeline on a test audio
AUDIO_PATH = '/data/fwsr/glm-asr/data/buckeye/audio/s0902b_011.wav'
TEXT = "i do have about people yeah who are having all these kids and don't take care of them at all"

# Run alignment through the full pipeline, capturing encoder output
# Hook the thinker model
hook_outputs = {}

def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            hook_outputs[name] = output[0].detach()
        else:
            hook_outputs[name] = output.detach()
    return hook

# Register hooks on decoder layers
for i, layer in enumerate(thinker.model.layers):
    layer.register_forward_hook(make_hook(f'dec_layer{i:02d}'))

# Also hook the final norm
thinker.model.norm.register_forward_hook(make_hook('dec_final_norm'))

# Setup input construction matching C++
import wave
with wave.open(AUDIO_PATH, 'rb') as wf:
    n = wf.getnframes()
    raw = wf.readframes(n)
wav = np.array(struct.unpack(f'<{n}h', raw), dtype=np.float32) / 32768.0
peak = np.abs(wav).max()
if peak > 1.0:
    wav = wav / peak
wav = np.clip(wav, -1.0, 1.0)

with torch.no_grad():
    # Run full alignment
    result = model.align(audio=AUDIO_PATH, text=TEXT, language="English")

# Now dump the captured hook outputs
print("Dumping decoder states...")
for name in sorted(hook_outputs.keys()):
    dump_tensor(hook_outputs[name], name)

# Also dump logits by running thinker directly with the same inputs
# Need to reconstruct the exact inputs used by the processor
words = model.aligner_processor.tokenize_chinese_mixed(TEXT) if TEXT else []
language = "English"
word_list, aligner_input_text = model.aligner_processor.encode_timestamp(TEXT, language)
inputs = model.processor(text=aligner_input_text, audio=AUDIO_PATH, return_tensors="pt", padding=True)
inputs = inputs.to(model.model.device).to(model.model.dtype)

with torch.no_grad():
    logits = thinker(**inputs).logits
    dump_tensor(logits, 'decoder_logits')
    dump_tensor(inputs['input_ids'], 'input_ids')
    
    # Get the timestamps from argmax
    output_ids = logits.argmax(dim=-1)
    ts_mask = inputs['input_ids'] == model.timestamp_token_id
    if ts_mask.any():
        ts_values = output_ids[ts_mask].cpu().numpy()
        print(f"Python timestamps (argmax): {ts_values.tolist()}")
        print(f"Python timestamps (ms): {(ts_values * 80).tolist()}")

print(f"\nAll dumps written to {OUT_DIR}/")
