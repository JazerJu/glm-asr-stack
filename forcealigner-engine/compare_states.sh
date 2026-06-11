#!/bin/bash
# compare_states.sh — Compare C++ vs Python encoder/decoder states for cut.wav
# Usage: bash compare_states.sh

set -e
ENGINE_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="/data/.cache/huggingface/hub/models--Qwen--Qwen3-ForcedAligner-0.6B/snapshots/c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
AUDIO="/data/fwsr/glm-asr/GLM-ASR/cut.wav"
TEXT="Let's get started. Welcome everyone to Claude Code Best Practices. In this talk, I am going to talk about kind of what Claude Code is at a high level. Then we'll peer under the hood a little."
PYTHON="/data/fwsr/glm-asr/GLM-ASR/venv/bin/python"
OUT_DIR="/tmp/state_compare"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/py" "$OUT_DIR/cpp"

echo "=== Step 1: Run C++ engine with dumps ==="
cd "$ENGINE_DIR"
FA_DUMP_ENCODER_DIR="$OUT_DIR/cpp" FA_DUMP_DECODER_DIR="$OUT_DIR/cpp" \
  ./force_aligner --model "$MODEL_DIR" --audio "$AUDIO" --text "$TEXT" --lang English 2>&1 | grep -E "dump|Done|Audio tokens|Seq"

echo ""
echo "=== Step 2: Run Python export ==="
$PYTHON "$ENGINE_DIR/export_py_states.py" "$OUT_DIR/py" 2>&1 | grep -v "WARNING"

echo ""
echo "=== Step 3: Compare ==="
$PYTHON -c "
import os, numpy as np

def load(path, cols):
    raw = np.fromfile(path, dtype=np.uint16)
    rows = len(raw) // cols
    import torch
    return torch.from_numpy(raw[:rows*cols].copy()).view(torch.bfloat16).float().numpy().reshape(rows, cols)

py_d = '$OUT_DIR/py'
cpp_d = '$OUT_DIR/cpp'

for name, cols in [('mel_input.bin',128), ('encoder_output.bin',1024), ('dec_input_embeds.bin',1024), ('dec_layer00.bin',1024)]:
    py_p = os.path.join(py_d, name)
    cpp_p = os.path.join(cpp_d, name)
    # map Python dec_layer00_output -> dec_layer00
    if name == 'dec_layer00.bin' and not os.path.exists(py_p):
        py_p = os.path.join(py_d, 'dec_layer00_output.bin')
    if not os.path.exists(py_p) or not os.path.exists(cpp_p):
        print(f'{name}: MISSING (py={os.path.exists(py_p)}, cpp={os.path.exists(cpp_p)})')
        continue
    py = load(py_p, cols); cpp = load(cpp_p, cols)
    if py.shape != cpp.shape:
        print(f'{name}: SHAPE MISMATCH py={py.shape} cpp={cpp.shape}')
        continue
    d = np.abs(py - cpp)
    c = np.dot(py.flatten(),cpp.flatten())/(np.linalg.norm(py)*np.linalg.norm(cpp)+1e-10)
    nz = np.isnan(cpp).sum()
    print(f'{name}: shape={py.shape}, cosine={c:.6f}, mean_diff={d.mean():.6f}, max_diff={d.max():.6f}, NaN={nz}')

    # Per-row cosine for key positions: first, last of each chunk (every 13 rows)
    for r_label, r in [('first',0), ('1st chunk last',12), ('2nd chunk first',13), ('2nd chunk last',25), ('mid', py.shape[0]//2), ('last', py.shape[0]-1)]:
        if r < py.shape[0]:
            rc = np.dot(py[r], cpp[r])/(np.linalg.norm(py[r])*np.linalg.norm(cpp[r])+1e-10)
            flag = 'OK' if rc>0.999 else ('~' if rc>0.99 else '!!')
            print(f'  row {r:4d} ({r_label:20s}): cosine={rc:.6f} {flag}')
    print()
"
