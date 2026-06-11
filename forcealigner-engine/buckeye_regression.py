#!/usr/bin/env python3
"""Buckeye forced alignment regression test.
Runs Python Qwen3ForcedAligner and C++ engine on short segments,
compares against human ground truth using AAS (Accumulated Average Shift).
"""
import json
import subprocess
import sys
import os
import time
import tempfile

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "buckeye")
MANIFEST = os.path.join(DATA_DIR, "manifest.json")
ENGINE = os.path.join(os.path.dirname(__file__), "force_aligner")
MODEL_DIR = "/data/.cache/huggingface/hub/models--Qwen--Qwen3-ForcedAligner-0.6B"
# Resolve snapshot
for d in os.listdir(os.path.join(MODEL_DIR, "snapshots")):
    MODEL_PATH = os.path.join(MODEL_DIR, "snapshots", d)
    break

PYTHON = "/data/fwsr/glm-asr/GLM-ASR/venv/bin/python"

def load_manifest():
    with open(MANIFEST) as f:
        return json.load(f)

def pick_segments(data, max_duration=10.0, min_words=4, max_words=20, limit=20):
    segs = [s for s in data["samples"]
            if s["duration_s"] <= max_duration
            and min_words <= s["num_words"] <= max_words]
    # Deterministic pick: sort by id, take first `limit`
    segs.sort(key=lambda s: s["id"])
    return segs[:min(limit, len(segs))]

def parse_engine_output(text):
    """Parse tab-separated: word\\tstart\\tend"""
    results = []
    for line in text.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) == 3:
            results.append({
                "word": parts[0],
                "start_ms": float(parts[1]) * 1000,
                "end_ms": float(parts[2]) * 1000,
            })
    return results

def run_cpp_engine(segments):
    """Run C++ engine in daemon mode on all segments."""
    print(f"[C++] Starting engine daemon...")
    # Build stdin input
    stdin_lines = []
    for seg in segments:
        audio_path = os.path.join(DATA_DIR, seg["audio"])
        stdin_lines.append(f"{audio_path}\t{seg['transcript']}\tEnglish")
    stdin_data = "\n".join(stdin_lines) + "\n"

    t0 = time.time()
    proc = subprocess.run(
        [ENGINE, "--model", MODEL_PATH, "--daemon"],
        input=stdin_data,
        capture_output=True,
        text=True,
        timeout=300,
    )
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"[C++] Engine failed: {proc.stderr}", file=sys.stderr)
        sys.exit(1)

    # Parse outputs — each segment separated by blank line
    outputs = proc.stdout.strip().split("\n\n") if "\n\n" in proc.stdout else proc.stdout.strip().split("get\t")
    # Better: split by segment boundaries
    all_lines = proc.stdout.strip().split("\n")

    # Match outputs to segments by counting
    results_per_segment = []
    current = []
    seg_idx = 0
    for line in all_lines:
        line = line.strip()
        if not line:
            if current:
                results_per_segment.append(parse_engine_output("\n".join(current)))
                current = []
            continue
        current.append(line)
    if current:
        results_per_segment.append(parse_engine_output("\n".join(current)))

    print(f"[C++] Processed {len(results_per_segment)} segments in {elapsed:.1f}s "
          f"({elapsed/len(segments)*1000:.0f}ms/segment)")

    if len(results_per_segment) != len(segments):
        print(f"[C++] WARNING: output segments ({len(results_per_segment)}) != input ({len(segments)})")
        # Pad or truncate
        while len(results_per_segment) < len(segments):
            results_per_segment.append([])

    return results_per_segment

def run_python_engine(segments):
    """Run Python reference one segment at a time."""
    print(f"[Python] Running Python reference on {len(segments)} segments...")

    # Write a helper script
    helper = """
import sys
import json
import os
import warnings
import logging
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TQDM_DISABLE"] = "1"
sys.path.insert(0, '/data/ASR模型/Qwen3-ASR')
from qwen_asr import Qwen3ForcedAligner
import torch

MODEL_PATH = "/data/.cache/huggingface/hub/models--Qwen--Qwen3-ForcedAligner-0.6B/snapshots/c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
model = Qwen3ForcedAligner.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

manifest_path = sys.argv[1]
indices = [int(x) for x in sys.argv[2].split(",")]

with open(manifest_path) as f:
    data = json.load(f)

results = []
for idx in indices:
    seg = data["samples"][idx]
    audio_path = seg["audio_path"]
    try:
        res = model.align(audio=audio_path, text=seg["transcript"], language="English")
        words = []
        for w in res[0]:
            words.append({"word": w.text, "start_ms": w.start_time * 1000, "end_ms": w.end_time * 1000})
        results.append(words)
    except Exception as e:
        print(f"ERROR on {seg['id']}: {e}", file=sys.stderr)
        results.append([])

print(json.dumps(results))
"""
    # Build segment info for the helper
    seg_data = []
    for i, seg in enumerate(segments):
        seg_data.append({
            "id": seg["id"],
            "audio_path": os.path.join(DATA_DIR, seg["audio"]),
            "transcript": seg["transcript"],
        })

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"samples": seg_data}, f)
        tmp_manifest = f.name

    t0 = time.time()
    proc = subprocess.run(
        [PYTHON, "-c", helper, tmp_manifest, ",".join(str(i) for i in range(len(segments)))],
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed = time.time() - t0
    os.unlink(tmp_manifest)

    if proc.returncode != 0:
        print(f"[Python] Failed: {proc.stderr}", file=sys.stderr)
        # Fall back: return empty results
        return [[] for _ in segments]

    try:
        stdout = proc.stdout
        # Find the JSON array start (skip any warnings that leaked to stdout)
        json_start = stdout.find("[")
        if json_start >= 0:
            results = json.loads(stdout[json_start:])
        else:
            results = [[] for _ in segments]
    except:
        print(f"[Python] Bad output: {proc.stdout[:200]}", file=sys.stderr)
        return [[] for _ in segments]

    print(f"[Python] Processed {len(results)} segments in {elapsed:.1f}s "
          f"({elapsed/len(segments)*1000:.0f}ms/segment)")
    return results

def compute_aas(predicted, ground_truth):
    """Compute Accumulated Average Shift (mean abs error in ms) between predicted and GT words.
    Matches by word position (not by word text, since normalization may differ).
    """
    if not predicted or not ground_truth:
        return None, 0

    n = min(len(predicted), len(ground_truth))
    if n == 0:
        return None, 0

    total_shift = 0.0
    count = 0
    for i in range(n):
        gt = ground_truth[i]
        pr = predicted[i]
        # Average of start and end absolute errors
        start_err = abs(pr["start_ms"] - gt["start_ms"])
        end_err = abs(pr["end_ms"] - gt["end_ms"])
        total_shift += (start_err + end_err) / 2.0
        count += 1

    return total_shift / count if count > 0 else None, count

def main():
    data = load_manifest()
    segments = pick_segments(data, max_duration=10.0, min_words=4, max_words=20, limit=39)

    print(f"=== Buckeye Regression Test ===")
    print(f"Segments: {len(segments)}")
    print()

    # Run both engines
    cpp_results = run_cpp_engine(segments)
    python_results = run_python_engine(segments)

    # Compare
    print()
    print(f"{'ID':<16} {'GT_w':>5} {'C++_w':>5} {'Py_w':>5}  {'C++_AAS':>8}  {'Py_AAS':>8}  {'C++vsGT':>8}")
    print("-" * 80)

    total_cpp_aas = []
    total_py_aas = []

    for i, seg in enumerate(segments):
        gt_words = seg.get("words", [])
        cpp_words = cpp_results[i] if i < len(cpp_results) else []
        py_words = python_results[i] if i < len(python_results) else []

        cpp_aas, cpp_n = compute_aas(cpp_words, gt_words)
        py_aas, py_n = compute_aas(py_words, gt_words)

        cpp_str = f"{cpp_aas:.1f}" if cpp_aas is not None else "N/A"
        py_str = f"{py_aas:.1f}" if py_aas is not None else "N/A"

        # C++ vs GT diff
        cpp_vs_gt = ""
        if cpp_aas is not None and py_aas is not None:
            diff = cpp_aas - py_aas
            cpp_vs_gt = f"{diff:+.1f}"
            total_cpp_aas.append(cpp_aas)
            total_py_aas.append(py_aas)

        print(f"{seg['id']:<16} {len(gt_words):>5} {len(cpp_words):>5} {len(py_words):>5}  "
              f"{cpp_str:>8}  {py_str:>8}  {cpp_vs_gt:>8}")

    print("-" * 80)
    if total_cpp_aas:
        avg_cpp = sum(total_cpp_aas) / len(total_cpp_aas)
        avg_py = sum(total_py_aas) / len(total_py_aas)
        print(f"{'MEAN':<16} {'':>5} {'':>5} {'':>5}  {avg_cpp:>8.1f}  {avg_py:>8.1f}  {avg_cpp - avg_py:+8.1f}")
        print()
        print(f"C++ AAS: {avg_cpp:.1f} ms | Python AAS: {avg_py:.1f} ms | Δ: {avg_cpp - avg_py:+.1f} ms")
        if abs(avg_cpp - avg_py) < 50:
            print("✅ C++ engine within 50ms of Python reference")
        else:
            print("⚠️  C++ engine differs from Python by >50ms — investigate")
    else:
        print("No comparable results")

if __name__ == "__main__":
    main()
