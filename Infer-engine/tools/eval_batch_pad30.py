"""
Pad30s evaluation — 256 audio files padded to 30s, using daemon batch mode.
Timing is infer-only: LOAD/MEL excluded, INFER TIME from submit to done.

Usage:
    GLMASR_MAX_NEW_TOKENS=1 python3 tools/eval_batch_pad30.py
"""

import argparse
import glob
import io
import os
import re
import subprocess
import sys
import time

import numpy as np
import soundfile as sf


ENGINE_BIN = "./glm_asr_infer"
MODEL_DIR = "/data/fwsr/glm-asr/GLM-ASR/glm-asr-nvfp4-awq"
MEL_FILTERS = "/data/fwsr/glm-asr/output/mel_filters.bin"
PCM_DIR = "/dev/shm/ie_pad30"
PAD_SECS = 30    # every file is exactly 30s
HF_CACHE = "/data/.cache/huggingface"
HF_DATASET = "hf-audio/esb-datasets-test-only-sorted"
HF_CONFIG = "librispeech"


def decode_audio_value(audio_value) -> tuple[np.ndarray, int]:
    if isinstance(audio_value, dict):
        if audio_value.get("array") is not None:
            data = np.asarray(audio_value["array"], dtype=np.float32)
            sr = int(audio_value.get("sampling_rate") or 16000)
        elif audio_value.get("bytes") is not None:
            data, sr = sf.read(io.BytesIO(audio_value["bytes"]))
            data = np.asarray(data, dtype=np.float32)
        elif audio_value.get("path"):
            data, sr = sf.read(audio_value["path"])
            data = np.asarray(data, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported audio field keys: {sorted(audio_value.keys())}")
    else:
        raise TypeError(f"Unsupported audio field type: {type(audio_value)!r}")
    if data.ndim == 2:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float32), sr


def prepare_pad30_from_hf(
    out_dir: str,
    vllm_dir: str,
    num: int,
    start: int,
    dataset: str,
    config: str,
    split: str,
    hf_cache: str,
    force: bool,
) -> list[str]:
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install it with `pip install datasets soundfile pyarrow`."
        ) from exc

    os.makedirs(out_dir, exist_ok=True)
    if vllm_dir:
        os.makedirs(vllm_dir, exist_ok=True)
    existing = sorted(glob.glob(os.path.join(out_dir, "pad30_*.bin")))
    existing_vllm = sorted(glob.glob(os.path.join(vllm_dir, "pad30_*.flac"))) if vllm_dir else []
    if len(existing) >= num and (not vllm_dir or len(existing_vllm) >= num) and not force:
        return existing[:num]

    ds = load_dataset(dataset, config, split=split, cache_dir=hf_cache)
    if "audio" in ds.features:
        ds = ds.cast_column("audio", Audio(decode=False))

    target_len = PAD_SECS * 16000
    paths = []
    row = start
    while len(paths) < num and row < len(ds):
        item = ds[row]
        pcm, sr = decode_audio_value(item["audio"])
        if sr != 16000:
            raise RuntimeError(f"Expected 16 kHz audio after dataset decode, got {sr} Hz at row {row}")
        padded = np.zeros(target_len, dtype=np.float32)
        n = min(target_len, len(pcm))
        padded[:n] = pcm[:n]
        path = os.path.join(out_dir, f"pad30_{len(paths):04d}.bin")
        padded.tofile(path)
        if vllm_dir:
            flac_path = os.path.join(vllm_dir, f"pad30_{len(paths):04d}.flac")
            sf.write(flac_path, padded, 16000)
        paths.append(path)
        row += 1

    if len(paths) < num:
        raise RuntimeError(f"Only prepared {len(paths)} pad30 files from {dataset}/{config}:{split}")
    return paths


def parse_engine_output(lines: list[str], bin_files: list[str]) -> dict[str, str]:
    """Parse engine stdout lines to extract transcriptions keyed by bin file path."""
    results = {}
    pattern = re.compile(r"^(.+):\s*(.+)$")

    for line in lines:
        line = line.strip()
        match = pattern.match(line)
        if match:
            path, text = match.groups()
            results[path] = text

    return results


class EngineDaemon:
    """Long-running engine process — model loaded once, batch submissions via stdin/stdout."""

    def __init__(self, engine: str, model: str, mel: str):
        self.proc = subprocess.Popen(
            [engine, "--daemon"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._send(f"LOAD {model}")
        self._drain_until_ready()
        self._send(f"MEL {mel}")
        self._drain_until_ready()

    def _send(self, line: str):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _drain_until_ready(self):
        lines = []
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            if line == "READY":
                return lines
            lines.append(line)
        return lines

    def run_batch(
        self, bin_files: list[str], timeout: int = 600
    ) -> tuple[list[str], str, float, int]:
        self._send(f"BATCH {len(bin_files)}")
        for bf in bin_files:
            self._send(bf)
        self._send("RUN")

        all_lines = self._drain_until_ready()

        infer_ms = 0.0
        infer_line = ""
        output_tokens = 0
        output_lines = []
        for line in all_lines:
            if line.startswith("INFER_TIME:"):
                infer_line = line
                m = re.match(r"INFER_TIME:\s+([\d.]+)\s+ms", line)
                if m:
                    infer_ms = float(m.group(1))
            elif line.startswith("OUTPUT_TOKENS:"):
                m = re.match(r"OUTPUT_TOKENS:\s+(\d+)", line)
                if m:
                    output_tokens = int(m.group(1))
            else:
                output_lines.append(line)

        return output_lines, infer_line, infer_ms, output_tokens

    def close(self):
        try:
            self._send("QUIT")
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
            self.proc.wait()


def main():
    parser = argparse.ArgumentParser(description="Evaluate GLM-ASR on pad30s PCM")
    parser.add_argument("--num", type=int, default=256, help="Number of samples")
    parser.add_argument("--audio-dir", default=PCM_DIR, help="Directory containing pad30_*.bin files")
    parser.add_argument("--prepare-from-hf", action="store_true", help="Create pad30_*.bin files from a HuggingFace dataset when needed")
    parser.add_argument("--prepare-vllm-dir", default="", help="Also write pad30_*.flac files for vLLM comparison")
    parser.add_argument("--prepare-force", action="store_true", help="Overwrite existing pad30 files during HF preparation")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare pad30 files and exit without running the engine")
    parser.add_argument("--warmup", type=int, default=0, help="Run this many warmup samples before the timed batch")
    parser.add_argument("--dataset", default=HF_DATASET, help="HuggingFace dataset id for --prepare-from-hf")
    parser.add_argument("--dataset-config", default=HF_CONFIG, help="HuggingFace dataset config")
    parser.add_argument("--split", default="test.clean", help="HuggingFace split")
    parser.add_argument("--start", type=int, default=0, help="Start row for --prepare-from-hf")
    parser.add_argument("--hf-cache", default=HF_CACHE, help="HuggingFace cache directory")
    parser.add_argument("--engine", default=ENGINE_BIN, help="Path to glm_asr_infer binary")
    parser.add_argument("--model", default=MODEL_DIR, help="Model directory")
    parser.add_argument("--mel", default=MEL_FILTERS, help="Mel filters file")
    args = parser.parse_args()

    requested = args.num + max(args.warmup, 0)
    if args.prepare_from_hf:
        pcm_files = prepare_pad30_from_hf(
            args.audio_dir,
            args.prepare_vllm_dir,
            requested,
            args.start,
            args.dataset,
            args.dataset_config,
            args.split,
            args.hf_cache,
            args.prepare_force,
        )
    else:
        pcm_files = sorted(glob.glob(os.path.join(args.audio_dir, "pad30_*.bin")))[:requested]
    total_n = len(pcm_files)
    if total_n == 0:
        print(f"No PCM files in {args.audio_dir}")
        print("Use --prepare-from-hf to create pad30 files from a HuggingFace dataset.")
        sys.exit(1)
    if total_n < requested:
        print(f"Need {requested} PCM files in {args.audio_dir}, found {total_n}")
        print("Use --prepare-from-hf or --prepare-force to create enough pad30 files.")
        sys.exit(1)

    warmup_files = pcm_files[: args.warmup] if args.warmup > 0 else []
    timed_files = pcm_files[args.warmup : args.warmup + args.num]
    n = len(timed_files)

    print(f"Using {n} timed pad30s PCM files from {args.audio_dir}")
    if args.warmup > 0:
        print(f"Using {len(warmup_files)} separate warmup pad30s PCM files")
    print(f"Each is {PAD_SECS}s → total timed audio = {n * PAD_SECS}s")
    if args.prepare_only:
        if args.prepare_vllm_dir:
            vllm_files = sorted(glob.glob(os.path.join(args.prepare_vllm_dir, "pad30_*.flac")))
            print(f"Prepared {len(vllm_files)} vLLM FLAC files in {args.prepare_vllm_dir}")
        return

    print("\nStarting engine daemon...")
    daemon = EngineDaemon(args.engine, args.model, args.mel)
    print("Engine loaded.\n")

    if warmup_files:
        print(f"WARMUP: {len(warmup_files)} samples")
        daemon.run_batch(warmup_files)

    print(f"RUN: {n} samples in single batch")
    output_lines, infer_line, infer_ms, output_tokens = daemon.run_batch(timed_files)

    daemon.close()

    print(f"\n{'=' * 60}")
    results = parse_engine_output(output_lines, timed_files)
    ok = 0
    for i, f in enumerate(timed_files):
        pred = results.get(f, "[NO OUTPUT]")
        if pred and pred != "[NO OUTPUT]":
            ok += 1
        if i < 10 or i % 40 == 0:
            print(f"  [{i}] {pred[:80]}")

    total_audio_s = n * PAD_SECS
    rtfx = total_audio_s / (infer_ms / 1000.0) if infer_ms > 0 else 0
    tok_s = output_tokens / (infer_ms / 1000.0) if infer_ms > 0 else 0
    tok_sample = output_tokens / n if n > 0 else 0
    print(f"\n{'=' * 60}")
    print(f"INFER TIME: {infer_ms/1000:.1f}s ({n} samples)")
    print(f"AVG: {infer_ms/n:.1f} ms/sample | RTFx: {rtfx:.1f}x")
    print(f"OUTPUT TOKENS: {output_tokens} | TOK/s: {tok_s:.1f} | TOK/sample: {tok_sample:.2f}")
    print(f"ok={ok}/{n}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
