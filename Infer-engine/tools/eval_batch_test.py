"""
LibriSpeech evaluation script for GLM-ASR C inference engine.

Reads audio from a local HuggingFace Arrow cache when available, otherwise falls
back to datasets.load_dataset(). Exports selected rows as raw float32 PCM and
runs the C engine in daemon mode.

Usage:
    python3 eval_batch_test.py --split test.clean --num 4
    python3 eval_batch_test.py --split test.other --num 80
    python3 eval_batch_test.py --split test.clean --all       # full set
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import soundfile as sf


HF_CACHE = "/data/.cache/huggingface"
HF_DATASET = "hf-audio/esb-datasets-test-only-sorted"
HF_CONFIG = "librispeech"
ARROW_DIR = os.path.join(
    HF_CACHE,
    "datasets/hf-audio___esb-datasets-test-only-sorted/librispeech/0.0.0/f04e38ac30d7f497f7cb58a29c4996e81f2c4e67",
)

SPLIT_TO_FILE = {
    "test.clean": "esb-datasets-test-only-sorted-test.clean.arrow",
    "test.other": "esb-datasets-test-only-sorted-test.other.arrow",
}

ENGINE_BIN = "./Infer-engine/glm_asr_infer"
MODEL_DIR = "./GLM-ASR/glm-asr-nvfp4-awq"
MEL_FILTERS = "./output/mel_filters.bin"

MAX_AUDIO_SECS = 30.0
ENGINE_MAX_BATCH = 0


def load_arrow_table(split: str) -> Any:
    import pyarrow as pa

    filename = SPLIT_TO_FILE.get(split)
    if not filename:
        print(f"Unknown split: {split}. Available: {list(SPLIT_TO_FILE.keys())}")
        sys.exit(1)

    path = os.path.join(ARROW_DIR, filename)
    if not os.path.exists(path):
        print(f"Arrow file not found: {path}")
        print(f"Set HF_HOME={HF_CACHE} or download the dataset first.")
        sys.exit(1)

    reader = pa.ipc.open_stream(path)
    return reader.read_all()


def try_load_arrow_table(split: str, arrow_dir: str) -> Any | None:
    filename = SPLIT_TO_FILE.get(split)
    if not filename:
        return None
    path = os.path.join(arrow_dir, filename)
    if not os.path.exists(path):
        return None
    import pyarrow as pa

    reader = pa.ipc.open_stream(path)
    return reader.read_all()


def decode_flac_to_f32(flac_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode FLAC bytes from arrow to mono float32 numpy array."""
    import io

    data, sr = sf.read(io.BytesIO(flac_bytes))

    if len(data.shape) == 2:
        data = data.mean(axis=1)

    return data.astype(np.float32), sr


def decode_audio_value(audio_value) -> tuple[np.ndarray, int]:
    """Decode a HF datasets audio field into mono float32 PCM."""
    if isinstance(audio_value, dict):
        if audio_value.get("array") is not None:
            data = np.asarray(audio_value["array"], dtype=np.float32)
            sr = int(audio_value.get("sampling_rate") or 16000)
        elif audio_value.get("bytes") is not None:
            data, sr = decode_flac_to_f32(audio_value["bytes"])
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


def load_hf_dataset(dataset: str, config: str, split: str, cache_dir: str):
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install it with `pip install datasets soundfile pyarrow` "
            "or pre-populate the local Arrow cache."
        ) from exc

    ds = load_dataset(dataset, config, split=split, cache_dir=cache_dir)
    if "audio" in ds.features:
        ds = ds.cast_column("audio", Audio(decode=False))
    return ds


def parse_engine_output(lines: list[str], bin_files: list[str]) -> dict[str, str]:
    """Parse engine stdout lines to extract transcriptions keyed by bin file path."""
    results = {}
    pattern = re.compile(r"^(.+\.bin):\s*(.+)$")

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
    parser = argparse.ArgumentParser(description="Evaluate GLM-ASR on LibriSpeech")
    parser.add_argument("--split", default="test.clean", help="Dataset split")
    parser.add_argument("--dataset", default=HF_DATASET, help="HuggingFace dataset id")
    parser.add_argument("--dataset-config", default=HF_CONFIG, help="HuggingFace dataset config")
    parser.add_argument("--hf-cache", default=HF_CACHE, help="HuggingFace cache directory")
    parser.add_argument(
        "--arrow-dir",
        default=ARROW_DIR,
        help="Use this local Arrow directory when present before falling back to datasets.load_dataset",
    )
    parser.add_argument("--num", type=int, default=4, help="Number of samples")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument(
        "--all", action="store_true", help="Process all samples in the split"
    )
    parser.add_argument(
        "--engine", default=ENGINE_BIN, help="Path to glm_asr_infer binary"
    )
    parser.add_argument("--model", default=MODEL_DIR, help="Model directory")
    parser.add_argument("--mel", default=MEL_FILTERS, help="Mel filters file")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep temp .bin files")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=ENGINE_MAX_BATCH,
        help="Max samples per engine run (default: all selected samples in one RUN)",
    )
    args = parser.parse_args()

    total_start_time = time.time()
    table = try_load_arrow_table(args.split, args.arrow_dir)
    ds = None
    if table is not None:
        print(f"Loading {args.split} from local Arrow cache: {args.arrow_dir}")
        total = table.num_rows
    else:
        print(
            f"Loading {args.dataset}/{args.dataset_config} split={args.split} "
            f"via HuggingFace datasets..."
        )
        ds = load_hf_dataset(args.dataset, args.dataset_config, args.split, args.hf_cache)
        total = len(ds)
    print(f"  {total} total rows")

    if table is not None:
        audio_col = table.column("audio")
        text_col = table.column("text")
        length_col = table.column("audio_length_s") if "audio_length_s" in table.column_names else None
    else:
        audio_col = text_col = length_col = None

    start = args.start
    if args.all:
        count = total - start
    else:
        count = args.num
    end = min(start + count, total)
    count = end - start

    print(f"  Evaluating rows {start}..{end - 1} ({count} samples)\n")

    tmp_dir = tempfile.mkdtemp(prefix="glm_asr_eval_")
    samples = []
    skipped = 0

    for i in range(start, end):
        if table is not None:
            audio_value = audio_col[i].as_py()
            text = text_col[i].as_py()
            if length_col is not None:
                length_s = float(length_col[i].as_py())
            else:
                pcm_probe, sr_probe = decode_audio_value(audio_value)
                length_s = len(pcm_probe) / float(sr_probe)
                audio_value = {"array": pcm_probe, "sampling_rate": sr_probe}
        else:
            row = ds[i]
            audio_value = row["audio"]
            text = row.get("text") or row.get("sentence") or row.get("transcript") or ""
            length_s = float(row.get("audio_length_s") or 0.0)

        pcm, sr = decode_audio_value(audio_value)
        if length_s <= 0.0:
            length_s = len(pcm) / float(sr)
        if length_s > MAX_AUDIO_SECS:
            print(f"  [SKIP] row {i}: {length_s:.1f}s exceeds {MAX_AUDIO_SECS}s limit")
            skipped += 1
            continue
        if sr != 16000:
            raise RuntimeError(f"Expected 16 kHz audio after dataset decode, got {sr} Hz at row {i}")
        bin_path = os.path.join(tmp_dir, f"sample_{i}.bin")
        pcm.tofile(bin_path)

        samples.append((bin_path, text, i, length_s))
        print(f"  row {i}: {length_s:.1f}s, {len(pcm)} samples")

    if not samples:
        print("\nNo valid samples to evaluate.")
        return

    effective_batch_size = len(samples) if args.batch_size <= 0 else args.batch_size

    print(f"\n{'=' * 60}")
    if args.batch_size <= 0:
        print(f"Processing {len(samples)} samples in a single daemon RUN")
    else:
        print(
            f"Processing {len(samples)} samples (max {effective_batch_size} per batch)"
        )
    print(f"{'=' * 60}\n")

    print("Starting engine daemon...")
    daemon = EngineDaemon(args.engine, args.model, args.mel)
    print("Engine loaded.\n")

    all_results = []
    total_infer_ms = 0.0
    total_infer_samples = 0
    total_output_tokens = 0

    for batch_start in range(0, len(samples), effective_batch_size):
        batch_end = min(batch_start + effective_batch_size, len(samples))
        batch = samples[batch_start:batch_end]
        bin_files = [s[0] for s in batch]

        if effective_batch_size < len(samples):
            batch_num = batch_start // effective_batch_size + 1
            print(
                f"Batch {batch_num}: samples {batch_start}-{batch_end - 1} ({len(batch)} files)"
            )
        else:
            print(f"RUN: samples {batch_start}-{batch_end - 1} ({len(batch)} files)")

        output_lines, infer_line, infer_ms, output_tokens = daemon.run_batch(bin_files)

        if infer_ms > 0:
            total_infer_ms += infer_ms
            total_infer_samples += len(batch)
            total_output_tokens += output_tokens
            print(f"  {infer_line}")

        parsed = parse_engine_output(output_lines, bin_files)

        for bin_path, gt, row_idx, length_s in batch:
            pred = parsed.get(bin_path, "[NO OUTPUT]")
            all_results.append((row_idx, pred, gt))
            print(f"  row {row_idx}: {pred[:80]}...")

    daemon.close()

    print(f"\n{'=' * 60}")
    print(f"Results Summary ({len(all_results)} successful)")
    print(f"{'=' * 60}")
    for row_idx, pred, gt in all_results:
        print(f"\n[{row_idx}] PRED: {pred}")
        print(f"[{row_idx}] TRUE: {gt}")

    total_elapsed = time.time() - total_start_time
    hours = int(total_elapsed // 3600)
    minutes = int((total_elapsed % 3600) // 60)
    seconds = total_elapsed % 60

    if hours > 0:
        time_str = f"{hours}h {minutes}m {seconds:.1f}s"
    elif minutes > 0:
        time_str = f"{minutes}m {seconds:.1f}s"
    else:
        time_str = f"{seconds:.1f}s"

    print(f"\n{'=' * 60}")
    print(f"TOTAL TIME: {time_str}")
    if total_infer_samples > 0:
        avg_ms = total_infer_ms / total_infer_samples
        total_audio_s = sum(s[3] for s in [s for s in samples if s[1]])
        rtfx = total_audio_s / (total_infer_ms / 1000.0) if total_infer_ms > 0 else 0
        print(
            f"INFER TIME: {total_infer_ms / 1000:.1f}s ({total_infer_samples} samples)"
        )
        print(f"AVG: {avg_ms:.1f} ms/sample | RTFx: {rtfx:.1f}x")
        tok_s = total_output_tokens / (total_infer_ms / 1000.0) if total_infer_ms > 0 else 0
        tok_sample = total_output_tokens / total_infer_samples if total_infer_samples > 0 else 0
        print(
            f"OUTPUT TOKENS: {total_output_tokens} | TOK/s: {tok_s:.1f} | TOK/sample: {tok_sample:.2f}"
        )
    print(f"{'=' * 60}")

    if not args.keep_tmp:
        import shutil

        shutil.rmtree(tmp_dir)
        print(f"\nCleaned up {tmp_dir}")
    else:
        print(f"\nTemp files kept in {tmp_dir}")


if __name__ == "__main__":
    main()
