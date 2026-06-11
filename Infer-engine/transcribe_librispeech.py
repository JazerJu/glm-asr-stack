#!/usr/bin/env python3
"""Run the 2611 mixed-length LibriSpeech Infer-engine benchmark."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_REPO = "zai-org/GLM-ASR-Nano-2512"
DEFAULT_MODEL_DIR = ROOT / "models" / "GLM-ASR-Nano-2512"
DEFAULT_MEL = ROOT / "resources" / "mel_filters.bin"


def env_default(name: str, default: str) -> str:
    return os.environ.get(name, default)


def run_tee(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def require_path(path: str, kind: str) -> None:
    p = Path(path)
    if kind == "dir" and not p.is_dir():
        raise SystemExit(f"ERROR: model directory not found: {path}")
    if kind == "file" and not p.is_file():
        raise SystemExit(f"ERROR: mel filters file not found: {path}")


def download_models(model_dir: Path, mel_path: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise SystemExit("Missing dependency: huggingface_hub. Run `pip install -r requirements.txt`.") from exc

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"==> Downloading {DEFAULT_MODEL_REPO} to {model_dir}")
    snapshot_download(repo_id=DEFAULT_MODEL_REPO, local_dir=str(model_dir), local_dir_use_symlinks=False)

    if not mel_path.exists():
        print(f"==> Downloading mel_filters.bin to {mel_path}")
        mel_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            hf_hub_download(repo_id=DEFAULT_MODEL_REPO, filename="mel_filters.bin", local_dir=str(mel_path.parent))
        except Exception:
            fallback = ROOT / "resources" / "mel_filters.bin"
            if fallback.exists():
                shutil.copy2(fallback, mel_path)
            else:
                raise


def infer_env() -> dict[str, str]:
    env = os.environ.copy()
    defaults = {
        "GLMASR_VRAM_UTIL": "0.9",
        "GLMASR_MAX_SEQS": "256",
        "GLMASR_MAX_NEW_TOKENS": "256",
        "GLMASR_MAX_BATCHED_TOKENS": "10000",
        "GLMASR_ENCODER_FA_PIPELINED": "1",
        "GLMASR_ENCODER_FA_VLLM": "1",
        "GLMASR_ENCODER_FA_LONG_MIN_SEQ": "1",
        "GLMASR_ENCODER_FA_MIN_SEQ": "1",
    }
    for key, value in defaults.items():
        env.setdefault(key, value)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="2611 mixed-length LibriSpeech Infer-engine benchmark")
    parser.add_argument("--engine", default="./glm_asr_infer")
    parser.add_argument("--model", default=os.environ.get("MODEL_BF16", str(DEFAULT_MODEL_DIR)))
    parser.add_argument("--mel", default=os.environ.get("MEL_FILTERS", str(DEFAULT_MEL)))
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--start", type=int, default=int(env_default("GLMASR_BENCH_MIXED_START", "9")))
    parser.add_argument("--num", type=int, default=int(env_default("GLMASR_BENCH_MIXED_NUM", "2611")))
    parser.add_argument("--out", default=env_default("GLMASR_BENCH_OUT", "bench_out/librispeech_2611"))
    parser.add_argument("--dataset", default=env_default("GLMASR_BENCH_DATASET", "hf-audio/esb-datasets-test-only-sorted"))
    parser.add_argument("--dataset-config", default=env_default("GLMASR_BENCH_DATASET_CONFIG", "librispeech"))
    parser.add_argument("--split", default=env_default("GLMASR_BENCH_SPLIT", "test.clean"))
    parser.add_argument("--hf-cache", default=os.environ.get("HF_HOME", "/data/.cache/huggingface"))
    args = parser.parse_args()

    if args.download_models:
        download_models(Path(args.model), Path(args.mel))

    require_path(args.model, "dir")
    require_path(args.mel, "file")
    if not Path(args.engine).is_file():
        raise SystemExit(
            "ERROR: ./glm_asr_infer not found. Build it first:\n"
            "  cmake -S . -B build_cmake -DCUDA_ROOT=/usr/local/cuda-12.8 "
            '-DGLMASR_CUDA_ARCH=sm_120 -DGLMASR_PYTHON_EXECUTABLE="$(which python3)"\n'
            "  cmake --build build_cmake -j\n"
            "  cmake --build build_cmake --target cutedsl_fa2_bridge -j"
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==> Infer-engine 2611 mixed-length LibriSpeech benchmark")
    run_tee(
        [
            sys.executable,
            "tools/eval_batch_test.py",
            "--engine",
            args.engine,
            "--model",
            args.model,
            "--mel",
            args.mel,
            "--dataset",
            args.dataset,
            "--dataset-config",
            args.dataset_config,
            "--split",
            args.split,
            "--start",
            str(args.start),
            "--num",
            str(args.num),
            "--hf-cache",
            args.hf_cache,
        ],
        out_dir / "infer_librispeech_2611.log",
        env=infer_env(),
    )
    print(f"==> Benchmark log written to {out_dir / 'infer_librispeech_2611.log'}")


if __name__ == "__main__":
    main()
