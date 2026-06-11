#!/usr/bin/env python3
"""Run the pad30 Infer-engine vs vLLM comparison on the same HF samples."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT / "models" / "GLM-ASR-Nano-2512"
DEFAULT_MEL = ROOT / "resources" / "mel_filters.bin"
DEFAULT_MODEL_REPO = "zai-org/GLM-ASR-Nano-2512"


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


def post_json(url: str, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def transcribe_one(api_base: str, model: str, audio_path: Path, prompt: str, max_tokens: int, timeout: int) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio_url", "audio_url": {"url": f"file://{audio_path}"}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    t0 = time.perf_counter()
    out = post_json(f"{api_base.rstrip('/')}/chat/completions", payload, timeout)
    elapsed = time.perf_counter() - t0
    text = out["choices"][0]["message"].get("content", "")
    return {"path": str(audio_path), "text": text, "latency_s": elapsed, "ok": True}


def wait_for_vllm(port: int, timeout_s: int, log_path: Path) -> None:
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                resp.read()
            return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise SystemExit(f"ERROR: vLLM did not become ready. See {log_path}")


def terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=20)
    except Exception:
        proc.kill()
        proc.wait(timeout=20)


def infer_env() -> dict[str, str]:
    env = os.environ.copy()
    defaults = {
        "GLMASR_VRAM_UTIL": "0.9",
        "GLMASR_MAX_SEQS": "256",
        "GLMASR_MAX_NEW_TOKENS": "1",
        "GLMASR_MAX_BATCHED_TOKENS": "10000",
        "GLMASR_ENCODER_FA_PIPELINED": "1",
        "GLMASR_ENCODER_FA_VLLM": "1",
        "GLMASR_ENCODER_FA_LONG_MIN_SEQ": "1",
        "GLMASR_ENCODER_FA_MIN_SEQ": "1",
    }
    for key, value in defaults.items():
        env.setdefault(key, value)
    return env


def reexec_if_needed_for_offline_vllm(args: argparse.Namespace) -> None:
    if args.skip_vllm or args.vllm_backend != "offline":
        return
    if os.environ.get("GLMASR_VLLM_ENV_CLEAN") == "1":
        return
    if not os.environ.get("LD_LIBRARY_PATH") and not os.environ.get("PYTHONPATH"):
        return

    clean_env = os.environ.copy()
    clean_env.pop("LD_LIBRARY_PATH", None)
    clean_env.pop("PYTHONPATH", None)
    clean_env["GLMASR_VLLM_ENV_CLEAN"] = "1"
    print("==> Re-execing with clean LD_LIBRARY_PATH/PYTHONPATH for vLLM offline benchmark", flush=True)
    os.execve(sys.executable, [sys.executable, *sys.argv], clean_env)


def run_vllm_client(args: argparse.Namespace, audio_dir: Path, out_dir: Path) -> dict:
    files = sorted(audio_dir.glob("pad30_*.flac"))[: args.num]
    if len(files) < args.num:
        raise SystemExit(f"ERROR: expected {args.num} pad30_*.flac files in {audio_dir}, got {len(files)}")

    print("=" * 70)
    print("vLLM pad30 submission")
    print("=" * 70)
    print(f"API base:    http://127.0.0.1:{args.vllm_port}/v1")
    print(f"Audio dir:   {audio_dir}")
    print(f"Files:       {len(files)}")
    print(f"Workers:     {args.workers}")
    print(f"max_tokens:  {args.max_tokens}")
    print(f"Prompt:      {args.prompt!r}")
    print()

    results = []
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [
            pool.submit(
                transcribe_one,
                f"http://127.0.0.1:{args.vllm_port}/v1",
                "glm-asr",
                path,
                args.prompt,
                args.max_tokens,
                args.timeout,
            )
            for path in files
        ]
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            try:
                results.append(fut.result())
            except (urllib.error.URLError, TimeoutError, Exception) as exc:
                results.append({"ok": False, "error": repr(exc), "latency_s": 0.0})
            if i % 10 == 0 or i == len(futs):
                ok = sum(1 for r in results if r.get("ok"))
                print(f"  progress: {i}/{len(futs)} done | ok={ok}")
    wall_s = time.perf_counter() - t0

    ok_results = [r for r in results if r.get("ok")]
    latencies = sorted(r["latency_s"] for r in ok_results)
    total_audio_s = len(files) * 30.0
    summary = {
        "engine": "vllm",
        "success": len(ok_results),
        "failed": len(results) - len(ok_results),
        "total_audio_s": total_audio_s,
        "wall_time_s": wall_s,
        "rtfx_wall": total_audio_s / wall_s if wall_s > 0 else 0.0,
        "latency_mean_s": sum(latencies) / len(latencies) if latencies else 0.0,
        "latency_p50_s": latencies[len(latencies) // 2] if latencies else 0.0,
        "latency_p95_s": latencies[int(len(latencies) * 0.95) - 1] if latencies else 0.0,
        "results": results,
    }

    print("\n" + "=" * 70)
    print("vLLM Results")
    print("=" * 70)
    print(f"Successful:      {summary['success']}/{len(results)}")
    print(f"Total audio:     {total_audio_s:.1f}s")
    print(f"Wall time:       {wall_s:.3f}s")
    print(f"RTFx wall:       {summary['rtfx_wall']:.2f}x")
    print(f"Latency mean:    {summary['latency_mean_s']:.3f}s")
    print(f"Latency p50/p95: {summary['latency_p50_s']:.3f}s / {summary['latency_p95_s']:.3f}s")
    for i, r in enumerate(ok_results[:10]):
        print(f"[{i:02d}] {os.path.basename(r['path'])}: {r['text'][:80]}")

    output = out_dir / "vllm_pad30.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved: {output}")
    return summary


def run_vllm_offline(args: argparse.Namespace, audio_dir: Path, out_dir: Path) -> dict:
    requested = args.num + max(args.warmup, 0)
    all_files = sorted(audio_dir.glob("pad30_*.flac"))[:requested]
    if len(all_files) < requested:
        raise SystemExit(f"ERROR: expected {requested} pad30_*.flac files in {audio_dir}, got {len(all_files)}")
    warmup_files = all_files[: args.warmup] if args.warmup > 0 else []
    files = all_files[args.warmup : args.warmup + args.num]

    os.environ.pop("LD_LIBRARY_PATH", None)
    os.environ.pop("PYTHONPATH", None)

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit("ERROR: vllm is not installed. Run `pip install -r requirements.txt`.") from exc

    print("=" * 70)
    print("vLLM offline pad30 benchmark")
    print("=" * 70)
    print(f"Model:       {args.model}")
    print(f"Audio dir:   {audio_dir}")
    print(f"Files:       {len(files)} timed")
    if warmup_files:
        print(f"Warmup:      {len(warmup_files)} separate samples")
    print(f"max_tokens:  {args.max_tokens}")
    print(f"Prompt:      {args.prompt!r}")
    print()

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=float(args.vllm_gpu_memory_utilization),
        max_num_seqs=args.vllm_max_num_seqs,
        max_num_batched_tokens=args.vllm_max_batched_tokens,
        max_model_len=args.vllm_max_model_len,
        allowed_local_media_path=str(audio_dir),
        enable_prefix_caching=False,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
    )
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    {"type": "audio_url", "audio_url": {"url": f"file://{path}"}},
                ],
            }
        ]
        for path in files
    ]

    if warmup_files:
        warmup_messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": args.prompt},
                        {"type": "audio_url", "audio_url": {"url": f"file://{path}"}},
                    ],
                }
            ]
            for path in warmup_files
        ]
        print(f"WARMUP: {len(warmup_messages)} vLLM offline samples")
        llm.chat(
            warmup_messages,
            sampling_params=sampling,
            use_tqdm=False,
            chat_template_content_format="openai",
        )

    t0 = time.perf_counter()
    outputs = llm.chat(
        messages,
        sampling_params=sampling,
        use_tqdm=False,
        chat_template_content_format="openai",
    )
    wall_s = time.perf_counter() - t0

    results = []
    for path, out in zip(files, outputs, strict=True):
        text = out.outputs[0].text if out.outputs else ""
        results.append({"path": str(path), "text": text, "ok": bool(text)})

    ok_results = [r for r in results if r.get("ok")]
    total_audio_s = len(files) * 30.0
    summary = {
        "engine": "vllm-offline",
        "success": len(ok_results),
        "failed": len(results) - len(ok_results),
        "total_audio_s": total_audio_s,
        "wall_time_s": wall_s,
        "rtfx_wall": total_audio_s / wall_s if wall_s > 0 else 0.0,
        "results": results,
    }

    print("\n" + "=" * 70)
    print("vLLM Offline Results")
    print("=" * 70)
    print(f"Successful:  {summary['success']}/{len(results)}")
    print(f"Total audio: {total_audio_s:.1f}s")
    print(f"Wall time:   {wall_s:.3f}s")
    print(f"RTFx wall:   {summary['rtfx_wall']:.2f}x")
    for i, r in enumerate(ok_results[:10]):
        print(f"[{i:02d}] {os.path.basename(r['path'])}: {r['text'][:80]}")

    output = out_dir / "vllm_pad30_offline.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved: {output}")
    return summary


def start_vllm(args: argparse.Namespace, audio_dir: Path, out_dir: Path) -> subprocess.Popen:
    if shutil.which("vllm") is None:
        raise SystemExit("ERROR: vllm is not in PATH. Run `pip install -r requirements.txt` first.")

    log_path = out_dir / "vllm_server.log"
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("PYTHONPATH", None)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    cmd = [
        "vllm",
        "serve",
        args.model,
        "--served-model-name",
        "glm-asr",
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        args.vllm_gpu_memory_utilization,
        "--max-num-seqs",
        str(args.vllm_max_num_seqs),
        "--max-num-batched-tokens",
        str(args.vllm_max_batched_tokens),
        "--max-model-len",
        str(args.vllm_max_model_len),
        "--allowed-local-media-path",
        str(audio_dir),
        "--no-use-tqdm-on-load",
        "--no-enable-prefix-caching",
        "--port",
        str(args.vllm_port),
    ]
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    print(f"==> Waiting for vLLM on port {args.vllm_port}")
    wait_for_vllm(args.vllm_port, args.vllm_startup_timeout, log_path)
    return proc


def main() -> None:
    parser = argparse.ArgumentParser(description="Pad30 Infer-engine vs vLLM benchmark")
    parser.add_argument("--engine", default="./glm_asr_infer")
    parser.add_argument("--model", default=os.environ.get("MODEL_BF16", str(DEFAULT_MODEL_DIR)))
    parser.add_argument("--mel", default=os.environ.get("MEL_FILTERS", str(DEFAULT_MEL)))
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--num", type=int, default=int(env_default("GLMASR_BENCH_PAD30_NUM", "256")))
    parser.add_argument("--start", type=int, default=int(env_default("GLMASR_BENCH_PAD30_START", "0")))
    parser.add_argument("--ie-audio-dir", default=env_default("GLMASR_BENCH_PAD30_DIR", "/tmp/ie_pad30"))
    parser.add_argument("--vllm-audio-dir", default=env_default("GLMASR_BENCH_VLLM_PAD30_DIR", "/tmp/vllm_pad30"))
    parser.add_argument("--out", default=env_default("GLMASR_BENCH_OUT", "bench_out/pad30"))
    parser.add_argument("--dataset", default=env_default("GLMASR_BENCH_DATASET", "hf-audio/esb-datasets-test-only-sorted"))
    parser.add_argument("--dataset-config", default=env_default("GLMASR_BENCH_DATASET_CONFIG", "librispeech"))
    parser.add_argument("--split", default=env_default("GLMASR_BENCH_SPLIT", "test.clean"))
    parser.add_argument("--hf-cache", default=os.environ.get("HF_HOME", "/data/.cache/huggingface"))
    parser.add_argument("--prepare-force", action="store_true")
    parser.add_argument("--skip-infer", action="store_true", help="Do not run Infer-engine side")
    parser.add_argument("--skip-vllm", action="store_true", help="Do not run vLLM side")
    parser.add_argument("--vllm-backend", choices=("offline", "server"), default=env_default("VLLM_BENCH_BACKEND", "offline"))
    parser.add_argument("--keep-vllm-server", action="store_true", help="Assume vLLM is already running and do not start/stop it")
    parser.add_argument("--vllm-port", type=int, default=int(env_default("VLLM_PORT", "8000")))
    parser.add_argument("--vllm-max-batched-tokens", type=int, default=int(env_default("VLLM_MAX_BATCHED_TOKENS", "6000")))
    parser.add_argument("--vllm-max-num-seqs", type=int, default=int(env_default("VLLM_MAX_NUM_SEQS", "256")))
    parser.add_argument("--vllm-max-model-len", type=int, default=int(env_default("VLLM_MAX_MODEL_LEN", "2048")))
    parser.add_argument("--vllm-gpu-memory-utilization", default=env_default("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))
    parser.add_argument("--vllm-startup-timeout", type=int, default=180)
    parser.add_argument("--workers", type=int, default=int(env_default("VLLM_WORKERS", "256")))
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=int(env_default("GLMASR_BENCH_WARMUP", "16")))
    parser.add_argument("--prompt", default="Please transcribe")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    reexec_if_needed_for_offline_vllm(args)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ie_dir = Path(args.ie_audio_dir)
    vllm_dir = Path(args.vllm_audio_dir)

    if args.download_models:
        download_models(Path(args.model), Path(args.mel))

    if not Path(args.model).is_dir():
        raise SystemExit(f"ERROR: model directory not found: {args.model}")
    if not Path(args.mel).is_file():
        raise SystemExit(f"ERROR: mel filters file not found: {args.mel}")
    if not args.skip_infer and not Path(args.engine).is_file():
        raise SystemExit(f"ERROR: engine binary not found: {args.engine}")

    if not args.skip_infer:
        print("==> Infer-engine pad30 benchmark")
        run_tee(
            [
                sys.executable,
                "tools/eval_batch_pad30.py",
                "--engine",
                args.engine,
                "--model",
                args.model,
                "--mel",
                args.mel,
                "--num",
                str(args.num),
                "--start",
                str(args.start),
                "--audio-dir",
                str(ie_dir),
                "--prepare-from-hf",
                "--prepare-vllm-dir",
                str(vllm_dir),
                "--dataset",
                args.dataset,
                "--dataset-config",
                args.dataset_config,
                "--split",
                args.split,
                "--hf-cache",
                args.hf_cache,
                *(["--warmup", str(args.warmup)] if args.warmup > 0 else []),
                *(["--prepare-force"] if args.prepare_force else []),
            ],
            out_dir / "infer_pad30.log",
            env=infer_env(),
        )
    else:
        print("==> Preparing pad30 FLAC files for vLLM")
        run_tee(
            [
                sys.executable,
                "tools/eval_batch_pad30.py",
                "--engine",
                args.engine,
                "--model",
                args.model,
                "--mel",
                args.mel,
                "--num",
                str(args.num),
                "--start",
                str(args.start),
                "--audio-dir",
                str(ie_dir),
                "--prepare-from-hf",
                "--prepare-vllm-dir",
                str(vllm_dir),
                "--dataset",
                args.dataset,
                "--dataset-config",
                args.dataset_config,
                "--split",
                args.split,
                "--hf-cache",
                args.hf_cache,
                "--prepare-only",
                *(["--warmup", str(args.warmup)] if args.warmup > 0 else []),
                *(["--prepare-force"] if args.prepare_force else []),
            ],
            out_dir / "prepare_pad30.log",
            env=infer_env(),
        )

    if args.skip_vllm:
        print(f"==> Logs written to {out_dir}")
        return

    proc = None
    try:
        if args.vllm_backend == "offline":
            if args.keep_vllm_server:
                raise SystemExit("ERROR: --keep-vllm-server only applies to --vllm-backend server")
            run_vllm_offline(args, vllm_dir, out_dir)
        else:
            if args.keep_vllm_server:
                wait_for_vllm(args.vllm_port, 5, out_dir / "vllm_server.log")
            else:
                print("==> Starting vLLM server")
                proc = start_vllm(args, vllm_dir, out_dir)
            run_vllm_client(args, vllm_dir, out_dir)
    finally:
        if proc is not None:
            terminate_process(proc)

    print(f"==> Logs written to {out_dir}")


if __name__ == "__main__":
    main()
