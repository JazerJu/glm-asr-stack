#!/usr/bin/env python3
"""ForceAligner evaluation across Buckeye, LibriSpeech, AISHELL-1."""

from __future__ import annotations

import argparse
import getpass
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
import types
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


VENV_PYTHON = "/data/fwsr/glm-asr/GLM-ASR/venv/bin/python"
QWEN_ASR_ROOT = "/data/ASR模型/Qwen3-ASR"
DEFAULT_MODEL_PATH = "/data/.cache/huggingface/hub/models--Qwen--Qwen3-ForcedAligner-0.6B/snapshots/c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
DEFAULT_ENGINE_BIN = "/data/fwsr/glm-asr/asr-aligner-engine/force_aligner"
DEFAULT_ENGINE_DIR = "/data/fwsr/glm-asr/asr-aligner-engine"
DEFAULT_BUCKETS = (
    ("<10s", 0.0, 10.0),
    ("10-20s", 10.0, 20.0),
    ("20-30s", 20.0, 30.0),
    ("30s+", 30.0, math.inf),
)
FILLER_WORDS = {
    "uh", "um", "uhh", "umm", "er", "erm", "ah", "eh", "mm", "mhm",
    "huh", "hm", "oh", "like", "youknow", "i mean", "mmhm",
}
NORMALIZED_FILLER_WORDS = {
    re.sub(r"[^0-9a-z\u4e00-\u9fff']+", "", item.strip().lower().replace("’", "'"))
    for item in FILLER_WORDS
}


def maybe_reexec_into_venv() -> None:
    hidden_flag = "--_already_in_glm_asr_venv"
    if hidden_flag in sys.argv:
        sys.argv.remove(hidden_flag)
        return
    if not os.path.exists(VENV_PYTHON):
        return
    try:
        current = os.path.realpath(sys.executable)
        target = os.path.realpath(VENV_PYTHON)
    except OSError:
        return
    if current == target:
        return
    os.execv(VENV_PYTHON, [VENV_PYTHON, __file__, hidden_flag, *sys.argv[1:]])


maybe_reexec_into_venv()

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm


def load_pure_torch_aligner_class() -> type:
    package_root = Path(QWEN_ASR_ROOT) / "qwen_asr"
    pure_torch_root = package_root / "pure_torch"
    module_path = (pure_torch_root / "inference.py").resolve()
    trusted_root = Path(QWEN_ASR_ROOT).resolve()
    current_user = getpass.getuser()
    if trusted_root.is_symlink() or module_path.is_symlink():
        raise RuntimeError("Refusing to load qwen_asr pure_torch module from symlinked path")
    if trusted_root not in module_path.parents:
        raise RuntimeError(f"Module path escapes trusted QWEN_ASR_ROOT: {module_path}")
    if not module_path.exists():
        raise FileNotFoundError(f"PureTorch aligner module not found: {module_path}")
    module_owner = module_path.owner()
    if module_owner != current_user:
        raise RuntimeError(
            f"Refusing to execute untrusted module owned by {module_owner!r}; expected current user {current_user!r}"
        )
    if "qwen_asr" not in sys.modules:
        pkg = types.ModuleType("qwen_asr")
        pkg.__path__ = [str(package_root)]
        sys.modules["qwen_asr"] = pkg
    if "qwen_asr.pure_torch" not in sys.modules:
        subpkg = types.ModuleType("qwen_asr.pure_torch")
        subpkg.__path__ = [str(pure_torch_root)]
        sys.modules["qwen_asr.pure_torch"] = subpkg

    module_name = "qwen_asr.pure_torch.inference"
    if module_name in sys.modules:
        return sys.modules[module_name].PureTorchForcedAligner

    spec = spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load qwen_asr.pure_torch.inference")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.PureTorchForcedAligner


PureTorchForcedAligner = load_pure_torch_aligner_class()


WORD_LINE_RE = re.compile(r"^(?P<word>.+?)\t(?P<start>\d+(?:\.\d+)?)\t(?P<end>\d+(?:\.\d+)?)(?:\t.*)?$")
BENCH_RE = re.compile(
    r"\[bench\]\s+Total:\s+(?P<total>\d+(?:\.\d+)?)\s+ms\s+Mel:\s+(?P<mel>\d+(?:\.\d+)?)\s+ms\s+Encoder:\s+(?P<encoder>\d+(?:\.\d+)?)\s+ms\s+Decoder:\s+(?P<decoder>\d+(?:\.\d+)?)\s+ms"
)
SEQ_LEN_RE = re.compile(r"\[align\]\s+Sequence length:\s+(?P<seq_len>\d+)")
MEL_FRAMES_RE = re.compile(r"\[align\]\s+Computing mel spectrogram:\s+(?P<mel_frames>\d+)\s+frames")
TOKEN_COUNT_RE = re.compile(r"\[align\]\s+Audio tokens:\s+(?P<audio_tokens>\d+)")
DAEMON_END_MARKER = "[daemon] END_REQUEST"


@dataclass
class WordAlignment:
    text: str
    start_ms: int
    end_ms: int


@dataclass
class Sample:
    dataset: str
    sample_id: str
    audio_path: str
    text: str
    language: str
    duration_s: float
    ground_truth: Optional[List[WordAlignment]] = None


@dataclass
class CompareSummary:
    matched_words: int = 0
    ref_words: int = 0
    test_words: int = 0
    mismatched_words: int = 0
    skipped_ref_words: int = 0
    skipped_test_words: int = 0
    exact_word_matches: int = 0
    exact_start_matches: int = 0
    exact_end_matches: int = 0
    start_abs_errors: Optional[List[float]] = None
    end_abs_errors: Optional[List[float]] = None
    start_signed_errors: Optional[List[float]] = None
    end_signed_errors: Optional[List[float]] = None

    def __post_init__(self) -> None:
        self.start_abs_errors = self.start_abs_errors or []
        self.end_abs_errors = self.end_abs_errors or []
        self.start_signed_errors = self.start_signed_errors or []
        self.end_signed_errors = self.end_signed_errors or []


def duration_bucket(duration_s: float) -> str:
    for name, lower, upper in DEFAULT_BUCKETS:
        if lower <= duration_s < upper:
            return name
    return DEFAULT_BUCKETS[-1][0]


def normalize_word(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("’", "'")
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff']+", "", text)
    text = re.sub(r"^'+|'+$", "", text)
    return text


def is_content_word(text: str) -> bool:
    norm = normalize_word(text)
    if not norm:
        return False
    return norm not in NORMALIZED_FILLER_WORDS


def resolve_within_root(path_value: str, root: Path) -> Path:
    root_resolved = root.resolve()
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = root_resolved / candidate
    candidate = candidate.resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError(f"Path escapes dataset root: {candidate}")
    return candidate


def summarize(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def build_aligner(model_path: str, device: str) -> PureTorchForcedAligner:
    return PureTorchForcedAligner(model_path=model_path, device=device, dtype=torch.bfloat16)


def load_buckeye(dataset_root: str) -> List[Sample]:
    dataset_root_path = Path(dataset_root).resolve()
    manifest_path = dataset_root_path / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    samples: List[Sample] = []
    for row in manifest["samples"]:
        gt = [
            WordAlignment(
                text=word["word"],
                start_ms=int(round(float(word["start_ms"]))),
                end_ms=int(round(float(word["end_ms"]))),
            )
            for word in row.get("words", [])
        ]
        samples.append(
            Sample(
                dataset="buckeye",
                sample_id=row["id"],
                audio_path=str(resolve_within_root(row["audio"], dataset_root_path)),
                text=row["transcript"],
                language="English",
                duration_s=float(row["duration_s"]),
                ground_truth=gt,
            )
        )
    return samples


def load_librispeech(dataset_root: str) -> List[Sample]:
    dataset_root_path = Path(dataset_root).resolve()
    test_clean = dataset_root_path / "LibriSpeech" / "test-clean"
    samples: List[Sample] = []
    for trans_path in sorted(test_clean.glob("*/*/*.trans.txt")):
        with trans_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                utt_id, text = parts
                audio_path = resolve_within_root(f"LibriSpeech/test-clean/{trans_path.parent.relative_to(test_clean)}/{utt_id}.flac", dataset_root_path)
                if not audio_path.exists():
                    continue
                info = sf.info(str(audio_path))
                samples.append(
                    Sample(
                        dataset="librispeech",
                        sample_id=utt_id,
                        audio_path=str(audio_path),
                        text=text.lower(),
                        language="English",
                        duration_s=float(info.duration),
                    )
                )
    return samples


def load_aishell1(dataset_root: str) -> List[Sample]:
    dataset_root_path = Path(dataset_root).resolve()
    transcript_path = dataset_root_path / "transcript" / "aishell_transcript_v0.8.txt"
    transcript_map: Dict[str, str] = {}
    with transcript_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            transcript_map[parts[0]] = "".join(parts[1:])

    train_root = dataset_root_path / "train"
    samples: List[Sample] = []
    for audio_path in sorted(train_root.glob("S*/*.wav")):
        utt_id = audio_path.stem
        text = transcript_map.get(utt_id)
        if not text:
            continue
        info = sf.info(str(audio_path))
        samples.append(
            Sample(
                dataset="aishell1",
                sample_id=utt_id,
                audio_path=str(resolve_within_root(str(audio_path), dataset_root_path)),
                text=text,
                language="Chinese",
                duration_s=float(info.duration),
            )
        )
    return samples


def load_samples(dataset: str, root: str) -> List[Sample]:
    if dataset == "buckeye":
        return load_buckeye(root)
    if dataset == "librispeech":
        return load_librispeech(root)
    if dataset == "aishell1":
        return load_aishell1(root)
    raise ValueError(f"Unsupported dataset: {dataset}")


def maybe_prepare_wav(audio_path: str) -> Tuple[str, Optional[str]]:
    suffix = Path(audio_path).suffix.lower()
    if suffix == ".wav":
        return audio_path, None
    samples, sample_rate = sf.read(audio_path, dtype="float32")
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_path = temp.name
    temp.close()
    try:
        sf.write(temp_path, samples, sample_rate, subtype="PCM_16")
    except Exception:  # noqa: BLE001
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise
    return temp_path, temp_path


def aggregate_token_alignments(token_alignments: Sequence[WordAlignment]) -> List[WordAlignment]:
    """Merge BPE token-level timestamps into word-level, matching C++ aligner logic."""
    if not token_alignments:
        return []
    has_gh_prefix = any(aln.text.startswith(" ") for aln in token_alignments[1:])
    words: List[WordAlignment] = []
    buf = ""
    word_start_ms = 0

    for i, token in enumerate(token_alignments):
        starts_word = has_gh_prefix and token.text.startswith(" ")
        piece = token.text[1:] if starts_word else token.text

        flush = (starts_word and buf) if has_gh_prefix else (i > 0 and buf)
        if flush:
            word_end_ms = token_alignments[i - 1].end_ms
            words.append(WordAlignment(buf, word_start_ms, word_end_ms))
            buf = ""

        if not buf:
            word_start_ms = 0 if i == 0 else token_alignments[i - 1].end_ms

        buf += piece

    if buf:
        word_end_ms = token_alignments[-1].end_ms
        words.append(WordAlignment(buf, word_start_ms, word_end_ms))
    return words


def run_python_alignment(aligner: PureTorchForcedAligner, audio_path: str, text: str, language: str) -> Tuple[List[WordAlignment], float]:
    started = time.perf_counter()
    output = aligner.align(audio_path, text, language=language)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    token_alignments = [
        WordAlignment(
            text=item.text,
            start_ms=int(item.start_time_ms),
            end_ms=int(item.end_time_ms),
        )
        for item in output.alignments
    ]
    return aggregate_token_alignments(token_alignments), elapsed_ms


def parse_engine_output(stdout: str, stderr: str) -> Tuple[List[WordAlignment], Dict[str, Optional[float]]]:
    alignments: List[WordAlignment] = []
    combined = f"{stdout}\n{stderr}"
    for line in stdout.splitlines():
        match = WORD_LINE_RE.match(line.strip())
        if not match:
            continue
        alignments.append(
            WordAlignment(
                text=match.group("word"),
                start_ms=int(round(float(match.group("start")) * 1000.0)),
                end_ms=int(round(float(match.group("end")) * 1000.0)),
            )
        )

    bench_match = BENCH_RE.search(combined)
    seq_match = SEQ_LEN_RE.search(combined)
    mel_match = MEL_FRAMES_RE.search(combined)
    token_match = TOKEN_COUNT_RE.search(combined)
    bench = {
        "total_ms": float(bench_match.group("total")) if bench_match else None,
        "mel_ms": float(bench_match.group("mel")) if bench_match else None,
        "encoder_ms": float(bench_match.group("encoder")) if bench_match else None,
        "decoder_ms": float(bench_match.group("decoder")) if bench_match else None,
        "seq_len": int(seq_match.group("seq_len")) if seq_match else None,
        "mel_frames": int(mel_match.group("mel_frames")) if mel_match else None,
        "audio_tokens": int(token_match.group("audio_tokens")) if token_match else None,
    }
    return alignments, bench


class CppDaemon:
    def __init__(self, engine_bin: str, engine_dir: str, model_path: str, timeout_s: int) -> None:
        self.engine_bin = engine_bin
        self.engine_dir = engine_dir
        self.model_path = model_path
        self.timeout_s = timeout_s
        self.proc = self._start_proc()

    def _read_stdout_response(self) -> str:
        assert self.proc.stdout is not None
        lines: List[str] = []
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError("C++ daemon closed stdout unexpectedly")
            if line in {"\n", "\r\n"}:
                return "".join(lines)
            lines.append(line)

    def _read_stderr_response(self) -> str:
        assert self.proc.stderr is not None
        lines: List[str] = []
        while True:
            line = self.proc.stderr.readline()
            if line == "":
                if self.proc.poll() is not None:
                    raise RuntimeError("C++ daemon closed stderr unexpectedly")
                continue
            if line.rstrip("\r\n") == DAEMON_END_MARKER:
                return "".join(lines)
            lines.append(line)

    def _start_proc(self) -> subprocess.Popen:
        return subprocess.Popen(
            [self.engine_bin, "--daemon", "--model", self.model_path],
            cwd=self.engine_dir,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )

    def restart(self) -> None:
        try:
            self.close()
        except Exception:
            pass
        self.proc = self._start_proc()

    def run(self, audio_path: str, text: str, language: str) -> Tuple[List[WordAlignment], Dict[str, Optional[float]], float]:
        if self.proc.poll() is not None:
            self.restart()
        assert self.proc.stdin is not None

        started = time.perf_counter()
        try:
            self.proc.stdin.write(f"{audio_path}\t{text}\t{language}\n")
            self.proc.stdin.flush()
            stdout = self._read_stdout_response()
            stderr = self._read_stderr_response()
        except (BrokenPipeError, OSError):
            self.restart()
            self.proc.stdin.write(f"{audio_path}\t{text}\t{language}\n")
            self.proc.stdin.flush()
            stdout = self._read_stdout_response()
            stderr = self._read_stderr_response()
        wall_ms = (time.perf_counter() - started) * 1000.0

        alignments, bench = parse_engine_output(stdout, stderr)
        if not alignments:
            raise RuntimeError(f"No C++ alignment rows parsed for {audio_path}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
        return alignments, bench, wall_ms

    def close(self) -> None:
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                try:
                    self.proc.stdin.write("\n")
                    self.proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
                try:
                    self.proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)


def run_cpp_alignment(engine_bin: str, engine_dir: str, model_path: str, audio_path: str, text: str, language: str, timeout_s: int = 30) -> Tuple[List[WordAlignment], Dict[str, Optional[float]], float]:
    started = time.perf_counter()
    proc = subprocess.run(
        [engine_bin, "--model", model_path, "--audio", audio_path, "--text", text, "--lang", language],
        cwd=engine_dir,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    wall_ms = (time.perf_counter() - started) * 1000.0
    if proc.returncode != 0:
        raise RuntimeError(f"C++ aligner failed (code={proc.returncode}) for {audio_path}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    alignments, bench = parse_engine_output(proc.stdout, proc.stderr)
    if not alignments:
        raise RuntimeError(f"No C++ alignment rows parsed for {audio_path}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return alignments, bench, wall_ms


def compare_alignments(reference: Sequence[WordAlignment], candidate: Sequence[WordAlignment], content_only: bool = False) -> CompareSummary:
    ref_items = [item for item in reference if not content_only or is_content_word(item.text)]
    cand_items = [item for item in candidate if not content_only or is_content_word(item.text)]
    ref_items = [item for item in ref_items if normalize_word(item.text)]
    cand_items = [item for item in cand_items if normalize_word(item.text)]

    summary = CompareSummary(ref_words=len(ref_items), test_words=len(cand_items))
    summary.matched_words = min(len(ref_items), len(cand_items))
    summary.skipped_ref_words = max(0, len(ref_items) - summary.matched_words)
    summary.skipped_test_words = max(0, len(cand_items) - summary.matched_words)
    for ref_item, cand_item in zip(ref_items[: summary.matched_words], cand_items[: summary.matched_words]):
        start_delta = cand_item.start_ms - ref_item.start_ms
        end_delta = cand_item.end_ms - ref_item.end_ms
        words_match = normalize_word(ref_item.text) == normalize_word(cand_item.text)
        summary.exact_word_matches += int(words_match)
        summary.mismatched_words += int(not words_match)
        summary.exact_start_matches += int(start_delta == 0)
        summary.exact_end_matches += int(end_delta == 0)
        summary.start_abs_errors.append(abs(start_delta))
        summary.end_abs_errors.append(abs(end_delta))
        summary.start_signed_errors.append(float(start_delta))
        summary.end_signed_errors.append(float(end_delta))
    return summary


def add_summary(target: Dict[str, object], summary: CompareSummary, prefix: str) -> None:
    target[f"{prefix}_matched_words"] += summary.matched_words
    target[f"{prefix}_ref_words"] += summary.ref_words
    target[f"{prefix}_test_words"] += summary.test_words
    target[f"{prefix}_mismatched_words"] += summary.mismatched_words
    target[f"{prefix}_skipped_ref_words"] += summary.skipped_ref_words
    target[f"{prefix}_skipped_test_words"] += summary.skipped_test_words
    target[f"{prefix}_exact_word_matches"] += summary.exact_word_matches
    target[f"{prefix}_exact_start_matches"] += summary.exact_start_matches
    target[f"{prefix}_exact_end_matches"] += summary.exact_end_matches
    target[f"{prefix}_start_abs_errors"].extend(summary.start_abs_errors)
    target[f"{prefix}_end_abs_errors"].extend(summary.end_abs_errors)
    target[f"{prefix}_start_signed_errors"].extend(summary.start_signed_errors)
    target[f"{prefix}_end_signed_errors"].extend(summary.end_signed_errors)


def make_bucket_accumulator() -> Dict[str, object]:
    return {
        "samples": 0,
        "failures": 0,
        "python_wall_ms": [],
        "cpp_wall_ms": [],
        "encoder_ms": [],
        "seq_len": [],
        "duration_s": [],
        "py_cpp_matched_words": 0,
        "py_cpp_ref_words": 0,
        "py_cpp_test_words": 0,
        "py_cpp_mismatched_words": 0,
        "py_cpp_skipped_ref_words": 0,
        "py_cpp_skipped_test_words": 0,
        "py_cpp_exact_word_matches": 0,
        "py_cpp_exact_start_matches": 0,
        "py_cpp_exact_end_matches": 0,
        "py_cpp_start_abs_errors": [],
        "py_cpp_end_abs_errors": [],
        "py_cpp_start_signed_errors": [],
        "py_cpp_end_signed_errors": [],
        "gt_cpp_matched_words": 0,
        "gt_cpp_ref_words": 0,
        "gt_cpp_test_words": 0,
        "gt_cpp_mismatched_words": 0,
        "gt_cpp_skipped_ref_words": 0,
        "gt_cpp_skipped_test_words": 0,
        "gt_cpp_exact_word_matches": 0,
        "gt_cpp_exact_start_matches": 0,
        "gt_cpp_exact_end_matches": 0,
        "gt_cpp_start_abs_errors": [],
        "gt_cpp_end_abs_errors": [],
        "gt_cpp_start_signed_errors": [],
        "gt_cpp_end_signed_errors": [],
        "encoder_scaling": [],
    }


def finalize_bucket_stats(raw: Dict[str, object]) -> Dict[str, object]:
    return {
        "samples": raw["samples"],
        "failures": raw["failures"],
        "duration_s": summarize(raw["duration_s"]),
        "python_wall_ms": summarize(raw["python_wall_ms"]),
        "cpp_wall_ms": summarize(raw["cpp_wall_ms"]),
        "encoder_ms": summarize(raw["encoder_ms"]),
        "seq_len": summarize(raw["seq_len"]),
        "py_cpp": {
            "matched_words": raw["py_cpp_matched_words"],
            "ref_words": raw["py_cpp_ref_words"],
            "test_words": raw["py_cpp_test_words"],
            "mismatched_words": raw["py_cpp_mismatched_words"],
            "skipped_ref_words": raw["py_cpp_skipped_ref_words"],
            "skipped_test_words": raw["py_cpp_skipped_test_words"],
            "exact_word_matches": raw["py_cpp_exact_word_matches"],
            "exact_start_matches": raw["py_cpp_exact_start_matches"],
            "exact_end_matches": raw["py_cpp_exact_end_matches"],
            "start_abs_ms": summarize(raw["py_cpp_start_abs_errors"]),
            "end_abs_ms": summarize(raw["py_cpp_end_abs_errors"]),
            "start_signed_ms": summarize(raw["py_cpp_start_signed_errors"]),
            "end_signed_ms": summarize(raw["py_cpp_end_signed_errors"]),
        },
        "gt_cpp": {
            "matched_words": raw["gt_cpp_matched_words"],
            "ref_words": raw["gt_cpp_ref_words"],
            "test_words": raw["gt_cpp_test_words"],
            "mismatched_words": raw["gt_cpp_mismatched_words"],
            "skipped_ref_words": raw["gt_cpp_skipped_ref_words"],
            "skipped_test_words": raw["gt_cpp_skipped_test_words"],
            "exact_word_matches": raw["gt_cpp_exact_word_matches"],
            "exact_start_matches": raw["gt_cpp_exact_start_matches"],
            "exact_end_matches": raw["gt_cpp_exact_end_matches"],
            "start_abs_ms": summarize(raw["gt_cpp_start_abs_errors"]),
            "end_abs_ms": summarize(raw["gt_cpp_end_abs_errors"]),
            "start_signed_ms": summarize(raw["gt_cpp_start_signed_errors"]),
            "end_signed_ms": summarize(raw["gt_cpp_end_signed_errors"]),
        },
        "encoder_scaling": raw["encoder_scaling"],
    }


def render_bucket_line(name: str, bucket_stats: Dict[str, object], include_gt: bool) -> str:
    py_cpp = bucket_stats["py_cpp"]
    line = (
        f"  {name:<7} samples={bucket_stats['samples']:<4} failures={bucket_stats['failures']:<3} "
        f"py_cpp_start_mae={format_metric(py_cpp['start_abs_ms']['mean'])}ms "
        f"py_cpp_end_mae={format_metric(py_cpp['end_abs_ms']['mean'])}ms "
        f"word_match={py_cpp['exact_word_matches']}/{py_cpp['matched_words']} "
        f"word_mismatch={py_cpp['mismatched_words']} skipped={py_cpp['skipped_ref_words']}/{py_cpp['skipped_test_words']} "
        f"exact_start={py_cpp['exact_start_matches']}/{py_cpp['matched_words']} "
        f"exact_end={py_cpp['exact_end_matches']}/{py_cpp['matched_words']}"
    )
    if include_gt:
        gt_cpp = bucket_stats["gt_cpp"]
        line += (
            f" gt_start_mae={format_metric(gt_cpp['start_abs_ms']['mean'])}ms"
            f" gt_end_mae={format_metric(gt_cpp['end_abs_ms']['mean'])}ms"
            f" gt_word_match={gt_cpp['exact_word_matches']}/{gt_cpp['matched_words']}"
        )
    return line


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def estimate_total_runtime_ms(total_samples: int, observed_wall_ms: Sequence[float]) -> Optional[float]:
    if total_samples <= 0 or not observed_wall_ms:
        return None
    return float(total_samples * statistics.mean(observed_wall_ms))


def render_runtime_estimate(ms: Optional[float]) -> str:
    if ms is None:
        return "n/a"
    total_s = ms / 1000.0
    if total_s < 60:
        return f"{total_s:.1f}s"
    if total_s < 3600:
        return f"{total_s / 60.0:.1f}m"
    return f"{total_s / 3600.0:.2f}h"


def evaluate_dataset(
    dataset: str,
    samples: Sequence[Sample],
    total_inventory_samples: int,
    aligner: PureTorchForcedAligner,
) -> Dict[str, object]:
    accumulators = defaultdict(make_bucket_accumulator)
    failures: List[Dict[str, str]] = []
    observed_total_wall_ms: List[float] = []

    for sample in tqdm(samples, desc=f"eval:{dataset}"):
        bucket = duration_bucket(sample.duration_s)
        slot = accumulators[bucket]
        slot["samples"] += 1
        slot["duration_s"].append(sample.duration_s)

        temp_wav: Optional[str] = None
        try:
            sample_started = time.perf_counter()
            engine_audio_path, temp_wav = maybe_prepare_wav(sample.audio_path)
            py_words, py_wall_ms = run_python_alignment(aligner, engine_audio_path, sample.text, sample.language)
            cpp_words, bench, cpp_wall_ms = run_cpp_alignment(
                engine_bin=DEFAULT_ENGINE_BIN,
                engine_dir=DEFAULT_ENGINE_DIR,
                model_path=DEFAULT_MODEL_PATH,
                audio_path=engine_audio_path,
                text=sample.text,
                language=sample.language,
            )
            observed_total_wall_ms.append((time.perf_counter() - sample_started) * 1000.0)
            slot["python_wall_ms"].append(py_wall_ms)
            slot["cpp_wall_ms"].append(cpp_wall_ms)
            if bench.get("encoder_ms") is not None:
                slot["encoder_ms"].append(bench["encoder_ms"])
            if bench.get("seq_len") is not None:
                slot["seq_len"].append(bench["seq_len"])
            slot["encoder_scaling"].append(
                {
                    "sample_id": sample.sample_id,
                    "duration_s": sample.duration_s,
                    "seq_len": bench.get("seq_len"),
                    "audio_tokens": bench.get("audio_tokens"),
                    "mel_frames": bench.get("mel_frames"),
                    "encoder_ms": bench.get("encoder_ms"),
                }
            )

            py_cpp = compare_alignments(py_words, cpp_words, content_only=False)
            add_summary(slot, py_cpp, "py_cpp")

            if dataset == "buckeye" and sample.ground_truth:
                gt_cpp = compare_alignments(sample.ground_truth, cpp_words, content_only=True)
                add_summary(slot, gt_cpp, "gt_cpp")
        except Exception as exc:  # noqa: BLE001
            slot["failures"] += 1
            failures.append({"sample_id": sample.sample_id, "error": str(exc)})
        finally:
            if temp_wav and os.path.exists(temp_wav):
                os.unlink(temp_wav)

    per_bucket = {bucket: finalize_bucket_stats(acc) for bucket, acc in accumulators.items()}
    return {
        "dataset": dataset,
        "inventory_samples": total_inventory_samples,
        "selected_samples": len(samples),
        "evaluated_samples": sum(bucket["samples"] for bucket in accumulators.values()),
        "failed_samples": len(failures),
        "duration_buckets": per_bucket,
        "failures": failures[:20],
        "estimated_full_runtime_ms": estimate_total_runtime_ms(total_inventory_samples, observed_total_wall_ms),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["buckeye", "librispeech", "aishell1"], choices=["buckeye", "librispeech", "aishell1"])
    parser.add_argument("--buckeye-root", default="/data/fwsr/glm-asr/data/buckeye")
    parser.add_argument("--librispeech-root", default="/data/fwsr/glm-asr/data/librispeech")
    parser.add_argument("--aishell1-root", default="/data/fwsr/glm-asr/data/aishell1")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--engine-bin", default=DEFAULT_ENGINE_BIN)
    parser.add_argument("--engine-dir", default=DEFAULT_ENGINE_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples per dataset for smoke tests.")
    parser.add_argument("--json-out", default=None, help="Optional path to write full JSON results.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_roots = {
        "buckeye": args.buckeye_root,
        "librispeech": args.librispeech_root,
        "aishell1": args.aishell1_root,
    }
    inventory: Dict[str, List[Sample]] = {}
    selected: Dict[str, List[Sample]] = {}
    for dataset in args.datasets:
        samples = load_samples(dataset, dataset_roots[dataset])
        inventory[dataset] = samples
        if args.max_samples is not None:
            samples = samples[: args.max_samples]
        selected[dataset] = samples

    aligner = build_aligner(args.model_path, args.device)
    results = {
        "model_path": args.model_path,
        "engine_bin": args.engine_bin,
        "engine_dir": args.engine_dir,
        "device": args.device,
        "datasets": {},
    }

    for dataset in args.datasets:
        results["datasets"][dataset] = evaluate_dataset(
            dataset=dataset,
            samples=selected[dataset],
            total_inventory_samples=len(inventory[dataset]),
            aligner=aligner,
        )

    print("\n=== ForceAligner evaluation summary ===")
    for dataset in args.datasets:
        dataset_result = results["datasets"][dataset]
        print(
            f"\n[{dataset}] inventory={dataset_result['inventory_samples']} selected={dataset_result['selected_samples']} "
            f"evaluated={dataset_result['evaluated_samples']} "
            f"failures={dataset_result['failed_samples']} estimated_full_runtime={render_runtime_estimate(dataset_result['estimated_full_runtime_ms'])}"
        )
        include_gt = dataset == "buckeye"
        for bucket_name in [name for name, _, _ in DEFAULT_BUCKETS if name in dataset_result["duration_buckets"]]:
            print(render_bucket_line(bucket_name, dataset_result["duration_buckets"][bucket_name], include_gt))
        scaling_rows = []
        for bucket_stats in dataset_result["duration_buckets"].values():
            scaling_rows.extend(bucket_stats["encoder_scaling"])
        scaling_rows = [row for row in scaling_rows if row.get("seq_len") is not None and row.get("encoder_ms") is not None]
        if scaling_rows:
            scaling_rows = sorted(scaling_rows, key=lambda row: (row["seq_len"], row["duration_s"], row["sample_id"]))
            print("  encoder_scaling(seq_len -> encoder_ms):")
            for row in scaling_rows[:10]:
                print(
                    f"    {row['sample_id']}: seq_len={row['seq_len']} encoder_ms={row['encoder_ms']:.2f} "
                    f"audio_tokens={row['audio_tokens']} duration_s={row['duration_s']:.2f}"
                )
        if dataset_result["failures"]:
            print("  first_failures:")
            for failure in dataset_result["failures"][:5]:
                print(f"    {failure['sample_id']}: {failure['error'].splitlines()[0]}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nWrote JSON report to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
