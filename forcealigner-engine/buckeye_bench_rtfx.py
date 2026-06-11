#!/usr/bin/env python3
"""Buckeye full-dataset RTFx benchmark.
Feeds ALL segments to the C++ engine daemon, measures wall time vs audio duration.
Usage: ./buckeye_bench_rtfx.py [--limit N]
"""
import json, subprocess, sys, os, time

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "buckeye")
MANIFEST = os.path.join(DATA_DIR, "manifest.json")
ENGINE = os.path.join(os.path.dirname(__file__), "force_aligner")
MODEL_DIR = "/data/.cache/huggingface/hub/models--Qwen--Qwen3-ForcedAligner-0.6B"
for d in os.listdir(os.path.join(MODEL_DIR, "snapshots")):
    MODEL_PATH = os.path.join(MODEL_DIR, "snapshots", d)
    break

def main():
    limit = None
    for i, a in enumerate(sys.argv[1:]):
        if a == "--limit" and i + 1 < len(sys.argv) - 1:
            limit = int(sys.argv[i + 2])

    with open(MANIFEST) as f:
        data = json.load(f)

    segments = data["samples"]
    if limit:
        segments = segments[:limit]

    total_audio_s = sum(s["duration_s"] for s in segments)
    print(f"Segments: {len(segments)}")
    print(f"Total audio: {total_audio_s:.1f}s ({total_audio_s/60:.1f}min)")
    print()

    # Build daemon input
    stdin_lines = []
    for seg in segments:
        audio_path = os.path.join(DATA_DIR, seg["audio"])
        stdin_lines.append(f"{audio_path}\t{seg['transcript']}\tEnglish")
    stdin_data = "\n".join(stdin_lines) + "\n"

    print(f"Feeding {len(segments)} segments to daemon...")
    t0 = time.time()
    proc = subprocess.run(
        [ENGINE, "--model", MODEL_PATH, "--daemon"],
        input=stdin_data,
        capture_output=True,
        text=True,
        timeout=7200,  # 2h max
    )
    wall_s = time.time() - t0

    if proc.returncode != 0:
        print(f"Engine failed (rc={proc.returncode}):", file=sys.stderr)
        print(proc.stderr[:2000], file=sys.stderr)
        sys.exit(1)

    # Count output segments (blank-line separated)
    output_segs = [b for b in proc.stdout.strip().split("\n\n") if b.strip()]
    # Fallback: count non-blank line groups
    if len(output_segs) == 1:
        output_segs = [b for b in proc.stdout.strip().split("get\t") if b.strip()]

    rtfx = total_audio_s / wall_s
    ms_per_seg = wall_s / len(segments) * 1000

    print()
    print(f"=== RTFx Benchmark Results ===")
    print(f"Segments processed: {len(output_segs)} / {len(segments)}")
    print(f"Total audio:        {total_audio_s:.1f}s ({total_audio_s/60:.1f}min)")
    print(f"Wall time:          {wall_s:.1f}s ({wall_s/60:.1f}min)")
    print(f"RTFx:               {rtfx:.2f}x")
    print(f"Avg per segment:    {ms_per_seg:.0f}ms")
    if proc.stderr.strip():
        # Print any timing info from stderr
        for line in proc.stderr.strip().split("\n"):
            if any(k in line.lower() for k in ["time", "ms", "sec", "total", "init"]):
                print(f"  engine: {line.strip()}")

if __name__ == "__main__":
    main()
