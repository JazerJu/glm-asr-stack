#!/usr/bin/env python3
"""Step-by-step mel spectrogram comparison: C++ dump vs Python computation.

Traces the exact source of the ~0.97 raw log10 offset (9.28x power factor)
between C++ and Python mel spectrograms.
"""
import numpy as np
import torch
import struct, wave, os

AUDIO_PATH = '/data/fwsr/glm-asr/GLM-ASR/cut.wav'
MEL_FILTERS_PATH = '/data/fwsr/glm-asr/forcealigner-engine/mel_filterbank.bin'
CPP_MEL_PATH = '/tmp/enc_dump/cpp_conv/mel_input.bin'

N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128

def load_audio():
    with wave.open(AUDIO_PATH, 'rb') as wf:
        n = wf.getnframes()
        raw = wf.readframes(n)
    wav = np.array(struct.unpack(f'<{n}h', raw), dtype=np.float32) / 32768.0
    return torch.from_numpy(wav)

def load_cpp_mel():
    """Load C++ mel dump. C++ stores as [n_mels, num_frames] bf16."""
    raw = np.fromfile(CPP_MEL_PATH, dtype=np.uint16)
    # num_frames = 224000 / 160 = 1400
    mel = torch.from_numpy(raw.copy()).view(torch.bfloat16).float().reshape(N_MELS, 1400)
    return mel

def load_mel_filterbank():
    fb = np.fromfile(MEL_FILTERS_PATH, dtype=np.float32).reshape(N_MELS, N_FFT // 2 + 1)
    return torch.from_numpy(fb)

def main():
    print("=" * 70)
    print("MEL SPECTROGRAM DIAGNOSTIC: C++ vs Python step-by-step")
    print("=" * 70)

    wav = load_audio()
    print(f"\nAudio: {wav.shape[0]} samples ({wav.shape[0]/16000:.2f}s)")
    num_frames = wav.shape[0] // HOP_LENGTH
    bins = N_FFT // 2 + 1
    print(f"Expected frames: {num_frames}, bins: {bins}")

    mel_fb = load_mel_filterbank()
    print(f"Mel filterbank: {mel_fb.shape}")

    # ===================================================================
    # STEP 1: Hann window comparison
    # ===================================================================
    print("\n--- STEP 1: Hann Window ---")
    # Python (periodic)
    py_hann = torch.hann_window(N_FFT, periodic=True)
    # C++ formula: 0.5 - 0.5 * cos(2*pi*i/N)
    cpp_hann = torch.tensor([0.5 - 0.5 * np.cos(2.0 * np.pi * i / N_FFT) for i in range(N_FFT)])
    hann_diff = (py_hann - cpp_hann).abs().max().item()
    print(f"  Python hann sum: {py_hann.sum().item():.6f}")
    print(f"  C++ hann sum:    {cpp_hann.sum().item():.6f}")
    print(f"  Hann max diff:   {hann_diff:.2e}")
    print(f"  sum(hann^2):     {(py_hann**2).sum().item():.6f}")

    # ===================================================================
    # STEP 2: STFT / FFT comparison (frame 0 only)
    # ===================================================================
    print("\n--- STEP 2: STFT comparison (frame 0) ---")
    # Python torch.stft (one-sided, center=True)
    py_stft = torch.stft(wav, N_FFT, HOP_LENGTH, window=py_hann, center=True, return_complex=True)
    print(f"  torch.stft output shape: {py_stft.shape}")  # (201, T)
    print(f"  torch.stft num frames: {py_stft.shape[1]}")

    # Manual framing matching C++ (center=True, zero-pad out-of-range)
    frame_idx = 0
    center_sample = frame_idx * HOP_LENGTH  # = 0
    start = center_sample - N_FFT // 2  # = -200
    samples = torch.zeros(N_FFT)
    for j in range(N_FFT):
        sidx = start + j
        if 0 <= sidx < wav.shape[0]:
            samples[j] = wav[sidx]
    framed = samples * py_hann

    # Python stft frame 0
    py_frame0 = py_stft[:, 0]

    # Manual FFT of framed signal
    manual_fft = torch.fft.rfft(framed)

    print(f"  Frame 0 center at sample {center_sample}")
    print(f"  Manual FFT vs torch.stft frame 0:")
    fft_diff = (manual_fft.abs() - py_frame0.abs()).abs()
    print(f"    Max abs diff:  {fft_diff.max().item():.2e}")
    print(f"    Mean abs diff: {fft_diff.mean().item():.2e}")

    # Power spectrum comparison for frame 0
    py_power0 = py_frame0.abs() ** 2
    manual_power0 = manual_fft.abs() ** 2
    print(f"  Power spectrum frame 0:")
    print(f"    Python max power: {py_power0.max().item():.6f}")
    print(f"    Manual max power: {manual_power0.max().item():.6f}")
    print(f"    Power diff max:   {(py_power0 - manual_power0).abs().max().item():.2e}")

    # ===================================================================
    # STEP 3: Full mel spectrogram via torch.stft (Python reference)
    # ===================================================================
    print("\n--- STEP 3: Python mel spectrogram (torch.stft) ---")
    py_mag = py_stft.abs() ** 2  # (201, T)
    print(f"  Power spectrum shape: {py_mag.shape}")
    py_mel = torch.matmul(mel_fb, py_mag)  # (128, T)
    py_log_mel = torch.clamp(py_mel, min=1e-10).log10()

    # Drop last frame to match C++ (1400 frames)
    py_log_mel_trimmed = py_log_mel[:, :num_frames]
    print(f"  Python log_mel shape (trimmed): {py_log_mel_trimmed.shape}")
    print(f"  Python log_mel range: [{py_log_mel_trimmed.min().item():.4f}, {py_log_mel_trimmed.max().item():.4f}]")
    print(f"  Python log_mel mean: {py_log_mel_trimmed.mean().item():.4f}")

    # ===================================================================
    # STEP 4: C++ mel dump comparison
    # ===================================================================
    print("\n--- STEP 4: C++ mel dump comparison ---")
    cpp_mel = load_cpp_mel()
    print(f"  C++ mel shape: {cpp_mel.shape}")

    # The C++ mel is POST-processed: clamp(max-8) -> (x+4)/4
    # Undo post-process to get raw log_mel
    cpp_raw = cpp_mel * 4.0 - 4.0  # undo (x+4)/4
    # Note: clamping to max-8 means we can't fully undo for values near the clamp
    # but for the comparison, the offset should be consistent

    print(f"  C++ raw log_mel range: [{cpp_raw.min().item():.4f}, {cpp_raw.max().item():.4f}]")
    print(f"  C++ raw log_mel mean: {cpp_raw.mean().item():.4f}")
    print(f"  Python raw log_mel range: [{py_log_mel_trimmed.min().item():.4f}, {py_log_mel_trimmed.max().item():.4f}]")
    print(f"  Python raw log_mel mean: {py_log_mel_trimmed.mean().item():.4f}")

    # Direct diff in raw log10 domain
    raw_diff = cpp_raw - py_log_mel_trimmed
    print(f"\n  Raw log10 diff (C++ - Python):")
    print(f"    Mean:  {raw_diff.mean().item():.6f}")
    print(f"    Std:   {raw_diff.std().item():.6f}")
    print(f"    Min:   {raw_diff.min().item():.6f}")
    print(f"    Max:   {raw_diff.max().item():.6f}")

    # Linear power ratio
    power_ratio = 10 ** raw_diff.mean().item()
    print(f"    Power ratio (C++/Python): {power_ratio:.4f}x")

    # ===================================================================
    # STEP 5: Per-frame power comparison (spot check)
    # ===================================================================
    print("\n--- STEP 5: Per-frame power comparison ---")
    for frame_idx in [0, 100, 500, 1000, 1399]:
        if frame_idx >= num_frames:
            continue
        # Python power for this frame
        py_frame_power = py_mag[:, frame_idx]  # (201,) power spectrum
        py_frame_mel = mel_fb @ py_frame_power  # (128,) mel energy
        py_frame_log = torch.clamp(py_frame_mel, min=1e-10).log10()

        # C++ raw log mel for this frame
        cpp_frame_raw = cpp_raw[:, frame_idx]

        frame_diff = cpp_frame_raw - py_frame_log
        print(f"  Frame {frame_idx:4d}: diff mean={frame_diff.mean().item():.4f} "
              f"std={frame_diff.std().item():.4f} "
              f"ratio={10**frame_diff.mean().item():.3f}x")

    # ===================================================================
    # STEP 6: Check if difference is purely additive (constant offset)
    # ===================================================================
    print("\n--- STEP 6: Offset analysis ---")
    # If constant offset, then diff std should be ~0
    print(f"  Diff std: {raw_diff.std().item():.6f}")
    print(f"  Diff std / |mean|: {raw_diff.std().item() / abs(raw_diff.mean().item()):.6f}")

    # Check per-mel-bin offset
    print(f"\n  Per-mel-bin offset (frame-averaged):")
    bin_diffs = raw_diff.mean(dim=1)  # average over frames for each mel bin
    print(f"    Range: [{bin_diffs.min().item():.4f}, {bin_diffs.max().item():.4f}]")
    print(f"    Std:   {bin_diffs.std().item():.6f}")

    # Check first few mel bins
    for m in [0, 1, 2, 63, 64, 127]:
        print(f"    Bin {m:3d}: offset={bin_diffs[m].item():.4f}")

    # ===================================================================
    # STEP 7: Hypothesis test - is it a normalization factor?
    # ===================================================================
    print("\n--- STEP 7: Hypothesis tests ---")
    mean_offset = raw_diff.mean().item()
    print(f"  Observed offset: {mean_offset:.6f} (in raw log10)")
    print(f"  Power ratio:     {10**mean_offset:.4f}x")

    # Test: 2.0 factor (conjugate symmetry doubling)
    print(f"\n  log10(2) = {np.log10(2):.6f}  {'MATCH' if abs(mean_offset - np.log10(2)) < 0.05 else 'no'}")
    # Test: sum(hann^2)
    hann_sq_sum = (py_hann**2).sum().item()
    print(f"  log10(sum(hann^2)) = log10({hann_sq_sum:.1f}) = {np.log10(hann_sq_sum):.6f}  {'MATCH' if abs(mean_offset - np.log10(hann_sq_sum)) < 0.05 else 'no'}")
    # Test: N
    print(f"  log10(N) = log10({N_FFT}) = {np.log10(N_FFT):.6f}  {'MATCH' if abs(mean_offset - np.log10(N_FFT)) < 0.05 else 'no'}")
    # Test: N/2
    print(f"  log10(N/2) = log10({N_FFT/2}) = {np.log10(N_FFT/2):.6f}  {'MATCH' if abs(mean_offset - np.log10(N_FFT/2)) < 0.05 else 'no'}")
    # Test: 2/N
    print(f"  log10(2/N) = log10({2/N_FFT}) = {np.log10(2/N_FFT):.6f}  {'MATCH' if abs(mean_offset - np.log10(2/N_FFT)) < 0.05 else 'no'}")
    # Test: N^2
    print(f"  log10(N^2) = log10({N_FFT**2}) = {np.log10(N_FFT**2):.6f}  {'MATCH' if abs(mean_offset - np.log10(N_FFT**2)) < 0.05 else 'no'}")

    # Test: could it be that C++ uses |FFT|^2 while Python uses |FFT|?
    print(f"  log10(x^2) for any x = 2*log10(x) -- doesn't give constant offset unless |FFT| is constant")

    # ===================================================================
    # STEP 8: Direct FFT output comparison (frame 0)
    # ===================================================================
    print("\n--- STEP 8: Direct FFT output comparison (frame 0) ---")
    # C++ computes: hann-windowed samples -> cuFFT R2C -> power = Re^2 + Im^2
    # Python: torch.stft -> abs()^2

    # Manually compute frame 0 matching C++ exactly
    frame0_center = 0
    frame0_start = frame0_center - N_FFT // 2  # -200
    frame0_samples = torch.zeros(N_FFT)
    for j in range(N_FFT):
        sidx = frame0_start + j
        if 0 <= sidx < wav.shape[0]:
            frame0_samples[j] = wav[sidx]

    frame0_windowed = frame0_samples * py_hann
    frame0_fft = torch.fft.rfft(frame0_windowed)
    frame0_power = frame0_fft.real**2 + frame0_fft.imag**2  # match C++ power_spectrum_kernel

    # Python stft frame 0 power
    py_frame0_stft = py_stft[:, 0]
    py_frame0_power = py_frame0_stft.real**2 + py_frame0_stft.imag**2

    power_diff = frame0_power - py_frame0_power
    print(f"  Manual FFT vs torch.stft power (frame 0):")
    print(f"    Max diff:  {power_diff.abs().max().item():.2e}")
    print(f"    Mean diff: {power_diff.abs().mean().item():.2e}")
    print(f"    Max power: {frame0_power.max().item():.6f} vs {py_frame0_power.max().item():.6f}")

    # Mel projection for frame 0
    mel0_manual = mel_fb @ frame0_power
    mel0_py = mel_fb @ py_frame0_power
    log_mel0_manual = torch.clamp(mel0_manual, min=1e-10).log10()
    log_mel0_py = torch.clamp(mel0_py, min=1e-10).log10()
    print(f"\n  Mel projection frame 0:")
    print(f"    Manual log10 mel range: [{log_mel0_manual.min().item():.4f}, {log_mel0_manual.max().item():.4f}]")
    print(f"    Python log10 mel range: [{log_mel0_py.min().item():.4f}, {log_mel0_py.max().item():.4f}]")
    print(f"    Diff mean: {(log_mel0_manual - log_mel0_py).mean().item():.6f}")
    print(f"    Diff max:  {(log_mel0_manual - log_mel0_py).abs().max().item():.6f}")

    # C++ frame 0 raw
    cpp_frame0_raw = cpp_raw[:, 0]
    print(f"\n  C++ raw frame 0 range: [{cpp_frame0_raw.min().item():.4f}, {cpp_frame0_raw.max().item():.4f}]")
    print(f"  C++ vs Python (frame 0) offset: {(cpp_frame0_raw - log_mel0_py).mean().item():.4f}")

    # ===================================================================
    # STEP 9: Check if Python WhisperFeatureExtractor does something different
    # ===================================================================
    print("\n--- STEP 9: WhisperFeatureExtractor mel comparison ---")
    # The WhisperFeatureExtractor._torch_extract_fbank_features does:
    #   stft = torch.stft(waveform, n_fft, hop_length, window=window, return_complex=True)
    #   magnitudes = stft[..., :-1].abs() ** 2
    #   mel_spec = mel_filters.T @ magnitudes
    #
    # Key: mel_filters.T ! The filterbank is TRANSPOSED compared to our computation.
    # If mel_filters is (128, 201), then mel_filters.T is (201, 128)
    # (201, 128) @ magnitudes(201, T) -- this wouldn't work!
    # So mel_filters must be (201, 128) shape, making mel_filters.T = (128, 201)
    # Then (128, 201) @ (201, T) = (128, T) ✓

    # Let's check what the actual WhisperFeatureExtractor filterbank shape is
    try:
        from transformers import WhisperFeatureExtractor
        wfe = WhisperFeatureExtractor(feature_size=N_MELS, sampling_rate=16000,
                                       n_fft=N_FFT, hop_length=HOP_LENGTH)
        print(f"  WhisperFeatureExtractor mel_filters shape: {wfe.mel_filters.shape}")
        print(f"  Our mel_filterbank.bin shape: (128, 201)")

        # Compare filterbank values
        wfe_fb = torch.from_numpy(wfe.mel_filters).float()
        our_fb = mel_fb.float()
        if wfe_fb.shape == our_fb.T.shape:
            fb_diff = (wfe_fb - our_fb.T).abs()
            print(f"  Filterbank diff (WFE vs ours.T): max={fb_diff.max().item():.6f} mean={fb_diff.mean().item():.6f}")
        elif wfe_fb.shape == our_fb.shape:
            fb_diff = (wfe_fb - our_fb).abs()
            print(f"  Filterbank diff (WFE vs ours): max={fb_diff.max().item():.6f} mean={fb_diff.mean().item():.6f}")
        else:
            print(f"  Shape mismatch: WFE {wfe_fb.shape} vs ours {our_fb.shape}")
    except Exception as e:
        print(f"  Error loading WhisperFeatureExtractor: {e}")

    # ===================================================================
    # STEP 10: Check if Qwen3ForcedAligner uses a different mel path
    # ===================================================================
    print("\n--- STEP 10: Check Qwen3ForcedAligner mel computation ---")
    try:
        sys.path.insert(0, '/data/ASR模型/Qwen3-ASR')
        from qwen_asr import Qwen3ForcedAligner
        model = Qwen3ForcedAligner.from_pretrained('Qwen/Qwen3-ForcedAligner-0.6B',
                                                     dtype=torch.bfloat16, device_map='cuda:0')
        # Check how mel is computed
        tower = model.model.thinker.audio_tower
        print(f"  Audio tower type: {type(tower).__name__}")

        # Check if tower has its own mel computation
        if hasattr(tower, 'feature_extractor'):
            print(f"  Tower has feature_extractor: {type(tower.feature_extractor)}")
        if hasattr(tower, 'mel_filters'):
            print(f"  Tower mel_filters shape: {tower.mel_filters.shape if hasattr(tower.mel_filters, 'shape') else 'N/A'}")

        # Check model's processor/feature_extractor
        if hasattr(model, 'processor'):
            print(f"  Model processor: {type(model.processor)}")
        if hasattr(model, 'feature_extractor'):
            print(f"  Model feature_extractor: {type(model.feature_extractor)}")

        # Try to find how mel is computed in the forward path
        import inspect
        src = inspect.getsource(type(tower))
        # Find mel-related lines
        for i, line in enumerate(src.split('\n')):
            if 'mel' in line.lower() or 'stft' in line.lower() or 'feature' in line.lower():
                print(f"  L{i}: {line.strip()[:100]}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    main()
