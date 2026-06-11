# GLM-ASR InferEngine

<p align="center">
  <img src="assets/miner.png" width="128" alt="GLM-ASR InferEngine" />
</p>

<p align="center">
  <a href="README.en.md">English Version</a> | 中文版
</p>

`GLM-ASR InferEngine` ，一个使用 `GLM-ASR-Nano-2512` 的 CUDA C/C++ 高性能推理引擎。这个仓库只关注一件事：**在消费级 NVIDIA GPU 上尽可能快地跑 GLM-ASR 推理，并提供可复现 benchmark**。

完整集成ForceAligner的字幕、任务队列、Docker 运行时在单独项目：
> https://github.com/JazerJu/glm-asr-stack

# 1.编译 `glm_asr_infer`推理引擎

```bash
cmake -S . -B build_cmake \
  -DCUDA_ROOT=/usr/local/cuda-12.8 \
  -DGLMASR_CUDA_ARCH=sm_120 \
  -DGLMASR_PYTHON_EXECUTABLE="$(which python3)"
cmake --build build_cmake -j
cmake --build build_cmake --target cutedsl_fa2_bridge -j
```

产物包括：

```bash
./glm_asr_infer
cutedsl_fa/fa2_bridge.so
```

`glm_asr_infer` 是 C/CUDA 推理引擎；`fa2_bridge.so` 是运行时 `dlopen` 的 CuteDSL FA2/TMA 性能插件。`cutedsl_fa2_bridge` 会根据 `GLMASR_CUDA_ARCH` 自动选择对应的 CuteDSL kernel 源码，最终输出文件始终是：

```bash
cutedsl_fa/fa2_bridge.so
```

常用架构：

| GPU | CMake 参数 | CuteDSL 源码 |
|---|---|---|
| RTX 50 系 | `-DGLMASR_CUDA_ARCH=sm_120` | `cutedsl_fa/fa2_kernel.py` + `cutedsl_fa/fa_tma.py` |
| RTX 40 系 | `-DGLMASR_CUDA_ARCH=sm_89` | `cutedsl_fa/fa2_kernel_sm89.py` |
| RTX 30 系 | `-DGLMASR_CUDA_ARCH=sm_86` | `cutedsl_fa/fa2_kernel_sm89.py`，通过 `GLMASR_CUTEDSL_SM89_ARCH=sm_86` 编译 |

# 2.Infer-engine vs vLLM benchmark

因为VLLM的glm-asr backend处理音频时会默认pad至30s，因此benchmark时Infer-engine会进行同样的操作。

- 模型：`zai-org/GLM-ASR-Nano-2512` BF16原版。
- 预热：16 条独立 pad30 样本，不参与计时。
- vLLM：offline Python API，关闭 prefix cache。

| Engine | 输入 | Wall / infer window |
|---|---|---|
| **Infer-engine** | 256 x 30s pad30 raw PCM | **10.6s** |
| **vLLM 0.18.0** | 同一批 256 x 30s pad30 FLAC | **11.65s** |

复现命令：

```bash
pip install -r requirements.txt
cmake -S . -B build_cmake \
  -DCUDA_ROOT=/usr/local/cuda-12.8 \
  -DGLMASR_CUDA_ARCH=sm_120 \
  -DGLMASR_PYTHON_EXECUTABLE="$(which python3)"
cmake --build build_cmake -j
cmake --build build_cmake --target cutedsl_fa2_bridge -j
python3 bench_vllm_pad30.py --download-models
```

`bench_vllm_pad30.py` 会用同一批 HuggingFace 样本生成两份 pad30 输入：Infer-engine 使用 raw PCM `.bin`，vLLM 使用 `.flac`；随后先跑当前仓库生成的 `./glm_asr_infer`，再用 vLLM offline API 在同一个 Python 进程内跑同批请求。默认使用 16 条额外样本做独立 warmup，timed 的 256 条不会和 warmup 重复；vLLM offline 路径会显式关闭 prefix cache。vLLM 的音频预处理依赖其 GLM-ASR backend 内部的 `WhisperFeatureExtractor`。

只跑其中一边：

```bash
# 只跑 Infer-engine
python3 bench_vllm_pad30.py --download-models --skip-vllm

# 只跑 vLLM offline；会准备同批 pad30 FLAC
python3 bench_vllm_pad30.py --download-models --skip-infer

# 如需复现 OpenAI server 路径，可显式使用 server backend
python3 bench_vllm_pad30.py --download-models --skip-infer --vllm-backend server

# vLLM server 已经手动启动时，只提交请求
python3 bench_vllm_pad30.py --download-models --skip-infer --vllm-backend server --keep-vllm-server
```

# 3.实际案例：50 / 40 / 30 系显卡转录 11h 音频所需时长。

使用同一个 11h55min的音频 https://www.youtube.com/watch?v=86FAWCzIe_4 ， ASR + ForceAligner 两步转录，输出包含时间戳的完整srt文件。这里引入 ForceAligner生成时间戳，Infer-engine 只负责 ASR，即语音转文字；完整流程见 `glm-asr-stack`。

| GPU | Arch | ASR infer | ForceAligner | 两步核心时间 | 备注 |
|---|---|---|---|---|---|
| RTX 5070 Ti | SM120 | 约 95s | 约 65s | 约 160s | 本地 11h 两步实测，多个版本接近这个量级 |
| RTX 4090 | SM89 | **71.71s** | **52.89s** | **124.60s** | 远端 4090，SM89 CuteDSL FA2 |
| RTX 3090 Ti | SM86 | **105.35s** | **64.45s** | **169.80s** | 首次租用 3090 Ti，SM86 CuteDSL FA2 |

结论：4090 最快，5070 Ti 居中，3090 Ti 最慢。4090 相对 3090 Ti：ASR 约 `1.47x`，ForceAligner 约 `1.22x`，两步核心时间约 `1.36x`。

## 3.2 实际案例2：2611 条 长度不一音频的转录

数据来自HuggingFace librispeech 2611 条音频，默认 `start=9, num=2611`，从第九个音频开始，短于30s.

| GPU | Arch | Infer time | RTFx |
|---|---|---|---|
| RTX 5070 Ti | SM120 | **37.1s** | **516.5x** |

复现单机 转录：

```bash
python3 transcribe_librispeech.py --download-models
```

## 安装依赖

```bash
pip install -r requirements.txt
```


## Benchmark 脚本

```bash
# 第一张表：pad30，同批数据对比 Infer-engine 和 vLLM offline
python3 bench_vllm_pad30.py --download-models

# 第二张表：2611 条 mixed-length LibriSpeech，只测 Infer-engine
python3 transcribe_librispeech.py --download-models
```

默认数据集：

```text
dataset: hf-audio/esb-datasets-test-only-sorted
config:  librispeech
split:   test.clean
```

如果本地已有 HuggingFace Arrow cache，脚本会优先使用本地 cache；否则通过 `datasets.load_dataset()` 下载。

vLLM pad30 默认参数：

```bash
VLLM_BENCH_BACKEND=offline
VLLM_MAX_BATCHED_TOKENS=6000
VLLM_MAX_NUM_SEQS=256
VLLM_GPU_MEMORY_UTILIZATION=0.9
GLMASR_BENCH_WARMUP=16
```

可以通过环境变量覆盖。

## 依赖

构建：

- Linux x86_64
- NVIDIA GPU
- CUDA Toolkit 12.8 at `/usr/local/cuda-12.8`
- `gcc`, `cmake`

Benchmark Python 依赖：

```bash
pip install -r requirements.txt
```

## 运行参数

Infer-engine benchmark 默认环境：

```bash
GLMASR_VRAM_UTIL=0.9
GLMASR_MAX_SEQS=256
GLMASR_MAX_BATCHED_TOKENS=10000
GLMASR_ENCODER_FA_PIPELINED=1
GLMASR_ENCODER_FA_VLLM=1
GLMASR_ENCODER_FA_LONG_MIN_SEQ=1
GLMASR_ENCODER_FA_MIN_SEQ=1
python3 transcribe_librispeech.py --download-models
```

8GB 显存卡可以先降到：

```bash
GLMASR_MAX_SEQS=64
GLMASR_MAX_BATCHED_TOKENS=4000
GLMASR_VRAM_UTIL=0.85
python3 transcribe_librispeech.py --download-models
```

## 项目结构

```text
Infer-engine/
├── CMakeLists.txt     # 标准 CMake 构建入口
├── cuda/              # CUDA kernels: mel, encoder, decoder, paged KV, FA paths
├── include/           # C ABI headers and shared types
├── src/               # engine, scheduler, model loader, tokenizer, audio I/O
├── cutedsl_fa/        # CuTeDSL FA2 kernels and bridge code
├── tools/             # internal benchmark helpers used by transcribe_librispeech.py
├── transcribe_librispeech.py  # LibriSpeech transcription and comparison entry point
└── AGENTS.md          # detailed engineering notes and profiling history
```

## 模型与数据

本仓库不重新分发模型权重或 benchmark 音频。默认命令会自动下载：

- 模型：`zai-org/GLM-ASR-Nano-2512` -> `./models/GLM-ASR-Nano-2512`
- Mel filter：仓库内置 `./resources/mel_filters.bin`
- benchmark 数据：默认使用 `hf-audio/esb-datasets-test-only-sorted/librispeech`
