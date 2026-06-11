#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 sm86|sm89|sm120" >&2
  exit 2
fi
arch_name="$1"
case "$arch_name" in
  sm86) arch=sm_86 ;;
  sm89) arch=sm_89 ;;
  sm120) arch=sm_120 ;;
  *) echo "ERROR: unsupported arch '$arch_name'" >&2; exit 2 ;;
esac
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${GLM_ASR_FORCEALIGNER_SRC:-$ROOT/forcealigner-engine}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
cd "$SRC"
make clean >/tmp/glm_asr_forcealigner_${arch_name}_clean.log 2>&1 || true
make -j"$(nproc)" \
  NVCC="$CUDA_HOME/bin/nvcc" \
  CUDA_INC="$CUDA_HOME/include" \
  NVFLAGS="-O2 -std=c++17 -arch=$arch -Iinclude --compiler-options \"-Wall -Wno-unused-function\"" \
  LDFLAGS="-L$CUDA_HOME/lib64 -lcublas -lcublasLt -lcufft -lcudart -lcuda -lm"
mkdir -p "$ROOT/forcealigner-engine"
cp -f force_aligner "$ROOT/forcealigner-engine/force_aligner_${arch_name}"
cp -f mel_filterbank.bin "$ROOT/forcealigner-engine/mel_filterbank.bin"
chmod +x "$ROOT/forcealigner-engine/force_aligner_${arch_name}"
ln -sfn "force_aligner_${arch_name}" "$ROOT/forcealigner-engine/force_aligner"
echo "built $ROOT/forcealigner-engine/force_aligner_${arch_name}"
