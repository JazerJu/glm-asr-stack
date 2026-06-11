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
SRC="${GLM_ASR_INFER_SRC:-$ROOT/Infer-engine}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
cd "$SRC"
make clean >/tmp/glm_asr_infer_${arch_name}_clean.log 2>&1 || true
make -j"$(nproc)" \
  NVCC="$CUDA_HOME/bin/nvcc" \
  CUDA_INC="$CUDA_HOME/include" \
  NVFLAGS="-O2 -std=c++17 -arch=$arch -Iinclude --compiler-options \"-Wall -Wno-unused-function\"" \
  LDFLAGS="-L$CUDA_HOME/lib64 -lcublas -lcublasLt -lcufft -lcudart -lcuda -lm -ldl" \
  LDRPATH="-Xlinker --disable-new-dtags -Xlinker -rpath -Xlinker '$ROOT/lib' -Xlinker -rpath -Xlinker '$CUDA_HOME/lib64'"
mkdir -p "$ROOT/Infer-engine"
cp -f glm_asr_infer "$ROOT/Infer-engine/glm_asr_infer_${arch_name}"
chmod +x "$ROOT/Infer-engine/glm_asr_infer_${arch_name}"
ln -sfn "glm_asr_infer_${arch_name}" "$ROOT/Infer-engine/glm_asr_infer"
echo "built $ROOT/Infer-engine/glm_asr_infer_${arch_name}"
