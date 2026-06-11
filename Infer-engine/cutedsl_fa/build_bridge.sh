#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/cutedsl_fa"

CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.8}"
PYTHON="${PYTHON:-python3}"
GLMASR_CUDA_ARCH="${GLMASR_CUDA_ARCH:-sm_120}"

case "$GLMASR_CUDA_ARCH" in
  sm_120)
    echo "==> Building CuteDSL FA2 objects for sm_120"
    "$PYTHON" fa2_kernel.py

    echo "==> Building CuteDSL dense TMA object for sm_120"
    "$PYTHON" fa_tma.py

    BRIDGE_OBJECTS=(
      fa2_enc_varlen_fwd.o
      fa2_dec_varlen_fwd.o
      fa2_enc_dense_tma_fwd.o
    )
    ;;
  sm_89|sm_86)
    echo "==> Building CuteDSL FA2 objects for $GLMASR_CUDA_ARCH"
    GLMASR_CUTEDSL_SM89_ARCH="$GLMASR_CUDA_ARCH" "$PYTHON" fa2_kernel_sm89.py

    BRIDGE_OBJECTS=(
      sm89_build/fa2_enc_varlen_fwd.o
      sm89_build/fa2_dec_varlen_fwd.o
    )
    ;;
  *)
    echo "ERROR: unsupported GLMASR_CUDA_ARCH=$GLMASR_CUDA_ARCH; expected sm_120, sm_89, or sm_86" >&2
    exit 1
    ;;
esac

echo "==> Locating tvm_ffi and CUTLASS DSL runtime"
readarray -t PY_PATHS < <("$PYTHON" - <<'PY'
from pathlib import Path
import cutlass
import tvm_ffi

tvm_root = Path(tvm_ffi.__path__[0])
cutlass_file = Path(cutlass.__file__).resolve()
cutlass_static = None
for parent in cutlass_file.parents:
    cand = parent / "nvidia_cutlass_dsl" / "lib" / "libcuda_dialect_runtime_static.a"
    if cand.exists():
        cutlass_static = cand
        break
    cand = parent / "lib" / "libcuda_dialect_runtime_static.a"
    if cand.exists():
        cutlass_static = cand
        break
if cutlass_static is None:
    raise SystemExit("cannot find libcuda_dialect_runtime_static.a")

print(tvm_root / "include")
print(tvm_root / "lib")
print(cutlass_static)
PY
)

TVMFFI_INC="${PY_PATHS[0]}"
TVMFFI_LIB="${PY_PATHS[1]}"
CUTLASS_STATIC="${PY_PATHS[2]}"
TVMFFI_RPATH="${GLMASR_TVMFFI_RPATH:-$TVMFFI_LIB}"

echo "==> Patching CUTLASS DSL static runtime"
PATCH_DIR="build_bridge/patched_cuda_dialect"
rm -rf "$PATCH_DIR"
mkdir -p "$PATCH_DIR"
(
  cd "$PATCH_DIR"
  ar x "$CUTLASS_STATIC"
  for obj in *.o; do
    objcopy --redefine-sym cuda_dialect_init_library_once=cuda_dialect_init_ORIG "$obj" "p_$obj"
  done
  ar rcs ../libpatched3.a p_*.o
)

echo "==> Compiling override_init.o"
gcc -fPIC -O2 -c override_init.c -o override_init.o -I"$CUDA_ROOT/include"

echo "==> Linking fa2_bridge.so"
gcc -shared -fPIC -O2 -o fa2_bridge.so \
  fa2_bridge.c override_init.o \
  "${BRIDGE_OBJECTS[@]}" \
  -I"$CUDA_ROOT/include" -I"$TVMFFI_INC" \
  -L"$CUDA_ROOT/lib64" "$TVMFFI_LIB/libtvm_ffi.so" build_bridge/libpatched3.a \
  -lcuda -lcudart -ldl -lm -lpthread \
  -Wl,-rpath,"$TVMFFI_RPATH"

echo "==> Built cutedsl_fa/fa2_bridge.so"
