# pyright: reportGeneralTypeIssues=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportMissingTypeArgument=false
#
# Dense SM120 TMA experiment for the encoder attention path.
#
# This file deliberately stays separate from fa.py. The existing engine ABI is
# varlen/flat, while the optimized SM120 TMA fast path is dense batched
# non-causal attention. Keeping this as an isolated export lets us validate the
# kernel before adding a runtime equal-length dispatch in the C bridge.

import math
import importlib.util
import os
import sys
import types
from pathlib import Path

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack


OUT_DIR = Path("/data/fwsr/glm-asr/Infer-engine/cutedsl_fa")
OPT_CUTE_DIR = Path("/tmp/flash-attention-sm120-optimized/flash_attn/cute")
TMA_TILE_M = int(os.environ.get("GLMASR_TMA_TILE_M", "128"))
TMA_TILE_N = int(os.environ.get("GLMASR_TMA_TILE_N", "128"))
TMA_NUM_STAGES = int(os.environ.get("GLMASR_TMA_NUM_STAGES", "2"))
TMA_USE_TMA_Q = os.environ.get("GLMASR_TMA_USE_TMA_Q", "1") != "0"


def _install_flash_attn_aliases():
    vllm_pkg = sys.modules.get("vllm")
    if vllm_pkg is None:
        vllm_pkg = types.ModuleType("vllm")
        vllm_pkg.__path__ = [str(OPT_CUTE_DIR.parents[2])]
        sys.modules["vllm"] = vllm_pkg
    vllm_fa_pkg = sys.modules.get("vllm.vllm_flash_attn")
    if vllm_fa_pkg is None:
        vllm_fa_pkg = types.ModuleType("vllm.vllm_flash_attn")
        vllm_fa_pkg.__path__ = [str(OPT_CUTE_DIR.parent)]
        sys.modules["vllm.vllm_flash_attn"] = vllm_fa_pkg
    vllm_cute_pkg = sys.modules.get("vllm.vllm_flash_attn.cute")
    if vllm_cute_pkg is None:
        vllm_cute_pkg = types.ModuleType("vllm.vllm_flash_attn.cute")
        vllm_cute_pkg.__path__ = [str(OPT_CUTE_DIR)]
        vllm_cute_pkg.__file__ = str(OPT_CUTE_DIR / "__init__.py")
        sys.modules["vllm.vllm_flash_attn.cute"] = vllm_cute_pkg
    vllm_pkg.vllm_flash_attn = vllm_fa_pkg
    vllm_fa_pkg.cute = vllm_cute_pkg

    pkg = sys.modules.get("flash_attn")
    if pkg is None:
        pkg = types.ModuleType("flash_attn")
        pkg.__path__ = [str(OPT_CUTE_DIR.parent)]
        sys.modules["flash_attn"] = pkg
    cute_pkg = sys.modules.get("flash_attn.cute")
    if cute_pkg is None:
        cute_pkg = types.ModuleType("flash_attn.cute")
        cute_pkg.__path__ = [str(OPT_CUTE_DIR)]
        cute_pkg.__file__ = str(OPT_CUTE_DIR / "__init__.py")
        sys.modules["flash_attn.cute"] = cute_pkg
    pkg.cute = cute_pkg

    # The local optimized file was written against a CUTLASS DSL snapshot that
    # exported these enums from cutlass.cute.arch. The current DSL accepts the
    # corresponding string literals directly.
    import cutlass.cute.arch as arch

    if not hasattr(arch, "ProxyKind"):
        arch.ProxyKind = types.SimpleNamespace(async_shared="async.shared")
    if not hasattr(arch, "SharedSpace"):
        arch.SharedSpace = types.SimpleNamespace(shared_cta="cta")


def _load_optimized_class():
    mod_name = "cutedsl_fa._flash_fwd_sm120_tma_optimized"
    path = OPT_CUTE_DIR / "flash_fwd_sm120_tma_optimized.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod.FlashAttentionForwardSm120TMAOptimized


_install_flash_attn_aliases()
FlashAttentionForwardSm120TMAOptimized = _load_optimized_class()


def to_cute_tensor(t: torch.Tensor) -> cute.Tensor:
    return from_dlpack(t, enable_tvm_ffi=True, assumed_align=16).mark_layout_dynamic()


def _current_stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


@cute.jit
def fa2_enc_dense_tma_fwd(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mO: cute.Tensor,
    softmax_scale: Float32,
    seqlen: Int32,
    num_seqs: Int32,
    stream: cuda.CUstream,
):
    # Shapes are dense [B, S, H, D]. seqlen/num_seqs are explicit only to keep
    # the C ABI close to the existing varlen export and force dynamic layouts.
    _ = seqlen
    _ = num_seqs
    FlashAttentionForwardSm120TMAOptimized(
        cutlass.BFloat16,
        64,
        64,
        qhead_per_kvhead=1,
        is_causal=False,
        is_local=False,
        pack_gqa=False,
        tile_m=TMA_TILE_M,
        tile_n=TMA_TILE_N,
        num_stages=TMA_NUM_STAGES,
        num_threads=160,
        use_tma_Q=TMA_USE_TMA_Q,
    )(
        mQ,
        mK,
        mV,
        mO,
        None,
        softmax_scale,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        stream,
    )


def compile_export():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to compile/export the CuTe DSL kernel")
    device = torch.device("cuda")
    batch = 2
    seqlen = 512
    heads = 20
    head_dim = 64
    q = torch.empty((batch, seqlen, heads, head_dim), dtype=torch.bfloat16, device=device)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    o = torch.empty_like(q)
    compiled = cute.compile(
        fa2_enc_dense_tma_fwd,
        to_cute_tensor(q),
        to_cute_tensor(k),
        to_cute_tensor(v),
        to_cute_tensor(o),
        Float32(1.0 / math.sqrt(head_dim)),
        Int32(seqlen),
        Int32(batch),
        _current_stream(),
        options="--gpu-arch sm_120 --enable-tvm-ffi",
    )
    compiled.export_to_c(str(OUT_DIR / "fa2_enc_dense_tma_fwd.o"), function_name="fa2_enc_dense_tma_fwd")


if __name__ == "__main__":
    compile_export()
