# pyright: reportGeneralTypeIssues=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportInvalidTypeArguments=false, reportMissingTypeArgument=false, reportOperatorIssue=false, reportCallIssue=false, reportOptionalSubscript=false, reportUninitializedInstanceVariable=false

import math
from pathlib import Path
from types import SimpleNamespace

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack
from quack import layout_utils


TILE_N = 128
LOG2_E = math.log2(math.e)
OUT_DIR = Path("/data/fwsr/glm-asr/Infer-engine/cutedsl_fa")


def to_cute_tensor(t: torch.Tensor) -> cute.Tensor:
    return from_dlpack(t, enable_tvm_ffi=True, assumed_align=16).mark_layout_dynamic()


def assume_strides_128b_aligned(t: cute.Tensor):
    divby = 128 // t.element_type.width
    strides = tuple(cute.assume(s, divby=divby) for s in t.stride[:-1])
    return (*strides, t.stride[-1])


def assume_tensor_128b_aligned(t: cute.Tensor) -> cute.Tensor:
    return cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=assume_strides_128b_aligned(t)))


class FlashAttention2VarlenSm80:
    def __init__(self, head_dim: int, is_causal: bool, num_warps: int, tile_m: int = 64, tile_n: int = TILE_N):
        self.head_dim = head_dim
        self.head_dim_padded = ((head_dim + 31) // 32) * 32
        self.is_causal = is_causal
        self.num_warps = num_warps
        self.num_threads = num_warps * 32
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.dtype = cutlass.BFloat16

    def _make_acc_tensor_mn_view(self, acc: cute.Tensor) -> cute.Tensor:
        acc_layout_col_major = cute.make_layout(acc.layout.shape)
        acc_layout_mn = cute.make_layout(
            (
                (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),
                (acc_layout_col_major.shape[0][0], acc_layout_col_major.shape[2]),
            ),
            stride=(
                (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),
                (acc_layout_col_major.stride[0][0], acc_layout_col_major.stride[2]),
            ),
        )
        acc_layout_mn = cute.composition(acc.layout, acc_layout_mn)
        return cute.make_tensor(acc.iterator, acc_layout_mn)

    def _threadquad_reduce(self, val: cutlass.Float32, op):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31))
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31))
        return val

    def _threadquad_reduce_max(self, val: cutlass.Float32) -> cutlass.Float32:
        return self._threadquad_reduce(val, lambda x, y: cute.arch.fmax(x, y))

    def _threadquad_reduce_sum(self, val: cutlass.Float32) -> cutlass.Float32:
        return self._threadquad_reduce(val, lambda x, y: x + y)

    @cute.jit
    def normalize_softmax(self, acc_O: cute.Tensor, row_sum: cute.Tensor):
        acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
        for r in cutlass.range_constexpr(cute.size(row_sum)):
            row_sum[r] = self._threadquad_reduce_sum(row_sum[r])
            acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
            scale = 1.0 if acc_O_mn_row_is_zero_or_nan else cute.arch.rcp_approx(row_sum[r])
            acc_O_mn[r, None] = acc_O_mn[r, None].load() * scale

    def _make_smem_layouts(self):
        smem_k_block_size = 64 if self.head_dim_padded % 64 == 0 else 32
        swizzle_bits = 3 if smem_k_block_size == 64 else 2
        sQ_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size), stride=(smem_k_block_size, 1)),
        )
        sQ_layout = cute.tile_to_shape(sQ_layout_atom, (self.tile_m, self.head_dim_padded), (0, 1))
        sK_layout = cute.tile_to_shape(sQ_layout_atom, (self.tile_n, self.head_dim_padded), (0, 1))
        sV_layout = cute.tile_to_shape(sQ_layout_atom, (self.tile_n, self.head_dim_padded), (0, 1))
        sO_layout = sQ_layout
        return sQ_layout_atom, sQ_layout, sK_layout, sV_layout, sO_layout

    def _make_gmem_tiled_copies(self, sQ_layout_atom):
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype.width
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        t_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        t_layout = cute.make_ordered_layout(
            (self.num_threads // t_shape_dim_1, t_shape_dim_1),
            order=(1, 0),
        )
        v_layout = cute.make_layout((1, async_copy_elems))
        return (
            cute.make_tiled_copy_tv(atom_async_copy, t_layout, v_layout),
            cute.make_tiled_copy_tv(atom_universal_copy, t_layout, v_layout),
        )

    def _make_tiled_mma(self):
        return cute.make_tiled_mma(
            warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16)),
            (self.num_warps, 1, 1),
            permutation_mnk=(self.num_warps * 16, 16, 16),
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        softmax_scale: Float32,
        max_seqlen: Int32,
        num_seqs: Int32,
        mCuSeqlens: cute.Tensor,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(not (mQ.element_type == mK.element_type == mV.element_type == mO.element_type)):
            raise TypeError("All tensors must have the same dtype")
        if cutlass.const_expr(mQ.element_type != cutlass.BFloat16):
            raise TypeError("This kernel is BF16-only")
        mQ = assume_tensor_128b_aligned(mQ)
        mK = assume_tensor_128b_aligned(mK)
        mV = assume_tensor_128b_aligned(mV)
        mO = assume_tensor_128b_aligned(mO)

        sQ_layout_atom, sQ_layout, sK_layout, sV_layout, sO_layout = self._make_smem_layouts()
        gmem_tiled_copy_QKV, gmem_tiled_copy_O = self._make_gmem_tiled_copies(sQ_layout_atom)
        tiled_mma = self._make_tiled_mma()

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(sQ_layout)], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(sK_layout)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(sV_layout)], 1024]

        grid_dim = (cute.ceil_div(max_seqlen, self.tile_m), num_seqs, cute.size(mQ.shape[1]))
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mCuSeqlens,
            Float32(softmax_scale * LOG2_E),
            sQ_layout,
            sK_layout,
            sV_layout,
            sO_layout,
            gmem_tiled_copy_QKV,
            gmem_tiled_copy_O,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mCuSeqlens: cute.Tensor,
        softmax_scale_log2: Float32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_QKV: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, seq_idx, q_head = cute.arch.block_idx()

        q_start = mCuSeqlens[seq_idx]
        q_end = mCuSeqlens[seq_idx + 1]
        seqlen_q = q_end - q_start
        if m_block * self.tile_m < seqlen_q:
            num_q_heads = cute.size(mQ.shape[1])
            num_kv_heads = cute.size(mK.shape[1])
            qhead_per_kvhead = num_q_heads // num_kv_heads
            kv_head = q_head // qhead_per_kvhead

            n_block_max = cute.ceil_div(seqlen_q, self.tile_n)
            if self.is_causal:
                n_block_max = min(cute.ceil_div((m_block + 1) * self.tile_m, self.tile_n), n_block_max)
            n_block = n_block_max - 1

            q_seq = cute.domain_offset((q_start, 0), mQ[None, q_head, None])
            k_seq = cute.domain_offset((q_start, 0), mK[None, kv_head, None])
            v_seq = cute.domain_offset((q_start, 0), mV[None, kv_head, None])
            o_seq = cute.domain_offset((q_start, 0), mO[None, q_head, None])

            gQ = cute.local_tile(q_seq, (self.tile_m, self.head_dim_padded), (m_block, 0))
            gK = cute.local_tile(k_seq, (self.tile_n, self.head_dim_padded), (None, 0))
            gV = cute.local_tile(v_seq, (self.tile_n, self.head_dim_padded), (None, 0))

            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            sQ = storage.sQ.get_tensor(sQ_layout)
            sK = storage.sK.get_tensor(sK_layout)
            sV = storage.sV.get_tensor(sV_layout)
            sVt = cute.composition(
                sV,
                cute.make_layout((self.head_dim_padded, self.tile_n), stride=(self.tile_n, 1)),
            )

            gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(tidx)
            tQgQ = gmem_thr_copy_QKV.partition_S(gQ)
            tQsQ = gmem_thr_copy_QKV.partition_D(sQ)
            tKgK = gmem_thr_copy_QKV.partition_S(gK)
            tKsK = gmem_thr_copy_QKV.partition_D(sK)
            tVgV = gmem_thr_copy_QKV.partition_S(gV)
            tVsV = gmem_thr_copy_QKV.partition_D(sV)

            thr_mma = tiled_mma.get_slice(tidx)
            tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
            tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
            tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
            acc_shape_O = thr_mma.partition_shape_C((self.tile_m, self.head_dim_padded))
            acc_O = cute.make_fragment(acc_shape_O, Float32)
            acc_O.fill(0.0)

            smem_copy_atom_Q = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                self.dtype,
            )
            smem_copy_atom_K = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                self.dtype,
            )
            smem_copy_atom_V = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
                self.dtype,
            )
            smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q, tiled_mma)
            smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_K, tiled_mma)
            smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)
            smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
            smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
            smem_thr_copy_V = smem_tiled_copy_V.get_slice(tidx)
            tSsQ = smem_thr_copy_Q.partition_S(sQ)
            tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
            tSsK = smem_thr_copy_K.partition_S(sK)
            tSrK_copy_view = smem_thr_copy_K.retile(tSrK)
            tOsVt = smem_thr_copy_V.partition_S(sVt)
            tOrVt_copy_view = smem_thr_copy_V.retile(tOrVt)

            mcQ = cute.make_identity_tensor((seqlen_q, self.head_dim_padded))
            mcKV = cute.make_identity_tensor((seqlen_q, self.head_dim_padded))
            cQ = cute.local_tile(mcQ, (self.tile_m, self.head_dim_padded), (m_block, 0))
            cKV = cute.local_tile(mcKV, (self.tile_n, self.head_dim_padded), (n_block, 0))
            tQcQ = gmem_thr_copy_QKV.partition_S(cQ)
            tKVcKV = gmem_thr_copy_QKV.partition_S(cKV)

            tQpQ = cute.make_fragment(
                cute.make_layout(
                    (tQsQ.shape[0][1], cute.size(tQsQ, mode=[1]), cute.size(tQsQ, mode=[2])),
                    stride=(cute.size(tQsQ, mode=[2]), 0, 1),
                ),
                cutlass.Boolean,
            )
            tKVpKV = cute.make_fragment(
                cute.make_layout(
                    (tKsK.shape[0][1], cute.size(tKsK, mode=[1]), cute.size(tKsK, mode=[2])),
                    stride=(cute.size(tKsK, mode=[2]), 0, 1),
                ),
                cutlass.Boolean,
            )
            for rest_v in cutlass.range_constexpr(tQpQ.shape[0]):
                for rest_k in cutlass.range_constexpr(tQpQ.shape[2]):
                    tQpQ[rest_v, 0, rest_k] = cute.elem_less(tQcQ[(0, rest_v), 0, rest_k][1], self.head_dim)
            for rest_v in cutlass.range_constexpr(tKVpKV.shape[0]):
                for rest_k in cutlass.range_constexpr(tKVpKV.shape[2]):
                    tKVpKV[rest_v, 0, rest_k] = cute.elem_less(tKVcKV[(0, rest_v), 0, rest_k][1], self.head_dim)

            for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
                if cute.elem_less(tQcQ[0, m, 0][0], seqlen_q):
                    cute.copy(gmem_tiled_copy_QKV, tQgQ[None, m, None], tQsQ[None, m, None], pred=tQpQ[None, m, None])
                else:
                    tQsQ[None, m, None].fill(0)
            for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                if cute.elem_less(tKVcKV[0, n, 0][0], seqlen_q):
                    cute.copy(gmem_tiled_copy_QKV, tKgK[None, n, None, n_block], tKsK[None, n, None], pred=tKVpKV[None, n, None])
                else:
                    tKsK[None, n, None].fill(0)
            cute.arch.cp_async_commit_group()

            row_max = cute.make_fragment((acc_O.shape[0][0] * acc_O.shape[1]), Float32)
            row_sum = cute.make_fragment((acc_O.shape[0][0] * acc_O.shape[1]), Float32)
            row_max.fill(-Float32.inf)
            row_sum.fill(0.0)

            params = SimpleNamespace(
                seqlen_q=seqlen_q,
                m_block=m_block,
                n_block=n_block,
                tiled_mma=tiled_mma,
                thr_mma=thr_mma,
                tSrQ=tSrQ,
                tSrK=tSrK,
                tOrVt=tOrVt,
                acc_O=acc_O,
                gmem_tiled_copy_QKV=gmem_tiled_copy_QKV,
                tKgK=tKgK,
                tKsK=tKsK,
                tVgV=tVgV,
                tVsV=tVsV,
                tKVcKV=tKVcKV,
                tKVpKV=tKVpKV,
                smem_tiled_copy_Q=smem_tiled_copy_Q,
                smem_tiled_copy_K=smem_tiled_copy_K,
                smem_tiled_copy_V=smem_tiled_copy_V,
                tSsQ=tSsQ,
                tSrQ_copy_view=tSrQ_copy_view,
                tSsK=tSsK,
                tSrK_copy_view=tSrK_copy_view,
                tOsVt=tOsVt,
                tOrVt_copy_view=tOrVt_copy_view,
                row_max=row_max,
                row_sum=row_sum,
                softmax_scale_log2=softmax_scale_log2,
            )

            mask_steps = cute.ceil_div(self.tile_m, self.tile_n) if self.is_causal else 1
            for n_tile in cutlass.range_constexpr(mask_steps):
                n_block = n_block_max - n_tile - 1
                params.n_block = n_block
                if (not self.is_causal) or n_block >= 0:
                    self.compute_one_n_block(params, is_first_n_block=(n_tile == 0), in_mask_steps=True)
            for n_tile in range(mask_steps, n_block_max, 1):
                n_block = n_block_max - n_tile - 1
                params.n_block = n_block
                self.compute_one_n_block(params, is_first_n_block=False, in_mask_steps=(True if self.is_causal else False))

            self.normalize_softmax(acc_O, row_sum)

            rO = cute.make_fragment_like(acc_O, self.dtype)
            rO.store(acc_O.load().to(self.dtype))
            sO = cute.make_tensor(sQ.iterator, sO_layout)
            smem_copy_atom_O = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dtype)
            smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma)
            smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx)
            taccOrO = smem_thr_copy_O.retile(rO)
            taccOsO = smem_thr_copy_O.partition_D(sO)
            cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

            gO = cute.local_tile(o_seq, (self.tile_m, self.head_dim_padded), (m_block, 0))
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            tOsO = gmem_thr_copy_O.partition_S(sO)
            tOgO = gmem_thr_copy_O.partition_D(gO)
            tOrO = cute.make_fragment_like(tOgO, self.dtype)
            cute.arch.barrier()
            cute.copy(gmem_tiled_copy_O, tOsO, tOrO)

            cO = cute.local_tile(cute.make_identity_tensor((seqlen_q, self.head_dim_padded)), (self.tile_m, self.head_dim_padded), (m_block, 0))
            tOcO = gmem_thr_copy_O.partition_D(cO)
            tOpO = cute.make_fragment(
                cute.make_layout((tOgO.shape[0][1], tOgO.shape[1], tOgO.shape[2]), stride=(tOgO.shape[2], 0, 1)),
                cutlass.Boolean,
            )
            for rest_v in cutlass.range_constexpr(tOpO.shape[0]):
                for rest_n in cutlass.range_constexpr(cute.size(tOpO.shape[2])):
                    tOpO[rest_v, 0, rest_n] = cute.elem_less(tOcO[(0, rest_v), 0, rest_n][1], self.head_dim)
            for rest_m in cutlass.range_constexpr(cute.size(tOpO.shape[1])):
                if cute.elem_less(tOcO[0, rest_m, 0][0], seqlen_q):
                    cute.copy(gmem_tiled_copy_O, tOrO[None, rest_m, None], tOgO[None, rest_m, None], pred=tOpO[None, rest_m, None])

    @cute.jit
    def compute_one_n_block(self, params: SimpleNamespace, is_first_n_block: cutlass.Constexpr, in_mask_steps: cutlass.Constexpr):
        acc_shape_S = params.thr_mma.partition_shape_C((self.tile_m, self.tile_n))
        acc_S = cute.make_fragment(acc_shape_S, Float32)
        acc_S.fill(0.0)

        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        if is_first_n_block:
            for n in cutlass.range_constexpr(cute.size(params.tVsV.shape[1])):
                if cute.elem_less(params.tKVcKV[0, n, 0][0], params.seqlen_q):
                    cute.copy(
                        params.gmem_tiled_copy_QKV,
                        params.tVgV[None, n, None, params.n_block],
                        params.tVsV[None, n, None],
                        pred=params.tKVpKV[None, n, None],
                    )
                else:
                    params.tVsV[None, n, None].fill(0.0)
        else:
            cute.copy(
                params.gmem_tiled_copy_QKV,
                params.tVgV[None, None, None, params.n_block],
                params.tVsV,
                pred=params.tKVpKV,
            )
        cute.arch.cp_async_commit_group()

        cute.copy(params.smem_tiled_copy_Q, params.tSsQ[None, None, 0], params.tSrQ_copy_view[None, None, 0])
        cute.copy(params.smem_tiled_copy_K, params.tSsK[None, None, 0], params.tSrK_copy_view[None, None, 0])
        for k in cutlass.range_constexpr(cute.size(params.tSsQ.shape[2])):
            k_next = (k + 1) % cute.size(params.tSsQ.shape[2])
            cute.copy(params.smem_tiled_copy_Q, params.tSsQ[None, None, k_next], params.tSrQ_copy_view[None, None, k_next])
            cute.copy(params.smem_tiled_copy_K, params.tSsK[None, None, k_next], params.tSrK_copy_view[None, None, k_next])
            cute.gemm(params.tiled_mma, acc_S, params.tSrQ[None, None, k], params.tSrK[None, None, k], acc_S)

        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        if params.n_block > 0:
            cute.copy(
                params.gmem_tiled_copy_QKV,
                params.tKgK[None, None, None, params.n_block - 1],
                params.tKsK,
                pred=params.tKVpKV,
            )
            cute.arch.cp_async_commit_group()

        self.softmax_rescale_O(params, acc_S, is_first_n_block, in_mask_steps)

        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        rP_layout_divided = cute.logical_divide(rP.layout, (None, None, 2))
        rP_mma_view = cute.make_layout(
            ((rP_layout_divided.shape[0], rP_layout_divided.shape[2][0]), rP_layout_divided.shape[1], rP_layout_divided.shape[2][1]),
            stride=((rP_layout_divided.stride[0], rP_layout_divided.stride[2][0]), rP_layout_divided.stride[1], rP_layout_divided.stride[2][1]),
        )
        tOrS = cute.make_tensor(rP.iterator, rP_mma_view)
        cute.copy(params.smem_tiled_copy_V, params.tOsVt[None, None, 0], params.tOrVt_copy_view[None, None, 0])
        for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
            k_next = (k + 1) % cute.size(tOrS.shape[2])
            cute.copy(params.smem_tiled_copy_V, params.tOsVt[None, None, k_next], params.tOrVt_copy_view[None, None, k_next])
            cute.gemm(params.tiled_mma, params.acc_O, tOrS[None, None, k], params.tOrVt[None, None, k], params.acc_O)

    @cute.jit
    def apply_score_mask(self, params: SimpleNamespace, acc_S: cute.Tensor):
        acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        tScS_mn = layout_utils.reshape_acc_to_mn(params.thr_mma.partition_C(cS))
        t0ScS_mn = layout_utils.reshape_acc_to_mn(params.thr_mma.get_slice(0).partition_C(cS))
        thr_col_offset = tScS_mn[0][1]
        seqlen_col_limit = params.seqlen_q - params.n_block * self.tile_n - thr_col_offset

        if cutlass.const_expr(self.is_causal):
            causal_row_offset = 1 - params.n_block * self.tile_n - thr_col_offset
            for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                row_idx = tScS_mn[r, 0][0] + params.m_block * self.tile_m
                col_limit_right = cutlass.min(row_idx + causal_row_offset, seqlen_col_limit)
                for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                    acc_S_mn[r, c] = (
                        -Float32.inf
                        if t0ScS_mn[0, c][1] >= col_limit_right
                        else acc_S_mn[r, c]
                    )
        else:
            for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                oob = t0ScS_mn[0, c][1] >= seqlen_col_limit
                for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                    acc_S_mn[r, c] = -Float32.inf if oob else acc_S_mn[r, c]

    @cute.jit
    def softmax_rescale_O(self, params: SimpleNamespace, acc_S: cute.Tensor, is_first_n_block: cutlass.Constexpr, in_mask_steps: cutlass.Constexpr):
        if in_mask_steps:
            self.apply_score_mask(params, acc_S)
        acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
        acc_O_mn = self._make_acc_tensor_mn_view(params.acc_O)
        row_max_prev = None
        if cutlass.const_expr(not is_first_n_block):
            row_max_prev = cute.make_fragment_like(params.row_max, Float32)
            cute.basic_copy(params.row_max, row_max_prev)
        for r in cutlass.range_constexpr(cute.size(params.row_max)):
            acc_S_row = acc_S_mn[r, None].load()
            row_max_cur_row = acc_S_row.reduce(cute.ReductionOp.MAX, -Float32.inf, 0)
            row_max_cur_row = self._threadquad_reduce_max(row_max_cur_row)
            row_max_prev_row = None
            if cutlass.const_expr(not is_first_n_block):
                row_max_prev_row = row_max_prev[r]
                row_max_cur_row = cute.arch.fmax(row_max_prev_row, row_max_cur_row)
            if cutlass.const_expr(self.is_causal):
                row_max_cur_row = 0.0 if row_max_cur_row == -Float32.inf else row_max_cur_row
            acc_S_row_exp = cute.math.exp2(
                acc_S_row * params.softmax_scale_log2 - row_max_cur_row * params.softmax_scale_log2,
                fastmath=True,
            )
            acc_S_row_sum = acc_S_row_exp.reduce(cute.ReductionOp.ADD, Float32.zero, 0)
            if cutlass.const_expr(not is_first_n_block):
                prev_minus_cur_exp = cute.math.exp2(
                    row_max_prev_row * params.softmax_scale_log2 - row_max_cur_row * params.softmax_scale_log2,
                    fastmath=True,
                )
                acc_S_row_sum = acc_S_row_sum + params.row_sum[r] * prev_minus_cur_exp
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * prev_minus_cur_exp
            params.row_max[r] = row_max_cur_row
            params.row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None] = acc_S_row_exp


@cute.jit
def fa2_enc_varlen_fwd(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mO: cute.Tensor,
    softmax_scale: Float32,
    max_seqlen: Int32,
    num_seqs: Int32,
    mCuSeqlens: cute.Tensor,
    stream: cuda.CUstream,
):
    FlashAttention2VarlenSm80(head_dim=64, is_causal=False, num_warps=4, tile_m=128)(
        mQ, mK, mV, mO, softmax_scale, max_seqlen, num_seqs, mCuSeqlens, stream
    )


@cute.jit
def fa2_dec_varlen_fwd(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mO: cute.Tensor,
    softmax_scale: Float32,
    max_seqlen: Int32,
    num_seqs: Int32,
    mCuSeqlens: cute.Tensor,
    stream: cuda.CUstream,
):
    FlashAttention2VarlenSm80(head_dim=128, is_causal=True, num_warps=4, tile_n=96)(
        mQ, mK, mV, mO, softmax_scale, max_seqlen, num_seqs, mCuSeqlens, stream
    )


def _current_stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def _compile_and_export(
    fn,
    file_name: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    max_seqlen: int,
    num_seqs: int,
    cu_seqlens: torch.Tensor,
):
    compiled = cute.compile(
        fn,
        to_cute_tensor(q),
        to_cute_tensor(k),
        to_cute_tensor(v),
        to_cute_tensor(o),
        Float32(1.0 / math.sqrt(q.shape[-1])),
        Int32(max_seqlen),
        Int32(num_seqs),
        to_cute_tensor(cu_seqlens),
        _current_stream(),
        options="--gpu-arch sm_120 --enable-tvm-ffi",
    )
    compiled.export_to_c(str(OUT_DIR / f"{file_name}.o"), function_name=file_name)


def compile_exports():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to compile/export the CuTe DSL kernels")

    device = torch.device("cuda")

    enc_num_seqs = 2
    enc_cu = torch.tensor([0, 64, 128], dtype=torch.int32, device=device)
    enc_q = torch.empty((128, 20, 64), dtype=torch.bfloat16, device=device)
    enc_k = torch.empty((128, 20, 64), dtype=torch.bfloat16, device=device)
    enc_v = torch.empty((128, 20, 64), dtype=torch.bfloat16, device=device)
    enc_o = torch.empty_like(enc_q)
    _compile_and_export(
        fa2_enc_varlen_fwd,
        "fa2_enc_varlen_fwd",
        enc_q,
        enc_k,
        enc_v,
        enc_o,
        64,
        enc_num_seqs,
        enc_cu,
    )

    dec_num_seqs = 2
    dec_cu = torch.tensor([0, 64, 128], dtype=torch.int32, device=device)
    dec_q = torch.empty((128, 16, 128), dtype=torch.bfloat16, device=device)
    dec_k = torch.empty((128, 4, 128), dtype=torch.bfloat16, device=device)
    dec_v = torch.empty((128, 4, 128), dtype=torch.bfloat16, device=device)
    dec_o = torch.empty_like(dec_q)
    _compile_and_export(
        fa2_dec_varlen_fwd,
        "fa2_dec_varlen_fwd",
        dec_q,
        dec_k,
        dec_v,
        dec_o,
        64,
        dec_num_seqs,
        dec_cu,
    )


if __name__ == "__main__":
    compile_exports()
