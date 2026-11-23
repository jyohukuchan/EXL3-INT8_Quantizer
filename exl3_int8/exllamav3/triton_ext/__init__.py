import torch
import triton
import triton.language as tl


@triton.jit
def _decode_kernel(indices, output, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    state = tl.load(indices + offsets, offsets < n, other=0).to(tl.uint16).to(tl.uint32)
    product = state * 0x00DCD12D
    total = ((product & 255) + ((product >> 8) & 255) +
             ((product >> 16) & 255) + ((product >> 24) & 255))
    value = (total + 0x6400).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    scale = tl.full((), 0x1eee, tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    bias = tl.full((), 0xc931, tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    result = tl.fma(value, scale, bias).to(tl.float16).to(tl.float32)
    tl.store(output + offsets, result, offsets < n)


def exl3_mul1_multiplier():
    return 0x00DCD12D


def decode(input_indices, output_tiles, mcg, mul1):
    if mcg or not mul1:
        raise ValueError('mul24 のみ対応しています')
    if (input_indices.dtype not in (torch.int16, torch.uint16) or output_tiles.dtype not in (torch.float16, torch.float32)
            or input_indices.device != output_tiles.device or not input_indices.is_cuda
            or not input_indices.is_contiguous() or not output_tiles.is_contiguous()
            or input_indices.numel() != output_tiles.numel()):
        raise ValueError('入力と出力の dtype、device、サイズ、連続性が不正です')
    n = input_indices.numel()
    if n:
        _decode_kernel[(triton.cdiv(n, 256),)](input_indices, output_tiles, n, 256)


@triton.jit
def _pack_kernel(packed, unpacked, count: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tile = idx // (16 * K)
    word = (idx % (16 * K)) ^ 1
    result = tl.full((BLOCK,), 0, tl.uint32)
    for bit in range(16):
        pos = word * 16 + bit
        value = tl.load(unpacked + tile * 256 + pos // K, idx < count, other=0).to(tl.uint16).to(tl.uint32)
        result = (result << 1) | ((value >> (K - 1 - pos % K)) & 1)
    tl.store(packed + idx, result.to(tl.uint16), idx < count)


@triton.jit
def _unpack_kernel(unpacked, packed, count: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tile = idx // 256
    pos = ((idx % 256) * K + K - 16 + 256 * K) % (256 * K)
    result = tl.full((BLOCK,), 0, tl.uint32)
    for bit in range(16):
        at = (pos + bit) % (256 * K)
        value = tl.load(packed + tile * (16 * K) + ((at // 16) ^ 1), idx < count, other=0).to(tl.uint16).to(tl.uint32)
        result = (result << 1) | ((value >> (15 - at % 16)) & 1)
    tl.store(unpacked + idx, result.to(tl.uint16), idx < count)


@triton.jit
def _pack_signs_kernel(packed, unpacked, count: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    result = tl.full((BLOCK,), 0, tl.uint32)
    for bit in range(16):
        value = tl.load(unpacked + idx * 16 + bit, idx < count, other=0).to(tl.uint16, bitcast=True).to(tl.uint32)
        result |= ((value >> 15) & 1) << bit
    tl.store(packed + idx, result.to(tl.uint16), idx < count)


def _check_trellis(packed, unpacked, k):
    if (not 1 <= k <= 8 or packed.dtype not in (torch.int16, torch.uint16)
            or unpacked.dtype not in (torch.int16, torch.uint16)
            or packed.ndim != 3 or unpacked.ndim != 3
            or packed.shape[:2] != unpacked.shape[:2]
            or packed.shape[2] != 16 * k or unpacked.shape[2] != 256
            or packed.device != unpacked.device or not packed.is_cuda
            or not packed.is_contiguous() or not unpacked.is_contiguous()):
        raise ValueError('trellis の形状、dtype、device が不正です')


def pack_trellis(packed, unpacked, K):
    _check_trellis(packed, unpacked, K)
    count = packed.numel()
    if count:
        _pack_kernel[(triton.cdiv(count, 128),)](packed, unpacked, count, K, 128)


def unpack_trellis(unpacked, packed, K):
    _check_trellis(packed, unpacked, K)
    count = unpacked.numel()
    if count:
        _unpack_kernel[(triton.cdiv(count, 128),)](unpacked, packed, count, K, 128)


def pack_signs(packed, unpacked):
    if (packed.dtype not in (torch.int16, torch.uint16) or unpacked.dtype != torch.float16
            or packed.numel() * 16 != unpacked.numel() or packed.device != unpacked.device
            or not packed.is_cuda or not packed.is_contiguous() or not unpacked.is_contiguous()):
        raise ValueError('符号の入力と出力が不正です')
    count = packed.numel()
    if count:
        _pack_signs_kernel[(triton.cdiv(count, 128),)](packed, unpacked, count, 128)


@triton.jit
def _had_kernel(source, target, scales, COLS: tl.constexpr, SCALE: tl.constexpr,
                PRE: tl.constexpr, POST: tl.constexpr, HALF: tl.constexpr):
    block = tl.program_id(0)
    index = tl.arange(0, 128)
    offset = block * 128 + index
    value = tl.load(source + offset).to(tl.float32)
    if PRE:
        factor = tl.load(scales + (offset % COLS)).to(tl.float32)
        value = value * factor
        if HALF:
            value = value.to(tl.float16).to(tl.float32)
    for bit in tl.static_range(7):
        partner = tl.gather(value, index ^ (1 << bit), axis=0)
        value = tl.where((index & (1 << bit)) != 0, partner - value, value + partner)
    value = value * (SCALE * 0.088388347648)
    if HALF:
        value = value.to(tl.float16).to(tl.float32)
    if POST:
        value = value * tl.load(scales + (offset % COLS)).to(tl.float32)
    tl.store(target + offset, value)


def had_r_128(input, output, pre_scale, post_scale, scale):
    if (input.ndim != 2 or input.shape[1] % 128 or input.shape != output.shape
            or input.dtype not in (torch.float16, torch.float32) or output.dtype != input.dtype
            or not input.is_cuda or input.device != output.device
            or not input.is_contiguous() or not output.is_contiguous()):
        raise ValueError('Hadamard の入力と出力が不正です')
    factor = pre_scale if pre_scale is not None else post_scale
    if factor is not None and (factor.numel() != input.shape[1] or factor.dtype != torch.float16
                               or factor.device != input.device or not factor.is_contiguous()):
        raise ValueError('Hadamard のスケールが不正です')
    if input.numel():
        _had_kernel[(input.numel() // 128,)](input, output, factor if factor is not None else input,
                                            input.shape[1], scale, pre_scale is not None,
                                            pre_scale is None and post_scale is not None,
                                            input.dtype == torch.float16)


@triton.jit
def _reconstruct_kernel(output, packed, COUNT: tl.constexpr, COLS: tl.constexpr,
                        PACKED_COLS: tl.constexpr, N_OFFSET: tl.constexpr,
                        K: tl.constexpr, BLOCK: tl.constexpr, BATCH: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    batch = tl.program_id(1)
    if BATCH:
        address = tl.load(packed + batch)
        packed = address.to(tl.pointer_type(tl.int16))
        output += batch * COUNT
    row = idx // COLS
    col = idx % COLS
    tile = (row // 16) * PACKED_COLS + (col + N_OFFSET) // 16
    stream = (row % 16 // 8) * 128 + (col % 16) * 8 + row % 8
    pos = (stream * K + K - 16 + 256 * K) % (256 * K)
    state = tl.full((BLOCK,), 0, tl.uint32)
    for bit in range(16):
        at = (pos + bit) % (256 * K)
        word = tl.load(packed + tile * (16 * K) + ((at // 16) ^ 1), idx < COUNT, other=0).to(tl.uint16).to(tl.uint32)
        state = (state << 1) | ((word >> (15 - at % 16)) & 1)
    product = state * 0x00DCD12D
    total = ((product & 255) + ((product >> 8) & 255) +
             ((product >> 16) & 255) + ((product >> 24) & 255))
    value = (total + 0x6400).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    scale = tl.full((), 0x1eee, tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    bias = tl.full((), 0xc931, tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    tl.store(output + idx, tl.fma(value, scale, bias).to(tl.float16), idx < COUNT)


def reconstruct_slice(unpacked, packed, K, mcg, mul1, n_offset):
    if (not 1 <= K <= 8 or mcg or not mul1 or unpacked.ndim != 2 or packed.ndim != 3
            or unpacked.dtype != torch.float16 or packed.dtype not in (torch.int16, torch.uint16)
            or unpacked.shape[0] != packed.shape[0] * 16 or packed.shape[2] != 16 * K
            or unpacked.shape[1] % 128 or n_offset < 0 or n_offset % 128
            or n_offset + unpacked.shape[1] > packed.shape[1] * 16
            or not unpacked.is_cuda or unpacked.device != packed.device
            or not unpacked.is_contiguous() or not packed.is_contiguous()):
        raise ValueError('reconstruct の引数が不正です')
    count = unpacked.numel()
    if count:
        _reconstruct_kernel[(triton.cdiv(count, 256),)](unpacked, packed, count, unpacked.shape[1],
                                                        packed.shape[1], n_offset, K, 256, False)


def reconstruct(unpacked, packed, K, mcg, mul1):
    if unpacked.ndim != 2 or packed.ndim != 3 or unpacked.shape[1] != packed.shape[1] * 16:
        raise ValueError('reconstruct の形状が不正です')
    reconstruct_slice(unpacked, packed, K, mcg, mul1, 0)

from .viterbi import quantize_tiles, quantize_tiles_scratch


def had_paley(h):
    if h.ndim != 2 or h.shape[0] != h.shape[1] or h.dtype != torch.float16 or h.device.type != 'cpu' or not h.is_contiguous():
        raise ValueError('Paley 行列の出力が不正です')
    n = h.shape[0]
    p = n - 1
    if p < 2:
        raise ValueError('Paley 行列の次数が不正です')
    h[0, :] = 1
    h[1:, 0] = -1
    for i in range(p):
        for j in range(p):
            h[i + 1, j + 1] = 1 if i == j or pow((i - j) % p, (p - 1) // 2, p) == 1 else -1


def had_paley2(h):
    if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] % 2 or h.dtype != torch.float16 or h.device.type != 'cpu' or not h.is_contiguous():
        raise ValueError('Paley 行列の出力が不正です')
    n = h.shape[0]
    p = n // 2 - 1
    if p < 2:
        raise ValueError('Paley 行列の次数が不正です')
    for i in range(n // 2):
        for j in range(n // 2):
            if i == j:
                a, b, c, d = 1, -1, -1, -1
            elif i == 0 or j == 0 or pow((i - j) % p, (p - 1) // 2, p) == 1:
                a, b, c, d = 1, 1, 1, -1
            else:
                a, b, c, d = -1, -1, -1, 1
            h[2 * i, 2 * j] = a
            h[2 * i, 2 * j + 1] = b
            h[2 * i + 1, 2 * j] = c
            h[2 * i + 1, 2 * j + 1] = d


def had_r_128_batch(input, output, pre_table, post_table, ids, rows_per_expert, scale):
    if (input.ndim != 2 or input.shape != output.shape or rows_per_expert <= 0
            or input.shape[0] % rows_per_expert or (pre_table is None) == (post_table is None)
            or ids.dtype != torch.int64 or ids.numel() < input.shape[0] // rows_per_expert
            or ids.device != input.device or not ids.is_contiguous()):
        raise ValueError('Hadamard batch の引数が不正です')
    table = pre_table if pre_table is not None else post_table
    if table.ndim != 2 or table.shape[1] != input.shape[1] or table.dtype != torch.float16 or table.device != input.device or not table.is_contiguous():
        raise ValueError('Hadamard batch のテーブルが不正です')
    for expert, idx in enumerate(ids[:input.shape[0] // rows_per_expert].tolist()):
        if not 0 <= idx < table.shape[0]:
            raise ValueError('Hadamard batch の expert ID が不正です')
        rows = slice(expert * rows_per_expert, (expert + 1) * rows_per_expert)
        had_r_128(input[rows], output[rows], table[idx] if pre_table is not None else None,
                  table[idx] if post_table is not None else None, scale)


def test_distribution(input, dist_output, ref_output, min_value, max_value, mcg, mul1):
    if (input.dtype != torch.float32 or dist_output.dtype != torch.float32 or input.numel() == 0
            or min_value >= max_value or dist_output.numel() == 0 or mcg or not mul1
            or input.device != dist_output.device or not input.is_cuda):
        raise ValueError('分布の引数が不正です')
    def histogram(values, out):
        bins = ((values - min_value) / (max_value - min_value) * out.numel()).to(torch.int64).clamp_(0, out.numel() - 1)
        out.copy_(torch.bincount(bins.reshape(-1), minlength=out.numel()).to(torch.float32) / values.numel())
    histogram(input, dist_output)
    if ref_output is not None:
        if ref_output.shape != dist_output.shape or ref_output.dtype != torch.float32 or ref_output.device != input.device:
            raise ValueError('参照分布の出力が不正です')
        indices = torch.arange(65536, device=input.device, dtype=torch.int32).to(torch.uint16)
        values = torch.empty(65536, device=input.device, dtype=torch.float32)
        decode(indices, values, False, True)
        histogram(values, ref_output)


def reconstruct_batch(unpacked, packed_ptrs, K, mcg, mul1):
    if (unpacked.ndim != 3 or unpacked.dtype != torch.float16 or not unpacked.is_cuda
            or not unpacked.is_contiguous() or unpacked.shape[1] % 16 or unpacked.shape[2] % 128
            or packed_ptrs.dtype != torch.int64 or packed_ptrs.device != unpacked.device
            or not packed_ptrs.is_contiguous() or packed_ptrs.numel() != unpacked.shape[0]
            or not 1 <= K <= 8 or mcg or not mul1):
        raise ValueError('reconstruct batch の引数が不正です')
    count = unpacked.shape[1] * unpacked.shape[2]
    if count and unpacked.shape[0]:
        _reconstruct_kernel[(triton.cdiv(count, 256), unpacked.shape[0])](
            unpacked, packed_ptrs, count, unpacked.shape[2], unpacked.shape[2] // 16, 0, K, 256, True)


def reconstruct_had_slice(unpacked, packed, suh, svh, K, mcg, mul1, n_offset):
    rows, cols = unpacked.shape
    if (rows % 128 or suh.numel() < rows or svh.numel() < cols
            or suh.dtype != torch.float16 or svh.dtype != torch.float16
            or suh.device != unpacked.device or svh.device != unpacked.device):
        raise ValueError('reconstruct Hadamard のスケールが不正です')
    reconstructed = torch.empty_like(unpacked)
    reconstruct_slice(reconstructed, packed, K, mcg, mul1, n_offset)
    right = torch.empty_like(unpacked)
    had_r_128(reconstructed, right, None, svh[:cols], 1.0)
    transposed = right.T.contiguous()
    transformed = torch.empty_like(transposed)
    had_r_128(transposed, transformed, None, suh[:rows], 1.0)
    unpacked.copy_(transformed.T)


@triton.jit
def _copy_pointer_rows(out, ptrs, LENGTH: tl.constexpr, BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    idx = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ptr = tl.load(ptrs + batch).to(tl.pointer_type(tl.float16))
    value = tl.load(ptr + idx, idx < LENGTH, other=0)
    tl.store(out + batch * LENGTH + idx, value, idx < LENGTH)


def reconstruct_had_batch(unpacked, packed_ptrs, suh_ptrs, svh_ptrs, K, mcg, mul1):
    if unpacked.ndim != 3:
        raise ValueError('reconstruct Hadamard batch の形状が不正です')
    batch, rows, cols = unpacked.shape
    for ptrs in (suh_ptrs, svh_ptrs):
        if ptrs.dtype != torch.int64 or ptrs.device != unpacked.device or not ptrs.is_contiguous() or ptrs.numel() != batch:
            raise ValueError('reconstruct Hadamard batch のポインタが不正です')
    if rows % 128:
        raise ValueError('reconstruct Hadamard batch の行数が不正です')
    temp = torch.empty_like(unpacked)
    reconstruct_batch(temp, packed_ptrs, K, mcg, mul1)
    suh = torch.empty((batch, rows), dtype=torch.float16, device=unpacked.device)
    svh = torch.empty((batch, cols), dtype=torch.float16, device=unpacked.device)
    if batch and rows and cols:
        _copy_pointer_rows[(batch, triton.cdiv(rows, 256))](suh, suh_ptrs, rows, 256)
        _copy_pointer_rows[(batch, triton.cdiv(cols, 256))](svh, svh_ptrs, cols, 256)
        for i in range(batch):
            right = torch.empty_like(temp[i])
            had_r_128(temp[i], right, None, svh[i], 1.0)
            transposed = right.T.contiguous()
            transformed = torch.empty_like(transposed)
            had_r_128(transposed, transformed, None, suh[i], 1.0)
            unpacked[i].copy_(transformed.T)

from .ops import rms_norm, rms_norm_res_in, silu_mul
from .ops import (EXL3_MGEMV_MASKED_REDUCE_SUPPORTED, count_inf_nan,
                  exl3_gemv_int8_max_k, exl3_mgemv_masked_reduce_supported,
                  g_get_cc, g_get_num_sms, gated_rms_norm, hgemm,
                  hgemm_batched, hgemm_recon)
from .loader import (TensorLoadJob, stloader_close_file, stloader_deferred_cpu,
                     stloader_deferred_cuda, stloader_open_file, stloader_read)
from .ops import exl3_gemm, exl3_gemv
from .ops import (add, add_sigmoid_gate, add_sigmoid_gate_proj, deinterleave_qg,
                  gelu_mul, mul_sigmoid_, mul_sigmoid_broadcast_, mul_softplus_broadcast_,
                  relu_mul, relu2_mul, silu_oai_mul, softcap, xielu)
from .ops import rope
from .ops import BC_LinearEXL3, BC_LinearFP16
from .ops import BC_GatedMLP, BC_MLP
from .ops import (BC_GatedRMSNorm, gated_delta_net_fused_op_2, routing_ds3_nogroup,
                  routing_sel_norm, routing_std, routing_std_mapped)
from .ops import cuda_causal_conv1d_update
from .ops import blocksparse_mlp_routing, gr_mix, hc_apply, hc_head, hc_mix, hc_mix_num_chunks, ple_forward_streams, ple_gate
from .ops import BC_GatedDeltaNet, BC_GatedDeltaNetSplit
from .ops import cuda_recurrent_gated_delta_rule
from .moe import BC_BlockSparseMLP, exl3_mgemm, exl3_moe, exl3_moe_gather, exl3_moe_max_concurrency
