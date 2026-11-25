import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip('triton')

PATH = Path(__file__).resolve().parents[1] / 'exl3_int8/exllamav3/triton_ext/__init__.py'
spec = importlib.util.spec_from_file_location('exl3_triton_test', PATH)
backend = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = backend
spec.loader.exec_module(backend)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('index_dtype', [torch.int16, torch.uint16])
def test_decode_all_states(dtype, index_dtype):
    state = np.arange(65536, dtype=np.uint32)
    product = state * np.uint32(backend.exl3_mul1_multiplier())
    total = sum((product >> shift) & 255 for shift in (0, 8, 16, 24))
    values = (total + 0x6400).astype(np.uint16).view(np.float16).astype(np.float32)
    scale = np.array([0x1eee], dtype=np.uint16).view(np.float16).astype(np.float32)[0]
    bias = np.array([0xc931], dtype=np.uint16).view(np.float16).astype(np.float32)[0]
    expected = (values * scale + bias).astype(np.float16).astype(np.float32)
    indices = torch.from_numpy(state.astype(np.uint16)).cuda().view(index_dtype)
    output = torch.empty(65536, dtype=dtype, device='cuda')
    backend.decode(indices, output, False, True)
    np.testing.assert_array_equal(output.float().cpu().numpy(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
@pytest.mark.parametrize('k', range(1, 9))
def test_pack_trellis(k):
    rng = np.random.default_rng(k)
    values = rng.integers(0, 65536, size=(2, 3, 256), dtype=np.uint16)
    bits = ((values[..., :, None] >> np.arange(k - 1, -1, -1)) & 1).reshape(2, 3, 256 * k)
    words = (bits.reshape(2, 3, 16 * k, 16) * (1 << np.arange(15, -1, -1))).sum(axis=-1).astype(np.uint16)
    expected = words.reshape(2, 3, -1, 2)[..., ::-1].copy().reshape(2, 3, -1)
    src = torch.from_numpy(values).cuda().view(torch.int16)
    packed = torch.empty((2, 3, 16 * k), dtype=torch.int16, device='cuda')
    unpacked = torch.empty_like(src)
    backend.pack_trellis(packed, src, k)
    np.testing.assert_array_equal(packed.cpu().numpy().view(np.uint16), expected)
    backend.unpack_trellis(unpacked, packed, k)
    positions = (np.arange(256) * k + k - 16 + 256 * k) % (256 * k)
    expected_states = np.zeros((2, 3, 256), dtype=np.uint16)
    for bit in range(16):
        expected_states = (expected_states << 1) | bits[..., (positions + bit) % (256 * k)]
    np.testing.assert_array_equal(unpacked.cpu().numpy().view(np.uint16), expected_states)
    again = torch.empty_like(packed)
    backend.pack_trellis(again, unpacked, k)
    assert torch.equal(again, packed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
def test_pack_signs():
    values = torch.randn((7, 16), dtype=torch.float16, device='cuda')
    output = torch.empty(7, dtype=torch.int16, device='cuda')
    backend.pack_signs(output, values)
    expected = ((values.cpu().numpy().view(np.uint16) >> 15) * (1 << np.arange(16))).sum(axis=1).astype(np.uint16)
    np.testing.assert_array_equal(output.cpu().numpy().view(np.uint16), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_hadamard(dtype):
    x = torch.randn((3, 256), dtype=dtype, device='cuda')
    out = torch.empty_like(x)
    restored = torch.empty_like(x)
    backend.had_r_128(x, out, None, None, 1.0)
    backend.had_r_128(out, restored, None, None, 1.0)
    index = np.arange(128)
    matrix = np.array([[(-1) ** int(int(a & b).bit_count()) for b in index] for a in index], dtype=np.float32)
    reference = x.float().cpu().numpy().reshape(-1, 128) @ matrix / np.sqrt(128)
    np.testing.assert_allclose(out.float().cpu().numpy().reshape(-1, 128), reference, rtol=1e-3, atol=2e-3)
    torch.testing.assert_close(restored, x, rtol=1e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
@pytest.mark.parametrize('k', range(1, 9))
def test_reconstruct(k):
    rng = np.random.default_rng(k)
    packed = rng.integers(0, 65536, size=(2, 16, 16 * k), dtype=np.uint16)
    packed_gpu = torch.from_numpy(packed).cuda().view(torch.int16)
    out = torch.empty((32, 256), dtype=torch.float16, device='cuda')
    sliced = torch.empty((32, 128), dtype=torch.float16, device='cuda')
    backend.reconstruct(out, packed_gpu, k, False, True)
    backend.reconstruct_slice(sliced, packed_gpu, k, False, True, 128)
    np.testing.assert_array_equal(sliced.cpu().numpy(), out[:, 128:].cpu().numpy())
    words = packed.reshape(2, 16, -1, 2)[..., ::-1].copy().reshape(2, 16, -1)
    bits = ((words[..., :, None] >> np.arange(15, -1, -1)) & 1).reshape(2, 16, 256 * k)
    pos = (np.arange(256) * k + k - 16 + 256 * k) % (256 * k)
    states = np.zeros((2, 16, 256), dtype=np.uint32)
    for bit in range(16):
        states = (states << 1) | bits[..., (pos + bit) % (256 * k)].astype(np.uint32)
    product = states * np.uint32(0x00DCD12D)
    total = sum((product >> shift) & 255 for shift in (0, 8, 16, 24))
    value = (total + 0x6400).astype(np.uint16).view(np.float16).astype(np.float32)
    scale = np.array([0x1eee], dtype=np.uint16).view(np.float16).astype(np.float32)[0]
    bias = np.array([0xc931], dtype=np.uint16).view(np.float16).astype(np.float32)[0]
    decoded = (value * scale + bias).astype(np.float16)
    expected = np.empty((32, 256), dtype=np.float16)
    for t in range(256):
        expected[(np.arange(2) * 16 + 8 * (t // 128) + t % 8)[:, None],
                 (np.arange(16) * 16 + (t // 8) % 16)[None, :]] = decoded[:, :, t]
    np.testing.assert_array_equal(out.cpu().numpy(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
@pytest.mark.parametrize('k,length', [(k, 256) for k in range(1, 9)] + [(4, 160), (6, 160)])
def test_viterbi_basic(k, length):
    edges = 65536 >> k
    x = torch.randn((1, length), device='cuda')
    out = torch.empty_like(x)
    states = torch.empty_like(x, dtype=torch.int16)
    costs = torch.empty((1,), dtype=torch.float16, device='cuda')
    history = torch.empty((1, length, edges // (8 if k == 1 else 1)), dtype=torch.uint8, device='cuda')
    backend.quantize_tiles(x, out, states, costs, history, k, False, True, 0.0)
    decoded = torch.empty_like(x)
    backend.decode(states, decoded, False, True)
    torch.testing.assert_close(out, decoded, rtol=0, atol=0)
    assert torch.mean((out - x) ** 2).item() < 1.0
    raw = states.cpu().numpy().view(np.uint16).astype(np.uint32)[0]
    np.testing.assert_array_equal((np.roll(raw, 1) & ((1 << (16 - k)) - 1)), raw >> k)
    if length == 256:
        packed = torch.empty((1, 1, 16 * k), dtype=torch.int16, device='cuda')
        backend.pack_trellis(packed, states.view(1, 1, 256), k)
        unpacked = torch.empty_like(states).view(1, 1, 256)
        backend.unpack_trellis(unpacked, packed, k)
        assert torch.equal(unpacked.view_as(states), states)
        packed_tiles = packed.expand(1, 8, 16 * k).contiguous()
        reconstructed = torch.empty((16, 128), dtype=torch.float16, device='cuda')
        backend.reconstruct(reconstructed, packed_tiles, k, False, True)
        for t in range(256):
            row = 8 * (t // 128) + t % 8
            col = (t // 8) % 16
            assert reconstructed[row, col].item() == out[0, t].item()


def _viterbi_reference(values, k):
    length = len(values)
    edges = 65536 >> k
    edge = np.arange(edges, dtype=np.uint32)
    q = np.arange(1 << k, dtype=np.uint32)[:, None]
    state = (q << (16 - k)) | edge[None, :]
    prev = state >> k
    product = state * np.uint32(0x00DCD12D)
    total = sum((product >> shift) & 255 for shift in (0, 8, 16, 24))
    value = (total + 0x6400).astype(np.uint16).view(np.float16).astype(np.float64)
    scale = np.array([0x1eee], dtype=np.uint16).view(np.float16).astype(np.float64)[0]
    bias = np.array([0xc931], dtype=np.uint16).view(np.float16).astype(np.float64)[0]
    code = (value * scale + bias).astype(np.float16)
    ranks = []
    for i in range(edges):
        reverse = lambda x: int(f'{x:05b}'[::-1], 2)
        ranks.append((reverse((i & 1023) >> 5) << 10) | (reverse(i & 31) << 5) | (i >> 10))
    rank = np.array(ranks, dtype=np.uint32)
    def forward(roll, pre):
        history = np.empty((length, edges), dtype=np.uint16)
        cost = None
        for i in range(length):
            ri = (i + roll) % length
            diff = (code - np.float16(values[ri])).astype(np.float16).astype(np.float64)
            if i == 0:
                scores = (diff * diff).astype(np.float16)
                if pre is not None:
                    scores = np.where(prev == pre, scores, np.float16(np.inf))
            else:
                scores = (diff * diff + cost[prev].astype(np.float64)).astype(np.float16)
            choice = np.argmin(scores, axis=0)
            cost = scores[choice, edge]
            history[ri] = prev[choice, edge]
        return cost, history
    cost, history = forward(length // 2, None)
    cost_bits = cost.view(np.uint16).astype(np.uint32)
    end = int(np.argmin((cost_bits << 16) | rank))
    for i in range(length - 1, length // 2 - 1, -1):
        end = int(history[(i + length // 2) % length, end])
    _, history = forward(0, end)
    result = np.empty(length, dtype=np.uint16)
    for i in range(length - 1, -1, -1):
        previous = int(history[i, end])
        result[i] = (previous << k) | end
        end = previous
    return result


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
def test_viterbi_reference():
    k, length = 4, 160
    values = np.random.default_rng(123).normal(0, 0.6, length).astype(np.float32)
    source = torch.from_numpy(values).cuda().view(1, length)
    output = torch.empty_like(source)
    indices = torch.empty_like(source, dtype=torch.int16)
    costs = torch.empty(1, dtype=torch.float16, device='cuda')
    history = torch.empty((1, length, 65536 >> k), dtype=torch.uint8, device='cuda')
    backend.quantize_tiles(source, output, indices, costs, history, k, False, True, 0.0)
    np.testing.assert_array_equal(indices.cpu().numpy().view(np.uint16)[0], _viterbi_reference(values, k))


def test_paley():
    for fn, n in ((backend.had_paley, 12), (backend.had_paley2, 12)):
        h = torch.empty((n, n), dtype=torch.float16)
        fn(h)
        torch.testing.assert_close(h.float() @ h.float().T, torch.eye(n) * n, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
def test_hadamard_batch():
    x = torch.randn((6, 256), dtype=torch.float16, device='cuda')
    table = torch.randn((4, 256), dtype=torch.float16, device='cuda')
    ids = torch.tensor([2, 0, 3], dtype=torch.int64, device='cuda')
    output = torch.empty_like(x)
    ref = torch.empty_like(x)
    for pre in (True, False):
        backend.had_r_128_batch(x, output, table if pre else None, None if pre else table, ids, 2, 1.0)
        for i, expert in enumerate((2, 0, 3)):
            backend.had_r_128(x[i * 2:i * 2 + 2], ref[i * 2:i * 2 + 2], table[expert] if pre else None, None if pre else table[expert], 1.0)
        torch.testing.assert_close(output, ref, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
def test_distribution():
    values = torch.tensor([-3.0, -0.25, 0.25, 3.0], device='cuda')
    out = torch.empty(4, device='cuda')
    ref = torch.empty_like(out)
    backend.test_distribution(values, out, ref, -1, 1, False, True)
    torch.testing.assert_close(out, torch.full_like(out, 0.25))
    torch.testing.assert_close(ref.sum(), torch.ones((), device='cuda'))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
def test_reconstruct_batch_and_hadamard():
    k = 4
    packed = [torch.randint(-32768, 32767, (8, 8, 16 * k), dtype=torch.int16, device='cuda') for _ in range(2)]
    suh = [torch.randn(128, dtype=torch.float16, device='cuda') for _ in range(2)]
    svh = [torch.randn(128, dtype=torch.float16, device='cuda') for _ in range(2)]
    ptr = lambda tensors: torch.tensor([x.data_ptr() for x in tensors], dtype=torch.int64, device='cuda')
    output = torch.empty((2, 128, 128), dtype=torch.float16, device='cuda')
    expected = torch.empty_like(output)
    backend.reconstruct_batch(output, ptr(packed), k, False, True)
    for i in range(2):
        backend.reconstruct(expected[i], packed[i], k, False, True)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    raw = expected.float().cpu().numpy()
    backend.reconstruct_had_batch(output, ptr(packed), ptr(suh), ptr(svh), k, False, True)
    for i in range(2):
        backend.reconstruct_had_slice(expected[i], packed[i], suh[i], svh[i], k, False, True, 0)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    index = np.arange(128)
    h = np.array([[(-1) ** int(int(a & b).bit_count()) for b in index] for a in index], dtype=np.float32) / np.sqrt(128)
    for i in range(2):
        reference = (h @ raw[i] @ h) * suh[i].float().cpu().numpy()[:, None] * svh[i].float().cpu().numpy()[None, :]
        np.testing.assert_allclose(output[i].float().cpu().numpy(), reference, atol=0.02, rtol=0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
def test_viterbi_grid():
    k, length, step = 4, 160, 0.125
    x = torch.randn((1, length), device='cuda')
    out = torch.empty_like(x)
    states = torch.empty_like(x, dtype=torch.int16)
    costs = torch.empty(1, dtype=torch.float16, device='cuda')
    history = torch.empty((1, length, 65536 >> k), dtype=torch.uint8, device='cuda')
    backend.quantize_tiles(x, out, states, costs, history, k, False, True, step)
    decoded = torch.empty_like(x)
    backend.decode(states, decoded, False, True)
    expected = (torch.round(decoded / step) * step).half().float()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
def test_viterbi_batch():
    import time
    k, length, count = 4, 256, 64
    optimized, wave = backend.quantize_tiles_scratch(0, k, False, True, length)
    assert optimized and wave >= count
    x = torch.randn((count, length), device='cuda')
    out = torch.empty_like(x)
    states = torch.empty_like(x, dtype=torch.int16)
    costs = torch.empty(1, dtype=torch.float16, device='cuda')
    history = torch.empty((wave, length, 65536 >> k), dtype=torch.uint8, device='cuda')
    torch.cuda.synchronize()
    started = time.perf_counter()
    backend.quantize_tiles(x, out, states, costs, history, k, False, True, 0.0)
    torch.cuda.synchronize()
    print(f'Viterbi K={k}, tiles={count}, elapsed={time.perf_counter() - started:.3f}s')
    decoded = torch.empty_like(x)
    backend.decode(states, decoded, False, True)
    torch.testing.assert_close(out, decoded, rtol=0, atol=0)
    reference_out = torch.empty_like(x[:1])
    reference_states = torch.empty_like(states[:1])
    backend.quantize_tiles(x[:1], reference_out, reference_states, costs, history[:1], k, False, True, 0.0)
    torch.testing.assert_close(states[:1], reference_states, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
def test_rms_norm_silu():
    x = torch.randn((6, 128), dtype=torch.float16, device='cuda')
    weights = torch.randn((2, 128), dtype=torch.float16, device='cuda')
    output = torch.empty_like(x)
    backend.rms_norm(x, weights, output, 1e-5, 1.0, 0.7, False, False, 2)
    ref = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 1e-5) * 0.7
    ref *= weights.float()[torch.arange(6, device='cuda') % 2] + 1.0
    torch.testing.assert_close(output, ref.half(), atol=0.001, rtol=0.001)
    y = torch.randn_like(x)
    backend.silu_mul(x, y, output, 0.0)
    ref = torch.nn.functional.silu(x.float()).half().float() * y.float()
    torch.testing.assert_close(output, ref.half(), atol=0.001, rtol=0.001)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
@pytest.mark.parametrize('gate_first', [False, True])
@pytest.mark.parametrize('gate_act', [0, 1])
def test_gated_rms_norm(gate_first, gate_act):
    x = torch.randn((4, 128), dtype=torch.bfloat16, device='cuda')
    gate = torch.randn_like(x)
    w = torch.randn((2, 128), dtype=torch.bfloat16, device='cuda')
    result = torch.empty_like(x, dtype=torch.float32)
    backend.gated_rms_norm(x, w, result, gate, 1e-5, 0.5, 2, gate_first, gate_act)
    factor = torch.sigmoid(gate.float()) if gate_act else torch.nn.functional.silu(gate.float())
    values = x.float() * factor if gate_first else x.float()
    expected = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-5)
    expected *= w.float()[torch.arange(4, device='cuda') % 2] + 0.5
    if not gate_first:
        expected *= factor
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
def test_count_inf_nan():
    x = torch.tensor([float('inf'), float('-inf'), float('nan'), 1.0], device='cuda')
    y = torch.tensor([3, 4], dtype=torch.int64, device='cuda')
    backend.count_inf_nan(x, y)
    assert y.cpu().tolist() == [5, 5]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_hgemm(dtype):
    a = torch.randn((2, 9, 16), dtype=torch.float16, device='cuda')
    b = torch.randn((2, 16, 12), dtype=torch.float16, device='cuda')
    c = torch.empty((2, 9, 12), dtype=dtype, device='cuda')
    backend.hgemm_batched(a, b, c)
    expected = (a.float() @ b.float()).to(dtype)
    torch.testing.assert_close(c, expected, rtol=1e-3, atol=1e-3)
    backend.hgemm(a[0], b[0], c[0])
    torch.testing.assert_close(c[0], expected[0], rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('convert', [False, True])
def test_stloader(device, convert):
    with tempfile.TemporaryDirectory() as temp:
        original = torch.tensor([0.5, -2.0, 3.125, 8.0], dtype=torch.bfloat16)
        path = Path(temp) / 'tensors.bin'
        path.write_bytes(b'abcd' + original.view(torch.uint8).numpy().tobytes())
        handles = backend.stloader_open_file(str(path))
        try:
            direct = torch.empty(4, dtype=torch.bfloat16, device=device)
            backend.stloader_read(handles, 4, 8, direct)
            torch.testing.assert_close(direct.cpu(), original)
            output = torch.empty(4, dtype=torch.float16 if convert else torch.bfloat16, device=device)
            job = backend.TensorLoadJob(handles, 4, 8, output.data_ptr(), 8, convert, False, device == 'cuda', 0 if device == 'cuda' else -1)
            if device == 'cuda':
                backend.stloader_deferred_cuda([job], 4 * 1024 ** 2)
            else:
                backend.stloader_deferred_cpu([job])
            torch.testing.assert_close(output.cpu(), original.to(output.dtype))
        finally:
            backend.stloader_close_file(handles)

@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
@pytest.mark.parametrize('scales', [False, True])
def test_exl3_gemm(scales):
    k = 4
    a = torch.randn((3, 128), dtype=torch.float16, device='cuda')
    b = torch.randint(-32768, 32767, (8, 8, 16 * k), dtype=torch.int16, device='cuda')
    suh = torch.randn(128, dtype=torch.float16, device='cuda') if scales else None
    svh = torch.randn(128, dtype=torch.float16, device='cuda') if scales else None
    scratch = torch.empty_like(a) if scales else None
    actual = torch.empty((3, 128), dtype=torch.float16, device='cuda')
    backend.exl3_gemm(a, b, actual, suh, scratch, svh, -1, False, True, 0)
    w = torch.empty((128, 128), dtype=torch.float16, device='cuda')
    backend.reconstruct(w, b, k, False, True)
    transformed = torch.empty_like(a)
    if scales:
        backend.had_r_128(a, transformed, suh, None, 1.0)
    else:
        transformed.copy_(a)
    expected = transformed @ w
    if scales:
        backend.had_r_128(expected, expected, None, svh, 1.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_stloader_continues_partial_reads(tmp_path, monkeypatch):
    import os

    read = os.pread
    readv = os.preadv

    def partial(handle, size, offset):
        return read(handle, min(size, 3), offset)

    def partial_vector(handle, buffers, offset):
        return readv(handle, [buffers[0][:3]], offset)

    monkeypatch.setattr(os, 'pread', partial)
    monkeypatch.setattr(os, 'preadv', partial_vector)
    path = tmp_path / 'partial.bin'
    path.write_bytes(bytes(range(24)))
    handles = backend.stloader_open_file(str(path))
    try:
        target = torch.empty(12, dtype=torch.uint8)
        backend.stloader_read(handles, 5, target.numel(), target)
        assert target.tolist() == list(range(5, 17))
        backend.stloader_read(handles, 24, 0, target)
    finally:
        backend.stloader_close_file(handles)


def test_stloader_rejects_short_read():
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / 'short.bin'
        path.write_bytes(b'1234')
        handles = backend.stloader_open_file(str(path))
        try:
            destination = torch.empty(8, dtype=torch.uint8)
            with pytest.raises(EOFError):
                backend.stloader_read(handles, 0, 8, destination)
            job = backend.TensorLoadJob(handles, 0, 8, destination.data_ptr(), 4, False, False, False, -1)
            with pytest.raises(ValueError):
                backend.stloader_deferred_cpu([job])
        finally:
            backend.stloader_close_file(handles)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
@pytest.mark.parametrize('mode', [1, 2])
def test_rope(mode):
    q = torch.randn((2, 3, 2, 8), dtype=torch.float16, device='cuda')
    out = torch.empty_like(q)
    freq = torch.tensor([1.0, 0.1, 0.01, 0.001], device='cuda')
    positions = torch.tensor([2, 7], dtype=torch.int32, device='cuda')
    backend.rope(q, out, None, None, freq, 0, positions, None, mode, 1.0,
                 None, None, 1e-5, 0.0, 0.0, 1, 1, 0)
    angle = (positions[:, None] + torch.arange(3, device='cuda')).float()[..., None] * freq
    c, s = angle.cos().unsqueeze(2), angle.sin().unsqueeze(2)
    x = q.float()
    a = x[..., :4] if mode == 2 else x[..., ::2]
    b = x[..., 4:] if mode == 2 else x[..., 1::2]
    expected = torch.empty_like(q)
    if mode == 2:
        expected[..., :4], expected[..., 4:] = a * c - b * s, b * c + a * s
    else:
        expected[..., ::2], expected[..., 1::2] = a * c - b * s, b * c + a * s
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
def test_elementwise_helpers():
    x = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]], device='cuda')
    y = torch.full_like(x, 2.0)
    result = torch.empty_like(x, dtype=torch.float16)
    for fn, ref in ((backend.gelu_mul, torch.nn.functional.gelu(x, approximate='tanh')),
                    (backend.relu2_mul, x.relu().square()), (backend.relu_mul, x.relu())):
        fn(x, y, result, 0.0)
        torch.testing.assert_close(result, (ref * y).half(), atol=1e-3, rtol=1e-3)
    backend.silu_oai_mul(x, y, result, 1.0)
    gate = x.clamp(max=1.0)
    expected = 2.0 * gate * torch.sigmoid(1.702 * gate)
    torch.testing.assert_close(result, expected.half(), atol=1e-3, rtol=1e-3)
    qg = torch.arange(16, device='cuda').reshape(1, 16)
    q, g = torch.empty((1, 8), dtype=qg.dtype, device='cuda'), torch.empty((1, 8), dtype=qg.dtype, device='cuda')
    backend.deinterleave_qg(qg, q, g, 4)
    assert q.cpu().tolist() == [[0, 1, 2, 3, 8, 9, 10, 11]]
    assert g.cpu().tolist() == [[4, 5, 6, 7, 12, 13, 14, 15]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
def test_convert_qwen3_e2e():
    import json
    import os
    import subprocess
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        model, output, work = root / 'model', root / 'output', root / 'work'
        model.mkdir()
        work.mkdir()
        config = Qwen3Config(vocab_size=384, hidden_size=128, intermediate_size=256,
                             num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
                             head_dim=32, max_position_embeddings=512, tie_word_embeddings=False)
        Qwen3ForCausalLM(config).to(torch.bfloat16).save_pretrained(model)
        alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
        vocab = {piece: i for i, piece in enumerate(alphabet)}
        vocab.update({'[BOS]': 256, '[EOS]': 257, '[UNK]': 258})
        tokenizer = Tokenizer(models.BPE(vocab, [], unk_token='[UNK]'))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', bos_token='[BOS]', eos_token='[EOS]').save_pretrained(model)
        calibration = root / 'cal.safetensors'
        save_file({'input_ids': torch.randint(0, 256, (2, 64))}, str(calibration))
        quant = Path(__file__).resolve().parents[1]
        cmd = [sys.executable, str(quant / 'quantize.py'), 'convert', '--backend', 'triton',
               '--model', str(model), '--output', str(output), '--work', str(work), '--bpw', '4.0',
               '--calibration', str(calibration), '--cal-rows', '2', '--cal-cols', '64']
        subprocess.run(cmd, check=True, env=os.environ.copy(), timeout=600)
        subprocess.run([sys.executable, str(quant / 'validate_exl3.py'), str(output)], check=True, timeout=60)
        report = json.loads((output / 'quantize_report.json').read_text())
        assert report['backend'] == 'triton'
        assert report['extension']['multiplier'] == 0x00DCD12D
        assert {'cuda', 'rocm', 'triton'} <= report.keys()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
def test_gdn_gate_and_routing():
    b = torch.randn((2, 3, 4), device='cuda')
    a = torch.randn_like(b)
    bias = torch.randn(4, dtype=torch.bfloat16, device='cuda')
    log = torch.randn(4, dtype=torch.bfloat16, device='cuda')
    beta = torch.empty_like(b, dtype=torch.bfloat16)
    decay = torch.empty_like(b)
    backend.gated_delta_net_fused_op_2(b, a, bias, log, beta, decay, 0.8)
    raw = (torch.sigmoid(b) * 0.8).contiguous()
    torch.testing.assert_close(beta.float(), (raw.view(torch.int32) & -65536).view(torch.float32))
    torch.testing.assert_close(decay, -torch.nn.functional.softplus(a + bias.float()) * log.float().exp())
    hidden = torch.randn((3, 8), dtype=torch.float16, device='cuda')
    gate = torch.randn((8, 6), dtype=torch.float16, device='cuda')
    scores = torch.empty((3, 6), dtype=torch.float16, device='cuda')
    selected = torch.empty((3, 2), dtype=torch.int64, device='cuda')
    weights = torch.empty((3, 2), dtype=torch.float16, device='cuda')
    backend.routing_std(hidden, gate, scores, selected, weights, None, None, None)
    values, indices = torch.topk((hidden @ gate).float(), 2, dim=-1)
    torch.testing.assert_close(selected, indices)
    torch.testing.assert_close(weights, values.softmax(-1).half())
    backend.routing_ds3_nogroup(hidden, gate, scores, None, selected, weights, 1.5, None, 1)
    scores_ref = (hidden @ gate).float()
    indices = scores_ref.topk(2, dim=-1).indices
    values = torch.nn.functional.softplus(scores_ref.gather(-1, indices)).sqrt()
    torch.testing.assert_close(selected, indices)
    torch.testing.assert_close(weights, (values / values.sum(-1, keepdim=True) * 1.5).half())
    backend.routing_sel_norm(hidden, gate, scores, selected, weights, 1.5, None, 1)
    torch.testing.assert_close(weights, (values / values.sum(-1, keepdim=True) * 1.5).half())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
@pytest.mark.parametrize('history', [False, True])
def test_causal_conv(history):
    x = torch.randn((2, 8, 5), dtype=torch.bfloat16, device='cuda')
    state = torch.randn((3, 8, 12), dtype=torch.bfloat16, device='cuda')
    slots = torch.tensor([2, 0], dtype=torch.int32, device='cuda')
    weight = torch.randn((8, 4), dtype=torch.bfloat16, device='cuda')
    bias = torch.randn(8, dtype=torch.bfloat16, device='cuda')
    out = torch.empty((2, 5, 8), dtype=torch.bfloat16, device='cuda')
    old = state.clone()
    joined = torch.cat((old[slots.long(), :, :4], x), -1)
    ref = torch.nn.functional.silu(torch.nn.functional.conv1d(joined[..., 1:].float(), weight.float().unsqueeze(1), bias.float(), groups=8)).transpose(1, 2).to(torch.bfloat16)
    backend.cuda_causal_conv1d_update(x, state, slots, weight, bias, out, True, history)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)
    expected = old.clone()
    if history:
        expected[slots.long(), :, -9:] = joined
    else:
        expected[slots.long(), :, :4] = joined[..., -4:]
    torch.testing.assert_close(state, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
def test_hy4_hc_and_ple():
    rows, heads, dim = 2, 4, 16
    streams = torch.randn((rows, heads, dim), device='cuda')
    fn = torch.randn((2 * heads + heads * heads, heads * dim), device='cuda')
    base = torch.randn(2 * heads + heads * heads, device='cuda')
    scale = torch.randn(3, device='cuda')
    post = torch.empty((rows, heads), device='cuda')
    comb = torch.empty((rows, heads, heads), device='cuda')
    collapsed = torch.empty((rows, dim), dtype=torch.float16, device='cuda')
    backend.hc_mix(streams, fn, base, scale, 1e-5, 1e-6, 3, torch.empty(1, device='cuda'), post, comb, collapsed)
    x = streams.clone()
    y = torch.randn((rows, dim), device='cuda')
    expected = comb.transpose(1, 2) @ streams + post.unsqueeze(-1) * y.unsqueeze(1)
    backend.hc_apply(x, y, post, comb)
    torch.testing.assert_close(x, expected)
    gate = torch.randn((rows, heads), device='cuda')
    value = torch.randn((rows, dim), dtype=torch.float16, device='cuda')
    out = torch.empty_like(streams)
    backend.ple_gate(gate, value, out, 0.5)
    g = gate * 0.5
    expected = torch.sigmoid(g.sign() * g.abs().clamp(min=1e-6).sqrt()).unsqueeze(-1) * value.float().unsqueeze(1)
    torch.testing.assert_close(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
@pytest.mark.parametrize('low_disk', [False, True])
def test_convert_qwen35_e2e(low_disk):
    import json
    import os
    import subprocess
    from safetensors.torch import load_file, save_file
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3_5TextConfig, Qwen3_5ForCausalLM

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        model, output, work = root / 'model', root / 'output', root / 'work'
        model.mkdir()
        work.mkdir()
        config = Qwen3_5TextConfig(vocab_size=384, hidden_size=128, intermediate_size=256,
                                   num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                                   head_dim=32, linear_conv_kernel_dim=4, linear_num_key_heads=4,
                                   linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
                                   layer_types=['linear_attention', 'full_attention'], max_position_embeddings=512,
                                   tie_word_embeddings=False)
        Qwen3_5ForCausalLM(config).to(torch.bfloat16).save_pretrained(model)
        path = model / 'model.safetensors'
        tensors = load_file(path)
        tensors = {('model.language_model.' + key[len('model.'):] if key.startswith('model.') else key): value
                   for key, value in tensors.items()}
        save_file(tensors, path, metadata={'format': 'pt'})
        cfg_path = model / 'config.json'
        cfg = json.loads(cfg_path.read_text())
        cfg['full_attention_interval'] = 2
        cfg_path.write_text(json.dumps(cfg))
        alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
        vocab = {piece: i for i, piece in enumerate(alphabet)}
        vocab.update({'[BOS]': 256, '[EOS]': 257, '[UNK]': 258})
        tokenizer = Tokenizer(models.BPE(vocab, [], unk_token='[UNK]'))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', bos_token='[BOS]', eos_token='[EOS]').save_pretrained(model)
        calibration = root / 'cal.safetensors'
        save_file({'input_ids': torch.randint(0, 256, (2, 64))}, str(calibration))
        quant = Path(__file__).resolve().parents[1]
        cmd = [sys.executable, str(quant / 'quantize.py'), 'convert', '--backend', 'triton',
               '--model', str(model), '--output', str(output), '--work', str(work), '--bpw', '4.0',
               '--calibration', str(calibration), '--cal-rows', '2', '--cal-cols', '64']
        if low_disk:
            adapter = root / 'adapter'
            adapter.mkdir()
            prefix = 'base_model.model.model.language_model.layers.0.mlp.down_proj'
            save_file({prefix + '.lora_A.weight': torch.randn(2, 256) * 0.01,
                       prefix + '.lora_B.weight': torch.randn(128, 2) * 0.01},
                      str(adapter / 'adapter_model.safetensors'))
            (adapter / 'adapter_config.json').write_text(json.dumps({'r': 2, 'lora_alpha': 2}))
            (adapter / 'chat_template.jinja').write_text('adapter template')
            cmd.extend(['--low-disk', '--adapter', str(adapter)])
        subprocess.run(cmd, check=True, env=os.environ.copy(), timeout=600)
        subprocess.run([sys.executable, str(quant / 'validate_exl3.py'), str(output)], check=True, timeout=60)
        report = json.loads((output / 'quantize_report.json').read_text())
        assert report['backend'] == 'triton'
        if low_disk:
            assert report['low_disk'] is True and report['adapter'] == str(adapter)
            assert not (work / 'qtensors').exists()
            assert not (work / 'ckpt').exists()
            assert (output / 'chat_template.jinja').read_text() == 'adapter template'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
@pytest.mark.parametrize('history', [False, True])
def test_gdn_recurrent(history):
    b, length, kh, vh, kd, vd = 2, 3, 1, 2, 32, 16
    qkv = torch.randn((b, length, 2 * kh * kd + vh * vd), dtype=torch.bfloat16, device='cuda')
    g = torch.randn((b, length, vh), device='cuda') * 0.1
    beta = torch.rand((b, length, vh), dtype=torch.bfloat16, device='cuda')
    state = torch.randn((3, length if history else 1, vh, kd, vd), device='cuda')
    initial = state.clone()
    slots = torch.tensor([2, 0], dtype=torch.int32, device='cuda')
    out = torch.empty((b, length, vh, vd), dtype=torch.bfloat16, device='cuda')
    backend.cuda_recurrent_gated_delta_rule(qkv, g, beta, state, out, kh, vh, kd, vd, slots, history)
    q, k, v = qkv.float().split((kh * kd, kh * kd, vh * vd), dim=-1)
    q = q.reshape(b, length, kh, kd).repeat_interleave(2, 2)
    k = k.reshape(b, length, kh, kd).repeat_interleave(2, 2)
    v = v.reshape(b, length, vh, vd)
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    expected_state = initial[slots.long(), 0].clone()
    for t in range(length):
        expected_state = expected_state * g[:, t].exp()[..., None, None]
        delta = (v[:, t] - (expected_state * k[:, t, :, :, None]).sum(-2)) * beta[:, t].float().unsqueeze(-1)
        expected_state += k[:, t, :, :, None] * delta[:, :, None, :]
        expected = (expected_state * q[:, t, :, :, None]).sum(-2) / kd ** 0.5
        torch.testing.assert_close(out[:, t], expected.to(torch.bfloat16), atol=0, rtol=0)
        if history and t + 1 < length:
            torch.testing.assert_close(state[slots.long(), t + 1], expected_state)
    torch.testing.assert_close(state[slots.long(), 0], expected_state)


def test_gr_mix_reference():
    rows, heads, dim, rank = 2, 4, 8, 3
    streams = torch.randn(rows, heads, dim)
    weight = torch.randn(heads * dim).half()
    down = torch.randn(rank + heads, heads * dim).half()
    fn = (down.float() * weight.float()).half()
    up = torch.randn(heads, dim, rank).half()
    upt = up.reshape(heads, dim // 4, 4, rank).permute(0, 1, 3, 2).contiguous()
    dots = torch.empty(rows, rank + heads + 1, heads)
    post = torch.empty(rows, heads)
    mixed = torch.empty(rows, dim)
    backend.gr_mix(streams, fn, upt, weight, 1e-5, dots, post, mixed)
    inverse = torch.rsqrt(streams.square().mean(-1, keepdim=True) + 1e-5)
    normalized = streams * inverse * weight.float().reshape(heads, dim)
    projection = torch.einsum('rhd,mhd->rm', streams * inverse, fn.float().reshape(-1, heads, dim)) / heads
    expected = (normalized * torch.sigmoid(torch.einsum('rk,hdk->rhd',
                torch.nn.functional.silu(projection[:, :rank]), up.float()))).mean(1)
    torch.testing.assert_close(post, 2 * torch.sigmoid(projection[:, rank:]))
    torch.testing.assert_close(mixed, expected)


def test_ple_forward_streams_reference():
    batch, seq, heads, dim, embedding = 2, 3, 2, 8, 8
    streams = torch.randn(batch, seq, heads, dim)
    emb = torch.randn(batch, seq, embedding).half()
    key_w = torch.randn(embedding, heads * dim).half()
    value_w = torch.randn(embedding, dim).half()
    weights = [torch.randn(heads * dim).half() for _ in range(3)]
    conv_w = torch.randn(heads * dim, 1, 3).half()
    state = torch.randn(batch, heads * dim, 2).half()
    delta = torch.empty_like(streams)
    conv_stream = torch.empty(batch, heads * dim, seq + 2, dtype=torch.float16)
    backend.ple_forward_streams(streams, emb, key_w, value_w, *weights, conv_w,
                                state, 1e-5, 0.5, 1, delta, conv_stream)
    def norm(x, weight):
        x = x.float()
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * (weight.float().reshape(heads, dim) + 1)
    key = norm((emb @ key_w).reshape(batch, seq, heads, dim), weights[0])
    query = norm(streams, weights[1])
    gate = (key * query).sum(-1) * 0.5
    gate = torch.sigmoid(gate.sign() * gate.abs().clamp(min=1e-6).sqrt())
    gated = gate.unsqueeze(-1) * (emb @ value_w).float().unsqueeze(-2)
    normalized = norm(gated, weights[2]).half().reshape(batch, seq, heads * dim).transpose(1, 2)
    expected_stream = torch.cat((state, normalized), dim=-1)
    conv = torch.nn.functional.silu(torch.nn.functional.conv1d(expected_stream, conv_w, groups=heads * dim))
    torch.testing.assert_close(conv_stream, expected_stream)
    torch.testing.assert_close(delta, gated + conv.transpose(1, 2).reshape_as(gated))


def test_blocksparse_mlp_routing():
    from types import SimpleNamespace
    gate = torch.tensor([[1., 2., 0.], [0., -1., 2.]])
    config = SimpleNamespace(gate_tensor=gate, num_experts=3, num_experts_per_tok=2,
                             router_logits_bsz1=torch.empty(1, 3),
                             routing_weights_bsz1=torch.empty(1, 2),
                             selected_experts_bsz1=torch.empty(1, 2, dtype=torch.int64))
    y = torch.tensor([[1., 0.]])
    selected, weights = backend.blocksparse_mlp_routing(1, config, y, {})
    assert set(selected[0].tolist()) == {0, 1}
    torch.testing.assert_close(weights, torch.softmax((y @ gate).gather(1, selected), -1))
    selected, weights = backend.blocksparse_mlp_routing(1, config, y, {'activate_all_experts': True})
    assert selected.shape == (1, 3)
    torch.testing.assert_close(weights, torch.softmax((y @ gate).gather(1, selected), -1))


def _expert_weights(count, width=128, bits=2):
    packed = [torch.randint(-32768, 32767, (width // 16, width // 16, bits * 16),
                            dtype=torch.int16, device='cuda') for _ in range(count)]
    pre = [torch.randn(width, dtype=torch.float16, device='cuda') for _ in range(count)]
    post = [torch.randn(width, dtype=torch.float16, device='cuda') for _ in range(count)]
    pointers = [torch.tensor([t.data_ptr() for t in group], dtype=torch.int64, device='cuda')
                for group in (packed, pre, post)]
    return packed, pre, post, pointers


def _expert_reference(x, weights, expert, bits=2, output_dtype=torch.float16):
    packed, pre, post, _ = weights
    out = torch.empty((x.shape[0], post[expert].numel()), dtype=output_dtype, device=x.device)
    backend.exl3_gemm(x, packed[expert], out, pre[expert], torch.empty_like(x), post[expert],
                      -1, False, True, 0)
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
def test_exl3_mgemm_weighted_and_filtered():
    torch.manual_seed(19)
    weights = _expert_weights(3)
    ptrs = weights[3]
    x = torch.randn(1, 2, 128, dtype=torch.float16, device='cuda') * 0.2
    scratch = torch.empty(4, 2, 128, dtype=torch.float16, device='cuda')
    out = torch.empty(4, 2, 128, dtype=torch.float32, device='cuda')
    selected = torch.tensor([[0, 2], [1, 0]], dtype=torch.int64, device='cuda')
    scales = torch.tensor([[0.5, 0.25], [0.75, 0.2]], dtype=torch.float16, device='cuda')
    backend.exl3_mgemm(x, ptrs[0], out, ptrs[1], scratch, ptrs[2], selected, scales,
                       2, -1, False, True, 0, 2, 0, 2)
    expected0 = _expert_reference(x[0], weights, 0, output_dtype=torch.float32) * scales[0, 0]
    expected1 = (_expert_reference(x[0], weights, 1, output_dtype=torch.float32) * scales[1, 0]
                 + _expert_reference(x[0], weights, 0, output_dtype=torch.float32) * scales[1, 1])
    torch.testing.assert_close(out[0], expected0, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(out[1], expected1, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
@pytest.mark.parametrize('scratch_mode', [False, True])
@pytest.mark.parametrize('activation', [0, 1, 2])
def test_exl3_moe_reference(scratch_mode, activation):
    torch.manual_seed(23)
    gate, up, down = (_expert_weights(2) for _ in range(3))
    x = torch.randn(3, 128, dtype=torch.float16, device='cuda') * 0.1
    counts = torch.tensor([2, 2, 0], dtype=torch.int64, device='cuda')
    tokens = torch.tensor([0, 2, 0, 1], dtype=torch.int64, device='cuda')
    scales = torch.tensor([0.25, 0.5, 0.75, 0.4], dtype=torch.float16, device='cuda')
    out = torch.zeros((3, 128), dtype=torch.float32, device='cuda')
    tmp = [torch.empty(1, 4, 128, dtype=torch.float16, device='cuda') for _ in range(4)]
    scratch = torch.full((4, 128), float('nan'), device='cuda') if scratch_mode else None
    base = torch.tensor([0, 2], dtype=torch.int64, device='cuda') if scratch_mode else None
    backend.exl3_moe(x, out, counts, tokens, scales, *tmp, activation, 2, 2, 2,
                      *gate[3], *up[3], *down[3], False, True, False, True, False, True,
                      0.0, 2, scratch, base, 1, 4, 16)
    expected = torch.zeros_like(out)
    pieces = []
    for expert, (start, end) in enumerate(((0, 2), (2, 4))):
        rows = x.index_select(0, tokens[start:end])
        g = _expert_reference(rows, gate, expert)
        u = _expert_reference(rows, up, expert)
        if activation == 0:
            activated = (torch.nn.functional.silu(g) * u).half()
        elif activation == 1:
            activated = (torch.nn.functional.gelu(g, approximate='tanh') * u).half()
        else:
            activated = u.relu().square()
        packed, pre, post, _ = down
        transformed = torch.empty_like(activated)
        backend.had_r_128(activated, transformed, pre[expert], None, 1.0)
        reconstructed = torch.empty((128, 128), dtype=torch.float16, device='cuda')
        backend.reconstruct(reconstructed, packed[expert], 2, False, True)
        result = (transformed @ reconstructed).float()
        backend.had_r_128(result, result, None, post[expert], 1.0)
        result *= scales[start:end].float().unsqueeze(-1)
        pieces.append(result)
        expected.index_add_(0, tokens[start:end], result)
    if scratch_mode:
        torch.testing.assert_close(scratch, torch.cat(pieces), rtol=2e-3, atol=2e-3)
        assert not out.any()
        flat = torch.tensor([0, 1, 1, -1, 0, -1], dtype=torch.int64, device='cuda')
        order = torch.tensor([0, 2, 3, 0, 1, 0], dtype=torch.int64, device='cuda')
        starts = torch.tensor([0, 2], dtype=torch.int64, device='cuda')
        kinds = torch.ones(2, dtype=torch.int64, device='cuda')
        backend.exl3_moe_gather(out, scratch, flat, order, starts, base, kinds, scales)
        torch.testing.assert_close(out, expected, rtol=2e-3, atol=2e-3)
    else:
        torch.testing.assert_close(out, expected, rtol=2e-3, atol=2e-3)
    assert backend.exl3_moe_max_concurrency(0) >= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
def test_bc_blocksparse_mlp():
    torch.manual_seed(27)
    groups = [_expert_weights(2) for _ in range(3)]
    linears = [[backend.BC_LinearEXL3(p, a, b, 2, None, False, True,
                                      torch.empty((1, 128), dtype=torch.float16, device='cuda'))
                for p, a, b in zip(group[0], group[1], group[2])] for group in groups]
    temp = torch.empty((4, 128), dtype=torch.float16, device='cuda')
    out_single = torch.empty((4, 128), dtype=torch.float32, device='cuda')
    out_batch = torch.empty((4, 128), dtype=torch.float32, device='cuda')
    args = [temp] * 14 + [-1, -1]
    for group in groups:
        args.extend([*group[3], 2, False, True])
    args.extend([True, False, False, None, None, 0.0, *linears,
                 None, None, None, out_batch, None, None, None])
    args[8] = out_single
    bc = backend.BC_BlockSparseMLP(*args)
    x = torch.randn(2, 128, dtype=torch.float16, device='cuda') * 0.1
    indices = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64, device='cuda')
    scales = torch.tensor([[0.75, 0.25], [0.5, 0.5]], dtype=torch.float16, device='cuda')
    bc.run_bszN(x, indices, scales)
    expected = torch.zeros_like(out_batch[:2])
    for row in range(2):
        for slot, expert in enumerate(indices[row].tolist()):
            gate = _expert_reference(x[row:row + 1], groups[0], expert)
            up = _expert_reference(x[row:row + 1], groups[1], expert)
            activated = (torch.nn.functional.silu(gate) * up).half()
            value = _expert_reference(activated, groups[2], expert, output_dtype=torch.float32)
            expected[row] += value[0] * scales[row, slot].float()
    torch.testing.assert_close(out_batch[:2], expected, rtol=2e-3, atol=2e-3)
    bc.run_single_expert(x, 0)
    gate = _expert_reference(x, groups[0], 0)
    up = _expert_reference(x, groups[1], 0)
    expected_single = _expert_reference((torch.nn.functional.silu(gate) * up).half(), groups[2], 0,
                                        output_dtype=torch.float32)
    torch.testing.assert_close(out_single[:2], expected_single, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
def test_exl3_mgemm_sliced_output():
    torch.manual_seed(31)
    bits = 2
    packed = torch.randint(-32768, 32767, (8, 16, 16 * bits), dtype=torch.int16, device='cuda')
    scale_in = torch.randn(128, dtype=torch.float16, device='cuda')
    scale_out = torch.randn(256, dtype=torch.float16, device='cuda')
    x = torch.randn(1, 2, 128, dtype=torch.float16, device='cuda') * 0.1
    out = torch.empty((2, 256), dtype=torch.float16, device='cuda')
    fake = torch.empty((2, 2, 128), dtype=torch.float16, device='cuda')
    ptrs = torch.tensor([packed.data_ptr(), packed.data_ptr() + 8 * 16 * bits * 2],
                        dtype=torch.int64, device='cuda')
    pre = torch.tensor([scale_in.data_ptr()], dtype=torch.int64, device='cuda')
    post = torch.tensor([scale_out.data_ptr(), scale_out.data_ptr() + 128 * 2],
                        dtype=torch.int64, device='cuda')
    widths = torch.tensor([128, 128], dtype=torch.int32, device='cuda')
    dests = torch.tensor([out.data_ptr(), out.data_ptr() + 128 * 2], dtype=torch.int64, device='cuda')
    strides = torch.tensor([256, 256], dtype=torch.int32, device='cuda')
    sources = torch.tensor([0, 0], dtype=torch.int32, device='cuda')
    scratch = torch.empty_like(x)
    backend.exl3_mgemm(x, ptrs, fake, pre, scratch, post, None, None, bits, -1, False, True,
                       -1, -1, 0, 1, widths, dests, strides, sources, 1)
    expected = torch.empty_like(out)
    for i in range(2):
        part = torch.empty((2, 128), dtype=torch.float16, device='cuda')
        backend.exl3_gemm(x[0], packed[:, i * 8:(i + 1) * 8, :].contiguous(), part,
                          scale_in, torch.empty_like(x[0]), scale_out[i * 128:(i + 1) * 128],
                          -1, False, True, 0)
        expected[:, i * 128:(i + 1) * 128].copy_(part)
    torch.testing.assert_close(out, expected, rtol=2e-3, atol=2e-3)
