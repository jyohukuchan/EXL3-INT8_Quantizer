import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip('triton')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] >= 8, reason='Turing GPU が必要です')


def test_turing_cuda_parity():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'exl3_int8'))
    cuda = importlib.import_module('exllamav3_ext')
    path = root / 'exl3_int8/exllamav3/triton_ext/__init__.py'
    spec = importlib.util.spec_from_file_location('exl3_turing_triton_test', path)
    triton = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = triton
    spec.loader.exec_module(triton)
    torch.manual_seed(174)

    def close(a, b, label, atol=0, rtol=0):
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=label)
        print(label, 'max_diff', (a.float() - b.float()).abs().max().item(), flush=True)

    for k in (2, 3, 4, 6):
        print('K', k, flush=True)
        x = torch.randn((1, 256), device='cuda') * 0.5
        outs = []
        for ext in (cuda, triton):
            out = torch.empty_like(x)
            state = torch.empty_like(x, dtype=torch.int16)
            costs = torch.empty((1, 2, 65536 >> k), device='cuda', dtype=torch.float16)
            hist = torch.empty((1, 256, 65536 >> k), device='cuda', dtype=torch.int16 if ext is cuda else torch.uint8)
            ext.quantize_tiles(x, out, state, costs, hist, k, False, True, 0.0)
            outs.append((out, state))
        close(outs[0][1], outs[1][1], f'quantize state K={k}')
        close(outs[0][0], outs[1][0], f'quantize output K={k}')
        packed = torch.randint(-32768, 32767, (8, 8, 16 * k), dtype=torch.int16, device='cuda')
        rec = [torch.empty((128, 128), device='cuda', dtype=torch.float16) for _ in range(2)]
        for ext, w in zip((cuda, triton), rec): ext.reconstruct(w, packed, k, False, True)
        close(*rec, f'reconstruct K={k}')
        a = torch.randn((3, 128), device='cuda', dtype=torch.float16) * 0.1
        for scales in (False, True):
            su = torch.randn(128, device='cuda', dtype=torch.float16) if scales else None
            sv = torch.randn(128, device='cuda', dtype=torch.float16) if scales else None
            for dtype in (torch.float16, torch.float32):
                out = [torch.empty((3, 128), device='cuda', dtype=dtype) for _ in range(2)]
                for ext, y in zip((cuda, triton), out):
                    ext.exl3_gemm(a, packed, y, su, torch.empty_like(a) if scales else None, sv, 0, False, True, 0)
                close(*out, f'gemm K={k} scales={scales} dtype={dtype}', atol=0.003, rtol=0.003)
                out = [torch.empty((1, 128), device='cuda', dtype=dtype) for _ in range(2)]
                for ext, y in zip((cuda, triton), out):
                    ext.exl3_gemv(a[:1], packed, y, su, torch.empty_like(a[:1]) if scales else None, sv, False, True)
                close(*out, f'gemv K={k} scales={scales} dtype={dtype}', atol=0.003, rtol=0.003)
        groups = []
        for _ in range(3):
            weights = [torch.randint(-32768, 32767, (8, 8, 16 * k), dtype=torch.int16, device='cuda') for _ in range(2)]
            pre = [torch.randn(128, device='cuda', dtype=torch.float16) * 0.2 for _ in range(2)]
            post = [torch.randn(128, device='cuda', dtype=torch.float16) * 0.2 for _ in range(2)]
            ptrs = [torch.tensor([t.data_ptr() for t in group], dtype=torch.int64, device='cuda') for group in (weights, pre, post)]
            groups.append((weights, pre, post, ptrs))
        selected = torch.tensor([[0, 1]], dtype=torch.int64, device='cuda')
        routing = torch.tensor([[0.75, 0.25]], dtype=torch.float16, device='cuda')
        result = [torch.empty((2, 1, 128), device='cuda', dtype=torch.float32) for _ in range(2)]
        for ext, out in zip((cuda, triton), result):
            ext.exl3_mgemm(a[:1].view(1, 1, 128), groups[0][3][0], out, groups[0][3][1],
                            torch.empty((2, 1, 128), device='cuda', dtype=torch.float16), groups[0][3][2],
                            selected, routing, k, -1, False, True, -1, -1, 0, 1)
        close(result[0][:1], result[1][:1], f'mgemm K={k}', atol=0.003, rtol=0.003)
        counts = torch.tensor([1, 1, 0], dtype=torch.int64, device='cuda')
        tokens = torch.tensor([0, 0], dtype=torch.int64, device='cuda')
        route = routing.reshape(-1)
        result = [torch.zeros((1, 128), device='cuda', dtype=torch.float32) for _ in range(2)]
        for ext, out in zip((cuda, triton), result):
            tmp = [torch.empty((1, 2, 128), device='cuda', dtype=torch.float16) for _ in range(4)]
            ext.exl3_moe(a[:1], out, counts, tokens, route, *tmp, 0, k, k, k,
                          *groups[0][3], *groups[1][3], *groups[2][3], False, True, False, True, False, True,
                          0.0, 2, None, None, 1, 2, 16)
        close(*result, f'moe K={k}', atol=0.003, rtol=0.003)
    for k in (1, 5, 7, 8):
        packed = torch.randint(-32768, 32767, (8, 8, 16 * k), dtype=torch.int16, device='cuda')
        rec = [torch.empty((128, 128), device='cuda', dtype=torch.float16) for _ in range(2)]
        for ext, w in zip((cuda, triton), rec): ext.reconstruct(w, packed, k, False, True)
        close(*rec, f'reconstruct K={k}')
    print('T4 parity PASS', flush=True)


def test_turing_cooperative_moe():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'exl3_int8'))
    cuda = importlib.import_module('exllamav3_ext')
    torch.manual_seed(193)
    dev = 'cuda'
    k = 4
    weights = [torch.randint(-32768, 32767, (8, 8, k * 16), dtype=torch.int16, device=dev) for _ in range(3)]
    pre = [torch.randn(128, device=dev, dtype=torch.float16) * 0.2 for _ in range(3)]
    post = [torch.randn(128, device=dev, dtype=torch.float16) * 0.2 for _ in range(3)]
    pointers = [torch.tensor([t.data_ptr()], device=dev, dtype=torch.int64) for g in zip(weights, pre, post) for t in g]
    x = torch.randn((1, 128), device=dev, dtype=torch.float16) * 0.1
    sel = torch.zeros((1, 1), device=dev, dtype=torch.int64)
    rw = torch.tensor([[0.75]], device=dev, dtype=torch.float16)
    hg, hu, gg, gu, act = [torch.empty((1, 128), device=dev, dtype=torch.float16) for _ in range(5)]
    d = torch.empty((1, 128), device=dev, dtype=torch.float32)
    ctr = torch.zeros((7,), device=dev, dtype=torch.int32)
    out = torch.empty((1, 128), device=dev, dtype=torch.float32)
    cuda.exl3_moe_coop(x, sel, rw, -1, -1, 128, *pointers, None, None, None,
                       k, k, k, False, True, 0, 0., True, hg, hu, gg, gu, act, d, ctr, out, None, None)
    def linear(a, i, dtype):
        c = torch.empty((1, 128), dtype=dtype, device=dev)
        cuda.exl3_gemm(a, weights[i], c, pre[i], torch.empty_like(a), post[i], 0, False, True, 0)
        return c
    up = linear(x, 1, torch.float16).float()
    gate = linear(x, 0, torch.float16).float()
    activation = (torch.nn.functional.silu(gate) * up).half()
    ref = linear(activation, 2, torch.float32) * rw.float()
    torch.testing.assert_close(out, ref, atol=0.003, rtol=0.003)
    print('cooperative MoE PASS max_diff', (out-ref).abs().max().item())
