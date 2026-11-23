import torch
import triton
import triton.language as tl


@triton.jit
def _value(state, s0: tl.constexpr):
    product = state.to(tl.uint32) * 0x00DCD12D
    total = ((product & 255) + ((product >> 8) & 255) +
             ((product >> 16) & 255) + ((product >> 24) & 255))
    value = (total + 0x6400).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    scale = tl.full((), 0x1eee, tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    bias = tl.full((), 0xc931, tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    value = tl.fma(value, scale, bias).to(tl.float16).to(tl.float32)
    if s0 > 0:
        x = value / s0
        floor = tl.floor(x)
        fraction = x - floor
        rounded = floor + tl.where((fraction > 0.5) | ((fraction == 0.5) & ((floor.to(tl.int32) & 1) != 0)), 1.0, 0.0)
        value = (rounded * s0).to(tl.float16).to(tl.float32)
    return value


@triton.jit
def _step(input, costs, history, pre, E: tl.constexpr, L: tl.constexpr, K: tl.constexpr,
          STEP, ROLL: tl.constexpr, SECOND: tl.constexpr, S0: tl.constexpr,
          BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    edge = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = edge < E
    ri = (STEP + ROLL) % L
    weight = tl.load(input + tile * L + ri).to(tl.float16).to(tl.float32)
    if SECOND:
        pre_state = tl.load(pre + tile)
    best = tl.full((BLOCK,), float('inf'), tl.float32)
    best_prev = (edge >> K).to(tl.int32)
    for q in range(1 << K):
        state = (q << (16 - K)) | edge
        prev = state >> K
        diff = (_value(state, S0) - weight).to(tl.float16).to(tl.float32)
        if STEP == 0:
            error = (diff * diff).to(tl.float16).to(tl.float32)
            if SECOND:
                error = tl.where(prev == pre_state, error, float('inf'))
        else:
            previous = tl.load(costs + tile * 2 * E + ((STEP - 1) % 2) * E + prev, valid, other=float('inf')).to(tl.float32)
            error = tl.fma(diff, diff, previous).to(tl.float16).to(tl.float32)
        choose = error < best
        best = tl.where(choose, error, best)
        best_prev = tl.where(choose, prev, best_prev)
    tl.store(costs + tile * 2 * E + (STEP % 2) * E + edge, best, valid)
    if SECOND or ri < L // 2:
        branch = best_prev >> (16 - 2 * K)
        if K == 1:
            lane = tl.arange(0, BLOCK)
            packed = tl.full((BLOCK,), 0, tl.int32)
            for bit in tl.static_range(8):
                packed |= tl.gather(branch, (lane & ~7) + bit, axis=0) << bit
            tl.store(history + tile * L * (E // 8) + ri * (E // 8) + edge // 8,
                     packed.to(tl.uint8), valid & ((edge % 8) == 0))
        else:
            tl.store(history + tile * L * E + ri * E + edge, branch.to(tl.uint8), valid)


@triton.jit
def _best(costs, pre, E: tl.constexpr, L: tl.constexpr, BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    edge = tl.arange(0, BLOCK)
    value = tl.load(costs + tile * 2 * E + ((L - 1) % 2) * E + edge, edge < E, other=float('inf')).to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
    v = edge & 1023
    warp = v >> 5
    lane = v & 31
    rw = ((warp & 1) << 4) | ((warp & 2) << 2) | (warp & 4) | ((warp & 8) >> 2) | ((warp & 16) >> 4)
    rl = ((lane & 1) << 4) | ((lane & 2) << 2) | (lane & 4) | ((lane & 8) >> 2) | ((lane & 16) >> 4)
    rank = (rw << 10) | (rl << 5) | (edge >> 10)
    key = (value << 16) | rank
    best = tl.min(tl.where(edge < E, key, 0xffffffff), 0)
    chosen = tl.min(tl.where((key == best) & (edge < E), edge, 0x7fffffff), 0)
    tl.store(pre + tile, tl.where(best >= 0x7c000000, 0, chosen))


@triton.jit
def _trace(history, pre, output, indices, E: tl.constexpr, L: tl.constexpr, K: tl.constexpr,
           FIRST: tl.constexpr, S0: tl.constexpr):
    tile = tl.program_id(0)
    edge = tl.load(pre + tile)
    for step in range(L // 2 if FIRST else L):
        i = L - 1 - step
        ri = (i + (L // 2 if FIRST else 0)) % L
        if K == 1:
            packed = tl.load(history + tile * L * (E // 8) + ri * (E // 8) + edge // 8).to(tl.int32)
            q = (packed >> (edge % 8)) & 1
        else:
            q = tl.load(history + tile * L * E + ri * E + edge).to(tl.int32)
        prev = (q << (16 - 2 * K)) | (edge >> K)
        state = (prev << K) | edge
        edge = prev
        if not FIRST:
            tl.store(indices + tile * L + ri, state.to(tl.uint16))
            tl.store(output + tile * L + ri, _value(state, S0))
    if FIRST:
        tl.store(pre + tile, edge)


def quantize_tiles_scratch(device, K, mcg, mul1, L):
    if not 1 <= K <= 8 or mcg or not mul1 or L not in (160, 256):
        raise ValueError('mul24 と K=1..8、L=160 または 256 が必要です')
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    if K >= 4:
        wave = min(256, max(32, sms * 4))
    elif K == 3:
        wave = min(128, max(16, sms * 2))
    else:
        wave = min(32, max(8, sms))
    return True, wave


def quantize_tiles(input_tiles, output_tiles, output_indices, temp_costs, temp_edges, K, mcg, mul1, s0):
    if input_tiles.ndim != 2:
        raise ValueError('入力は 2 次元が必要です')
    n, length = input_tiles.shape
    quantize_tiles_scratch(input_tiles.device.index, K, mcg, mul1, length)
    edges = 65536 >> K
    if (input_tiles.dtype != torch.float32 or output_tiles.dtype != torch.float32
            or output_indices.dtype not in (torch.int16, torch.uint16)
            or input_tiles.shape != output_tiles.shape or input_tiles.shape != output_indices.shape
            or temp_costs.dtype != torch.float16 or temp_edges.dtype != torch.uint8
            or temp_costs.numel() < 1
            or temp_edges.ndim != 3 or tuple(temp_edges.shape[1:]) != (length, edges // (8 if K == 1 else 1))
            or not input_tiles.is_cuda or any(t.device != input_tiles.device or not t.is_contiguous()
                                               for t in (input_tiles, output_tiles, output_indices, temp_costs, temp_edges))):
        raise ValueError('量子化の入力、出力または scratch が不正です')
    wave = temp_edges.shape[0]
    if wave < 1:
        raise ValueError('scratch が空です')
    costs = torch.empty((min(wave, n), 2, edges), dtype=torch.float16, device=input_tiles.device)
    for start in range(0, n, wave):
        batch = min(wave, n - start)
        pre = torch.empty(batch, dtype=torch.int32, device=input_tiles.device)
        inp = input_tiles[start:start + batch]
        out = output_tiles[start:start + batch]
        idx = output_indices[start:start + batch]
        for step in range(length):
            _step[(batch, triton.cdiv(edges, 256))](inp, costs, temp_edges, pre, edges, length, K, step, length // 2, False, s0, 256)
        _best[(batch,)](costs, pre, edges, length, triton.next_power_of_2(edges))
        _trace[(batch,)](temp_edges, pre, out, idx, edges, length, K, True, s0)
        for step in range(length):
            _step[(batch, triton.cdiv(edges, 256))](inp, costs, temp_edges, pre, edges, length, K, step, 0, True, s0, 256)
        _trace[(batch,)](temp_edges, pre, out, idx, edges, length, K, False, s0)
