import torch
import torch.nn.functional as F


def rms_norm(x, w, y, epsilon, constant_bias, constant_scale, span_heads, add_residual, w_groups=1):
    shape = x.shape
    if span_heads:
        if x.ndim < 2 or w_groups != 1:
            raise ValueError('span_heads の形状が不正です')
        width = shape[-1] * shape[-2]
    else:
        width = shape[-1]
    if x.shape != y.shape or width % 4 or w_groups < 1:
        raise ValueError('RMSNorm の形状が不正です')
    values = x.float().reshape(-1, width)
    result = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + epsilon) * constant_scale
    if w is not None:
        if w.numel() != w_groups * width:
            raise ValueError('RMSNorm の重みが不正です')
        weights = w.float().reshape(w_groups, width)
        row = torch.arange(values.shape[0], device=x.device) % w_groups
        result = result * (weights[row] + constant_bias)
    if add_residual:
        result = result + y.float().reshape(-1, width)
    y.copy_(result.reshape(shape))


def rms_norm_res_in(x, w, y, r, epsilon, constant_bias, constant_scale):
    r.add_(x)
    rms_norm(r, w, y, epsilon, constant_bias, constant_scale, False, False)


def silu_mul(x, y, z, act_limit):
    activation = F.silu(x.float())
    other = y.float()
    if x.dtype == torch.float16:
        activation = activation.half().float()
    if act_limit != 0:
        activation = activation.clamp(max=act_limit)
        other = other.clamp(min=-act_limit, max=act_limit)
    z.copy_(activation * other)


def gated_rms_norm(x, w, y, g, epsilon, constant_bias, w_groups=1, gate_first=False, gate_act=0):
    if (x.shape != y.shape or x.shape != g.shape or x.ndim < 1 or w_groups < 1
            or w.numel() != w_groups * x.shape[-1] or gate_act not in (0, 1)):
        raise ValueError('gated RMSNorm の引数が不正です')
    width = x.shape[-1]
    values = x.float().reshape(-1, width)
    gates = g.float().reshape(-1, width)
    gates = torch.sigmoid(gates) if gate_act == 1 else F.silu(gates)
    if gate_first:
        values = values * gates
    norm = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + epsilon)
    weights = w.float().reshape(w_groups, width)
    rows = torch.arange(values.shape[0], device=x.device) % w_groups
    result = norm * (weights[rows] + constant_bias)
    if not gate_first:
        result = result * gates
    y.copy_(result.reshape_as(y))


def count_inf_nan(x, y):
    if x.dtype not in (torch.float16, torch.float32) or y.dtype != torch.int64 or y.numel() != 2 or x.device != y.device:
        raise ValueError('count_inf_nan の引数が不正です')
    y.add_(torch.stack((torch.isinf(x).sum(), torch.isnan(x).sum())).reshape_as(y))


def hgemm(a, b, c):
    c.copy_(torch.matmul(a.float(), b.float()) if c.dtype == torch.float32 else torch.matmul(a, b))


def hgemm_batched(a, w, c):
    c.copy_(torch.bmm(a.float(), w.float()) if c.dtype == torch.float32 else torch.bmm(a, w))


def hgemm_recon(a, b, c):
    hgemm(a, b, c)


def g_get_num_sms(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


def g_get_cc(device):
    major, minor = torch.cuda.get_device_capability(device)
    return 5 if major >= 10 else 4 if major >= 9 else 3 if major >= 8 and minor >= 9 else 2 if major >= 8 else 1


def exl3_gemv_int8_max_k(device):
    return 0


def exl3_mgemv_masked_reduce_supported():
    return False


EXL3_MGEMV_MASKED_REDUCE_SUPPORTED = False


def exl3_gemm(A, B, C, suh, A_had, svh, force_shape_idx, mcg, mul1, force_num_sms):
    from . import had_r_128, reconstruct
    if A.ndim < 2 or B.ndim != 3 or C.numel() // C.shape[-1] != A.numel() // A.shape[-1]:
        raise ValueError('EXL3 GEMM の形状が不正です')
    rows = A.numel() // A.shape[-1]
    a = A.reshape(rows, A.shape[-1])
    c = C.reshape(rows, C.shape[-1])
    weight = torch.empty((a.shape[-1], c.shape[-1]), dtype=torch.float16, device=A.device)
    reconstruct(weight, B, B.shape[-1] // 16, mcg, mul1)
    if suh is not None:
        transformed = A_had.reshape_as(a) if A_had is not None else torch.empty_like(a)
        had_r_128(a, transformed, suh, None, 1.0)
    else:
        transformed = a
    hgemm(transformed, weight, c)
    if svh is not None:
        had_r_128(c, c, None, svh, 1.0)
    return 0


def exl3_gemv(A, B, C, suh, A_had, svh, mcg, mul1):
    exl3_gemm(A, B, C, suh, A_had, svh, -1, mcg, mul1, 0)


def _activation_mul(x, y, z, act_limit, kind):
    values = x.float()
    if kind == 'gelu':
        act = F.gelu(values, approximate='tanh')
    elif kind == 'relu2':
        act = values.relu().square()
    else:
        act = values.relu()
    if x.dtype == torch.float16:
        act = act.half().float()
    other = y.float()
    if act_limit:
        act = act.clamp(max=act_limit)
        other = other.clamp(-act_limit, act_limit)
    z.copy_(act * other)


def gelu_mul(x, y, z, act_limit):
    _activation_mul(x, y, z, act_limit, 'gelu')


def relu2_mul(x, y, z, act_limit):
    _activation_mul(x, y, z, act_limit, 'relu2')


def relu_mul(x, y, z, act_limit):
    _activation_mul(x, y, z, act_limit, 'relu')


def silu_oai_mul(x, y, z, act_limit):
    gate, up = x.float(), y.float()
    if act_limit:
        gate, up = gate.clamp(max=act_limit), up.clamp(-act_limit, act_limit)
    z.copy_((up + 1.0) * gate * torch.sigmoid(1.702 * gate))


def softcap(x, y, softcap_factor):
    y.copy_(torch.tanh(x.float() / softcap_factor) * softcap_factor)


def add(x, y, z):
    torch.add(x, y, out=z)


def xielu(x, y, alpha_p, alpha_n):
    v = x.float()
    result = torch.where(v > 0, alpha_p.float() * v.square() + 0.5 * v,
                         (torch.expm1(v.clamp(max=-9.9838e-7)) - v) * alpha_n.float() + 0.5 * v)
    y.copy_(result.clamp(-65504, 65504))


def add_sigmoid_gate(x, y, z):
    z.add_((x.float().reshape(y.numel(), -1) * torch.sigmoid(y.float().reshape(-1, 1))).reshape_as(z))


def add_sigmoid_gate_proj(x, y, z, w):
    z.add_(x.float() * torch.sigmoid(y.float() @ w.float()))


def mul_sigmoid_(x, y):
    x.copy_(x.float() * torch.sigmoid(y.float()))


def mul_sigmoid_broadcast_(x, y):
    x.copy_((x.float().reshape(y.numel(), -1) * torch.sigmoid(y.float().reshape(-1, 1))).reshape_as(x))


def mul_softplus_broadcast_(x, y):
    x.copy_((x.float().reshape(y.numel(), -1) * F.softplus(y.float().reshape(-1, 1))).reshape_as(x))


def deinterleave_qg(qg, q, g, head_dim):
    reshaped = qg.reshape(*qg.shape[:-1], -1, 2, head_dim)
    q.copy_(reshaped[..., 0, :].reshape_as(q))
    g.copy_(reshaped[..., 1, :].reshape_as(g))


def rope(q, out_q, k, out_k, inv_freq, position, positions, position_ids, rope_mode,
         attn_factor, q_norm, k_norm, norm_eps, norm_constant_bias,
         llama_4_scaling_beta, llama_4_scaling_original, rotate_dims, rotate_offset):
    if q.ndim != 4 or rope_mode not in (1, 2) or rotate_dims < 1:
        raise ValueError('RoPE の引数が不正です')
    batch, length, _, width = q.shape
    part = inv_freq.shape[-1] * 2
    if rotate_offset < 0 or rotate_offset + part * rotate_dims > width or (positions is not None and position_ids is not None):
        raise ValueError('RoPE の位置または回転範囲が不正です')
    time = torch.arange(length, device=q.device, dtype=torch.long).expand(batch, -1)
    if positions is not None:
        pos = time + positions.to(torch.long).reshape(batch, 1)
    elif position_ids is not None:
        pos = position_ids.to(torch.long)
    else:
        pos = time + position

    def apply(source, target, norm, query):
        value = source.float()
        if norm is not None:
            value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + norm_eps)
            if norm.dtype == torch.float16:
                value = (value.half() * (norm + norm_constant_bias).half()).float()
            else:
                value = (value * (norm.float() + norm_constant_bias)).half().float()
        else:
            value = value.clone()
        for r in range(rotate_dims):
            pos_r = pos[..., r] if pos.ndim == 3 else pos
            if inv_freq.ndim == 1:
                angle = pos_r.float().unsqueeze(-1) * inv_freq.float()
            else:
                batch_idx = torch.arange(batch, device=q.device).view(batch, 1)
                angle = inv_freq[batch_idx, pos_r].float()
            cos = (angle.cos() * attn_factor).unsqueeze(2)
            sin = (angle.sin() * attn_factor).unsqueeze(2)
            section = value[..., rotate_offset + part * r:rotate_offset + part * (r + 1)]
            a = section[..., :part // 2] if rope_mode == 2 else section[..., ::2]
            b = section[..., part // 2:] if rope_mode == 2 else section[..., 1::2]
            first, second = (a * cos - b * sin).half(), (b * cos + a * sin).half()
            if rope_mode == 2:
                section[..., :part // 2], section[..., part // 2:] = first, second
            else:
                section[..., ::2], section[..., 1::2] = first, second
        if query and llama_4_scaling_beta > 0:
            base_pos = pos[..., 0] if pos.ndim == 3 else pos
            factor = 1 + llama_4_scaling_beta * torch.log1p(torch.div(base_pos, llama_4_scaling_original, rounding_mode='floor').float())
            value = (value.half().float() * factor[..., None, None]).half().float()
        target.copy_(value)
    apply(q, out_q, q_norm, True)
    if k is not None:
        if out_k is None:
            raise ValueError('RoPE の key 出力がありません')
        apply(k, out_k, k_norm, False)


class BC_LinearFP16:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def run(self, x, y):
        hgemm(x, self.weight, y)
        if self.bias is not None:
            y.add_(self.bias)


class BC_LinearEXL3:
    def __init__(self, trellis, suh, svh, K, bias, mcg, mul1, xh):
        self.trellis, self.suh, self.svh = trellis, suh, svh
        self.bias, self.mcg, self.mul1, self.xh = bias, mcg, mul1, xh

    def run(self, x, y):
        scratch = self.xh if self.xh.numel() == x.numel() else torch.empty_like(x)
        exl3_gemm(x, self.trellis, y, self.suh, scratch, self.svh, -1, self.mcg, self.mul1, 0)
        if self.bias is not None:
            y.add_(self.bias)

    def run_alloc(self, x, out_features, output_fp32):
        y = torch.empty((*x.shape[:-1], out_features), dtype=torch.float32 if output_fp32 else torch.float16, device=x.device)
        if out_features:
            self.run(x, y)
        return y


def BC_GatedMLP(*args, **kwargs):
    return None


def BC_MLP(*args, **kwargs):
    return None


class BC_GatedRMSNorm:
    def __init__(self, weight, rms_norm_eps, constant_bias, w_groups=1, gate_first=False, gate_act=0):
        self.weight = weight
        self.rms_norm_eps = rms_norm_eps
        self.constant_bias = constant_bias
        self.w_groups = w_groups
        self.gate_first = gate_first
        self.gate_act = gate_act

    def run(self, x, y, gate):
        gated_rms_norm(x, self.weight, y, gate, self.rms_norm_eps, self.constant_bias,
                       self.w_groups, self.gate_first, self.gate_act)


def gated_delta_net_fused_op_2(b, a, dt_bias, a_log, beta, g, beta_scale):
    raw = (torch.sigmoid(b.float()) * beta_scale).contiguous()
    truncated = (raw.view(torch.int32) & -65536).view(torch.float32)
    beta.copy_(truncated)
    g.copy_(-F.softplus(a.float() + dt_bias.float()) * a_log.float().exp())


def routing_std(hidden, gate, scores, topk_indices, topk_weights, per_expert_scale, gate_t, bias):
    scores.copy_(hidden @ gate)
    logits = scores.float()
    if bias is not None:
        logits = logits + bias.float()
    selected_logits, indices = torch.topk(logits, topk_indices.shape[-1], dim=-1)
    weights = torch.softmax(selected_logits, dim=-1)
    if per_expert_scale is not None:
        weights = weights * per_expert_scale.float()[indices]
    topk_indices.copy_(indices)
    topk_weights.copy_(weights)


def routing_std_mapped(hidden, gate, scores, topk_indices, topk_weights, per_expert_scale, gate_t, bias, expert_map):
    routing_std(hidden, gate, scores, topk_indices, topk_weights, per_expert_scale, gate_t, bias)
    topk_indices.copy_(expert_map[topk_indices].to(topk_indices.dtype))


def _routing_act(logits, act_fn):
    if act_fn == 0:
        return torch.sigmoid(logits)
    if act_fn == 1:
        return F.softplus(logits).sqrt()
    raise ValueError('ルーティングの活性化が不正です')


def routing_ds3_nogroup(hidden, gate, scores, bias, topk_indices, topk_weights, scaling_factor, gate_t, act_fn):
    scores.copy_(hidden @ gate)
    logits = scores.float()
    activated = _routing_act(logits, act_fn)
    choice = activated + bias.float() if bias is not None else logits
    indices = torch.topk(choice, topk_indices.shape[-1], dim=-1).indices
    weights = activated.gather(-1, indices)
    weights = weights * (scaling_factor / (weights.sum(dim=-1, keepdim=True) + 1e-20))
    topk_indices.copy_(indices)
    topk_weights.copy_(weights)


def routing_sel_norm(hidden, gate, scores, selected, weights, scaling_factor, gate_t, act_fn):
    scores.copy_(hidden @ gate)
    chosen = _routing_act(scores.float().gather(-1, selected), act_fn)
    weights.copy_(chosen * (scaling_factor / (chosen.sum(dim=-1, keepdim=True) + 1e-20)))


def cuda_causal_conv1d_update(x, conv_state, slots, weight, bias, out, activation, history):
    batch, dim, length = x.shape
    kernel = weight.shape[-1]
    if (weight.shape != (dim, kernel) or conv_state.ndim != 3 or conv_state.shape[1] != dim
            or conv_state.shape[-1] < kernel or out.shape != (batch, length, dim)):
        raise ValueError('causal convolution の引数が不正です')
    indices = torch.arange(batch, device=x.device) if slots is None else slots.long()
    initial = conv_state.index_select(0, indices)[..., :kernel]
    joined = torch.cat((initial, x), dim=-1)
    y = F.conv1d(joined[..., 1:].float(), weight.float().unsqueeze(1),
                 None if bias is None else bias.float(), groups=dim)
    if activation:
        y = F.silu(y)
    out.copy_(y.transpose(1, 2))
    updated = conv_state.index_select(0, indices)
    if history:
        write_size = min(updated.shape[-1], kernel + length)
        updated[..., -write_size:] = joined[..., -write_size:]
    else:
        updated[..., :kernel] = joined[..., -kernel:]
    conv_state.index_copy_(0, indices, updated)


def hc_mix_num_chunks(R, row_len):
    return 1


def hc_mix(streams, fn, base, scale, rms_eps, hc_eps, sinkhorn_iters, partials, post, comb, collapsed):
    rows, heads, dim = streams.shape
    flat = streams.float().reshape(rows, heads * dim)
    flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + rms_eps)
    mixes = F.linear(flat, fn.float())
    pre_w, post_w, comb_w = mixes.split([heads, heads, heads * heads], -1)
    pre_b, post_b, comb_b = base.float().split([heads, heads, heads * heads])
    pre = torch.sigmoid(pre_w * scale[0] + pre_b) + hc_eps
    post.copy_(2 * torch.sigmoid(post_w * scale[1] + post_b))
    matrix = comb_w.reshape(rows, heads, heads) * scale[2] + comb_b.reshape(heads, heads)
    matrix = torch.softmax(matrix, -1) + hc_eps
    matrix = matrix / (matrix.sum(-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        matrix = matrix / (matrix.sum(-1, keepdim=True) + hc_eps)
        matrix = matrix / (matrix.sum(-2, keepdim=True) + hc_eps)
    comb.copy_(matrix)
    collapsed.copy_((pre.unsqueeze(-1) * streams.float()).sum(1))


def hc_head(streams, fn, base, scale, rms_eps, hc_eps, partials, collapsed):
    flat = streams.float().flatten(1)
    norm = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + rms_eps)
    pre = torch.sigmoid(F.linear(norm, fn.float()) * scale + base) + hc_eps
    collapsed.copy_((pre.unsqueeze(-1) * streams.float()).sum(1))


def hc_apply(x, y, post, comb):
    residual = x.float() if comb is None else torch.bmm(comb.float().transpose(1, 2), x.float())
    x.copy_(residual + post.float().unsqueeze(-1) * y.float().unsqueeze(1))


def ple_gate(gate, value, out, gate_scale):
    g = gate.float() * gate_scale
    signed = g.sign() * g.abs().clamp(min=1e-6).sqrt()
    out.copy_(torch.sigmoid(signed).unsqueeze(-1) * value.float().unsqueeze(-2))


def BC_GatedDeltaNet(*args, **kwargs):
    return None


def BC_GatedDeltaNetSplit(*args, **kwargs):
    return None


def cuda_recurrent_gated_delta_rule(mixed_qkv, g, beta, recurrent_state, core_attn_out,
                                    num_k_heads, num_v_heads, k_head_dim, v_head_dim, slots, history):
    batch, length, _ = mixed_qkv.shape
    if num_v_heads % num_k_heads or (history and recurrent_state.shape[1] < length):
        raise ValueError('GDN の head または履歴サイズが不正です')
    key_size, value_size = num_k_heads * k_head_dim, num_v_heads * v_head_dim
    q, k, v = mixed_qkv.split((key_size, key_size, value_size), dim=-1)
    q = q.float().reshape(batch, length, num_k_heads, k_head_dim)
    k = k.float().reshape(batch, length, num_k_heads, k_head_dim)
    v = v.float().reshape(batch, length, num_v_heads, v_head_dim)
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    group = num_v_heads // num_k_heads
    q = q.repeat_interleave(group, dim=2)
    k = k.repeat_interleave(group, dim=2)
    indices = torch.arange(batch, device=mixed_qkv.device) if slots is None else slots.long()
    state = recurrent_state[:, 0].index_select(0, indices)
    for step in range(length):
        decay = g[:, step].float().exp()
        decay = decay.unsqueeze(-1).unsqueeze(-1) if decay.ndim == 2 else decay.unsqueeze(-1)
        state = state * decay
        delta = (v[:, step] - (state * k[:, step].unsqueeze(-1)).sum(-2)) * beta[:, step].float().unsqueeze(-1)
        state = state + k[:, step].unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, step].copy_((state * q[:, step].unsqueeze(-1)).sum(-2) * (k_head_dim ** -0.5))
        if history and step + 1 < length:
            recurrent_state[:, step + 1].index_copy_(0, indices, state)
    recurrent_state[:, 0].index_copy_(0, indices, state)


def gr_mix(streams, fn, upt, w, rms_eps, dots, post, mixed):
    rows, heads, dim = streams.shape
    rank = upt.shape[2]
    inverse = torch.rsqrt(streams.square().mean(dim=-1) + rms_eps)
    products = torch.einsum('rhd,mhd->rmh', streams, fn.float().reshape(-1, heads, dim))
    dots[:, :-1].copy_(products)
    dots[:, -1].copy_(streams.square().sum(dim=-1))
    projection = (products * inverse[:, None, :]).sum(dim=-1) / heads
    if post is not None:
        post.copy_(2 * torch.sigmoid(projection[:, rank:]))
    activated = F.silu(projection[:, :rank])
    up = upt.permute(0, 1, 3, 2).reshape(heads, dim, rank).float()
    gate = torch.sigmoid(torch.einsum('rk,hdk->rhd', activated, up))
    normalized = streams * inverse.unsqueeze(-1) * w.float().reshape(heads, dim)
    mixed.copy_((gate * normalized).mean(dim=1))


def ple_forward_streams(streams, emb, key_w, value_w, norm_key_w, norm_query_w,
                        norm_conv_w, conv_w, conv_state, rms_eps, gate_scale,
                        conv_dilation, delta, conv_stream):
    batch, seq, heads, dim = streams.shape
    key = (emb @ key_w).reshape(batch, seq, heads, dim)
    query = torch.empty_like(streams)
    key_norm = torch.empty_like(streams)
    rms_norm(key, norm_key_w, key_norm, rms_eps, 1.0, 1.0, False, False, heads)
    rms_norm(streams, norm_query_w, query, rms_eps, 1.0, 1.0, False, False, heads)
    gate = (query * key_norm).sum(dim=-1)
    gated = torch.empty_like(streams)
    ple_gate(gate, emb @ value_w, gated, gate_scale)
    normed = torch.empty_like(gated, dtype=emb.dtype)
    rms_norm(gated, norm_conv_w, normed, rms_eps, 1.0, 1.0, False, False, heads)
    state_len = conv_stream.shape[-1] - seq
    if state_len:
        if conv_state is None:
            conv_stream[..., :state_len].zero_()
        else:
            conv_stream[..., :state_len].copy_(conv_state)
    conv_stream[..., state_len:].copy_(normed.reshape(batch, seq, heads * dim).transpose(1, 2))
    conv = F.silu(F.conv1d(conv_stream, conv_w, groups=heads * dim, dilation=conv_dilation))
    delta.copy_(gated + conv.transpose(1, 2).reshape_as(gated))


def blocksparse_mlp_routing(bsz, cfg, y, params):
    all_experts = params.get('activate_all_experts', False)
    count = cfg.num_experts if all_experts else cfg.num_experts_per_tok
    logits = y @ cfg.gate_tensor
    weights, selected = torch.topk(logits, count, dim=-1, sorted=not (bsz == 1 and not all_experts))
    weights = torch.softmax(weights, dim=-1)
    if bsz == 1 and not all_experts:
        cfg.router_logits_bsz1.copy_(logits)
        cfg.routing_weights_bsz1.copy_(weights)
        cfg.selected_experts_bsz1.copy_(selected)
        return cfg.selected_experts_bsz1, cfg.routing_weights_bsz1
    return selected, weights
