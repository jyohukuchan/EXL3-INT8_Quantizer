import torch
import torch.nn.functional as F

from .loader import _device_bytes


def _pointer_view(address, shape, dtype, device):
    if not address:
        raise ValueError('tensor pointer が null です')
    count = 1
    for size in shape:
        count *= size
    size = torch.empty((), dtype=dtype).element_size()
    return _device_bytes(address, count * size, device.index or 0).view(dtype).view(shape)


def _linear(x, ptr, scale_in, scale_out, bits, mcg, mul1, width, stride=None, output_dtype=torch.float16, half_matmul=False):
    from . import reconstruct_slice, had_r_128
    rows, features = x.shape
    stride = width if stride is None else stride
    packed = _pointer_view(ptr, (features // 16, stride // 16, bits * 16), torch.int16, x.device)
    suh = _pointer_view(scale_in, (features,), torch.float16, x.device)
    svh = _pointer_view(scale_out, (width,), torch.float16, x.device)
    weight = torch.empty((features, width), dtype=torch.float16, device=x.device)
    reconstruct_slice(weight, packed, bits, mcg, mul1, 0)
    transformed = torch.empty_like(x)
    had_r_128(x, transformed, suh, None, 1.0)
    result = torch.empty((rows, width), dtype=output_dtype, device=x.device)
    result.copy_(transformed.float() @ weight.float() if output_dtype == torch.float32 and not half_matmul else transformed @ weight)
    had_r_128(result, result, None, svh, 1.0)
    return result, transformed


def exl3_mgemm(A, B, C, suh, A_had, svh, indices, weights, K, force_shape_idx, mcg, mul1,
                min_index, max_index, force_num_sms, num_tokens=1, size_n_list=None, c_ptrs=None,
                n_stride_list=None, had_src_list=None, num_had_src=0):
    if A.ndim != 3 or C.ndim != 3 or A.shape[1] != C.shape[1] or K < 1 or K > 8 or mcg or not mul1:
        raise ValueError('EXL3 MGEMM の引数が不正です')
    if (size_n_list is None) != (c_ptrs is None) or (had_src_list is None) != (n_stride_list is None):
        raise ValueError('EXL3 MGEMM の pointer 配列が不正です')
    if size_n_list is not None and (num_tokens != 1 or min_index >= 0 or weights is not None):
        raise ValueError('EXL3 MGEMM の可変幅と重み付き集約は併用できません')
    ptrs, ins, outs = B.tolist(), suh.tolist(), svh.tolist()
    widths = size_n_list.tolist() if size_n_list is not None else None
    dests = c_ptrs.tolist() if c_ptrs is not None else None
    strides = n_stride_list.tolist() if n_stride_list is not None else None
    sources = had_src_list.tolist() if had_src_list is not None else None
    total = max(A.shape[0], len(dests) if dests is not None else C.shape[0])
    selected = indices.reshape(-1).tolist() if indices is not None else list(range(total))
    if len(selected) < total:
        total = len(selected)
    w = weights.reshape(-1).tolist() if weights is not None else None
    if min_index >= 0:
        if indices is None:
            raise ValueError('EXL3 MGEMM の expert indices がありません')
        if num_tokens == 1:
            active = [(i - min_index, w[j] if w is not None else None) for j, i in enumerate(selected[:total]) if min_index <= i < max_index]
            selected = [v[0] for v in active]
            w = [v[1] for v in active] if w is not None else None
            total = len(active)
        else:
            selected = [i - min_index if min_index <= i < max_index else -1 for i in selected]
    if num_tokens < 1 or (weights is not None and total % num_tokens):
        raise ValueError('EXL3 MGEMM の token 数が不正です')
    output = []
    for j in range(total):
        idx = selected[j]
        if idx < 0:
            output.append(None)
            continue
        if idx >= len(ptrs):
            raise ValueError('EXL3 MGEMM の expert index が不正です')
        width = widths[idx] if widths is not None else C.shape[2]
        src = sources[idx] if sources is not None else idx
        if sources is not None and not 0 <= src < num_had_src:
            raise ValueError('EXL3 MGEMM の Hadamard source が不正です')
        x = A[0 if A.shape[0] == 1 else j]
        result, transformed = _linear(x, ptrs[idx], ins[src], outs[idx], K, mcg, mul1, width,
                                      strides[idx] if strides is not None else None, C.dtype)
        scratch_idx = src if sources is not None else j
        A_had.view(-1, *x.shape)[scratch_idx].copy_(transformed)
        if w is not None:
            result.mul_(w[j])
        if dests is not None:
            stride = strides[idx] if strides is not None else width
            target = _pointer_view(dests[idx], (A.shape[1], stride), C.dtype, C.device)
            target[:, :width].copy_(result)
        else:
            C[j].copy_(result)
        output.append(result)
    if w is not None:
        group = total // num_tokens
        for token in range(num_tokens):
            acc = torch.zeros_like(C[token])
            for result in output[token * group:(token + 1) * group]:
                if result is not None:
                    acc.add_(result)
            C[token].copy_(acc)
    return 0


def exl3_moe_max_concurrency(device):
    return max(1, torch.cuda.get_device_properties(device).multi_processor_count // 8)


def exl3_moe(hidden_state, output_state, expert_count, token_sorted, weight_sorted,
             temp_state_g, temp_state_u, temp_intermediate_g, temp_intermediate_u, act_function,
             K_gate, K_up, K_down, gate_ptrs_trellis, gate_ptrs_suh, gate_ptrs_svh,
             up_ptrs_trellis, up_ptrs_suh, up_ptrs_svh, down_ptrs_trellis, down_ptrs_suh,
             down_ptrs_svh, gate_mcg, gate_mul1, up_mcg, up_mul1, down_mcg, down_mul1,
             act_limit, num_active, output_scratch, fused_base, count_lo, count_hi, m_tile):
    if num_active == 0:
        return
    if act_function not in (0, 1, 2) or any((gate_mcg, up_mcg, down_mcg)) or not all((gate_mul1, up_mul1, down_mul1)):
        raise ValueError('EXL3 MoE の activation または codebook が不正です')
    if output_scratch is not None and fused_base is None:
        raise ValueError('EXL3 MoE の fused_base がありません')
    tables = [t.tolist() for t in (gate_ptrs_trellis, gate_ptrs_suh, gate_ptrs_svh,
                                   up_ptrs_trellis, up_ptrs_suh, up_ptrs_svh,
                                   down_ptrs_trellis, down_ptrs_suh, down_ptrs_svh)]
    counts = expert_count.tolist()[:-1]
    bases = fused_base.tolist() if fused_base is not None else None
    hidden = hidden_state.shape[1]
    intermediate = temp_intermediate_u.shape[2]
    start = 0
    for expert, count in enumerate(counts):
        end = start + count
        if count and count_lo <= count <= count_hi and count <= temp_state_g.shape[1]:
            tokens = token_sorted[start:end]
            source = hidden_state.index_select(0, tokens)
            up, _ = _linear(source, tables[3][expert], tables[4][expert], tables[5][expert], K_up, up_mcg, up_mul1, intermediate)
            if act_function == 2:
                activated = up.relu()
                if act_limit:
                    activated = activated.clamp(max=act_limit)
                    up = up.clamp(-act_limit, act_limit)
                activated = (activated * up).half()
            else:
                gate, _ = _linear(source, tables[0][expert], tables[1][expert], tables[2][expert], K_gate, gate_mcg, gate_mul1, intermediate)
                activated = F.silu(gate) if act_function == 0 else F.gelu(gate, approximate='tanh')
                if act_limit:
                    activated = activated.clamp(max=act_limit)
                    up = up.clamp(-act_limit, act_limit)
                activated = (activated * up).half()
            result, _ = _linear(activated, tables[6][expert], tables[7][expert], tables[8][expert], K_down, down_mcg, down_mul1, hidden, output_dtype=torch.float32, half_matmul=True)
            result.mul_(weight_sorted[start:end].float().unsqueeze(-1))
            if output_scratch is None:
                output_state.index_add_(0, tokens, result)
            else:
                output_scratch[bases[expert]:bases[expert] + count].copy_(result)
        start = end


class BC_BlockSparseMLP:
    def __init__(self, yh2, yh, interm_gu, interm_g, interm_u, interm_a, interm_a2,
                 out_d, out_d2, out_d_sh, coop_ctr, had_u, dq_temp_up, dq_temp_down,
                 min_expert, max_expert, gate_ptrs_trellis, gate_ptrs_suh, gate_ptrs_svh,
                 gate_K, gate_mcg, gate_mul1, up_ptrs_trellis, up_ptrs_suh, up_ptrs_svh,
                 up_K, up_mcg, up_mul1, down_ptrs_trellis, down_ptrs_suh, down_ptrs_svh,
                 down_K, down_mcg, down_mul1, act_silu, act_gelu, act_silu_oai,
                 shared_experts, shared_gate, act_limit, gates, ups, downs,
                 gu_trellis_ptr, gu_suh_ptr, gu_svh_ptr, out_bszn,
                 gate_bias_ptrs, up_bias_ptrs, down_bias_ptrs, act_relu2=False):
        self.interm = interm_u.shape[-1]
        self.out_d2, self.out_bszn, self.out_d_sh = out_d2, out_bszn, out_d_sh
        self.min_expert, self.max_expert = min_expert, max_expert
        self.gates, self.ups, self.downs = gates, ups, downs
        self.act_silu, self.act_gelu, self.act_silu_oai, self.act_relu2 = act_silu, act_gelu, act_silu_oai, act_relu2
        self.act_limit = act_limit
        self.shared_experts, self.shared_gate = shared_experts, shared_gate
        if not gates and not act_relu2:
            raise ValueError('gate がない expert には relu2 が必要です')

    def _project(self, linear, x, width, dtype):
        input_width = linear.suh.numel()
        if x.shape[-1] > input_width:
            raise ValueError('expert 入力の幅が不正です')
        if x.shape[-1] < input_width:
            x = F.pad(x, (0, input_width - x.shape[-1]))
        return linear.run_alloc(x.contiguous(), width, dtype == torch.float32)

    def _run_expert(self, x, expert, output_width):
        from .ops import silu_mul, gelu_mul, relu2_mul, silu_oai_mul, relu_mul
        up = self._project(self.ups[expert], x, self.interm, torch.float16)
        gate = self._project(self.gates[expert], x, self.interm, torch.float16) if self.gates else up
        activated = torch.empty_like(up)
        fn = (relu_mul if not self.gates else silu_oai_mul if self.act_silu_oai else
              gelu_mul if self.act_gelu else relu2_mul if self.act_relu2 else silu_mul)
        fn(gate, up, activated, self.act_limit)
        down = self.downs[expert]
        full = self._project(down, activated, down.svh.numel(), torch.float32)
        return full[..., :output_width]

    def run_single_expert(self, y, expert_idx):
        self.out_d2[:y.shape[0]].copy_(self._run_expert(y, expert_idx, self.out_d2.shape[-1]))

    def run_single_expert_dq(self, y, expert_idx, yh, interm, interm_a, out):
        out.copy_(self._run_expert(y, expert_idx, out.shape[-1]))

    def run_bszN(self, y, selected_experts, routing_weights):
        out = self.out_bszn[:y.shape[0]]
        out.zero_()
        indices = selected_experts.tolist()
        for row, choices in enumerate(indices):
            for slot, expert in enumerate(choices):
                if self.min_expert >= 0:
                    if not self.min_expert <= expert < self.max_expert:
                        continue
                    expert -= self.min_expert
                if not 0 <= expert < len(self.ups):
                    continue
                value = self._run_expert(y[row:row + 1], expert, out.shape[-1])
                out[row:row + 1].add_(value * routing_weights[row, slot].float())
        if self.shared_experts is not None:
            shared = self.out_d_sh[:, :y.shape[0]]
            self.shared_experts.run_bszN(y.unsqueeze(0), shared)
            value = shared.reshape(y.shape[0], -1).float()
            if self.shared_gate is not None:
                gate = torch.empty((y.shape[0], 1), dtype=y.dtype, device=y.device)
                self.shared_gate.run(y, gate)
                value = value * torch.sigmoid(gate.float())
            out.add_(value[:, :out.shape[-1]])


def exl3_moe_gather(output_state, output_scratch, flat_expert, inv_order, expert_start,
                    slot_base, slot_kind, weight_sorted):
    tokens = output_state.shape[0]
    if not tokens:
        return
    if flat_expert.numel() % tokens:
        raise ValueError('MoE gather の割り当て数が不正です')
    topk = flat_expert.numel() // tokens
    if topk > 32:
        raise ValueError('MoE gather の top-k が大きすぎます')
    experts = flat_expert.reshape(tokens, topk).tolist()
    positions = inv_order.reshape(tokens, topk).tolist()
    starts, bases, kinds = expert_start.tolist(), slot_base.tolist(), slot_kind.tolist()
    for token, choices in enumerate(experts):
        result = torch.zeros_like(output_state[token])
        for j, expert in enumerate(choices):
            if 0 <= expert < len(kinds) and kinds[expert]:
                pos = positions[token][j]
                value = output_scratch[bases[expert] + pos - starts[expert]]
                if kinds[expert] == 2:
                    value = value * weight_sorted[pos].float()
                result.add_(value)
        output_state[token].add_(result)
