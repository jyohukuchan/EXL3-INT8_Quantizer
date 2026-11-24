import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModelForImageTextToText, AutoTokenizer

from validate_exl3 import read_json, safe_path, shard_tensors, validate


ROOT = Path(__file__).resolve().parent
LIB = ROOT / 'exl3_int8' / 'exllamav3'


def calibration_inputs(tokenizer, seed, device):
    generator = torch.Generator().manual_seed(seed)
    directory = LIB / 'conversion' / 'standard_cal_data'
    streams = [tokenizer.encode((directory / name).read_text(), add_special_tokens=False)
               for name in ('wiki.utf8', 'code.utf8', 'multilingual.utf8', 'technical.utf8')]
    inputs = []
    for i in range(16):
        stream = streams[i % len(streams)]
        offset = torch.randint(len(stream) - 127, (), generator=generator).item()
        inputs.append(torch.tensor([stream[offset:offset + 128]], device=device))
    return inputs


def hadamard128(tensor):
    blocks = tensor.contiguous().view(-1, 128)
    for step in (1, 2, 4, 8, 16, 32, 64):
        left, right = blocks.view(-1, 2, step).unbind(1)
        blocks = torch.stack((left + right, left - right), 1)
    return blocks.reshape(tensor.shape) * (128 ** -0.5)


class ScaledLinear(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.su = torch.nn.Parameter(torch.ones(layer.in_features, device=layer.weight.device))
        self.sv = torch.nn.Parameter(torch.ones(layer.out_features, device=layer.weight.device))

    def forward(self, x):
        scaled = (x.float() * self.su).to(x.dtype)
        y = F.linear(scaled, self.layer.weight)
        y = (y.float() * self.sv).to(y.dtype)
        return y if self.layer.bias is None else y + self.layer.bias


def read_weight(tensors, name):
    with safe_open(tensors[name][2], framework='pt', device='cpu') as source:
        return source.get_tensor(name)


@torch.no_grad()
def install_student(model, tensors):
    spec = importlib.util.spec_from_file_location('exl3_int8_ref', LIB / 'util' / 'exl3_int8_ref.py')
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    roots = sorted(k.removesuffix('.trellis') for k in tensors if k.endswith('.trellis')
                   and (k.startswith('model.language_model.layers.') or k == 'lm_head.trellis'))
    if not roots:
        raise ValueError('蒸留する線形層がありません')
    for name in roots:
        layer = model.get_submodule(name)
        if name == 'lm_head' and layer.weight is model.get_input_embeddings().weight:
            layer.weight = torch.nn.Parameter(layer.weight.detach().clone(), requires_grad=False)
        packed = read_weight(tensors, name + '.trellis').numpy()
        device = layer.weight.device
        su = read_weight(tensors, name + '.suh').to(device).flatten()
        sv = read_weight(tensors, name + '.svh').to(device).flatten()
        if tuple(layer.weight.shape) != (packed.shape[1] * 16, packed.shape[0] * 16):
            raise ValueError(f'復号先とトレリスの形状が一致しません {name}')
        for start in range(0, packed.shape[1], 64):
            chunk = packed[:, start:start + 64]
            q = torch.from_numpy(ref.decode_q(chunk, mult=ref.MUL24)).to(device).float() * ref.S_FIX
            rows = slice(start * 16, (start + chunk.shape[1]) * 16)
            w = hadamard128(q) * su
            layer.weight[rows].copy_((hadamard128(w.T) * sv[rows]).T)
    for name, weight in model.named_parameters():
        if name in tensors and name.startswith('model.language_model.'):
            weight.copy_(read_weight(tensors, name))
    model.requires_grad_(False)
    scaled = {}
    for name in roots:
        scaled[name] = ScaledLinear(model.get_submodule(name))
        model.set_submodule(name, scaled[name])
    norms = {name: model.get_parameter(name).requires_grad_(True) for name in tensors
             if name.startswith('model.language_model.') and name.endswith('norm.weight')}
    return scaled, norms


def train_scales(model, inputs, teacher):
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=0.0002)
    losses = []
    for step in range(32):
        i = step % len(inputs)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=inputs[i], use_cache=False).logits[0].float()
        target = teacher[i].to(logits.device).float().log_softmax(-1)
        loss = F.kl_div(logits.log_softmax(-1), target, log_target=True, reduction='batchmean')
        if not torch.isfinite(loss):
            raise ValueError('蒸留損失が非有限です')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(loss.detach().item())
    return losses


@torch.no_grad()
def write_student(bundle, output, tensors, scaled, norms, report):
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / 'bundle'
        shutil.copytree(bundle, stage)
        for key, (_, shape, path, offset) in tensors.items():
            name, _, suffix = key.rpartition('.')
            if name in scaled and suffix in ('suh', 'svh'):
                original = read_weight(tensors, key)
                factor = scaled[name].su if suffix == 'suh' else scaled[name].sv
                value = (original.float() * factor.cpu().reshape(shape)).to(original.dtype)
            elif key in norms:
                value = norms[key].to(device='cpu', dtype=read_weight(tensors, key).dtype)
            else:
                continue
            if list(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError(f'保存するテンソルが不正です {key}')
            with (stage / path.relative_to(bundle)).open('r+b') as f:
                f.seek(offset)
                f.write(value.contiguous().view(torch.uint8).numpy().tobytes())
        validate(stage)
        (stage / 'distill_report.json').write_text(json.dumps(report, indent=2) + '\n')
        stage.rename(output)


def distill(args):
    source = args.model.resolve(strict=True)
    bundle = args.bundle.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(bundle) or output.is_relative_to(source):
        raise ValueError('出力先には入力ディレクトリ外の新しいパスを指定してください')
    validate(bundle)
    config = json.loads((source / 'config.json').read_text())
    if config.get('model_type') != 'qwen3_5' or config.get('text_config', {}).get('num_experts', 0):
        raise ValueError('蒸留には Qwen3.5 系の dense BF16 モデルが必要です')
    index = bundle / 'model.safetensors.index.json'
    shards = set(read_json(index)['weight_map'].values()) if index.exists() else {'model.safetensors'}
    tensors = {}
    for shard in sorted(shards):
        tensors.update(shard_tensors(safe_path(bundle, shard)))
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(source)
    inputs = calibration_inputs(tokenizer, args.seed, args.device)
    model = AutoModelForImageTextToText.from_pretrained(
        source, dtype=torch.bfloat16, attn_implementation='eager').eval().to(args.device)
    with torch.no_grad():
        teacher = [model(input_ids=x, use_cache=False).logits[0].cpu() for x in inputs]
    scaled, norms = install_student(model, tensors)
    losses = train_scales(model, inputs, teacher)
    report = dict(model=str(source), bundle=str(bundle), seed=args.seed, steps=len(losses),
                  rows=len(inputs), columns=128, learning_rate=0.0002, loss=losses,
                  linears=len(scaled), norms=len(norms), device=args.device, torch=torch.__version__)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_student(bundle, output, tensors, scaled, norms, report)
