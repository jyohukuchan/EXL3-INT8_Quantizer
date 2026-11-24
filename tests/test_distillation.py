import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file

def test_distill_changes_only_scales_and_norms_in_sharded_bundle(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from distillation import write_student
    from validate_exl3 import MUL24, shard_tensors

    bundle = tmp_path / 'input'
    bundle.mkdir()
    (bundle / 'config.json').write_text(json.dumps({'quantization_config': {'quant_method': 'exl3'}}))
    original = {'x.trellis': torch.zeros(8, 8, 48, dtype=torch.int16),
                'x.suh': torch.ones(128, dtype=torch.float16),
                'x.svh': torch.ones(128, dtype=torch.float16),
                'x.mul1': torch.tensor(MUL24, dtype=torch.int32),
                'input_norm.weight': torch.zeros(128, dtype=torch.bfloat16)}
    shard_names = {name: f'part-{i % 2}.safetensors' for i, name in enumerate(original)}
    tensors = {}
    for shard in sorted(set(shard_names.values())):
        path = bundle / shard
        save_file({key: value for key, value in original.items() if shard_names[key] == shard}, path)
        tensors.update(shard_tensors(path))
    (bundle / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': shard_names}))
    scales = {'x': SimpleNamespace(su=torch.full((128,), 2.0), sv=torch.full((128,), 3.0))}
    norms = {'input_norm.weight': torch.full((128,), 0.25)}
    out = tmp_path / 'output'
    write_student(bundle, out, tensors, scales, norms, {})
    expected = {**original, 'x.suh': original['x.suh'] * 2, 'x.svh': original['x.svh'] * 3,
                'input_norm.weight': norms['input_norm.weight'].bfloat16()}
    for key, shard in shard_names.items():
        assert torch.equal(load_file(bundle / shard)[key], original[key])
        assert torch.equal(load_file(out / shard)[key], expected[key])
