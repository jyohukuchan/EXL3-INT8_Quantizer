import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / 'exl3_int8'
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))
os.environ.setdefault('EXL3_BACKEND', 'triton')
if not hasattr(sys.modules.get('exllamav3_ext'), 'silu_mul'):
    sys.modules.pop('exllamav3_ext', None)

import exllamav3
from exllamav3 import Config, Model, Cache
from exllamav3.conversion.low_disk import ShardManager, plan_shards
from exllamav3.loader.safetensors import SafetensorsCollection
from exllamav3.modules.attention_fn.bc_attn import build_bc_attn
from exllamav3.util.rope import RopeSettings, RopeStyle, RoPE
import importlib.util

_val_spec = importlib.util.spec_from_file_location('validate_exl3', ROOT / 'validate_exl3.py')
_validator = importlib.util.module_from_spec(_val_spec)
_val_spec.loader.exec_module(_validator)


def _hy_mt3_config_dict(variant: str = '1.8b') -> dict:
    rope_scaling = {
        'type': 'dynamic',
        'alpha': 1000.0,
        'factor': 1.0,
        'beta_fast': 32,
        'beta_slow': 1,
        'mscale': 1.0,
        'mscale_all_dim': 1.0,
    }
    if variant == '1.8b':
        return {
            'architectures': ['HunYuanDenseV1ForCausalLM'],
            'model_type': 'hunyuan_v1_dense',
            'hidden_size': 2048,
            'intermediate_size': 6144,
            'num_hidden_layers': 32,
            'num_attention_heads': 16,
            'num_key_value_heads': 4,
            'head_dim': 128,
            'vocab_size': 120818,
            'tie_word_embeddings': True,
            'rms_norm_eps': 1e-5,
            'hidden_act': 'silu',
            'attention_bias': False,
            'mlp_bias': False,
            'use_qk_norm': True,
            'use_cla': False,
            'rope_theta': 10000.0,
            'rope_scaling': rope_scaling,
            'max_position_embeddings': 32768,
        }
    return {
        'architectures': ['HunYuanDenseV1ForCausalLM'],
        'model_type': 'hunyuan_v1_dense',
        'hidden_size': 4096,
        'intermediate_size': 14336,
        'num_hidden_layers': 32,
        'num_attention_heads': 32,
        'num_key_value_heads': 8,
        'head_dim': 128,
        'vocab_size': 128167,
        'tie_word_embeddings': True,
        'rms_norm_eps': 1e-5,
        'hidden_act': 'silu',
        'attention_bias': False,
        'mlp_bias': False,
        'use_qk_norm': True,
        'use_cla': False,
        'rope_theta': 10000.0,
        'rope_scaling': rope_scaling,
        'max_position_embeddings': 32768,
    }


def _write_tokenizer(directory: Path, vocab_size: int) -> None:
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers import models, pre_tokenizers

    vocab = {'<unk>': 0, '<bos>': 1, '<eos>': 2, '<pad>': 3}
    next_id = 4
    for code in range(32, 127):
        ch = chr(code)
        if ch not in vocab:
            vocab[ch] = next_id
            next_id += 1
    for code in ('\n', '\t', ' '):
        if code not in vocab:
            vocab[code] = next_id
            next_id += 1
    while next_id < vocab_size:
        vocab[f'<tok_{next_id}>'] = next_id
        next_id += 1
    tok = HFTokenizer(models.WordLevel(vocab=vocab, unk_token='<unk>'))
    tok.pre_tokenizer = pre_tokenizers.Split('', behavior='isolated')
    tok.save(str(directory / 'tokenizer.json'))
    (directory / 'tokenizer_config.json').write_text(
        json.dumps({'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>', 'unk_token': '<unk>'})
    )


def _make_synthetic_weights(cfg: dict, seed: int = 42, dtype: torch.dtype = torch.bfloat16) -> dict:
    gen = torch.Generator(device='cpu').manual_seed(seed)
    h = cfg['hidden_size']
    interm = cfg['intermediate_size']
    nq = cfg['num_attention_heads']
    nkv = cfg['num_key_value_heads']
    hd = cfg['head_dim']
    vocab = cfg['vocab_size']
    n_layers = cfg['num_hidden_layers']

    def randw(shape, scale=0.05):
        return (torch.randn(shape, generator=gen, dtype=torch.float32) * scale).to(dtype)

    def randn_norm(dim):
        return (1.0 + torch.randn((dim,), generator=gen, dtype=torch.float32) * 0.1).to(dtype)

    tensors = {
        'model.embed_tokens.weight': randw((vocab, h), 0.1),
        'model.norm.weight': randn_norm(h),
    }
    for i in range(n_layers):
        p = f'model.layers.{i}'
        tensors[f'{p}.input_layernorm.weight'] = randn_norm(h)
        tensors[f'{p}.self_attn.q_proj.weight'] = randw((nq * hd, h))
        tensors[f'{p}.self_attn.k_proj.weight'] = randw((nkv * hd, h))
        tensors[f'{p}.self_attn.v_proj.weight'] = randw((nkv * hd, h))
        tensors[f'{p}.self_attn.o_proj.weight'] = randw((h, nq * hd))
        if cfg.get('use_qk_norm', True):
            tensors[f'{p}.self_attn.query_layernorm.weight'] = randn_norm(hd)
            tensors[f'{p}.self_attn.key_layernorm.weight'] = randn_norm(hd)
        tensors[f'{p}.post_attention_layernorm.weight'] = randn_norm(h)
        tensors[f'{p}.mlp.gate_proj.weight'] = randw((interm, h))
        tensors[f'{p}.mlp.up_proj.weight'] = randw((interm, h))
        tensors[f'{p}.mlp.down_proj.weight'] = randw((h, interm))
    return tensors


def _build_synthetic_dir(
    directory: Path,
    num_layers: int = 2,
    hidden_size: int = 256,
    intermediate_size: int = 512,
    num_q_heads: int = 2,
    num_kv_heads: int = 2,
    head_dim: int = 128,
    vocab_size: int = 300,
    seed: int = 42,
    sharded: bool = False,
) -> dict:
    cfg = {
        'architectures': ['HunYuanDenseV1ForCausalLM'],
        'model_type': 'hunyuan_v1_dense',
        'hidden_size': hidden_size,
        'intermediate_size': intermediate_size,
        'num_hidden_layers': num_layers,
        'num_attention_heads': num_q_heads,
        'num_key_value_heads': num_kv_heads,
        'head_dim': head_dim,
        'vocab_size': vocab_size,
        'tie_word_embeddings': True,
        'rms_norm_eps': 1e-5,
        'hidden_act': 'silu',
        'attention_bias': False,
        'mlp_bias': False,
        'use_qk_norm': True,
        'use_cla': False,
        'rope_theta': 10000.0,
        'rope_scaling': {
            'type': 'dynamic',
            'alpha': 1000.0,
            'factor': 1.0,
            'beta_fast': 32,
            'beta_slow': 1,
            'mscale': 1.0,
            'mscale_all_dim': 1.0,
        },
        'max_position_embeddings': 4096,
        'bos_token_id': 1,
        'eos_token_id': 2,
        'pad_token_id': 3,
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'config.json').write_text(json.dumps(cfg, indent=2))
    _write_tokenizer(directory, vocab_size)
    tensors = _make_synthetic_weights(cfg, seed=seed)
    if sharded:
        shard0 = {k: v for k, v in tensors.items() if k.startswith('model.embed_tokens') or k.startswith('model.layers.0.')}
        shard1 = {k: v for k, v in tensors.items() if k not in shard0}
        save_file(shard0, str(directory / 'model-00001-of-00002.safetensors'))
        save_file(shard1, str(directory / 'model-00002-of-00002.safetensors'))
        weight_map = {k: 'model-00001-of-00002.safetensors' for k in shard0}
        weight_map.update({k: 'model-00002-of-00002.safetensors' for k in shard1})
        (directory / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': weight_map}, indent=2))
    else:
        save_file(tensors, str(directory / 'model.safetensors'))
    return cfg


@pytest.mark.parametrize('variant', ['1.8b', '7b'])
def test_hunyuan_dense_config_and_model_structure(tmp_path, variant):
    cfg_dict = _hy_mt3_config_dict(variant)
    (tmp_path / 'config.json').write_text(json.dumps(cfg_dict))
    save_file({'model.layers.0.self_attn.query_layernorm.weight': torch.ones(128, dtype=torch.bfloat16)}, str(tmp_path / 'model.safetensors'))

    config = Config.from_directory(str(tmp_path))
    assert config.architecture == 'HunYuanDenseV1ForCausalLM'
    assert config.hidden_size == cfg_dict['hidden_size']
    assert config.intermediate_size == cfg_dict['intermediate_size']
    assert config.num_hidden_layers == 32
    assert config.num_q_heads == cfg_dict['num_attention_heads']
    assert config.num_kv_heads == cfg_dict['num_key_value_heads']
    assert config.head_dim == 128
    assert config.vocab_size == cfg_dict['vocab_size']
    assert config.tie_word_embeddings is True
    assert config.use_qk_norm is True
    assert config.rms_norm_eps == pytest.approx(1e-5)

    model = Model.from_config(config)
    assert len(model.modules) == 1 + 32 + 2
    assert model.modules[0].key == 'model.embed_tokens'
    assert model.modules[-2].key == 'model.norm'
    assert model.modules[-1].key == 'lm_head'
    assert model.modules[-1].alt_key == 'model.embed_tokens'

    block0 = model.modules[1]
    assert block0.attn.q_norm.key == 'model.layers.0.self_attn.query_layernorm'
    assert block0.attn.k_norm.key == 'model.layers.0.self_attn.key_layernorm'
    assert block0.attn.post_rope_norm is True


def test_hunyuan_dense_qk_norm_detection_and_rope_parameters(tmp_path):
    no_flag_dir = tmp_path / 'no_flag'
    no_flag_dir.mkdir()
    cfg = _hy_mt3_config_dict('1.8b')
    cfg.pop('use_qk_norm')
    cfg['rope_parameters'] = {**cfg.pop('rope_scaling'), 'rope_type': 'dynamic', 'rope_theta': cfg.pop('rope_theta')}
    (no_flag_dir / 'config.json').write_text(json.dumps(cfg))
    save_file({'model.layers.0.self_attn.query_layernorm.weight': torch.ones(128, dtype=torch.bfloat16)}, str(no_flag_dir / 'model.safetensors'))

    c1 = Config.from_directory(str(no_flag_dir))
    assert c1.use_qk_norm is True
    r1 = RoPE('cpu', c1.rope_settings)
    base = 10000.0 * (1000.0 ** (128 / 126))
    expected_inv = 1.0 / (base ** (torch.arange(0, 128, 2, dtype=torch.int64).float() / 128))
    assert torch.equal(r1.inv_freq, expected_inv)

    disabled_dir = tmp_path / 'disabled'
    disabled_dir.mkdir()
    cfg_disabled = _hy_mt3_config_dict('1.8b')
    cfg_disabled['use_qk_norm'] = False
    (disabled_dir / 'config.json').write_text(json.dumps(cfg_disabled))
    save_file({}, str(disabled_dir / 'model.safetensors'))
    c2 = Config.from_directory(str(disabled_dir))
    assert c2.use_qk_norm is False
    m2 = Model.from_config(c2)
    assert m2.modules[1].attn.q_norm is None
    assert m2.modules[1].attn.k_norm is None
    assert m2.modules[1].attn.post_rope_norm is False


def test_hunyuan_dense_rejects_cla_and_biases(tmp_path):
    for key in ('use_cla', 'attention_bias', 'mlp_bias'):
        d = tmp_path / key
        d.mkdir()
        cfg = _hy_mt3_config_dict('1.8b')
        cfg[key] = True
        (d / 'config.json').write_text(json.dumps(cfg))
        save_file({}, str(d / 'model.safetensors'))
        with pytest.raises(ValueError, match=key):
            Config.from_directory(str(d))


def test_hunyuan_dense_shard_plan_keeps_tied_embed_until_lm_head(tmp_path):
    cfg = _build_synthetic_dir(tmp_path, num_layers=2, sharded=True)
    config = Config.from_directory(str(tmp_path))
    model = Model.from_config(config)
    manager = ShardManager(tmp_path, lambda name, path: None)
    plan_shards(manager, config.stc, model.modules, set())
    shard0 = str((tmp_path / 'model-00001-of-00002.safetensors').resolve())
    shard1 = str((tmp_path / 'model-00002-of-00002.safetensors').resolve())
    assert manager.last_use[shard0] == len(model.modules) - 1
    assert manager.last_use[shard1] == len(model.modules) - 2
    assert cfg['tie_word_embeddings'] is True


def test_ntk_alpha_rope_matches_transformers_reference():
    transformers_cfg_mod = pytest.importorskip('transformers.models.hunyuan_v1_dense.configuration_hunyuan_v1_dense')
    transformers_mod = pytest.importorskip('transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense')

    hf_cfg = transformers_cfg_mod.HunYuanDenseV1Config(
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=128,
        vocab_size=120818,
        rms_norm_eps=1e-5,
        rope_parameters={
            'rope_type': 'dynamic',
            'alpha': 1000.0,
            'factor': 1.0,
            'beta_fast': 32,
            'beta_slow': 1,
            'mscale': 1.0,
            'mscale_all_dim': 1.0,
            'rope_theta': 10000.0,
        },
    )
    hf_rope = transformers_mod.HunYuanDenseV1RotaryEmbedding(hf_cfg)
    assert hf_rope.rope_type == 'ntk_alpha'

    rs = RopeSettings(
        head_dim=128,
        rope_theta=10000.0,
        rope_scaling={
            'type': 'dynamic',
            'alpha': 1000.0,
            'factor': 1.0,
            'beta_fast': 32,
            'beta_slow': 1,
            'mscale': 1.0,
            'mscale_all_dim': 1.0,
        },
        rope_style=RopeStyle.NEOX,
    )
    exl_rope = RoPE('cpu', rs)
    assert torch.equal(exl_rope.inv_freq, hf_rope.inv_freq)
    assert exl_rope.attn_factor == 1.0

    pos_ids = torch.tensor([[0, 5, 17, 1023]], dtype=torch.long)
    dummy = torch.zeros((1, 4, 128), dtype=torch.float32)
    hf_cos, hf_sin = hf_rope(dummy, pos_ids)

    torch.manual_seed(7)
    q = torch.randn((1, 4, 16, 128), dtype=torch.float16)
    k = torch.randn((1, 4, 4, 128), dtype=torch.float16)
    hf_q, hf_k = transformers_mod.apply_rotary_pos_emb(
        q.float().transpose(1, 2),
        k.float().transpose(1, 2),
        hf_cos,
        hf_sin,
    )
    hf_q = hf_q.transpose(1, 2).half()
    hf_k = hf_k.transpose(1, 2).half()

    exl_q, exl_k = exl_rope.apply_torch(q, k, position_ids=pos_ids)
    torch.testing.assert_close(exl_q, hf_q, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(exl_k, hf_k, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
def test_post_rope_qk_norm_and_forward_match_transformers(tmp_path):
    transformers_cfg_mod = pytest.importorskip('transformers.models.hunyuan_v1_dense.configuration_hunyuan_v1_dense')
    transformers_mod = pytest.importorskip('transformers.models.hunyuan_v1_dense.modeling_hunyuan_v1_dense')

    model_dir = tmp_path / 'hf_model'
    cfg = _build_synthetic_dir(
        model_dir,
        num_layers=2,
        hidden_size=256,
        intermediate_size=512,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=128,
        vocab_size=300,
        seed=123,
    )

    hf_cfg = transformers_cfg_mod.HunYuanDenseV1Config(
        hidden_size=cfg['hidden_size'],
        intermediate_size=cfg['intermediate_size'],
        num_hidden_layers=cfg['num_hidden_layers'],
        num_attention_heads=cfg['num_attention_heads'],
        num_key_value_heads=cfg['num_key_value_heads'],
        head_dim=cfg['head_dim'],
        vocab_size=cfg['vocab_size'],
        tie_word_embeddings=True,
        rms_norm_eps=cfg['rms_norm_eps'],
        hidden_act=cfg['hidden_act'],
        attention_bias=False,
        mlp_bias=False,
        rope_parameters={**cfg['rope_scaling'], 'rope_type': 'dynamic', 'rope_theta': cfg['rope_theta']},
        max_position_embeddings=cfg['max_position_embeddings'],
    )
    hf_model = transformers_mod.HunYuanDenseV1ForCausalLM.from_pretrained(str(model_dir), config=hf_cfg, dtype=torch.float32)
    hf_model.eval().cuda()

    exl_config = Config.from_directory(str(model_dir))
    exl_model = Model.from_config(exl_config)
    cache = Cache(exl_model, max_num_tokens=256)
    exl_model.load(device='cuda:0')
    try:
        for block in exl_model.modules[1:-2]:
            assert block.attn.q_norm_tensor is None
            assert block.attn.k_norm_tensor is None

        input_ids = torch.tensor([[1, 42, 17, 255, 88, 9]], dtype=torch.long)
        with torch.no_grad():
            hf_out = hf_model(input_ids.cuda(), use_cache=False).logits.cpu().float()

        exl_out_nc = exl_model.forward(input_ids, {'attn_mode': 'flash_attn_nc'}).cpu().float()
        assert exl_out_nc.shape == (1, 6, 384)
        assert (exl_out_nc[..., 300:] == 0).all()
        exl_out_nc = exl_out_nc[..., :300]
        torch.testing.assert_close(exl_out_nc, hf_out, rtol=2e-2, atol=2e-2)
        cos_sim = torch.nn.functional.cosine_similarity(exl_out_nc.flatten(), hf_out.flatten(), dim=0).item()
        assert cos_sim > 0.9999

        for block in exl_model.modules[1:-2]:
            layer = cache.layers[block.attn.layer_idx, 0]
            assert build_bc_attn(block.attn, layer) is None

        logits_prefill = exl_model.forward(
            input_ids[:, :4],
            {'attn_mode': 'flash_attn', 'cache': cache, 'batch_shape': (1, 256), 'past_len': 0},
        ).cpu().float()[..., :300]
        logits_step1 = exl_model.forward(
            input_ids[:, 4:5],
            {'attn_mode': 'flash_attn', 'cache': cache, 'batch_shape': (1, 256), 'past_len': 4},
        ).cpu().float()[..., :300]
        logits_step2 = exl_model.forward(
            input_ids[:, 5:6],
            {'attn_mode': 'flash_attn', 'cache': cache, 'batch_shape': (1, 256), 'past_len': 5},
        ).cpu().float()[..., :300]
        exl_out_cached = torch.cat([logits_prefill, logits_step1, logits_step2], dim=1)
        torch.testing.assert_close(exl_out_cached, hf_out, rtol=2e-2, atol=2e-2)
        cos_sim_cached = torch.nn.functional.cosine_similarity(exl_out_cached.flatten(), hf_out.flatten(), dim=0).item()
        assert cos_sim_cached > 0.9999

        attn0 = exl_model.modules[1].attn
        attn0.post_rope_norm = False
        attn0.q_norm_tensor = attn0.q_norm.weight.data
        attn0.k_norm_tensor = attn0.k_norm.weight.data
        wrong_out = exl_model.forward(input_ids, {'attn_mode': 'flash_attn_nc'}).cpu().float()[..., :300]
        max_wrong_diff = (wrong_out - hf_out).abs().max().item()
        max_right_diff = (exl_out_nc - hf_out).abs().max().item()
        assert max_wrong_diff > max_right_diff * 5.0
    finally:
        exl_model.unload()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU が必要です')
@pytest.mark.parametrize('low_disk', [False, True])
def test_quantize_and_validate_hunyuan_dense(tmp_path, low_disk):
    model_dir = tmp_path / 'model'
    work_dir = tmp_path / 'work'
    out_dir = tmp_path / 'out'
    _build_synthetic_dir(
        model_dir,
        num_layers=2,
        hidden_size=256,
        intermediate_size=512,
        num_q_heads=2,
        num_kv_heads=2,
        head_dim=128,
        vocab_size=300,
        seed=99,
        sharded=low_disk,
    )

    cmd = [
        sys.executable,
        str(ROOT / 'quantize.py'),
        'convert',
        '--backend', 'triton',
        '--model', str(model_dir),
        '--output', str(out_dir),
        '--work', str(work_dir),
        '--bpw', '4.0',
        '--cal-rows', '4',
        '--cal-cols', '32',
    ]
    if low_disk:
        cmd.extend(['--low-disk', '--shard-size', '1'])

    subprocess.run(cmd, cwd=str(ROOT), check=True)

    count = _validator.validate(out_dir)
    assert count == 2 * 7 + 1

    report = json.loads((out_dir / 'quantize_report.json').read_text())
    assert report['extension']['multiplier'] == 0x00DCD12D
    assert report['backend'] == 'triton'

    q_cfg = Config.from_directory(str(out_dir))
    q_model = Model.from_config(q_cfg)
    cache = Cache(q_model, max_num_tokens=256)
    q_model.load(device='cuda:0')
    try:
        input_ids = torch.tensor([[1, 10, 20, 30]], dtype=torch.long)
        logits_nc = q_model.forward(input_ids, {'attn_mode': 'flash_attn_nc'})
        assert logits_nc.shape == (1, 4, 384)
        assert torch.isfinite(logits_nc[..., :300]).all()

        logits_cached = q_model.forward(
            input_ids,
            {'attn_mode': 'flash_attn', 'cache': cache, 'batch_shape': (1, 256), 'past_len': 0},
        )
        assert logits_cached.shape == (1, 4, 384)
        assert torch.isfinite(logits_cached[..., :300]).all()
    finally:
        q_model.unload()
