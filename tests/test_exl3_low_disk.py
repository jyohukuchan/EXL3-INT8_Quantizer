import ctypes
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest

import torch
from safetensors.torch import save_file, load_file


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'exl3_int8/exllamav3'
for name, path in [('exl3_test', PACKAGE), ('exl3_test.loader', PACKAGE / 'loader'), ('exl3_test.conversion', PACKAGE / 'conversion')]:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module

ext = types.ModuleType('exllamav3_ext')
ext.exl3_mul1_multiplier = lambda: 0x00DCD12D
ext.stloader_open_file = lambda name: open(name, 'rb')
ext.stloader_close_file = lambda handle: handle.close()
def read(handle, offset, size, tensor):
    handle.seek(offset)
    data = handle.read(size)
    ctypes.memmove(tensor.data_ptr(), data, size)
ext.stloader_read = read
class Job:
    def __init__(self, handle, offset, size, destination, capacity, bf16, fp32, cuda, device):
        self.handle, self.offset, self.size, self.destination = handle, offset, size, destination
        if bf16 or fp32:
            raise AssertionError('conversion unsupported by test mock')
ext.TensorLoadJob = Job
def deferred(jobs):
    for job in jobs:
        job.handle.seek(job.offset)
        ctypes.memmove(job.destination, job.handle.read(job.size), job.size)
ext.stloader_deferred_cpu = deferred
sys.modules['exllamav3_ext'] = ext
ext_module = types.ModuleType('exl3_test.ext')
ext_module.exllamav3_ext = ext
sys.modules['exl3_test.ext'] = ext_module
loader = importlib.import_module('exl3_test.loader.safetensors')
low_disk = importlib.import_module('exl3_test.conversion.low_disk')


class LowDiskTest(unittest.TestCase):
    def test_fetch_and_release_preserves_preexisting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            remote = root / 'remote'
            local = root / 'local'
            remote.mkdir()
            local.mkdir()
            save_file({'a.weight': torch.ones(2)}, str(remote / 'a.safetensors'))
            save_file({'b.weight': torch.ones(2)}, str(remote / 'b.safetensors'))
            shutil.copyfile(remote / 'b.safetensors', local / 'b.safetensors')
            (local / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a.weight': 'a.safetensors', 'b.weight': 'b.safetensors'}}))
            manager = low_disk.ShardManager(local, lambda name, path: shutil.copyfile(remote / name, path))
            collection = loader.SafetensorsCollection(str(local), load_method='python', shard_fetcher=manager.fetch)
            manager.last_use = {str(local / 'a.safetensors'): 0, str(local / 'b.safetensors'): 0}
            self.assertTrue(torch.equal(collection.get_tensor('a.weight'), torch.ones(2)))
            manager.release(0, collection)
            self.assertFalse((local / 'a.safetensors').exists())
            self.assertTrue((local / 'b.safetensors').exists())

    def test_checkpoint_storage_and_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'ckpt').mkdir()
            (root / 'ckpt_old').mkdir()
            (root / 'ckpt_old' / 'stale').write_text('old')
            args = {'work_dir': temp, 'low_disk': True, 'checkpoint_dtype': 'bf16'}
            def save_dict(name, data, options):
                self.assertFalse((root / 'ckpt_old').exists())
                (root / name).write_text(json.dumps(data))
            def save_tensor(value, name, options):
                save_file({'tensor': value}, str(root / name))
            state = torch.tensor([1.25, 2.5], dtype=torch.float32)
            self.assertTrue(low_disk.save_checkpoint(args, {'next_module_idx': 1}, state, torch.tensor([1]), save_dict, save_tensor))
            stored = load_file(str(root / 'ckpt/state.safetensors'))['tensor']
            self.assertEqual(stored.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(stored.float(), state))
            args['no_checkpoint'] = True
            self.assertFalse(low_disk.save_checkpoint(args, {}, state, state, save_dict, save_tensor))
            self.assertFalse((root / 'ckpt_new').exists())

    def test_compile_releases_and_normalizes(self):
        from unittest.mock import patch
        from safetensors import safe_open
        module_package = types.ModuleType('exl3_test.modules')
        class Module:
            def __init__(self, key):
                self.key = key
            def __iter__(self):
                return iter([self])
            def get_compile_sizes(self, stc):
                return [stc.get_tensor_size(self.key)]
            def get_compile_tensors(self, stc):
                return {self.key: stc.get_tensor(self.key)}
        class Linear(Module):
            pass
        module_package.Module = Module
        module_package.Linear = Linear
        memory = types.ModuleType('exl3_test.util.memory')
        memory.free_mem = lambda: None
        model_package = types.ModuleType('exl3_test.model')
        model_package.__path__ = []
        class OutputConfig:
            @staticmethod
            def from_directory(directory):
                return types.SimpleNamespace(stc=loader.SafetensorsCollection(directory, load_method='python'))
        class OutputModel:
            @staticmethod
            def from_config(config):
                class Leaf:
                    modules = []
                    def __init__(self, key):
                        self.key = key
                    def quant_format_id(self):
                        return None
                return [Leaf(key) for key in config.stc.tensor_file_map]
        model_package.Config = OutputConfig
        model_package.Model = OutputModel
        trie = types.ModuleType('marisa_trie')
        class Trie:
            def __init__(self, keys):
                self.items = list(keys)
            def keys(self, prefix):
                return [key for key in self.items if key.startswith(prefix)]
        trie.Trie = Trie
        decision = types.ModuleType('exl3_test.model.decision')
        decision.package_decision_weights = lambda *args: None
        with patch.dict(sys.modules, {'exl3_test.modules': module_package, 'exl3_test.util.memory': memory, 'marisa_trie': trie, 'exl3_test.model': model_package, 'exl3_test.model.decision': decision}):
            sys.modules.pop('exl3_test.conversion.compile', None)
            compile_module = importlib.import_module('exl3_test.conversion.compile')
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source, work, out = (root / x for x in ('source', 'work', 'out'))
                source.mkdir()
                (source / 'config.json').write_text('{}')
                (work / 'qtensors').mkdir(parents=True)
                first = 'model.layers.0.linear_attn.A_log'
                second = 'model.layers.1.linear_attn.norm.weight'
                for i, key in enumerate((first, second)):
                    save_file({key: torch.ones(300000)}, str(work / 'qtensors' / f'{i}.safetensors'))
                class Config:
                    model_classes = {}
                    def get_tensor_name_fixes(self):
                        return {}
                model = types.SimpleNamespace(modules=[Module(first), Module(second)])
                adapter_dir = root / 'adapter'
                adapter_dir.mkdir()
                (adapter_dir / 'chat_template.jinja').write_text('adapter template')
                (adapter_dir / 'adapter_config.json').write_text(json.dumps({'r': 1, 'lora_alpha': 1}))
                adapter = low_disk.LoraAdapter(adapter_dir, [])
                config = Config()
                config.stc = types.SimpleNamespace(weight_adapter=adapter)
                args = {'in_dir': str(source), 'out_dir': str(out), 'work_dir': str(work), 'shard_size': 1, 'low_disk': True, 'inline_normalize_gdn': True, 'final_bits': 4, 'head_bits': 6, 'adapter': 'example/adapter', 'adapter_dir': str(adapter_dir)}
                original_save = compile_module.save_file
                def checked_save(tensors, filename):
                    if '00002-of-00002' in filename:
                        self.assertFalse((work / 'qtensors/0.safetensors').exists())
                    original_save(tensors, filename)
                compile_module.save_file = checked_save
                compile_module.compile_model(args, model, config, None)
                self.assertEqual((out / 'chat_template.jinja').read_text(), 'adapter template')
                self.assertEqual(json.loads((out / 'config.json').read_text())['quantization_config']['adapter'], 'example/adapter')
                self.assertEqual(list((work / 'qtensors').glob('*.safetensors')), [])
                metadata = json.loads((out / 'quantization_config.json').read_text())['tensor_storage']
                self.assertEqual(metadata[first]['stored_tensors'][first], {'dtype': 'torch.bfloat16', 'n_bytes': 600000, 'shape': [300000]})
                self.assertEqual(metadata[second]['stored_tensors'][second], {'dtype': 'torch.bfloat16', 'n_bytes': 600000, 'shape': [300000]})
            sys.modules.pop('exl3_test.conversion.compile', None)
            sys.modules.pop('exl3_test.conversion.quant_config', None)

    def test_lora_scaling_and_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = 'model.language_model.layers.0.mlp.down_proj.weight'
            replacement = 'model.embed_tokens.weight'
            prefix = 'base_model.model.model.language_model.layers.0.mlp.down_proj'
            a = torch.tensor([[1., 2.], [3., 4.]])
            b = torch.tensor([[2., 1.], [4., 3.]])
            save_file({prefix + '.lora_A.weight': a, prefix + '.lora_B.weight': b,
                       'base_model.model.model.embed_tokens.modules_to_save.default.weight': torch.full((2, 2), 7.)},
                      str(root / 'adapter_model.safetensors'))
            cfg = {'r': 8, 'lora_alpha': 8, 'rank_pattern': {'down_proj': 2},
                   'alpha_pattern': {'down_proj': 6}, 'modules_to_save': ['embed_tokens']}
            for rslora in (False, True):
                cfg['use_rslora'] = rslora
                (root / 'adapter_config.json').write_text(json.dumps(cfg))
                adapter = low_disk.LoraAdapter(root, [target, replacement])
                scale = 6 / (2 ** .5 if rslora else 2)
                weight = torch.tensor([[1., 2.], [3., 4.]])
                self.assertTrue(torch.equal(adapter.apply(target, weight), weight + scale * (b @ a)))
                self.assertTrue(torch.equal(adapter.apply(replacement, weight), torch.full((2, 2), 7.)))

    def test_shard_plan_keeps_extra_and_unknown(self):
        class Module:
            def __init__(self, key):
                self.key = key
                self.alt_key = None
            def __iter__(self):
                return iter([self])
        with tempfile.TemporaryDirectory() as temp:
            manager = low_disk.ShardManager(temp, lambda name, path: None)
            collection = types.SimpleNamespace(tensor_file_map={'layer.0.weight': str(Path(temp) / 'a.safetensors'),
                'layer.1.weight': str(Path(temp) / 'b.safetensors'), 'extra': str(Path(temp) / 'b.safetensors'),
                'unknown': str(Path(temp) / 'c.safetensors')})
            low_disk.plan_shards(manager, collection, [Module('layer.0'), Module('layer.1')], {'extra'})
            self.assertEqual(manager.last_use[str(Path(temp) / 'a.safetensors')], 0)
            self.assertEqual(manager.last_use[str(Path(temp) / 'b.safetensors')], 2)
            self.assertEqual(manager.last_use[str(Path(temp) / 'c.safetensors')], 2)

    def test_lora_direct_and_deferred(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = 'model.language_model.layers.0.mlp.down_proj.weight'
            prefix = 'base_model.model.model.language_model.layers.0.mlp.down_proj'
            weight = torch.tensor([[1., 2.], [3., 4.]])
            a = torch.tensor([[1., 2.]])
            b = torch.tensor([[3.], [4.]])
            save_file({target: weight, 'other.weight': torch.tensor([[9., 8.]])}, str(root / 'model.safetensors'))
            adapter_dir = root / 'adapter'
            adapter_dir.mkdir()
            save_file({prefix + '.lora_A.weight': a, prefix + '.lora_B.weight': b}, str(adapter_dir / 'adapter_model.safetensors'))
            (adapter_dir / 'adapter_config.json').write_text(json.dumps({'r': 1, 'lora_alpha': 2, 'rank_pattern': {'down_proj': 1}, 'alpha_pattern': {'down_proj': 3}, 'use_rslora': True}))
            (adapter_dir / 'chat_template.jinja').write_text('test')
            adapter = low_disk.LoraAdapter(adapter_dir, [target])
            self.assertEqual(len(adapter.pairs), 1)
            collection = loader.SafetensorsCollection(str(root), weight_adapter=adapter)
            expected = weight + 3 * (b @ a)
            self.assertTrue(torch.equal(collection.get_tensor(target), expected))
            collection.begin_deferred_load()
            actual = collection.get_tensor(target)
            other = collection.get_tensor('other.weight')
            collection.end_deferred_load()
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(other, torch.tensor([[9., 8.]])))
            collection.close()
            out = root / 'out'
            out.mkdir()
            adapter.copy_tokenizer_files(out)
            self.assertEqual((out / 'chat_template.jinja').read_text(), 'test')


if __name__ == '__main__':
    unittest.main()
