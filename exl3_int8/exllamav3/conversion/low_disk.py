import json
import math
import shutil
from pathlib import Path

import torch
from safetensors import safe_open


class LoraAdapter:
    def __init__(self, directory, base_keys, identifier=None):
        self.directory = Path(directory)
        self.identifier = identifier or str(directory)
        self.config = json.loads((self.directory / 'adapter_config.json').read_text())
        if self.config.get('use_dora') or self.config.get('fan_in_fan_out') or self.config.get('bias', 'none') != 'none':
            raise ValueError('Unsupported PEFT adapter configuration')
        self.pairs = {}
        self.replacements = {}
        keys = set(base_keys)
        entries = {}
        for path in sorted(self.directory.glob('*.safetensors')):
            with safe_open(path, framework='pt', device='cpu') as f:
                for name in f.keys():
                    if name in entries:
                        raise ValueError(f'Duplicate adapter tensor: {name}')
                    entries[name] = path
        def resolve(name):
            candidates = [name]
            while candidates[-1].startswith(('base_model.model.', 'base_model.')):
                old = candidates[-1]
                candidates.append(old[len('base_model.model.'):] if old.startswith('base_model.model.') else old[len('base_model.'):])
            matches = [x for x in candidates if x in keys]
            if len(matches) != 1:
                raise ValueError(f'Cannot uniquely resolve adapter tensor {name}: {matches}')
            return matches[0]
        for name, path in entries.items():
            if '.lora_A.' in name:
                before, after = name.split('.lora_A.', 1)
                b_name = before + '.lora_B.' + after
                if b_name not in entries:
                    raise ValueError(f'Missing LoRA B tensor: {name}')
                target = resolve(before + '.weight')
                if target in self.pairs:
                    raise ValueError(f'Duplicate LoRA target: {target}')
                module = target.removesuffix('.weight')
                rank = self._pattern('rank_pattern', module, self.config['r'])
                alpha = self._pattern('alpha_pattern', module, self.config['lora_alpha'])
                if not isinstance(rank, int) or rank <= 0:
                    raise ValueError(f'Invalid LoRA rank for {target}: {rank}')
                scale = alpha / (math.sqrt(rank) if self.config.get('use_rslora') else rank)
                self.pairs[target] = ((path, name), (entries[b_name], b_name), scale, rank)
            elif '.modules_to_save.' in name:
                before, after = name.split('.modules_to_save.', 1)
                parts = after.split('.', 1)
                if len(parts) != 2:
                    raise ValueError(f'Invalid modules_to_save tensor: {name}')
                target = resolve(before + '.' + parts[1])
                if target in self.replacements:
                    raise ValueError(f'Duplicate replacement: {target}')
                self.replacements[target] = (path, name)
            elif any(name.endswith('.' + module + '.weight') or name.endswith('.' + module + '.bias') for module in (self.config.get('modules_to_save') or [])):
                target = resolve(name)
                if target in self.replacements:
                    raise ValueError(f'Duplicate replacement: {target}')
                self.replacements[target] = (path, name)
            elif '.lora_B.' not in name:
                raise ValueError(f'Unsupported adapter tensor: {name}')
        for name in entries:
            if '.lora_B.' in name and name.replace('.lora_B.', '.lora_A.') not in entries:
                raise ValueError(f'Missing LoRA A tensor: {name}')

    def _pattern(self, field, module, default):
        patterns = self.config.get(field) or {}
        matches = [(len(k), v) for k, v in patterns.items() if module == k or module.endswith('.' + k)]
        return max(matches)[1] if matches else default

    def has(self, key):
        return key in self.pairs or key in self.replacements

    def _tensor(self, item, device):
        path, name = item
        with safe_open(path, framework='pt', device='cpu') as f:
            return f.get_tensor(name).to(device)

    def apply(self, key, tensor):
        if key in self.replacements:
            result = self._tensor(self.replacements[key], tensor.device)
            if result.shape != tensor.shape:
                raise ValueError(f'Adapter replacement shape mismatch for {key}')
            return result.to(tensor.dtype)
        if key not in self.pairs:
            return tensor
        a_ref, b_ref, scale, rank = self.pairs[key]
        a = self._tensor(a_ref, tensor.device).float()
        b = self._tensor(b_ref, tensor.device).float()
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank or (b.shape[0], a.shape[1]) != tuple(tensor.shape):
            raise ValueError(f'LoRA shape mismatch for {key}')
        return (tensor.float() + scale * (b @ a)).to(tensor.dtype)

    def copy_tokenizer_files(self, out_dir):
        names = ('chat_template.jinja', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json', 'tokenizer.model', 'vocab.json', 'merges.txt')
        for name in names:
            source = self.directory / name
            if source.is_file():
                shutil.copy2(source, Path(out_dir) / name)


class ShardManager:
    def __init__(self, directory, fetcher):
        self.directory = Path(directory).resolve()
        self.fetcher = fetcher
        self.fetched = set()
        self.last_use = {}

    def fetch(self, filename):
        path = Path(filename).resolve()
        if path.parent != self.directory or path.suffix != '.safetensors':
            raise ValueError(f'Invalid shard path: {filename}')
        if not path.exists():
            self.fetcher(path.name, path)
            if not path.is_file():
                raise FileNotFoundError(path)
            self.fetched.add(path)
        return str(path)

    def read_header(self, filename, fixes):
        if hasattr(self.fetcher, "read_header"):
            return self.fetcher.read_header(Path(filename).name, fixes)
        from ..loader.safetensors import read_header
        return read_header(self.fetch(filename), fixes)

    def release(self, index, collection):
        for path in tuple(self.fetched):
            if self.last_use.get(str(path), float('inf')) <= index:
                collection.evict_shard(str(path))
                path.unlink(missing_ok=True)
                self.fetched.remove(path)


class HFShardFetcher:
    def __init__(self, repo_id, revision=None):
        self.repo_id = repo_id
        self.revision = revision

    def download(self, name, destination):
        import tempfile
        from huggingface_hub import hf_hub_download
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as cache:
            source = hf_hub_download(self.repo_id, name, revision=self.revision, cache_dir=cache)
            Path(source).resolve().replace(destination)

    def read_header(self, name, fixes):
        import struct
        import requests
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers
        from ..loader.safetensors import MAX_HEADER_SIZE, validate_header
        url = hf_hub_url(self.repo_id, name, revision=self.revision)
        headers = build_hf_headers()
        headers['Range'] = 'bytes=0-7'
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        if response.status_code != 206 or len(response.content) != 8:
            raise ValueError(f'Server does not support range requests for {name}')
        total = int(response.headers['Content-Range'].split('/')[-1])
        size = struct.unpack('<Q', response.content)[0]
        if not 0 < size <= MAX_HEADER_SIZE or size + 8 > total:
            raise ValueError(f'Invalid safetensors header size for {name}')
        headers['Range'] = f'bytes=8-{size + 7}'
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        if response.status_code != 206 or len(response.content) != size:
            raise ValueError(f'Invalid safetensors header response for {name}')
        header = json.loads(response.content)
        validate_header(header, name, size + 8, total)
        for key in list(header):
            for old, new in fixes.items():
                if key.endswith(old):
                    header[key[:-len(old)] + new] = header.pop(key)
                    break
        header['_header_offset'] = size + 8
        return header

    def __call__(self, name, destination):
        self.download(name, destination)

    def prepare(self, directory, adapter=False):
        from huggingface_hub import list_repo_files
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        for name in list_repo_files(self.repo_id, revision=self.revision):
            path = Path(name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError(f'Invalid repository filename: {name}')
            if not adapter and (name.endswith(('.safetensors', '.bin', '.pt', '.pth', '.ckpt'))):
                continue
            if adapter and name.endswith(('.bin', '.pt', '.pth', '.ckpt')):
                continue
            destination = root / path
            if not destination.exists():
                self.download(name, destination)


def plan_shards(manager, collection, modules, extra_keys):
    prefixes = {}
    for index, module in enumerate(modules):
        for item in module:
            for key in (item.key, getattr(item, 'alt_key', None)):
                keys = [key] if isinstance(key, str) else key if isinstance(key, (tuple, list)) else []
                for prefix in keys:
                    if isinstance(prefix, str):
                        prefixes[prefix] = max(index, prefixes.get(prefix, -1))
    last = {}
    for key, filename in collection.tensor_file_map.items():
        parts = key.split('.')
        matches = [prefixes['.'.join(parts[:end])] for end in range(1, len(parts) + 1)
                   if '.'.join(parts[:end]) in prefixes]
        key_last = len(modules) if key in extra_keys or not matches else max(matches)
        filename = str(Path(filename).resolve())
        last[filename] = max(last.get(filename, -1), key_last)
    manager.last_use = last


def save_checkpoint(args, job_state, state, original_input_ids, save_dict, save_tensor):
    if args.get('no_checkpoint'):
        return False
    work = Path(args['work_dir'])
    current, old, new = (work / name for name in ('ckpt', 'ckpt_old', 'ckpt_new'))
    if args.get('low_disk') and old.exists():
        shutil.rmtree(old)
    new.mkdir(exist_ok=True)
    dtype = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.get('checkpoint_dtype', 'fp32')]
    def cast(t):
        return t.to(dtype) if t.is_floating_point() and dtype != torch.float32 else t
    checkpoint_state = [cast(t) for t in state] if isinstance(state, list) else cast(state)
    save_dict('ckpt_new/job.json', job_state, args)
    save_tensor(checkpoint_state, 'ckpt_new/state.safetensors', args)
    save_tensor(original_input_ids, 'ckpt_new/original_input_ids.safetensors', args)
    if old.exists():
        shutil.rmtree(old)
    current.rename(old)
    new.rename(current)
    return True
