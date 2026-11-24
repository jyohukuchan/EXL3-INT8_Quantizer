import argparse
import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MULTIPLIER = 0x00DCD12D


def verify_extension(backend=None):
    if backend is None:
        backend = resolve_backend(os.environ.get('EXL3_BACKEND', 'auto'))
    sys.path.insert(0, str(ROOT))
    if backend == 'triton':
        ext = importlib.import_module('exllamav3.ext')
        value = ext.exllamav3_ext.exl3_mul1_multiplier()
        path = Path(ext.__file__).resolve()
    else:
        import torch
        ext = importlib.import_module('exllamav3_ext')
        value = ext.exl3_mul1_multiplier()
        path = Path(ext.__file__).resolve()
        if path.parent != ROOT:
            raise RuntimeError(f'想定外の拡張が読み込まれました: {path}')
    if value != MULTIPLIER:
        raise RuntimeError(f'拡張の multiplier が不正です: {value:#x}')
    return {'multiplier': value, 'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def resolve_backend(value):
    if value != 'auto':
        return value
    import torch
    if torch.version.hip:
        return 'rocm'
    if torch.version.cuda:
        return 'cuda'
    raise RuntimeError('torch から CUDA または ROCm を検出できません')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=('auto', 'rocm', 'cuda', 'triton'),
                        default=os.environ.get('EXL3_BACKEND', 'auto'))
    args = parser.parse_args()
    if args.backend not in ('auto', 'rocm', 'cuda', 'triton'):
        parser.error('EXL3_BACKEND は auto、rocm、cuda、triton のいずれかにしてください')
    backend = resolve_backend(args.backend)
    os.environ['EXL3_BACKEND'] = backend
    os.environ['EXL3_MUL1_MULT'] = '0x00DCD12D'
    if backend != 'triton':
        env = os.environ.copy()
        if backend == 'rocm':
            env.setdefault('PYTORCH_ROCM_ARCH', 'gfx1201')
            env.setdefault('MAX_JOBS', '8')
        env.pop('EXLLAMA_NOCOMPILE', None)
        subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace', '--force'], cwd=ROOT, env=env, check=True)
    info = verify_extension(backend)
    print(f"multiplier: {info['multiplier']:#010x}\n拡張: {info['path']}\nSHA-256: {info['sha256']}")


if __name__ == '__main__':
    main()
