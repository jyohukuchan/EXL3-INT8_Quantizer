import importlib
import os

backend = os.environ.get('EXL3_BACKEND', 'auto').lower()
if backend not in ('auto', 'rocm', 'cuda', 'triton'):
    raise ValueError(f'EXL3_BACKEND が不正です: {backend}')

if backend == 'triton':
    exllamav3_ext = importlib.import_module('.triton_ext', __package__)
elif backend == 'auto':
    try:
        exllamav3_ext = importlib.import_module('exllamav3_ext')
    except ImportError:
        exllamav3_ext = importlib.import_module('.triton_ext', __package__)
else:
    exllamav3_ext = importlib.import_module('exllamav3_ext')

if exllamav3_ext.exl3_mul1_multiplier() != 0x00DCD12D:
    raise RuntimeError('EXL3 拡張の multiplier が mul24 ではありません')
