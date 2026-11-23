import ast
import importlib.util
import os
from pathlib import Path
import random
import re
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / 'exl3_int8'
EXT = PKG / 'exllamav3/exllamav3_ext'


def build_module():
    spec = importlib.util.spec_from_file_location('exl3_build_ext', PKG / 'build_ext.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sources_for(rocm):
    tree = ast.parse((PKG / 'setup.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_collect_sources']
    nodes += [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ROCM_EXCLUDE' for t in n.targets)]
    ns = {'os': os, '__file__': str(PKG / 'setup.py'), 'sources_dir': str(EXT), 'IS_ROCM': rocm}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'setup.py', 'exec'), ns)
    return {str((PKG / x).resolve().relative_to(EXT)) for x in ns['_collect_sources']()}


class CudaBuildTest(unittest.TestCase):
    def test_sources_and_bindings(self):
        cuda, rocm = sources_for(False), sources_for(True)
        all_cuda = {str(p.relative_to(EXT)) for p in EXT.rglob('*') if p.is_file() and p.suffix in ('.c', '.cpp', '.cu') and 'rocm' not in p.relative_to(EXT).parts}
        self.assertEqual(cuda, all_cuda)
        required = {'quant/quantize.cu', 'quant/reconstruct.cu', 'quant/exl3_gemm.cu',
                    'quant/exl3_gemv.cu', 'quant/exl3_gemv_int8.cu', 'quant/exl3_kernel_map.cu',
                    'quant/exl3_moe.cu', 'quant/exl3_moe_coop.cu', 'rope.cu', 'hgemm.cu',
                    'hgemm_f16acc.cu', 'graph.cu', 'cuda_drv.cpp', 'cpu/moe_handoff.cu'}
        self.assertTrue(required <= cuda)
        self.assertTrue(required.isdisjoint(rocm))
        self.assertTrue(any(p.startswith('parallel/') for p in cuda))
        self.assertTrue(any(p.startswith('quant/comp_units/') for p in cuda))
        self.assertTrue(any(p.startswith('rocm/quant/comp_units_rdna/') for p in rocm))
        definitions = '\n'.join((EXT / p).read_text(errors='replace') for p in cuda)
        for symbol in ('exl3_mul1_multiplier', 'reconstruct', 'reconstruct_slice', 'reconstruct_batch',
                       'reconstruct_had_slice', 'reconstruct_had_batch', 'exl3_gemm', 'exl3_gemv',
                       'exl3_mgemm', 'exl3_moe', 'exl3_moe_coop', 'quantize_tiles'):
            self.assertRegex(definitions, r'\b(?:void|int|int64_t|bool)\s+' + symbol + r'\s*\([^;{}]*\)\s*\{')
            self.assertIn('"' + symbol + '"', (EXT / 'bindings.cpp').read_text())

    def test_no_literal_multiplier_in_quant_code(self):
        for p in (EXT / 'quant').rglob('*'):
            if p.suffix not in ('.cu', '.cuh'):
                continue
            s = re.sub(r'/\*.*?\*/|//[^\n]*', '', p.read_text(), flags=re.S)
            s = re.sub(r'#define\s+EXL3_MUL1_MULT\s+0x83DCD12Du', '', s)
            self.assertNotRegex(s, r'(?i)0x83dcd12d', str(p))

    def test_tile_mapping(self):
        source = (EXT / 'quant/exl3_dq.cuh').read_text()
        for expression in ('8 * (lane >> 2) + 2 * (lane & 3)', 'base + 128', 'base + 64', 'base + 192'):
            self.assertIn(expression, source)
        rng = random.Random(5)
        for bits in range(1, 9):
            values = [rng.randrange(1 << bits) for _ in range(256)]
            words = [0] * (bits * 8)
            for i, value in enumerate(values):
                for j in range(bits):
                    bit = (value >> (bits - 1 - j)) & 1
                    pos = i * bits + j
                    words[pos // 32] |= bit << (31 - pos % 32)
            matrix = [[None] * 16 for _ in range(16)]
            for lane in range(32):
                base = 8 * (lane >> 2) + 2 * (lane & 3)
                for group, offset in enumerate((0, 128, 64, 192)):
                    for j in range(2):
                        t = base + offset + j
                        b0 = (t + 257) * bits - 16
                        b1 = b0 + 16
                        i0, i1 = b0 // 32, (b1 - 1) // 32
                        shift = (i1 + 1) * 32 - b1
                        a, b = words[i0 % len(words)], words[i1 % len(words)]
                        value = (((a << 32) | b) >> shift) & 0xffff
                        row = 2 * (lane % 4) + (8 if group % 2 else 0) + j
                        col = lane // 4 + (8 if group >= 2 else 0)
                        self.assertIsNone(matrix[row][col])
                        matrix[row][col] = value
                        expected = 0
                        for k in range(16):
                            position = (t * bits + bits - 16 + k) % (256 * bits)
                            expected = (expected << 1) | ((words[position // 32] >> (31 - position % 32)) & 1)
                        self.assertEqual(value, expected)
                        self.assertEqual(t, 8 * (col + 16 * (row // 8)) + row % 8)
            self.assertTrue(all(value is not None for row in matrix for value in row))

    def test_build_backend(self):
        mod = build_module()
        for backend in ('rocm', 'cuda', 'triton'):
            with self.subTest(backend=backend), mock.patch.dict(os.environ, {}, clear=True), \
                 mock.patch.object(sys, 'argv', ['build_ext.py', '--backend', backend]), \
                 mock.patch.object(mod.subprocess, 'run') as run, \
                 mock.patch.object(mod, 'verify_extension', return_value={'multiplier': mod.MULTIPLIER, 'path': 'x', 'sha256': 'x'}):
                mod.main()
                self.assertEqual(run.called, backend != 'triton')
                if run.called:
                    env = run.call_args.kwargs['env']
                    self.assertEqual(env['EXL3_BACKEND'], backend)
                    self.assertEqual(env['EXL3_MUL1_MULT'], '0x00DCD12D')
                    if backend == 'rocm':
                        self.assertEqual((env['PYTORCH_ROCM_ARCH'], env['MAX_JOBS']), ('gfx1201', '8'))
        for hip, cuda, expected in [('7.0', None, 'rocm'), (None, '13.0', 'cuda')]:
            with mock.patch.dict(sys.modules, {'torch': mock.Mock(version=mock.Mock(hip=hip, cuda=cuda))}):
                self.assertEqual(mod.resolve_backend('auto'), expected)
        with mock.patch.dict(os.environ, {'EXL3_BACKEND': 'triton'}), \
             mock.patch.object(sys, 'argv', ['build_ext.py']), \
             mock.patch.object(mod.subprocess, 'run') as run, \
             mock.patch.object(mod, 'verify_extension', return_value={'multiplier': mod.MULTIPLIER, 'path': 'x', 'sha256': 'x'}):
            mod.main()
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
