import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

QUANT = Path(__file__).resolve().parent
EXT = QUANT / 'exl3_int8'
sys.path.insert(0, str(EXT))
from build_ext import verify_extension


def run(*args, env=None):
    subprocess.run([str(x) for x in args], check=True, env=env)


def normalize_gdn_weights(output):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    converted = set()
    for path in sorted(output.glob('*.safetensors')):
        with safe_open(path, framework='pt', device='cpu') as source:
            names = [name for name in source.keys()
                     if name.endswith(('.linear_attn.norm.weight', '.linear_attn.A_log'))
                     and source.get_slice(name).get_dtype() == 'F32']
            metadata = source.metadata()
        if not names:
            continue
        tensors = load_file(path, device='cpu')
        for name in names:
            tensors[name] = tensors[name].to(torch.bfloat16)
        with tempfile.TemporaryDirectory(prefix='.shelf-', dir=output) as temp:
            target = Path(temp) / path.name
            save_file(tensors, str(target), metadata=metadata)
            os.replace(target, path)
        converted.update(names)
    config_path = output / 'quantization_config.json'
    if converted and config_path.exists():
        config = json.loads(config_path.read_text())
        for module in config.get('tensor_storage', {}).values():
            for name, tensor in module.get('stored_tensors', {}).items():
                if name in converted:
                    tensor['dtype'] = 'torch.bfloat16'
                    tensor['n_bytes'] //= 2
        config_path.write_text(json.dumps(config, indent=4) + '\n')




def triton_version():
    try:
        import triton
        return triton.__version__
    except ImportError:
        return None


def resolve_backend(backend):
    env = os.environ.copy()
    env['EXL3_BACKEND'] = backend
    script = (
        "import json; from exllamav3.ext import exllamav3_ext as ext; "
        "print(json.dumps({'backend': 'triton' if ext.__name__ == 'exllamav3.triton_ext' "
        "else 'native', 'multiplier': ext.exl3_mul1_multiplier(), 'path': ext.__file__}))"
    )
    env['PYTHONPATH'] = str(EXT) + os.pathsep + env.get('PYTHONPATH', '')
    output = subprocess.check_output([sys.executable, '-c', script], env=env, text=True)
    info = json.loads(output.splitlines()[-1])
    if info['multiplier'] != 0x00DCD12D:
        raise RuntimeError('拡張の multiplier が mul24 ではありません')
    if info['backend'] == 'native':
        import torch
        info.update(verify_extension('rocm' if torch.version.hip else 'cuda'))
        info['backend'] = 'rocm' if torch.version.hip else 'cuda'
    return info


def convert(args):
    import torch
    info = resolve_backend(args.backend)
    started = datetime.now(timezone.utc).isoformat()
    env = os.environ.copy()
    env.update(SHELF_QUANTIZE_SEED=str(args.seed), OMP_NUM_THREADS=str(args.threads), EXL3_BACKEND=args.backend)
    if info['backend'] == 'triton':
        env['EXL3_QKV_SLICE'] = '0'
        script = ("import sys; sys.path.insert(0, sys.argv.pop(1)); "
                  "from exllamav3.conversion.convert_model import parser, main, prepare; "
                  "args, state, ok, error = prepare(parser.parse_args()); "
                  "main(args, state) if ok else sys.exit(' !! Error: ' + str(error))")
        cmd = [sys.executable, '-c', script, EXT]
    else:
        cmd = [sys.executable, EXT / 'convert.py']
    cmd.extend(['-i', args.model, '-o', args.output,
                '-w', args.work, '-b', args.bpw, '-cb', 'mul1'])
    for option, value in [('-cd', args.calibration), ('-cr', args.cal_rows), ('-cc', args.cal_cols)]:
        if value is not None:
            cmd.extend([option, value])
    for option, value in [('--hf-repo', args.hf_repo), ('--hf-revision', args.hf_revision),
                          ('--adapter', args.adapter), ('--adapter-revision', args.adapter_revision),
                          ('--checkpoint-dtype', args.checkpoint_dtype), ('--shard_size', args.shard_size)]:
        if value is not None:
            cmd.extend([option, value])
    if args.low_disk:
        cmd.append('--low-disk')
    if args.no_checkpoint:
        cmd.append('--no-checkpoint')
    run(*cmd, env=env)
    normalize_gdn_weights(args.output)
    run(sys.executable, QUANT / 'validate_exl3.py', args.output)
    report = {'model': str(args.model.resolve()), 'output': str(args.output.resolve()),
              'work': str(args.work.resolve()), 'bpw': args.bpw, 'threads': args.threads,
              'seed': args.seed, 'low_disk': args.low_disk, 'hf_repo': args.hf_repo,
              'hf_revision': args.hf_revision, 'adapter': args.adapter, 'adapter_revision': args.adapter_revision,
              'checkpoint_dtype': args.checkpoint_dtype, 'no_checkpoint': args.no_checkpoint,
              'shard_size': args.shard_size, 'calibration': str(args.calibration.resolve()) if args.calibration else 'bundled',
              'cal_rows': args.cal_rows or 250, 'cal_cols': args.cal_cols or 2048,
              'source_commit': 'b922ccbd558237b13cc2e3bb25b87bc7cb2cbfe3',
              'pytorch': torch.__version__, 'backend': info['backend'], 'cuda': torch.version.cuda,
              'rocm': torch.version.hip, 'triton': triton_version(), 'extension': info,
              'started_at': started, 'finished_at': datetime.now(timezone.utc).isoformat()}
    (args.output / 'quantize_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')


def main():
    p = argparse.ArgumentParser(description='EXL3-INT8 mul24 量子化ツール')
    sub = p.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build-ext', help='拡張をビルドして検証')
    build.add_argument('--backend', choices=['auto', 'rocm', 'cuda', 'triton'], default='auto')
    c = sub.add_parser('convert', help='モデルを量子化して検証')
    c.add_argument('--backend', choices=['auto', 'rocm', 'cuda', 'triton'], default='auto')
    for name in ('model', 'output', 'work'):
        c.add_argument('--' + name, type=Path, required=True)
    c.add_argument('--bpw', type=float, required=True)
    c.add_argument('--threads', type=int, default=8)
    c.add_argument('--seed', type=int, default=0)
    c.add_argument('--calibration', type=Path)
    c.add_argument('--cal-rows', type=int)
    c.add_argument('--cal-cols', type=int)
    c.add_argument('--low-disk', action='store_true')
    c.add_argument('--hf-repo')
    c.add_argument('--hf-revision')
    c.add_argument('--adapter')
    c.add_argument('--adapter-revision')
    c.add_argument('--checkpoint-dtype', choices=['fp32', 'bf16', 'fp16'])
    c.add_argument('--no-checkpoint', action='store_true')
    c.add_argument('--shard-size', type=int)
    d = sub.add_parser('distill', help='Qwen3.5 系の量子化済みモデルのスケールと Norm を KL 蒸留')
    for name in ('model', 'bundle', 'output'):
        d.add_argument('--' + name, type=Path, required=True)
    d.add_argument('--seed', type=int, default=52)
    d.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    v = sub.add_parser('validate', help='bundle を検証')
    v.add_argument('bundle', type=Path)
    args = p.parse_args()
    if args.command == 'build-ext':
        if args.backend == 'triton':
            print(json.dumps(resolve_backend('triton'), ensure_ascii=False))
        elif args.backend == 'cuda':
            env = os.environ.copy()
            env.update(EXL3_BACKEND='cuda', EXL3_MUL1_MULT='0x00DCD12D')
            subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace', '--force'], cwd=EXT, env=env, check=True)
            print(json.dumps(resolve_backend('cuda'), ensure_ascii=False))
        else:
            run(sys.executable, EXT / 'build_ext.py', '--backend', args.backend)
    elif args.command == 'distill':
        from distillation import distill
        distill(args)
    elif args.command == 'validate':
        run(sys.executable, QUANT / 'validate_exl3.py', args.bundle)
    elif args.command == 'convert':
        if args.threads < 1 or args.seed < 0 or (args.shard_size is not None and args.shard_size < 1):
            p.error('threads と shard-size は正数、seed は 0 以上にしてください')
        convert(args)


if __name__ == '__main__':
    main()
