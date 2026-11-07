"""Guarded, detached per-GPU execution of a preregistered calibration grid."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def probe():
    record = {'at': datetime.now(timezone.utc).isoformat()}
    for name, command in [('nproc', ['nproc']), ('uptime', ['uptime']),
            ('memory', ['free', '-g']), ('gpus', ['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total', '--format=csv']),
            ('processes', ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv'])]:
        record[name] = subprocess.check_output(command, text=True)
    record['owners'] = {}
    for line in record['processes'].splitlines()[1:]:
        if line.strip():
            pid = line.split(',')[0].strip()
            result = subprocess.run(['ps', '-o', 'user=', '-p', pid], text=True, capture_output=True)
            record['owners'][pid] = dict(user=result.stdout.strip(), returncode=result.returncode)
    return record


def rows_on_gpu(record, gpu):
    gpu_uuid = next(line.split(',')[1].strip() for line in record['gpus'].splitlines()[1:]
                    if line.split(',')[0].strip() == str(gpu))
    return [int(line.split(',')[0]) for line in record['processes'].splitlines()[1:]
            if line.strip() and line.split(',')[1].strip() == gpu_uuid]


def execute(configuration, gpu, output, events):
    output.mkdir(parents=True, exist_ok=True)
    trial_id = configuration['id']
    def event(name, **kwargs):
        row = dict(at=datetime.now(timezone.utc).isoformat(), event=name, trial_id=trial_id, gpu=gpu, **kwargs)
        with events.open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    while shutil.disk_usage(output).free < 500 * 1024 ** 2:
        event('waiting_for_disk', free_bytes=shutil.disk_usage(output).free)
        if Path('out/STOP').exists():
            raise RuntimeError('stop requested while waiting for sufficient scratch disk')
        time.sleep(5)
    before = probe()
    event('before_launch', probe=before, free_disk_bytes=shutil.disk_usage(output).free)
    if rows_on_gpu(before, gpu):
        raise RuntimeError('selected GPU has existing processes; refusing launch')
    argv = [sys.executable, '-u', configuration.get('entrypoint', 'research/unlearning_matching/calibration_study.py'),
            '--data-dir', 'data', '--base-runs-dir', 'out/prepared', '--out-dir', str(output),
            '--original', 'original.pt', '--gpu', str(gpu), '--threads', '2',
            '--seed', str(configuration['seed']), '--split-seed', str(configuration['split_seed']),
            '--schedules', configuration['schedule'], '--lr', str(configuration['lr']),
            '--forget-weight', str(configuration['forget_weight']),
            '--anchor-weight', str(configuration['anchor_weight']), '--epochs', str(configuration['epochs']),
            '--bn-mode', configuration['bn_mode'], '--target-kind', configuration['target_kind'],
            '--loss', configuration['loss']]
    argv.extend(configuration.get('extra_args', []))
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
    with (output / 'run.log').open('w') as stream:
        process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        event('launched', pid=process.pid, command=argv, expected_script=Path(argv[2]).name)
        deadline = time.monotonic() + 60
        while process.poll() is None:
            after = probe()
            pids = rows_on_gpu(after, gpu)
            if pids:
                event('after_launch', probe=after, expected_pid=process.pid)
                if any(pid != process.pid or after['owners'][str(pid)]['user'] != 'root' for pid in pids):
                    process.terminate()
                    process.wait(timeout=30)
                    raise RuntimeError('GPU ownership changed after launch; stopped only this process')
                break
            if time.monotonic() > deadline:
                process.terminate()
                process.wait(timeout=30)
                raise RuntimeError('GPU initialization timed out')
            time.sleep(.5)
        code = process.wait()
    event('completed', returncode=code, probe=probe())
    if code:
        raise RuntimeError(f'{trial_id} failed with {code}')
    run = output / f"seed{configuration['seed']}_split{configuration['split_seed']}" / configuration['schedule']
    if not (run / 'summary.json').exists():
        raise RuntimeError(f'{trial_id} exited without a complete summary; possibly stopped early')
    summary = json.loads((run / 'summary.json').read_text())
    best = json.loads((run / 'best.json').read_text())
    if best != summary['selection'] or summary['epochs'] != configuration['epochs']:
        raise RuntimeError(f'{trial_id} summary and best selection disagree')
    history = [json.loads(line) for line in (run / 'history.jsonl').read_text().splitlines()]
    selected = [row for row in history if row['epoch'] == best['epoch'] and
                row['stage'] == ('initial' if best['epoch'] == 0 else
                                 ('after_retain' if configuration['schedule'] == 'A' else 'epoch_end'))]
    if len(selected) != 1 or selected[0]['metrics']['selection_J'] != best['selection_J']:
        raise RuntimeError(f'{trial_id} selected epoch is inconsistent with history')
    if not (run / 'best.pt').exists() or not (run / 'last.pt').exists():
        raise RuntimeError(f'{trial_id} checkpoint pair missing')
    (output / 'completed.json').write_text(json.dumps(dict(id=trial_id, gpu=gpu, completed_utc=datetime.now(timezone.utc).isoformat())) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grid', type=Path, default=Path('research/unlearning_matching/study_round1.json'))
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--worker-count', type=int, default=2)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    grid = json.loads(a.grid.read_text())
    for source, digest in grid['sources'].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'preregistered source hash differs: {source}')
    root = Path('out/optimization')
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'dispatcher_registry.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
                                    script='calibration_dispatch.py', gpu=a.gpu)) + '\n')
    events = root / f'dispatch_gpu{a.gpu}{"_smoke" if a.smoke else ""}.jsonl'
    candidates = [grid['smoke']] if a.smoke else [c for i, c in enumerate(grid['candidates']) if i % a.worker_count == a.gpu]
    for config in candidates:
        if Path('out/STOP').exists():
            break
        output = root / config['id']
        if (output / 'completed.json').exists():
            continue
        execute(config, a.gpu, output, events)


if __name__ == '__main__':
    main()
