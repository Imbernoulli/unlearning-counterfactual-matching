"""Run explicitly registered commands sequentially on an owned clean GPU.

This is an infrastructure wrapper: it never selects an algorithm or reads
metrics. Commands, GPU assignments and expected output paths come from a JSON
job list; all process IDs and before/after resource probes are recorded.
"""
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
from calibration_dispatch import probe, rows_on_gpu


def now():return datetime.now(timezone.utc).isoformat()


def register(pid, script, gpu, job):
    path=Path('out/optimization/auxiliary_process_registry.jsonl');path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:stream.write(json.dumps(dict(pid=pid,expected_script=script,gpu=gpu,job=job,at=now()))+'\n')


def execute(job,gpu):
    output=Path(job['log_dir']);output.mkdir(parents=True,exist_ok=True)
    done=output/'completed.json'
    if done.exists():return
    argv=[str(value).replace('{gpu}',str(gpu)) for value in job['argv']]
    # The executable may be explicit, but every GPU child is a known Python entrypoint.
    expected=job['expected_script']
    if not any(Path(value).name==expected for value in argv):raise ValueError('registered script absent from command')
    while shutil.disk_usage(output).free<500*1024**2:
        if Path('out/STOP').exists():return
        time.sleep(5)
    before=probe()
    if rows_on_gpu(before,gpu):raise RuntimeError('GPU is occupied; refusing job '+job['id'])
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
    start=time.monotonic();record=dict(id=job['id'],gpu=gpu,argv=argv,started_utc=now(),before=before)
    with (output/'run.log').open('w') as log:
        child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        register(child.pid,expected,gpu,job['id']);record['pid']=child.pid
        deadline=time.monotonic()+60
        while child.poll() is None:
            after=probe();pids=rows_on_gpu(after,gpu)
            if pids:
                if any(pid!=child.pid or after['owners'][str(pid)]['user']!='root' for pid in pids):
                    child.terminate();child.wait(timeout=30);raise RuntimeError('GPU attribution changed after launch')
                record['after_launch']=after;break
            if time.monotonic()>deadline:
                child.terminate();child.wait(timeout=30);raise RuntimeError('job failed to initialize GPU within 60 seconds')
            time.sleep(.5)
        (output/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
        code=child.wait()
    record.update(returncode=code,ended_utc=now(),wall_seconds=time.monotonic()-start,after_exit=probe())
    (output/'finished.json').write_text(json.dumps(record,indent=2)+'\n')
    if code:raise RuntimeError(job['id']+' returned '+str(code))
    for path in job.get('required_outputs',[]):
        if not Path(path).is_file():raise RuntimeError('required output missing: '+path)
    done.write_text(json.dumps(dict(id=job['id'],gpu=gpu,completed_utc=now()))+'\n')
    print(json.dumps(dict(event='job_completed',id=job['id'],gpu=gpu,wall_seconds=record['wall_seconds'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--jobs',type=Path,required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    register(os.getpid(),'gpu_job_queue.py',a.gpu,str(a.jobs))
    configuration=json.loads(a.jobs.read_text())
    for path,expected in configuration.get('sources',{}).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
            raise ValueError('registered command source changed: '+path)
    for job in configuration['jobs']:
        if int(job['gpu'])!=a.gpu:continue
        if Path('out/STOP').exists():break
        execute(job,a.gpu)


if __name__=='__main__':main()
