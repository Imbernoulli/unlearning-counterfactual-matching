"""Run the preregistered unlearning study in one owned two-GPU Daytona sandbox.

The controller is intended to run detached. It makes at most one create call,
downloads measured artifacts, and deletes only the sandbox it created.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import tarfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel"
RUN_ID = "unlearning-project-20261106"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def capacity_module():
    spec = importlib.util.spec_from_file_location("unlearning_capacity", ROOT / "scripts/daytona/capacity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REMOTE_DRIVER = r'''import json,os,pathlib,subprocess,time,traceback
root=pathlib.Path('/root/study');os.chdir(root)
out=root/'out';out.mkdir(exist_ok=True)
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1')
def record(name,value):
 (out/name).write_text(json.dumps(value,indent=2)+'\n')
def preflight():
 rows={}
 for cmd in [['nproc'],['uptime'],['free','-g'],['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total','--format=csv'],['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv']]:
  value=subprocess.check_output(cmd,text=True);rows[' '.join(cmd)]=value
  if '--query-compute-apps=pid,gpu_uuid,used_memory' in cmd:
   for line in value.splitlines()[1:]:
    if line.strip():
     pid=line.split(',')[0].strip();rows['owner '+pid]=subprocess.check_output(['ps','-o','user=','-p',pid],text=True)
 return rows
def base(seed):
 return ['python','-u','research/unlearning_matching/run_experiments.py','--data-dir','data','--original','original.pt','--out-dir','out/runs','--seed',str(seed),'--split-seed',str(seed),'--amp','--threads','2']
def launch(seed,phase,gpu):
 record('preflight_%s_%s.json'%(seed,phase),preflight())
 log=(out/('%s_%s.log'%(seed,phase))).open('w')
 p=subprocess.Popen(base(seed)+['--phase',phase,'--gpu',str(gpu)],stdout=log,stderr=subprocess.STDOUT,env=env)
 time.sleep(2);record('postflight_%s_%s.json'%(seed,phase),preflight())
 return p,log
def await_job(job,seed,phase):
 p,log=job;rc=p.wait();log.close()
 if rc:raise RuntimeError('%s seed %s failed with rc %s'%(phase,seed,rc))
try:
 record('resources.json',preflight())
 subprocess.run(['python','-c','import torch,numpy,sklearn,json;print(json.dumps(dict(torch=torch.__version__,numpy=numpy.__version__,sklearn=sklearn.__version__)))'],stdout=(out/'versions.json').open('w'),check=True,env=env)
 subprocess.run(base(42)+['--phase','prepare'],check=True,env=env,stdout=(out/'42_prepare.log').open('w'),stderr=subprocess.STDOUT)
 reference=launch(42,'reference',0)
 diagnostics=launch(42,'diagnostics',1);await_job(diagnostics,42,'diagnostics')
 methods=launch(42,'experiments',1)
 # Wait for both; retrieve logs continuously through the outer controller.
 await_job(methods,42,'experiments');await_job(reference,42,'reference')
 evaluation=launch(42,'evaluate',0);await_job(evaluation,42,'evaluate')
 record('seed42_complete.json',{'at':time.time(),'status':'complete'})
 if os.environ.get('UNLEARNING_EXTRA_SEEDS')=='1':
  jobs=[(43,launch(43,'all',0)),(44,launch(44,'all',1))]
  for seed,job in jobs:await_job(job,seed,'all')
 record('complete.json',{'status':'complete','at':time.time()})
except BaseException:
 record('failure.json',{'status':'failed','at':time.time(),'traceback':traceback.format_exc()})
finally:
 record('finished.json',{'at':time.time()})
'''

CONTINUATION_DRIVER = r'''import json,os,pathlib,subprocess,time,traceback
root=pathlib.Path('/root/study');os.chdir(root);out=root/'out'
def record(name,value): (out/name).write_text(json.dumps(value,indent=2)+'\n')
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1')
while not (out/'finished.json').exists(): time.sleep(2)
if (out/'failure.json').exists():
 record('continuation_aborted.json',{'reason':'seed42 failed'});raise SystemExit(1)
(out/'finished.json').unlink();(out/'complete.json').unlink(missing_ok=True)
record('continuation_running.json',{'seeds':[43,44],'runner_changed':False,'at':time.time()})
try:
 jobs=[]
 for gpu,seed in enumerate([43,44]):
  rows={}
  for cmd in [['nproc'],['uptime'],['free','-g'],['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv']]:
   val=subprocess.check_output(cmd,text=True);rows[' '.join(cmd)]=val
   if '--query-compute-apps=pid,gpu_uuid,used_memory' in cmd:
    for line in val.splitlines()[1:]:
     if line.strip():
      pid=line.split(',')[0].strip();rows['owner '+pid]=subprocess.check_output(['ps','-o','user=','-p',pid],text=True)
  record('preflight_%s_all.json'%seed,rows)
  cmd=['python','-u','research/unlearning_matching/run_experiments.py','--data-dir','data','--original','original.pt','--out-dir','out/runs','--seed',str(seed),'--split-seed',str(seed),'--amp','--threads','2','--phase','all','--gpu',str(gpu)]
  log=(out/('%s_all.log'%seed)).open('w');p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env);jobs.append((seed,p,log))
  time.sleep(2)
  record('postflight_%s_all.json'%seed,{'nvidia-smi':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv'],text=True)})
 for seed,p,log in jobs:
  rc=p.wait();log.close()
  if rc: raise RuntimeError('seed %s exited %s'%(seed,rc))
 record('complete.json',{'status':'complete','seeds':[42,43,44],'at':time.time()})
except BaseException:
 record('failure.json',{'status':'failed','at':time.time(),'traceback':traceback.format_exc()})
finally:
 record('finished.json',{'at':time.time()})
'''

FIXED_QUANTILE_DRIVER = r'''import json,os,pathlib,subprocess,time,traceback,shutil
root=pathlib.Path('/root/study');os.chdir(root);out=root/'out'
def record(name,value): (out/name).write_text(json.dumps(value,indent=2)+'\n')
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1')
while not (out/'finished.json').exists() or not (out/'runs/seed44_split44/results.json').exists():
 if (out/'failure.json').exists():
  record('fixed_quantile_aborted.json',{'reason':'base experiment failed'});raise SystemExit(1)
 time.sleep(2)
(out/'finished.json').unlink();(out/'complete.json').rename(out/'base_complete.json')
record('fixed_quantile_running.json',{'seeds':[42,43,44],'loss':'both','base_runner_changed':False,'at':time.time()})
try:
 for name in ['research','tasks']:
  shutil.copytree(root/name,out/'source'/name,dirs_exist_ok=True)
 jobs=[]
 def launch(seed,gpu):
  rows={}
  for cmd in [['nproc'],['uptime'],['free','-g'],['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv']]:
   val=subprocess.check_output(cmd,text=True);rows[' '.join(cmd)]=val
   if '--query-compute-apps=pid,gpu_uuid,used_memory' in cmd:
    for line in val.splitlines()[1:]:
     if line.strip():
      pid=line.split(',')[0].strip();rows['owner '+pid]=subprocess.check_output(['ps','-o','user=','-p',pid],text=True)
  record('preflight_%s_fixed_quantile.json'%seed,rows)
  cmd=['python','-u','research/unlearning_matching/fixed_quantile.py','--data-dir','data','--original','original.pt','--base-runs-dir','out/runs','--out-dir','out/fixed_quantile','--seed',str(seed),'--split-seed',str(seed),'--threads','2','--phase','all','--loss','both','--gpu',str(gpu)]
  log=(out/('%s_fixed_quantile.log'%seed)).open('w');p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
  time.sleep(2);record('postflight_%s_fixed_quantile.json'%seed,{'nvidia-smi':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv'],text=True)})
  return p,log
 first=launch(42,0);second=launch(43,1)
 rc=first[0].wait();first[1].close()
 if rc:raise RuntimeError('fixed quantile42 failed %s'%rc)
 third=launch(44,0)
 for seed,job in [(43,second),(44,third)]:
  rc=job[0].wait();job[1].close()
  if rc:raise RuntimeError('fixed quantile%s failed %s'%(seed,rc))
 record('complete.json',{'status':'complete','base_seeds':[42,43,44],'fixed_quantile_seeds':[42,43,44],'at':time.time()})
except BaseException:
 record('failure.json',{'status':'failed','at':time.time(),'traceback':traceback.format_exc()})
finally:
 record('fixed_quantile_finished.json',{'at':time.time()});record('finished.json',{'at':time.time()})
'''


def append_fixed_quantile(args, client):
    sandbox = client.get(args.sandbox_id)
    if sandbox.labels.get("mlsbench-run-id") != RUN_ID or sandbox.labels.get("purpose") != "unlearning-project":
        raise RuntimeError("fixed-quantile continuation requires this study's exact sandbox")
    path = ROOT / "research/unlearning_matching/fixed_quantile.py"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != "760a2147acee205d2c34c0fbc4113be60161cea7bde74dd977bab4d81f625c79":
        raise RuntimeError("fixed-quantile source is not the preregistered revision")
    sandbox.fs.upload_file(str(path), "/root/study/research/unlearning_matching/fixed_quantile.py")
    sandbox.fs.upload_file(FIXED_QUANTILE_DRIVER.encode(), "/root/study/fixed_quantile_driver.py")
    save(args.out_dir / "fixed_quantile_upload.json", dict(at=stamp(), code_sha256=digest,
        frozen_commit="bde1754754", sandbox_id=sandbox.id, seeds=[42, 43, 44], loss="both"))
    response = sandbox.process.exec("nohup python -u /root/study/fixed_quantile_driver.py > /root/study/fixed_quantile_driver.log 2>&1 < /dev/null &", timeout=30)
    if response.exit_code:
        raise RuntimeError("fixed-quantile driver failed to launch")
    while True:
        response = sandbox.process.exec("test -f /root/study/out/fixed_quantile_finished.json && echo FIXED_FINISHED; test -f /root/study/out/fixed_quantile_aborted.json && echo FIXED_ABORTED; tail -n 1 /root/study/out/43_all.log 2>/dev/null; tail -n 1 /root/study/out/44_all.log 2>/dev/null; tail -n 1 /root/study/out/42_fixed_quantile.log 2>/dev/null; tail -n 1 /root/study/out/43_fixed_quantile.log 2>/dev/null; tail -n 1 /root/study/out/44_fixed_quantile.log 2>/dev/null", timeout=30)
        (args.out_dir / "fixed_quantile_progress.log").write_text(response.result)
        print(response.result[-1800:], flush=True)
        if "FIXED_FINISHED" in response.result or "FIXED_ABORTED" in response.result:
            (args.out_dir / "finish_allowed").touch()
            return
        sandbox.refresh_activity()
        time.sleep(30)


def continue_seeds(args, client):
    """Pause only our controller, append fixed seeds, then restore its cleanup loop."""
    sandbox = client.get(args.sandbox_id)
    if sandbox.labels.get("mlsbench-run-id") != RUN_ID or sandbox.labels.get("purpose") != "unlearning-project":
        raise RuntimeError("continuation requires this study's exact sandbox")
    controller_pid = int((args.out_dir / "controller.pid").read_text())
    if Path(f"/proc/{controller_pid}").stat().st_uid != os.getuid():
        raise RuntimeError("controller PID is not ours")
    os.kill(controller_pid, signal.SIGSTOP)
    try:
        sandbox.fs.upload_file(CONTINUATION_DRIVER.encode(), "/root/study/continue_seeds.py")
        response = sandbox.process.exec("nohup python -u /root/study/continue_seeds.py > /root/study/continuation.log 2>&1 < /dev/null &", timeout=30)
        if response.exit_code:
            raise RuntimeError("continuation driver failed to launch")
        deadline = time.monotonic() + 1200
        while True:
            response = sandbox.process.exec("test -f /root/study/out/continuation_running.json && echo CONTINUATION_RUNNING; test -f /root/study/out/continuation_aborted.json && echo CONTINUATION_ABORTED", timeout=30)
            if "CONTINUATION_RUNNING" in response.result:
                save(args.out_dir / "continuation_control.json", dict(
                    controller_pid=controller_pid, sandbox_id=sandbox.id, seeds=[43, 44],
                    runner_changed=False, configuration_changed=False, resumed_at=stamp()))
                return
            if "CONTINUATION_ABORTED" in response.result:
                raise RuntimeError("seed42 failed; continuation aborted")
            if time.monotonic() > deadline:
                raise TimeoutError("seed42 did not finish before continuation deadline")
            sandbox.refresh_activity()
            time.sleep(10)
    finally:
        os.kill(controller_pid, signal.SIGCONT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / ".local-saves/unlearning-project-20261106/daytona")
    parser.add_argument("--data", type=Path, default=ROOT / ".local-saves/unlearning-project-20261106/data/cifar10.npz")
    parser.add_argument("--original", type=Path, default=Path("/srv/home/bohanlyu/data_mlsbench/unlearn-lab/original/cifar10_resnet18_original.pt"))
    parser.add_argument("--extra-seeds", action="store_true")
    parser.add_argument("--continue-seeds", action="store_true")
    parser.add_argument("--fixed-quantile-existing", action="store_true")
    parser.add_argument("--sandbox-id")
    parser.add_argument("--deadline-hours", type=float, default=3.)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.out_dir / "controller_status.json"
    state = dict(started_at=stamp(), status="preparing", run_id=RUN_ID, image=IMAGE)
    if not args.continue_seeds and not args.fixed_quantile_existing:
        save(status_path, state)
    cap = capacity_module()
    before = cap.capacity()
    save(args.out_dir / "capacity_before.json", before)
    if before.get("free", 0) < 2:
        raise RuntimeError("Daytona reports fewer than two free GPUs")
    from daytona import Daytona, DaytonaConfig, CreateSandboxFromImageParams, Resources, Image, GpuType, ListSandboxesQuery
    client = Daytona(DaytonaConfig(api_key=cap.api_key(), api_url=cap.API, target="earth"))
    if args.fixed_quantile_existing:
        append_fixed_quantile(args, client)
        return
    if args.continue_seeds:
        continue_seeds(args, client)
        return
    name = "unlearning-study-" + uuid.uuid4().hex[:10]
    labels = {"user": "bohan", "purpose": "unlearning-project", "mlsbench-run-id": RUN_ID, "controller-name": name}
    payload = args.out_dir / "payload.tar"
    paths = ["research/unlearning_matching/run_experiments.py", "tasks/security-machine-unlearning/scripts/common.py", "tasks/security-machine-unlearning/scripts/evaluate.py"]
    paths += [f"tasks/security-machine-unlearning/edits/{method}.edit.py" for method in ("original", "ft", "ga", "rl", "l1_sparse", "salun")]
    manifest = {}
    with tarfile.open(payload, "w") as archive:
        for rel in paths:
            data = (ROOT / rel).read_bytes()
            manifest[rel] = hashlib.sha256(data).hexdigest()
            archive.add(ROOT / rel, arcname=rel)
        archive.add(args.data, arcname="data/cifar10.npz")
        archive.add(args.original, arcname="original.pt")
    save(args.out_dir / "payload_manifest.json", manifest)
    sandbox = None
    try:
        state.update(status="creating", sandbox_name=name, create_attempted_at=stamp(), maximum_create_calls=1)
        save(status_path, state)
        # One create attempt only. A timeout is reconciled by exact unique name.
        try:
            sandbox = client.create(CreateSandboxFromImageParams(
                name=name, image=Image.base(IMAGE), resources=Resources(cpu=16, memory=64, disk=30, gpu=2, gpu_type=[GpuType.H200, GpuType.H100]),
                labels=labels, auto_stop_interval=60, auto_delete_interval=0, public=False, spot=False), timeout=600)
        except BaseException:
            for found in client.list(ListSandboxesQuery(name=name, labels=labels, limit=10)):
                if found.name == name and all((found.labels or {}).get(k) == v for k, v in labels.items()):
                    sandbox = found
                    break
            if sandbox is None:
                raise
            sandbox.wait_for_sandbox_start(timeout=600)
        state.update(status="bootstrapping", sandbox_id=sandbox.id)
        save(status_path, state)
        print(json.dumps(state), flush=True)
        # STARTED can precede toolbox readiness. Retry a read-only trivial exec.
        ready_deadline = time.monotonic() + 180
        while True:
            try:
                sandbox.process.exec("true", timeout=15)
                break
            except Exception:
                if time.monotonic() > ready_deadline:
                    raise
                time.sleep(5)
        sandbox.fs.upload_file(str(payload), "/tmp/unlearning_payload.tar")
        bootstrap = "mkdir -p /root/study && tar -xf /tmp/unlearning_payload.tar -C /root/study && rm /tmp/unlearning_payload.tar && python -m pip install --no-cache-dir numpy==1.26.4 scikit-learn==1.5.2"
        response = sandbox.process.exec(bootstrap, timeout=300)
        (args.out_dir / "bootstrap.log").write_text(response.result)
        if response.exit_code:
            raise RuntimeError(f"bootstrap exited {response.exit_code}")
        sandbox.fs.upload_file(REMOTE_DRIVER.encode(), "/root/study/remote_driver.py")
        environment = "UNLEARNING_EXTRA_SEEDS=1 " if args.extra_seeds else ""
        response = sandbox.process.exec(environment + "nohup python -u /root/study/remote_driver.py > /root/study/driver.log 2>&1 < /dev/null &", timeout=30)
        if response.exit_code:
            raise RuntimeError("remote driver launch failed")
        state.update(status="running", running_at=stamp())
        save(status_path, state)
        deadline = time.monotonic() + 3600 * args.deadline_hours
        while True:
            response = sandbox.process.exec("test -f /root/study/out/finished.json && echo FINISHED; tail -n 4 /root/study/driver.log; tail -n 1 /root/study/out/42_reference.log 2>/dev/null; tail -n 1 /root/study/out/42_experiments.log 2>/dev/null", timeout=30)
            (args.out_dir / "latest_progress.log").write_text(response.result)
            state["last_checked_at"] = stamp()
            save(status_path, state)
            print(response.result[-1800:], flush=True)
            if "FINISHED" in response.result:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("study exceeded its declared controller deadline")
            sandbox.refresh_activity()
            time.sleep(30)
        state["status"] = "collecting"
        save(status_path, state)
    except BaseException as exc:
        state.update(status="failed", exception_type=type(exc).__name__, failure_at=stamp())
        save(status_path, state)
        raise
    finally:
        if sandbox is not None:
            try:
                response = sandbox.process.exec("tar -czf /tmp/unlearning_artifacts.tgz -C /root/study out driver.log remote_driver.py", timeout=180)
                if response.exit_code == 0:
                    sandbox.fs.download_file("/tmp/unlearning_artifacts.tgz", str(args.out_dir / "artifacts.tgz"))
                    subprocess.run(["tar", "-xzf", str(args.out_dir / "artifacts.tgz"), "-C", str(args.out_dir)], check=True)
                    state["artifacts_collected_at"] = stamp()
            finally:
                # This exact ID came from this invocation's create, never a broad reaper.
                client.delete(sandbox, timeout=120, wait=True)
                code, detail = cap.rest("/sandbox/" + sandbox.id)
                state.update(sandbox_deleted_at=stamp(), deletion_check_http=code,
                             deletion_check_state=detail.get("state") if isinstance(detail, dict) else None)
                if code != 404 and not (isinstance(detail, dict) and detail.get("state") in ("destroyed", "deleted")):
                    raise RuntimeError("sandbox deletion not yet confirmed")
                if state["status"] == "collecting":
                    state["status"] = "complete" if (args.out_dir / "out/complete.json").exists() else "failed"
                save(status_path, state)
        payload.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
