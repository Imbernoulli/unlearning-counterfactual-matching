"""An owned, reusable two-H200 workspace for calibration-only optimization.

No test images or retrained models are uploaded by create. A detached watchdog
collects artifacts and deletes this exact allocation at the stated UTC cutoff.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import uuid

from daytona_run import capacity_module

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "unlearning-optimization-20261106"
PURPOSE = "unlearning-calibration-optimization"
IMAGE = "pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel"


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def client_and_capacity():
    from daytona import Daytona, DaytonaConfig
    cap = capacity_module()
    return Daytona(DaytonaConfig(api_key=cap.api_key(), api_url=cap.API, target="earth")), cap


def owned(client, state):
    sandbox = client.get(state["sandbox_id"])
    if sandbox.id != state["sandbox_id"] or any((sandbox.labels or {}).get(k) != v for k, v in state["labels"].items()):
        raise RuntimeError("sandbox attribution does not match this session")
    return sandbox


PROBE = r'''import subprocess,json
rows={}
for cmd in [['nproc'],['uptime'],['free','-g'],['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total','--format=csv'],['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv']]:
 value=subprocess.check_output(cmd,text=True);rows[' '.join(cmd)]=value
 if '--query-compute-apps=pid,gpu_uuid,used_memory' in cmd:
  for line in value.splitlines()[1:]:
   if line.strip():
    pid=line.split(',')[0].strip();rows['owner '+pid]=subprocess.check_output(['ps','-o','user=','-p',pid],text=True)
print(json.dumps(rows,indent=2))
'''

STOP_WRITERS = r'''from pathlib import Path
import json,os,pwd,signal,time
root=Path('/root/study/out');root.mkdir(parents=True,exist_ok=True);(root/'STOP').touch()
registry={}
for path in (root/'optimization').glob('dispatch_gpu*.jsonl'):
 for line in path.read_text().splitlines():
  try:record=json.loads(line)
  except json.JSONDecodeError:continue
  if record.get('event')=='launched':registry[int(record['pid'])]=record.get('expected_script','calibration_study.py')
extra=root/'optimization/dispatcher_registry.jsonl'
if extra.exists():
 for line in extra.read_text().splitlines():
  record=json.loads(line);registry[int(record['pid'])]='calibration_dispatch.py'
extra=root/'optimization/auxiliary_process_registry.jsonl'
if extra.exists():
 for line in extra.read_text().splitlines():
  record=json.loads(line);registry[int(record['pid'])]=record['expected_script']
def alive():
 result=[]
 for pid,expected in registry.items():
  proc=Path('/proc')/str(pid)
  try:
   if proc.stat().st_uid != os.getuid():raise RuntimeError('registered process owner changed')
   command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
   # A departed PID may have been reused: unrelated processes are never touched.
   if expected in command and (proc/'stat').read_text().split()[2]!='Z':result.append(pid)
  except FileNotFoundError:continue
 return result
deadline=time.monotonic()+60
while alive() and time.monotonic()<deadline:time.sleep(.25)
remaining=alive()
for pid in remaining:os.kill(pid,signal.SIGTERM)
deadline=time.monotonic()+10
while alive() and time.monotonic()<deadline:time.sleep(.25)
if alive():raise RuntimeError('owned writers did not stop; refusing inconsistent archive')
print(json.dumps(dict(registered_pids=list(registry),terminated_after_grace=remaining,writers_remaining=[])))
'''


def create(args):
    from daytona import CreateSandboxFromImageParams, Resources, GpuType, Image, ListSandboxesQuery
    import numpy as np
    args.directory.mkdir(parents=True, exist_ok=True)
    path = args.directory / "session.json"
    if path.exists():
        raise RuntimeError("session.json already exists; do not allocate a duplicate")
    client, cap = client_and_capacity()
    capacity = cap.capacity()
    save(args.directory / "capacity_before.json", capacity)
    if capacity.get("free", 0) < 2:
        raise RuntimeError("fewer than two free Daytona GPUs")
    data = ROOT / ".local-saves/unlearning-project-20261106/data/cifar10.npz"
    with np.load(data, allow_pickle=False) as z:
        x, y = z["x_train"], z["y_train"].astype(np.int64)
    h = hashlib.sha256(); h.update(x.tobytes()); h.update(y.tobytes())
    if h.hexdigest() != "af4f586337eda0c26f82c4fd1864403d1ee0385e2ea10df41f8d40c8a70c8b89":
        raise RuntimeError("training data hash mismatch")
    train_only = args.directory / "cifar10_train_only.npz"
    np.savez(train_only, x_train=x, y_train=y)
    payload = args.directory / "payload.tar"
    sources = ["research/unlearning_matching/run_experiments.py", "research/unlearning_matching/fixed_quantile.py",
               "tasks/security-machine-unlearning/scripts/common.py", "tasks/security-machine-unlearning/scripts/evaluate.py"]
    sources += [f"tasks/security-machine-unlearning/edits/{m}.edit.py" for m in ("original", "ft", "ga", "rl", "l1_sparse", "salun")]
    manifest = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}
    original = Path("/srv/home/bohanlyu/data_mlsbench/unlearn-lab/original/cifar10_resnet18_original.pt")
    with tarfile.open(payload, "w") as archive:
        for source in sources:
            archive.add(ROOT / source, arcname=source)
        archive.add(train_only, arcname="data/cifar10.npz")
        archive.add(original, arcname="original.pt")
    save(args.directory / "input_manifest.json", dict(source_sha256=manifest, train_sha256=h.hexdigest(),
        data_keys=["x_train", "y_train"], test_uploaded=False, retrain_uploaded=False))
    name = "unlearning-opt-" + uuid.uuid4().hex[:10]
    labels = {"user": "bohan", "purpose": PURPOSE, "mlsbench-run-id": RUN_ID, "controller-name": name}
    state = dict(status="creating", created_at=now(), name=name, labels=labels, image=IMAGE,
                 cleanup_utc=args.cleanup_utc, maximum_create_calls=1)
    save(path, state)
    sandbox = None
    try:
        # The TTL is a provider-side backstop before the user's hard stop.
        user_end = datetime.fromisoformat("2026-11-07T05:14:30+00:00")
        ttl = max(1, int((user_end - datetime.now(timezone.utc)).total_seconds() // 60))
        try:
            sandbox = client.create(CreateSandboxFromImageParams(name=name, image=Image.base(IMAGE),
                resources=Resources(cpu=16, memory=64, disk=args.disk or 80, gpu=2, gpu_type=GpuType.H200),
                # This provider requires ephemeral GPU allocations: stopping
                # them deletes them. Never stop a live GPU sandbox to resize it.
                labels=labels, auto_stop_interval=60, auto_delete_interval=0, ttl_minutes=ttl,
                public=False, spot=False), timeout=600)
        except BaseException:
            for candidate in client.list(ListSandboxesQuery(name=name, labels=labels, limit=10)):
                if candidate.name == name and all((candidate.labels or {}).get(k) == v for k, v in labels.items()):
                    sandbox = candidate
                    break
            if sandbox is None:
                raise
            sandbox.wait_for_sandbox_start(timeout=600)
        state.update(status="bootstrapping", sandbox_id=sandbox.id)
        save(path, state)
        with (args.directory / "watchdog.log").open("w") as log:
            p = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "watch", "--directory", str(args.directory)],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (args.directory / "watchdog.pid").write_text(str(p.pid))
        deadline = time.monotonic() + 180
        while True:
            try:
                sandbox.process.exec("true", timeout=15)
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise
                time.sleep(5)
        sandbox.fs.upload_file(str(payload), "/tmp/unlearning_opt_payload.tar")
        response = sandbox.process.exec("mkdir -p /root/study && tar -xf /tmp/unlearning_opt_payload.tar -C /root/study && rm /tmp/unlearning_opt_payload.tar && python -m pip install --no-cache-dir numpy==1.26.4 scikit-learn==1.5.2", timeout=300)
        (args.directory / "bootstrap.log").write_text(response.result)
        if response.exit_code:
            raise RuntimeError("bootstrap dependency installation failed")
        sandbox.fs.upload_file(PROBE.encode(), "/root/study/resource_probe.py")
        response = sandbox.process.exec("python /root/study/resource_probe.py", timeout=30)
        (args.directory / "resources.json").write_text(response.result)
        for seed in [42, 43, 44]:
            command = "OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 python -u research/unlearning_matching/run_experiments.py --phase prepare --amp --data-dir data --out-dir out/prepared --original original.pt --seed %d --split-seed %d --threads 2" % (seed, seed)
            response = sandbox.process.exec(command, cwd="/root/study", timeout=60)
            (args.directory / f"prepare_seed{seed}.log").write_text(response.result)
            if response.exit_code:
                raise RuntimeError("split preparation failed")
        response = sandbox.process.exec("python -c \"import numpy as np,json,torch,sklearn;z=np.load('/root/study/data/cifar10.npz');assert sorted(z.files)==['x_train','y_train'];print(json.dumps(dict(keys=z.files,torch=torch.__version__,numpy=np.__version__,sklearn=sklearn.__version__)))\"", timeout=30)
        (args.directory / "versions_data_seal.json").write_text(response.result)
        if response.exit_code:
            raise RuntimeError("train-only data seal failed")
        state.update(status="ready", ready_at=now(), remote_root="/root/study", prepared_root="/root/study/out/prepared")
        save(path, state)
        print(json.dumps(state), flush=True)
    except BaseException:
        if sandbox is not None:
            client.delete(sandbox, timeout=120, wait=True)
            state.update(status="failed_deleted", failed_at=now())
            save(path, state)
        raise
    finally:
        payload.unlink(missing_ok=True)
        train_only.unlink(missing_ok=True)


def collect_and_delete(args, client, cap, state):
    with (args.directory / "cleanup.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = json.loads((args.directory / "session.json").read_text())
        if current.get("status") in ("deleted", "failed_deleted"):
            return
        sandbox = owned(client, state)
        try:
            response = sandbox.process.exec("python - <<'STOP_WRITERS'\n" + STOP_WRITERS + "\nSTOP_WRITERS", timeout=90)
            (args.directory / "writer_shutdown.json").write_text(response.result)
            if response.exit_code:
                raise RuntimeError("registered writers did not stop cleanly")
            response = sandbox.process.exec("tar -czf /tmp/unlearning_optimization_artifacts.tgz -C /root/study out research tasks", timeout=240)
            if response.exit_code:
                raise RuntimeError("optimization artifact archive failed")
            archive = args.directory / "artifacts.tgz"
            sandbox.fs.download_file("/tmp/unlearning_optimization_artifacts.tgz", str(archive))
            subprocess.run(["tar", "-xzf", str(archive), "-C", str(args.directory)], check=True)
            current["artifacts_collected_at"] = now()
        finally:
            client.delete(sandbox, timeout=120, wait=True)
            code, detail = cap.rest("/sandbox/" + sandbox.id)
            current.update(status="deleted", deleted_at=now(), deletion_check_http=code,
                           deletion_check_state=detail.get("state") if isinstance(detail, dict) else None)
            save(args.directory / "session.json", current)
            if code != 404:
                raise RuntimeError("sandbox deletion not yet confirmed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["create", "status", "upload", "download", "exec", "probe", "finish", "watch"])
    p.add_argument("--directory", type=Path, default=ROOT / ".local-saves/unlearning-project-20261106/optimization/daytona")
    p.add_argument("--cleanup-utc", default="2026-11-07T05:10:00+00:00")
    p.add_argument("--local", type=Path)
    p.add_argument("--remote")
    p.add_argument("--command")
    p.add_argument("--command-file", type=Path)
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--disk", type=int)
    a = p.parse_args()
    a.directory = a.directory.resolve()
    if a.action == "create":
        create(a); return
    state = json.loads((a.directory / "session.json").read_text())
    if a.action == "status":
        print(json.dumps(state, indent=2)); return
    if a.action == "finish":
        (a.directory / "finish_requested").touch(); return
    client, cap = client_and_capacity()
    if a.action == "watch":
        cutoff = datetime.fromisoformat(state["cleanup_utc"])
        while datetime.now(timezone.utc) < cutoff and not (a.directory / "finish_requested").exists():
            current = json.loads((a.directory / "session.json").read_text())
            if current.get("status") in ("deleted", "failed_deleted"):
                return
            try:
                owned(client, state).refresh_activity()
            except Exception as error:
                # A transient keepalive failure must not disable deadline cleanup.
                print(json.dumps(dict(at=now(), event="keepalive_error", error=str(error))), flush=True)
            time.sleep(30)
        collect_and_delete(a, client, cap, state); return
    sandbox = owned(client, state)
    if a.action == "upload":
        sandbox.fs.upload_file(str(a.local), a.remote)
        with (a.directory / "uploads.jsonl").open("a") as f:
            f.write(json.dumps(dict(at=now(), local=str(a.local), remote=a.remote,
                sha256=hashlib.sha256(a.local.read_bytes()).hexdigest())) + "\n")
    elif a.action == "download":
        a.local.parent.mkdir(parents=True, exist_ok=True)
        sandbox.fs.download_file(a.remote, str(a.local))
    else:
        command = "python /root/study/resource_probe.py" if a.action == "probe" else (a.command_file.read_text() if a.command_file else a.command)
        response = sandbox.process.exec(command, cwd="/root/study", timeout=a.timeout)
        print(response.result, end="", flush=True)
        raise SystemExit(response.exit_code or 0)


if __name__ == "__main__":
    main()
