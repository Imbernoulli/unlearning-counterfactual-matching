"""Wait for an entirely idle GPU 4, then run the independent unlearning study.

This CPU-only supervisor never stops an existing job.  Only the exact child
PID created here may be stopped if another job appears on GPU 4.  Run this
supervisor detached; it exits when all seeds finish or an experiment fails.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


GPU = 4
ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).with_name("run_experiments.py")
STOP = threading.Event()


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(event, **values):
    print(json.dumps(dict(time=now(), event=event, **values)), flush=True)


def command(argv):
    return subprocess.check_output(argv, text=True, timeout=20).strip()


def write_status(path, state, **values):
    obj = dict(time=now(), state=state, supervisor_pid=os.getpid(), gpu=GPU, **values)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)
    return obj


def inventory():
    """Inspect ALL GPU process owners, including jobs on other cards."""
    gpu_csv = command(["nvidia-smi", "--query-gpu=index,uuid,memory.used",
                       "--format=csv"])
    app_csv = command(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                       "--format=csv"])
    cards = list(csv.DictReader(io.StringIO(gpu_csv), skipinitialspace=True))
    card = next(row for row in cards if int(row["index"]) == GPU)
    processes = []
    for row in csv.DictReader(io.StringIO(app_csv), skipinitialspace=True):
        pid = int(row["pid"])
        owner = subprocess.run(["ps", "-o", "user=", "-p", str(pid)],
                               text=True, capture_output=True, timeout=10)
        processes.append(dict(pid=pid, gpu_uuid=row["gpu_uuid"].strip(),
                              used_memory=row.get("used_gpu_memory [MiB]", ""),
                              owner=owner.stdout.strip() or "exited-or-unknown"))
    memory_key = next(k for k in card if k.startswith("memory.used"))
    memory_mib = int(card[memory_key].split()[0])
    return dict(gpu=GPU, uuid=card["uuid"].strip(), memory_mib=memory_mib,
                processes=processes,
                occupants=[p for p in processes if p["gpu_uuid"] == card["uuid"].strip()])


def resources():
    # Keep the requested command-level resource evidence in the supervisor log.
    snapshot = dict(nproc=int(command(["nproc"])), uptime=command(["uptime"]),
                    free_g=command(["free", "-g"]), load=os.getloadavg())
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        mem[key] = int(value.split()[0])
    snapshot["available_gib"] = mem["MemAvailable"] / 1024 ** 2
    snapshot["ready"] = (snapshot["load"][0] < .75 * snapshot["nproc"]
                         and snapshot["available_gib"] >= 16)
    return snapshot


def lease():
    directory = Path.home() / ".cache/mlsb/gpu-leases"
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / f"gpu{GPU}.lock").open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def stop_own_child(child):
    """Never act on a process that was not created by this supervisor."""
    if child is None or child.poll() is not None:
        return
    if Path(f"/proc/{child.pid}").stat().st_uid != os.geteuid():
        raise RuntimeError(f"refusing to stop child PID {child.pid}: owner changed")
    emit("stop_own_child", pid=child.pid)
    child.terminate()
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        # Popen still owns this unreaped PID, which cannot have been reused.
        child.kill()
        child.wait(timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seeds", default="42,43,44")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if not seeds or len(seeds) != len(set(seeds)):
        parser.error("provide a nonempty list of distinct integer seeds")
    args.python = args.python.expanduser().resolve()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    if not args.python.is_file() or not (args.data_dir / "cifar10.npz").is_file():
        parser.error("Python interpreter and prepared cifar10.npz must exist")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.out_dir / "queue_status.json"
    supervisor_lock = (args.out_dir / "queue.lock").open("a+")
    try:
        fcntl.flock(supervisor_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("another supervisor already owns this output directory")
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: STOP.set())
    startup_hash = sha256(RUNNER)
    completed, child = [], None
    emit("start", seeds=seeds, runner_sha256=startup_hash, resources=resources())
    try:
        for seed in seeds:
            log_path = args.out_dir / f"seed{seed}_split{seed}.log"
            while not STOP.is_set():
                held = None
                try:
                    view = inventory()
                    machine = resources()
                    if view["occupants"] or view["memory_mib"] >= 1024 or not machine["ready"]:
                        write_status(status_path, "waiting", seed=seed, completed=completed,
                                     inventory=view, resources=machine, log=str(log_path))
                        emit("waiting", seed=seed, occupants=view["occupants"],
                             memory_mib=view["memory_mib"], cpu_ready=machine["ready"])
                        STOP.wait(30)
                        continue
                    held = lease()
                    if held is None:
                        write_status(status_path, "waiting_for_lease", seed=seed,
                                     completed=completed, log=str(log_path))
                        STOP.wait(30)
                        continue
                    # Race check after acquiring the shared lease, before launch.
                    view = inventory()
                    if view["occupants"] or view["memory_mib"] >= 1024:
                        emit("occupied_after_lease", seed=seed, inventory=view)
                        continue
                    current_hash = sha256(RUNNER)
                    argv = [str(args.python), "-u", str(RUNNER), "--phase", "all",
                            "--data-dir", str(args.data_dir), "--out-dir", str(args.out_dir),
                            "--seed", str(seed), "--split-seed", str(seed),
                            "--gpu", str(GPU), "--threads", "2", "--amp"]
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(GPU),
                               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                               OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
                    with log_path.open("a") as logfile:
                        logfile.write(json.dumps(dict(event="queue_launch", time=now(),
                            argv=argv, runner_sha256=current_hash)) + "\n")
                        logfile.flush()
                        child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=logfile,
                                                 stderr=subprocess.STDOUT, start_new_session=True)
                    emit("launched", seed=seed, pid=child.pid, log=str(log_path),
                         runner_sha256=current_hash, changed_since_queue=current_hash != startup_hash,
                         resources=machine, inventory=view)
                    collision = False
                    STOP.wait(2)
                    while child.poll() is None and not STOP.is_set():
                        view = inventory()
                        others = [p for p in view["occupants"] if p["pid"] != child.pid]
                        if others:
                            emit("collision", seed=seed, own_pid=child.pid, others=others)
                            stop_own_child(child)
                            collision = True
                            write_status(status_path, "waiting_after_collision", seed=seed,
                                         completed=completed, inventory=view, log=str(log_path))
                            break
                        write_status(status_path, "running", seed=seed, child_pid=child.pid,
                                     completed=completed, inventory=view, log=str(log_path),
                                     runner_sha256=current_hash)
                        STOP.wait(30)
                    if STOP.is_set():
                        stop_own_child(child)
                        break
                    if collision:
                        child = None
                        continue
                    returncode = child.wait()
                    child = None
                    if returncode:
                        write_status(status_path, "failed", seed=seed, returncode=returncode,
                                     completed=completed, log=str(log_path))
                        emit("failed", seed=seed, returncode=returncode, log=str(log_path))
                        return returncode if 0 < returncode < 256 else 1
                    completed.append(seed)
                    write_status(status_path, "seed_completed", seed=seed, completed=completed,
                                 log=str(log_path))
                    emit("seed_completed", seed=seed)
                    break
                finally:
                    if held is not None:
                        held.close()
            if STOP.is_set():
                write_status(status_path, "stopped", seed=seed, completed=completed)
                return 130
        write_status(status_path, "completed", completed=completed)
        emit("completed", seeds=completed)
        return 0
    except Exception as error:
        stop_own_child(child)
        write_status(status_path, "failed", completed=completed,
                     error=f"{type(error).__name__}: {error}")
        raise
    finally:
        supervisor_lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
