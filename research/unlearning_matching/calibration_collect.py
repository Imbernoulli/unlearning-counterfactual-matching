"""Incrementally recover completed calibration trials, validating every artifact."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time
from daytona_session import ROOT, client_and_capacity, owned, save


LIST = """from pathlib import Path
import json
root=Path('/root/study')
print(json.dumps([str(p.parent.parent.parent.relative_to(root)) for p in sorted((root/'out/optimization').glob('*/seed*_split*/*/summary.json'))]))
"""
ARCHIVE = """from pathlib import Path
import json,tarfile,hashlib
root=Path('/root/study');trial=root/TRIAL
assert trial.parent==root/'out/optimization'
files=[p for p in sorted(trial.rglob('*')) if p.is_file() and (p.suffix in ('.json','.jsonl','.log','.npz','.py') or p.name=='best.pt')]
assert any(p.name=='best.pt' for p in files)
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
archive=Path('/tmp/calibration_'+trial.name+'.tgz')
with tarfile.open(archive,'w:gz') as tar:
 for path in files:tar.add(path,arcname=str(path.relative_to(root)))
print(json.dumps(dict(archive=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=manifest,
                     last_checkpoints=[str(p) for p in trial.rglob('last.pt')])))
"""


def remote_python(sandbox, source, timeout=60):
    result=sandbox.process.exec("python - <<'COLLECT'\n"+source+"\nCOLLECT",cwd='/root/study',timeout=timeout)
    if result.exit_code:raise RuntimeError(result.result)
    return json.loads(result.result.strip().splitlines()[-1])


def validate_trial(trial):
    summaries=list(trial.glob('seed*_split*/*/summary.json'))
    if len(summaries)!=1:raise ValueError('expected exactly one complete schedule per candidate')
    path=summaries[0];summary=json.loads(path.read_text());best=json.loads(path.with_name('best.json').read_text())
    history=[json.loads(line) for line in path.with_name('history.jsonl').read_text().splitlines()]
    stage='after_retain' if summary['schedule']=='A' else 'epoch_end'
    endpoints=[row for row in history if row['stage']==stage]
    if [row['epoch'] for row in endpoints]!=list(range(1,summary['epochs']+1)):
        raise ValueError('history does not contain every completed epoch endpoint')
    selection_stage='initial' if best['epoch']==0 else stage
    selected=[row for row in history if row['epoch']==best['epoch'] and row['stage']==selection_stage]
    if len(selected)!=1 or best!=summary['selection'] or best['selection_J']!=selected[0]['metrics']['selection_J']:
        raise ValueError('best selection does not match summary and exact history entry')
    if not path.with_name('best.pt').exists():raise ValueError('best checkpoint not recovered')
    metrics=selected[0]['metrics']
    return dict(id=trial.name,epoch=best['epoch'],J=best['selection_J'],
        KS=metrics['forget_selection_confidence_ks'],error_gap_pp=metrics['absolute_forget_selection_error_gap_pp'],
        retain_drift_pp=metrics['retain_subset_accuracy_drift_pp'],selection_drift_pp=metrics['selection_accuracy_drift_pp'],
        train_s=summary['train_seconds'],wall_s=summary['wall_seconds'],bn_delta=metrics['bn_buffer_max_delta'],
        fallback=best['is_original_fallback'],source_sha256=summary['source_sha256'],
        checkpoint=str(path.with_name('best.pt')),method=summary['schedule'],lr=summary['lr'],
        alpha=summary['forget_weight'],loss=summary['loss'],target_kind=summary['target_kind'],bn_mode=summary['bn_mode'])


def collect(args):
    client,cap=client_and_capacity()
    state=json.loads((args.directory/'session.json').read_text())
    if state['status'] in ('deleted','failed_deleted'):return False
    sandbox=owned(client,state)
    trials=remote_python(sandbox,LIST)
    for relative in trials:
        trial=args.directory/relative
        integrity=trial/'recovered_integrity.json'
        if integrity.exists():continue
        details=remote_python(sandbox,ARCHIVE.replace('TRIAL',repr(relative)))
        archive=args.directory/(trial.name+'.tgz')
        sandbox.fs.download_file(details['archive'],str(archive))
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=details['sha256']:
            raise ValueError('downloaded calibration archive checksum mismatch')
        with tarfile.open(archive) as tar:tar.extractall(args.directory,filter='data')
        for path,digest in details['files'].items():
            if hashlib.sha256((args.directory/path).read_bytes()).hexdigest()!=digest:
                raise ValueError('downloaded calibration member checksum mismatch: '+path)
        row=validate_trial(trial)
        save(integrity,dict(collected_utc=datetime.now(timezone.utc).isoformat(),**details,validation=row))
        print(json.dumps(dict(event='trial_recovered',**row)),flush=True)
        removable=list(details['last_checkpoints']) if args.delete_last else []
        if args.delete_best:
            removable.extend('/root/study/'+path for path in details['files'] if path.endswith('/best.pt'))
        if removable:
            command="from pathlib import Path\npaths="+repr(removable)+"\nfor name in paths:\n p=Path(name);assert p.name in ('last.pt','best.pt') and str(p).startswith('/root/study/out/optimization/');p.unlink(missing_ok=True)\nprint('null')"
            remote_python(sandbox,command)
        remote_python(sandbox,"from pathlib import Path\nPath("+repr(details['archive'])+").unlink(missing_ok=True)\nprint('null')")
        archive.unlink()
    rows=[]
    for trial in sorted((args.directory/'out/optimization').glob('*')):
        if (trial/'recovered_integrity.json').exists():rows.append(validate_trial(trial))
    save(args.directory/'calibration_table.json',dict(updated_utc=datetime.now(timezone.utc).isoformat(),rows=rows))
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,default=ROOT/'.local-saves/unlearning-project-20261106/optimization/daytona')
    p.add_argument('--once',action='store_true');p.add_argument('--delete-last',action='store_true')
    p.add_argument('--delete-best',action='store_true',help='remove redundant remote best only after complete local SHA verification')
    a=p.parse_args();a.directory=a.directory.resolve()
    while not (a.directory/'collector.stop').exists():
        try:
            if not collect(a):return
        except Exception as error:print(json.dumps(dict(event='collection_error',at=datetime.now(timezone.utc).isoformat(),error=str(error))),flush=True)
        if a.once:return
        time.sleep(15)


if __name__=='__main__':main()
