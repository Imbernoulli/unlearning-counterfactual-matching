"""Calibration-only SalUn warmstart followed by unchanged joint-energy updates.

This new development round is separate from the earlier frozen energy study.
SalUn supplies the fixed held-out target; Original remains the diagnostic reference.
Only the update initialization is SalUn. Its BatchNorm buffers stay frozen;
all parameters can update. Selection includes unchanged SalUn at epoch zero,
with utility constraints relative to SalUn, and never reads audit/test/Retrain.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def load(name):
    spec = importlib.util.spec_from_file_location('salun_residual_' + name[:-3], HERE / name)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def utility_feasible(metrics, warmstart_summary):
    return (metrics['retain_subset_accuracy_drift_from_warmstart_pp'] >= -.25 - 1e-8
            and metrics['selection_accuracy_drift_from_warmstart_pp'] >= -.5 - 1e-8)


def install(study, warmstart, warmstart_metadata, warmstart_sha):
    import numpy as np
    import torch
    energy_preregister, energy_schedule = study.preregister, study.study_schedule
    energy_targets = study.study_targets
    original_diagnostics = study.Diagnostics
    original_dump = study.base.dump
    metadata_file = warmstart.with_suffix('.json')
    provenance = dict(residual_wrapper_sha256=SOURCE_SHA256,
        warmstart_file=str(warmstart.resolve()), warmstart_sha256=warmstart_sha,
        warmstart_metadata_sha256=hashlib.sha256(metadata_file.read_bytes()).hexdigest(),
        initialization='inherited SalUn checkpoint, including its BatchNorm buffers',
        frozen_target_teacher='SalUn warmstart checkpoint on held-out target fold',
        diagnostic_teacher='Original checkpoint, unchanged',
        protocol_version=2,
        residual_selection='minimum feasible selection J including unchanged SalUn epoch0',
        residual_retain_degradation_max_pp=.25, residual_selection_degradation_max_pp=.5,
        residual_feasibility_reference='SalUn warmstart', residual_bn_reference='SalUn warmstart',
        study_kind='new_salun_residual_development_round')
    state = torch.load(warmstart, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not state or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('warmstart must be a plain model state dict')

    def preregister(a):
        if a.schedules != ['C'] or a.bn_mode != 'frozen':
            raise ValueError('residual round requires C and frozen BatchNorm')
        if any(warmstart_metadata.get(k) != getattr(a, k) for k in ('seed', 'split_seed')):
            raise ValueError('SalUn warmstart seed/deletion split differs')
        if warmstart_metadata.get('test_used_during_training') is not False:
            raise ValueError('SalUn provenance must exclude test during training')
        with tempfile.TemporaryDirectory(prefix='salun-residual-preregister-') as directory:
            shadow = SimpleNamespace(**{**vars(a), 'run_dir': Path(directory)})
            energy_preregister(shadow)
            config = json.loads((Path(directory)/'preregister.json').read_text())['configuration']
        config.update(provenance)
        config['selection_rule'] = dict(retain_accuracy_loss_max_pp=.25,
            selection_accuracy_loss_max_pp=.5, relative_to='SalUn warmstart',
            objective='max(abs(error_forget-error_selection), pooled_confidence_KS); probability units',
            within_run_tie='earlier epoch, including0', cross_run_tie='earlier epoch, smaller lr, stable id',
            fallback='unchanged SalUn epoch0 competes with all feasible endpoints')
        path = a.run_dir/'preregister.json'
        if path.exists():
            raise FileExistsError('residual run directories are immutable; use an explicit retry directory')
        original_dump(path, dict(created_utc=datetime.now(timezone.utc).isoformat(), configuration=config))
        snapshot=a.run_dir/'source_snapshot';snapshot.mkdir(exist_ok=True)
        for filename in ('salun_residual_study.py','energy_study.py','calibration_study.py'):
            (snapshot/filename).write_bytes((HERE/filename).read_bytes())

    class ResidualDiagnostics(original_diagnostics):
        def __init__(self, a, data, original_model):
            super().__init__(a, data, original_model)
            model=study.base.model_from(state).eval()
            self.warmstart_summary=self.summarize(self.distributions(model))
            self.warmstart_bn=study.bn_buffers(model)
            original_dump(a.run_dir/'warmstart_diagnostics.json',self.warmstart_summary)
            del model

        def record(self, model, destination, epoch, stage):
            # Inherited recorder insists its frozen buffers equal Original.
            # Evaluate that recorder in inherited mode, then separately enforce
            # exact equality against SalUn below; no model forward mode changes.
            mode=self.a.bn_mode;self.a.bn_mode='inherited'
            try:
                record=super().record(model,destination,epoch,stage)
            finally:
                self.a.bn_mode=mode
            metrics=record['metrics'];summary=record['distributions_summary']
            for dataset,key in [('retain_subset','retain_subset'),('selection','selection')]:
                metrics[key+'_accuracy_drift_from_warmstart_pp']=(
                    summary[dataset]['all']['accuracy_pct']-self.warmstart_summary[dataset]['all']['accuracy_pct'])
            current=study.bn_buffers(model)
            delta=max(float((current[k].double()-v.double()).abs().max()) for k,v in self.warmstart_bn.items())
            metrics.update(bn_buffer_max_delta_from_warmstart=delta,
                bn_buffers_equal_warmstart=delta==0.,feasible=utility_feasible(metrics,self.warmstart_summary))
            if delta!=0.:raise RuntimeError('SalUn BatchNorm buffers changed')
            # Replace only this call's just-appended diagnostic record, keeping
            # inherited Original-relative drift/buffer fields truthful.
            history=destination/'history.jsonl';lines=history.read_text().splitlines()
            lines[-1]=json.dumps(record);history.write_text('\n'.join(lines)+'\n')
            study.base.log('salun_residual_diagnostics',epoch=epoch,stage=stage,**metrics)
            return record

    def schedule(a,name,data,teacher,targets,diagnostics,_original_state):
        # Original teacher/targets/diagnostics remain intact; only trainable
        # initialization passed to the unchanged loop changes.
        energy_schedule(a,name,data,teacher,targets,diagnostics,state)
        destination=a.run_dir/name
        path=destination/'summary.json'
        if not path.exists():return
        summary=json.loads(path.read_text())
        initial=summary['initial_calibration'];selected=summary['selection']
        # Epoch0 is always feasible relative to itself and wins exact ties.
        if selected.get('is_original_fallback') or initial['selection_J']<=selected['selection_J']:
            torch.save(state,destination/'best.pt.tmp');os.replace(destination/'best.pt.tmp',destination/'best.pt')
            selected=dict(epoch=0,selection_J=initial['selection_J'],feasible=True,
                is_original_fallback=False,is_warmstart_fallback=True,checkpoint='best.pt',
                reason='unchanged SalUn selected by registered epoch0 comparison',
                schedule=name,lr=a.lr,**provenance)
        else:
            selected.update(is_original_fallback=False,is_warmstart_fallback=False,**provenance)
        selected.update(retain_subset_accuracy_drift_pp=(initial['retain_subset_accuracy_drift_pp'] if selected['epoch']==0 else selected['retain_subset_accuracy_drift_pp']),
            selection_accuracy_drift_pp=(initial['selection_accuracy_drift_pp'] if selected['epoch']==0 else selected['selection_accuracy_drift_pp']))
        summary.update(selection=selected,**provenance)
        study.base.dump(destination/'best.json',selected)
        study.base.dump(path,summary)
        last=json.loads((destination/'last.json').read_text());last.update(provenance)
        study.base.dump(destination/'last.json',last)

    def targets(a, original_teacher, data):
        # This pass reads target-only arrays through the unchanged energy hook.
        # The model used by diagnostics remains the separate Original teacher.
        warmstart_teacher=study.base.model_from(state).eval().requires_grad_(False)
        try:
            result=energy_targets(a,warmstart_teacher,data)
        finally:
            del warmstart_teacher
        path=a.run_dir/'energy_target_metadata.json'
        target_metadata=json.loads(path.read_text())
        target_metadata.update(target_teacher='SalUn warmstart',
            target_teacher_sha256=warmstart_sha,residual_wrapper_sha256=SOURCE_SHA256,
            diagnostic_teacher='Original checkpoint',protocol_version=2)
        original_dump(path,target_metadata)
        return result

    study.preregister,study.study_schedule=preregister,schedule
    study.study_targets=targets
    study.Diagnostics=ResidualDiagnostics
    return provenance


def self_test():
    assert utility_feasible(dict(retain_subset_accuracy_drift_from_warmstart_pp=-.25,
        selection_accuracy_drift_from_warmstart_pp=-.5),{})
    assert not utility_feasible(dict(retain_subset_accuracy_drift_from_warmstart_pp=-.251,
        selection_accuracy_drift_from_warmstart_pp=0),{})
    assert not utility_feasible(dict(retain_subset_accuracy_drift_from_warmstart_pp=0,
        selection_accuracy_drift_from_warmstart_pp=-.501),{})
    energy=load('energy_study.py');energy.self_test()
    assert not energy.torch.cuda.is_initialized()
    print('SalUn residual CPU checks passed: utility bounds; unchanged energy checks; CUDA uninitialized')


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--warmstart',type=Path)
    parser.add_argument('--original-manifest',type=Path)
    parser.add_argument('--self-test',action='store_true')
    options,remaining=parser.parse_known_args()
    if options.self_test:
        if options.warmstart or options.original_manifest or remaining:raise ValueError('self-test takes no other options')
        self_test();return
    if options.warmstart is None:raise ValueError('--warmstart SalUn.pt is required')
    warmstart=options.warmstart.resolve(strict=True)
    meta=json.loads(warmstart.with_suffix('.json').read_text())
    if options.original_manifest:
        helper=load('with_original_manifest.py')
        manifest,checkpoint=helper.manifest_pin(options.original_manifest)
        energy=helper.load_and_pin('energy_study.py',manifest,checkpoint,options.original_manifest)
        if any(arg=='--original' or arg.startswith('--original=') for arg in remaining):raise ValueError('omit --original with --original-manifest')
        remaining=['--original',str(checkpoint),*remaining]
    else:
        energy=load('energy_study.py')
    study=energy.load_harness();energy.install_hooks(study,'joint')
    install(study,warmstart,meta,hashlib.sha256(warmstart.read_bytes()).hexdigest())
    sys.argv=[sys.argv[0],*remaining];study.main()

if __name__=='__main__':main()
