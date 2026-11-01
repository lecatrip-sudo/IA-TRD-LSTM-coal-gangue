"""Immutable R/S runner. Frozen V55 input processing; no held-case fit access."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import residual_core as rc

ROOT=Path('E:/第三篇')
OUT=Path(__file__).resolve().parents[1]
OLD=ROOT/'outputs/forecast_v55'

def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def record(path):return dict(path=Path(path).name,sha256=sha(path))

def run(row,frozen,out):
    started=now();start=time.monotonic()
    config=dict(**frozen['config'],**row,protocol_sha256=frozen['protocol_sha256'],frozen_sha256=sha(OUT/'frozen.json'))
    write(out/'config.json',config)
    for name in ['environment.json','test_evidence.json','frozen.json','input_manifest.json','PROTOCOL.md']:
        shutil.copyfile(OUT/name,out/name)
    for name in ['residual_core.py','run_residual.py']:
        shutil.copyfile(OUT/'src'/name,out/name)
    shutil.copyfile(rc.LEGACY_PATH,out/'legacy_core.py')
    manifest=dict(schema='light.run_manifest.v3',run_id=row['run_id'],matrix_row=row['run_id'],status='running',started_at=started,
        seed=dict(role='randomness_estimation',value=row['seed']),command=[sys.executable,str(Path(__file__).resolve()),'--run-id',row['run_id']],
        config=record(out/'config.json'),environment=record(out/'environment.json'),
        code=dict(commit=frozen['code_state']['commit'],dirty=True,diff_sha256=frozen['code_state']['scoped_diff_sha256'],
                  files=[record(out/p) for p in ['residual_core.py','run_residual.py','legacy_core.py']]),
        inputs=[dict(**record(out/'input_manifest.json'),role='source_lineage_and_hashes',source_revision='V55 frozen raw CFD files, unchanged')],
        provenance=dict(frozen_sha256=sha(OUT/'frozen.json'),prior_evaluation_access=True,public_preregistration=False),
        completion=dict(status='NOT_EVALUATED',evidence_artifacts=[]))
    write(out/'manifest.json',manifest)
    try:
        if os.environ.get('PYTHONHASHSEED')!=str(row['seed']):raise RuntimeError('PYTHONHASHSEED not set before process start')
        if not (sys.version.startswith('3.11.9') and torch.__version__=='2.10.0+cpu' and np.__version__=='2.4.2' and pd.__version__=='3.0.0'):
            raise RuntimeError('Historical environment drift')
        torch.set_num_interop_threads(1)
        for path,expected in frozen['code_sha256'].items():
            if sha(path)!=expected:raise RuntimeError('Frozen code changed: '+path)
        train_cases=[c for c in config['cases'] if c!=row['held_case']]
        if len(train_cases)!=5:raise RuntimeError('Invalid five-case split')
        data={}
        for c in train_cases:
            data[c]={}
            for h in [24,48,72]:
                p=ROOT/f'shuju/shuju/{h}h/{c}.csv'
                if sha(p)!=frozen['source_sha256'][p.relative_to(ROOT).as_posix()]:raise RuntimeError('Training source changed')
                data[c][h]=rc.legacy.load_csv(p)
        prepared=rc.prepare_training(data,train_cases,row['arm'])
        cs=prepared['cs'];ts=prepared['ts'];scale=prepared['scale']
        scalers=dict(coords=cs.record(),temperature=ts.record(),train_cases=train_cases)
        oldpath=OLD/'runs'/f'{row["held_case"]}_s{row["seed"]}_l0p0_a'
        oldscalers=json.loads((oldpath/'scalers.json').read_text(encoding='utf-8'))
        if scalers!=oldscalers:raise RuntimeError('Input scalers differ from archived A2')
        write(out/'scalers.json',dict(**scalers,output=dict(arm=row['arm'],scale_K=scale,scale_float32_K=float(np.float32(scale)),s_v_K=prepared['s_v'],
            s_v_fit_ids=[f'{c}/24h->48h' for c in train_cases],definition='sqrt(mean_case(mean_row((T48-T24)^2)))',fallback_if_le_1e_9_K=1.)))
        def epoch_callback(curve):
            frame=pd.DataFrame(curve);frame['train_mse_K2']=frame.train_mse_normalized*scale**2
            frame.to_csv(out/'epochs.csv',index=False,float_format='%.17g')
            print(json.dumps(dict(event='epoch',**curve[-1])),flush=True)
        print(json.dumps(dict(event='train_start',run_id=row['run_id'],n_train=len(prepared['target']),s_T=float(ts.span[0]),s_v=prepared['s_v'],output_scale=scale)),flush=True)
        result=rc.fit_model(prepared['coords'],prepared['seq'],prepared['target'],row['seed'],epochs=20,batch_size=1024,epoch_callback=epoch_callback)
        model=result['model']
        oldguard=json.loads((oldpath/'guardrails.json').read_text(encoding='utf-8'))
        if result['initial_digest']!=oldguard['initial_sha256']:raise RuntimeError('Initialization differs from same-seed A2')
        torch.save(model.state_dict(),out/'model.pt');model_saved_at=now()
        del data,prepared
        held={};evaluation_read_started_at=now()
        for h in [48,72,96]:
            p=ROOT/f'shuju/shuju/{h}h/{row["held_case"]}.csv'
            if sha(p)!=frozen['source_sha256'][p.relative_to(ROOT).as_posix()]:raise RuntimeError('Evaluation source changed')
            held[h]=rc.legacy.load_csv(p)
        if not all(np.array_equal(held[48][0],held[h][0]) for h in [72,96]):raise RuntimeError('Evaluation alignment failure')
        xyz=held[72][0];x=torch.from_numpy(cs.transform(xyz))
        sequence=torch.from_numpy(np.stack([ts.transform(held[h][1]) for h in [48,72]],axis=1))
        model.eval();parts=[]
        with torch.no_grad():
            for i in range(0,len(x),4096):parts.append(model(x[i:i+4096],sequence[i:i+4096]).numpy())
        z=np.vstack(parts).reshape(-1)
        pred=rc.decode_increment(z,held[72][1],scale).astype(np.float64)
        truth=held[96][1].reshape(-1);persistence=held[72][1].reshape(-1);linear=2*persistence-held[48][1].reshape(-1)
        frame=pd.DataFrame(xyz,columns=['X','Y','Z']);frame.insert(0,'row_id',np.arange(len(xyz)))
        for name,arr in [('T48_K',held[48][1].reshape(-1)),('T72_K',persistence),('T96_K',truth),('predicted_increment_normalized',z),('Tpred_K',pred),('persistence_K',persistence),('linear_K',linear)]:frame[name]=arr
        frame.to_csv(out/'predictions.csv.gz',index=False,float_format='%.17g',compression={'method':'gzip','mtime':0})
        dynamic=np.abs(persistence-held[48][1].reshape(-1))>.1;raw=[]
        for name,p in [(row['arm'],pred),('persistence',persistence),('linear',linear)]:
            raw.append(dict(method=name,stratum='all_rows',**rc.legacy.metrics(truth,p)))
            raw.append(dict(method=name,stratum='equal_coordinate_weight',**rc.legacy.coordinate_metrics(xyz,truth,p)))
            for label,mask in [('dynamic_inputs_gt0p1K',dynamic),('low_change_inputs_le0p1K',~dynamic)]:
                if mask.any():raw.append(dict(method=name,stratum=label,**rc.legacy.metrics(truth[mask],p[mask])))
        write(out/'raw_metrics.json',raw)
        (out/'raw_metrics.jsonl').write_text('\n'.join(json.dumps(r,allow_nan=False) for r in raw)+'\n',encoding='utf-8')
        guard=dict(train_cases=train_cases,held_case=row['held_case'],held_excluded_from_fit=row['held_case'] not in train_cases,
            input_scaler_matches_A2=True,initialization_matches_A2=True,initial_sha256=result['initial_digest'],s_v_uses_only_train24and48=True,
            no96_in_fit=True,training_inputs_h=[24,48],training_target_increment_h=[48,72],evaluation_target_increment_h=[72,96],row_pooled_loss=True,
            model_saved_at=model_saved_at,evaluation_read_started_at=evaluation_read_started_at,order_model_saved_before_evaluation=model_saved_at<=evaluation_read_started_at,
            completed_epochs=len(result['epochs']),n_train=result['n_train'],n_test=len(xyz),n_dynamic=int(dynamic.sum()),finite_outputs=True,
            max_preclip_gradient_norm=result['max_grad'],dtype='float32',output_dtype='float32',device='cpu',intraop_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
            no_clipping_at_inference=True,shared_coordinates_possible=True,strictly_independent_rows_claim=False)
        if guard['completed_epochs']!=20 or not guard['order_model_saved_before_evaluation']:raise RuntimeError('Protocol completion failure')
        write(out/'guardrails.json',guard)
        manifest.update(status='completed',ended_at=now(),wall_s=time.monotonic()-start,termination=dict(reason='natural_exit',exit_code=0,signal=None),
            completion=dict(status='PASS',oracle=['20 predeclared epochs completed','saved final model before held-case read','finite predictions and exact input parity'],evidence_artifacts=['test_evidence','raw_metrics','predictions','guardrail_evidence']))
        print(json.dumps(dict(event='complete',run_id=row['run_id'],wall_s=manifest['wall_s'],metrics=raw[0])),flush=True)
    except BaseException as exc:
        write(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),at=now()))
        manifest.update(status='failed',ended_at=now(),wall_s=time.monotonic()-start,termination=dict(reason='error',exit_code=1,signal=None),completion=dict(status='FAIL',evidence_artifacts=['failure']))
        write(out/'manifest.json',manifest)
        raise
    write(out/'manifest.json',manifest)

def finalize_manifest(out):
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    names=dict(stdout='stdout.log',stderr='stderr.log',raw_metrics='raw_metrics.jsonl',predictions='predictions.csv.gz',test_evidence='test_evidence.json',guardrail_evidence='guardrails.json',failure='failure.json',weights='model.pt',scalers='scalers.json',training_curve='epochs.csv')
    manifest['artifacts']={k:record(out/v) for k,v in names.items() if (out/v).exists()}
    manifest['all_artifact_sha256']={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}
    write(out/'manifest.json',manifest)

def launch():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id');parser.add_argument('--queue',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();frozen=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'))
    if args.run_id:
        row=next(r for r in frozen['matrix'] if r['run_id']==args.run_id);run(row,frozen,OUT/'runs'/row['run_id']);return
    if not args.queue:raise SystemExit('Specify --queue or --run-id')
    start=time.monotonic();runs=OUT/'runs';runs.mkdir(exist_ok=True)
    progress=[]
    for row in frozen['matrix']:
        out=runs/row['run_id']
        if out.exists():
            status=json.loads((out/'manifest.json').read_text(encoding='utf-8')).get('status') if (out/'manifest.json').exists() else 'unmanifested'
            if args.resume and status=='completed':continue
            raise RuntimeError('Refusing to overwrite existing run: '+str(out))
        if time.monotonic()-start>43200:
            write(OUT/'queue_budget_stop.json',dict(at=now(),next_run=row['run_id'],reason='12 h queue guard; matrix incomplete'));raise SystemExit(2)
        out.mkdir();env=dict(os.environ,PYTHONHASHSEED=str(row['seed']),PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        with (out/'stdout.log').open('w',encoding='utf-8') as stdout,(out/'stderr.log').open('w',encoding='utf-8') as stderr:
            proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--run-id',row['run_id']],cwd=OUT,env=env,stdout=stdout,stderr=stderr)
        finalize_manifest(out)
        progress.append(dict(run_id=row['run_id'],exit_code=proc.returncode,queue_elapsed_s=time.monotonic()-start))
        write(OUT/'queue_progress.json',progress);print(json.dumps(progress[-1]),flush=True)
        if proc.returncode:raise SystemExit(proc.returncode)
    write(OUT/'queue_completed.json',dict(at=now(),wall_s=time.monotonic()-start,expected_runs=36))

if __name__=='__main__':launch()
