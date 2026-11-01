"""Immutable V55 fold runner. Test targets are read only after final fitting."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset,DataLoader
import core

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/forecast_v55'

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def run(row,frozen,out):
    cfg=frozen['config']
    core.seed_all(row['seed'])
    torch.set_num_interop_threads(1)
    if os.environ.get('PYTHONHASHSEED')!=str(row['seed']):raise RuntimeError('Launcher must set PYTHONHASHSEED before process start')
    started=now();t0=time.monotonic()
    manifest=dict(schema='v55.local_run_manifest.v1',run_id=row['run_id'],matrix_row=row,started_at=started,
      frozen_sha256=sha(OUT/'frozen.json'),code_sha256=frozen['code_sha256'],source_sha256=frozen['source_sha256'],
      environment_sha256=frozen['environment_sha256'],argv=sys.argv,status='running',seed=row['seed'],device='cpu',dtype='float32')
    write(out/'manifest.json',manifest);write(out/'config.json',dict(**cfg,**row))
    try:
      for p,h in frozen['code_sha256'].items():
        if sha(ROOT/p)!=h:raise RuntimeError('Frozen code changed: '+p)
      train_cases=[c for c in cfg['cases'] if c!=row['held_case']]
      data={};fit_ids=core.training_ids(cfg['cases'],row['held_case'])
      for c in train_cases:
        data[c]={}
        for hour in [24,48,72]:
          p=ROOT/f'shuju/shuju/{hour}h/{c}.csv'
          if sha(p)!=frozen['source_sha256'][str(p.relative_to(ROOT)).replace('\\','/')]:raise RuntimeError('Source changed')
          data[c][hour]=core.load_csv(p)
        for hour in [48,72]:
          if not np.array_equal(data[c][24][0],data[c][hour][0]):raise RuntimeError('Training coordinate mismatch')
      cs=core.Scaler().fit(np.vstack([data[c][24][0] for c in train_cases]),[f'{c}/24h' for c in train_cases])
      ts=core.Scaler().fit(np.vstack([data[c][h][1] for c in train_cases for h in [24,48,72]]),fit_ids)
      coords=np.vstack([cs.transform(data[c][24][0]) for c in train_cases])
      seq=np.vstack([np.stack([ts.transform(data[c][h][1]) for h in [24,48]],axis=1) for c in train_cases])
      target=np.vstack([ts.transform(data[c][72][1]) for c in train_cases])
      write(out/'scalers.json',dict(coords=cs.record(),temperature=ts.record(),train_cases=train_cases))
      generator=torch.Generator().manual_seed(row['seed'])
      ds=TensorDataset(torch.from_numpy(coords),torch.from_numpy(seq),torch.from_numpy(target))
      loader=DataLoader(ds,batch_size=cfg['batch_size'],shuffle=True,num_workers=0,drop_last=False,
                        generator=generator,worker_init_fn=core.seed_worker)
      model=core.Model();optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
      # Content digest of initialization supports exactly paired lambda comparisons.
      initial_digest=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
      print(json.dumps(dict(event='train_start',held=row['held_case'],seed=row['seed'],penalty=row['penalty'],n_train=len(ds),batches=len(loader),initial_sha256=initial_digest)),flush=True)
      epochs=[];max_grad=0.
      for ep in range(1,cfg['epochs']+1):
        et=time.monotonic();model.train();mse_sum=0.;pen_sum=0.;n=0
        for xb,sb,yb in loader:
          optimizer.zero_grad(set_to_none=True);pred=model(xb,sb)
          mse=torch.mean((pred-yb)**2);pen=torch.mean(torch.abs(pred-sb[:,-1,:]))
          loss=mse+row['penalty']*pen
          if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
          loss.backward()
          norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
          max_grad=max(max_grad,float(norm));optimizer.step()
          n+=len(xb);mse_sum+=float(mse.detach())*len(xb);pen_sum+=float(pen.detach())*len(xb)
        rec=dict(epoch=ep,n=n,train_mse_normalized=mse_sum/n,train_mse_K2=(mse_sum/n)*float(ts.span[0])**2,
                 train_temporal_MAE_normalized=pen_sum/n,wall_s=time.monotonic()-et)
        epochs.append(rec);pd.DataFrame(epochs).to_csv(out/'epochs.csv',index=False)
        print(json.dumps(rec),flush=True)
      torch.save(model.state_dict(),out/'model.pt')
      model_saved_at=now()
      # The entire held case, including observed input arrays, is loaded only after fitting.
      held={}
      for hour in [48,72,96]:
        p=ROOT/f'shuju/shuju/{hour}h/{row["held_case"]}.csv'
        if sha(p)!=frozen['source_sha256'][str(p.relative_to(ROOT)).replace('\\','/')]:raise RuntimeError('Evaluation source changed')
        held[hour]=core.load_csv(p)
      if not all(np.array_equal(held[48][0],held[h][0]) for h in [72,96]):raise RuntimeError('Evaluation row alignment failure')
      eval_read_at=now();xyz=held[72][0];x=torch.from_numpy(cs.transform(xyz))
      s=torch.from_numpy(np.stack([ts.transform(held[h][1]) for h in [48,72]],axis=1))
      model.eval();preds=[]
      with torch.no_grad():
        for i in range(0,len(x),4096):preds.append(model(x[i:i+4096],s[i:i+4096]).numpy())
      pred=ts.inverse(np.vstack(preds)).reshape(-1).astype(np.float64)
      truth=held[96][1].reshape(-1);persistence=held[72][1].reshape(-1);linear=2*persistence-held[48][1].reshape(-1)
      frame=pd.DataFrame(xyz,columns=['X','Y','Z']);frame.insert(0,'row_id',np.arange(len(xyz)))
      frame['T48_K']=held[48][1].reshape(-1);frame['T72_K']=persistence;frame['T96_K']=truth
      frame['Tpred_K']=pred;frame['persistence_K']=persistence;frame['linear_K']=linear
      frame.to_csv(out/'predictions.csv.gz',index=False,float_format='%.17g',compression={'method':'gzip','mtime':0})
      raw=[];dynamic=np.abs(persistence-held[48][1].reshape(-1))>cfg['dynamic_input_change_K']
      for name,p in [('LSTM',pred),('persistence',persistence),('linear',linear)]:
        raw.append(dict(method=name,stratum='all_rows',**core.metrics(truth,p)))
        raw.append(dict(method=name,stratum='equal_coordinate_weight',**core.coordinate_metrics(xyz,truth,p)))
        for label,mask in [('dynamic_inputs_gt0p1K',dynamic),('low_change_inputs_le0p1K',~dynamic)]:
          if mask.any():raw.append(dict(method=name,stratum=label,**core.metrics(truth[mask],p[mask])))
      write(out/'raw_metrics.json',raw)
      guard=dict(train_cases=train_cases,held_case=row['held_case'],fit_ids=fit_ids,held_excluded_from_fit=row['held_case'] not in train_cases,
        no96_in_fit=all('96h' not in v for v in fit_ids),row_alignment=True,finite_outputs=True,completed_epochs=len(epochs),
        initial_sha256=initial_digest,max_preclip_gradient_norm=max_grad,model_saved_at=model_saved_at,evaluation_read_at=eval_read_at,
        order_model_saved_before_evaluation=model_saved_at<=eval_read_at,n_test=len(xyz),n_dynamic=int(dynamic.sum()),
        shared_coordinates_possible=True,strictly_independent_rows_claim=False)
      write(out/'guardrails.json',guard)
      manifest.update(status='completed',termination='fixed_epochs_completed',ended_at=now(),wall_s=time.monotonic()-t0,
        artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['manifest.json','stdout.log','stderr.log']})
      write(out/'manifest.json',manifest)
      print(json.dumps(dict(event='complete',wall_s=manifest['wall_s'],metrics=raw[:1])),flush=True)
    except BaseException as e:
      write(out/'failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),at=now()))
      manifest.update(status='failed',termination='exception',ended_at=now(),wall_s=time.monotonic()-t0)
      write(out/'manifest.json',manifest);raise

def launch():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id');parser.add_argument('--queue',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();frozen=json.loads((OUT/'frozen.json').read_text(encoding='utf-8'))
    if args.run_id:
      row=next(r for r in frozen['matrix'] if r['run_id']==args.run_id);run(row,frozen,OUT/'runs'/args.run_id);return
    if not args.queue:raise SystemExit('Specify --queue or --run-id')
    t0=time.monotonic();runs=OUT/'runs';runs.mkdir(exist_ok=True)
    progress=[]
    for row in frozen['matrix']:
      out=runs/row['run_id']
      if out.exists():
        status=json.loads((out/'manifest.json').read_text())['status'] if (out/'manifest.json').exists() else 'unmanifested'
        if args.resume and status=='completed':continue
        raise RuntimeError('Existing run cannot be overwritten: '+row['run_id'])
      elapsed=time.monotonic()-t0
      if elapsed>frozen['config']['training_wall_budget_s']:
        write(OUT/'queue_budget_stop.json',dict(at=now(),elapsed_s=elapsed,next_run=row['run_id'],reason='Frozen budget reached; require amendment'))
        raise SystemExit(2)
      out.mkdir();env=dict(os.environ,PYTHONHASHSEED=str(row['seed']),PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
      with (out/'stdout.log').open('w',encoding='utf-8') as stdout,(out/'stderr.log').open('w',encoding='utf-8') as stderr:
        proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--run-id',row['run_id']],cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
      progress.append(dict(run_id=row['run_id'],exit_code=proc.returncode,queue_elapsed_s=time.monotonic()-t0))
      write(OUT/'queue_progress.json',progress)
      print(json.dumps(progress[-1]),flush=True)
      if proc.returncode:raise SystemExit(proc.returncode)
    write(OUT/'queue_completed.json',dict(at=now(),wall_s=time.monotonic()-t0,expected_runs=len(frozen['matrix'])))

if __name__=='__main__':launch()
