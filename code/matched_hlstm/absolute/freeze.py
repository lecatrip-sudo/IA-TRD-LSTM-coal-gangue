"""Freeze local matrix, identities, and real unit-test evidence before training."""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/forecast_v55'
CASES = ['80mm','150mm','300mm','0.25kxl','0.5kxl','0.75kxl']

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def write(p,x):
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')

if __name__=='__main__':
    if (OUT/'frozen.json').exists():raise SystemExit('Already frozen; refusing overwrite')
    result=subprocess.run([sys.executable,str(ROOT/'scripts/forecast_v55/test_core.py')],capture_output=True,text=True,encoding='utf-8',errors='replace')
    (OUT/'test_green.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode:raise SystemExit(result.returncode)
    files={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted((ROOT/'scripts/forecast_v55').glob('*.py'))}
    sources={f'shuju/shuju/{t}h/{c}.csv':sha(ROOT/f'shuju/shuju/{t}h/{c}.csv') for c in CASES for t in [24,48,72,96]}
    cfg=dict(cases=CASES,train_inputs_h=[24,48],train_target_h=72,test_inputs_h=[48,72],test_target_h=96,
      seeds=[42,43,44],penalties=[0.,.02],epochs=20,batch_size=1024,learning_rate=.001,gradient_clip=1.,
      hidden=128,layers=2,dropout=.1,cpu_threads=4,device='cpu',dtype='float32',amp=False,
      checkpoint='fixed_final_epoch',scheduler=None,validation_selection=False,training_wall_budget_s=7200,
      primary='row_MAE_K',sensitivity='inverse-coordinate-multiplicity row-error weighting',
      dynamic_input_change_K=.1,
      prior_test_access=True,preregistered=False)
    rows=[]
    for c in CASES:
      for s in cfg['seeds']:
       for lam in cfg['penalties']:
        rows.append(dict(run_id=f'{c}_s{s}_l{str(lam).replace(".","p")}_a',held_case=c,seed=s,penalty=lam,seed_role='randomness_estimation'))
    rows.append(dict(run_id='80mm_s42_l0p0_b',held_case='80mm',seed=42,penalty=0.,seed_role='fixed_repro_repeat'))
    env=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),processor=platform.processor(),
      torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,device='cpu',threads=4)
    write(OUT/'environment.json',env)
    git={}
    for label,args in [('commit',['rev-parse','HEAD']),('dirty',['status','--porcelain'])]:
      r=subprocess.run(['git',*args],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
      git[label]=dict(exit_code=r.returncode,output=r.stdout)
    write(OUT/'git_state.json',git)
    audit=ROOT/'.review_work/revision_v55_audit/audit.json'
    frozen=dict(frozen_at=datetime.now(timezone.utc).isoformat(),config=cfg,matrix=rows,source_sha256=sources,code_sha256=files,
      plan_sha256=sha(OUT/'PLAN.md'),environment_sha256=sha(OUT/'environment.json'),
      data_audit={'path':str(audit.relative_to(ROOT)),'sha256':sha(audit)},test_evidence={'exit_code':0,'path':'test_green.log','sha256':sha(OUT/'test_green.log')},
      limitations=['Post-hoc already-seen96h','One training target time and one evaluation target time','Case-label holdout not unseen-coordinate split','No canonical Light passport'])
    write(OUT/'frozen.json',frozen)
    print(json.dumps({'frozen_at':frozen['frozen_at'],'matrix_rows':len(rows),'source_files':len(sources)}))
