"""Increment-output extension; original model/input scalers remain unchanged."""
import importlib.util
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

LEGACY_PATH = Path('E:/第三篇/outputs/forecast_v55/frozen_code/core.py')
spec = importlib.util.spec_from_file_location('v55_core', LEGACY_PATH)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)

def increment_scale(cases):
    if not cases:
        raise ValueError('No training cases')
    means=[]
    for case in cases:
        diff=legacy.finite(np.asarray(case[48][1],dtype=np.float64)-np.asarray(case[24][1],dtype=np.float64))
        means.append(float(np.mean(diff**2)))
    scale=float(np.sqrt(np.mean(means)))
    if not np.isfinite(scale):
        raise ValueError('Nonfinite increment scale')
    return scale if scale>1e-9 else 1.0

def prepare_training(data, ordered_train_cases, arm):
    if arm not in ['R','S'] or not ordered_train_cases:
        raise ValueError('Unsupported arm or empty training set')
    for c in ordered_train_cases:
        for h in [48,72]:
            if not np.array_equal(data[c][24][0],data[c][h][0]):
                raise ValueError('Training coordinate mismatch')
    fit_ids=[f'{c}/{h}h' for c in ordered_train_cases for h in [24,48,72]]
    cs=legacy.Scaler().fit(np.vstack([data[c][24][0] for c in ordered_train_cases]),[f'{c}/24h' for c in ordered_train_cases])
    ts=legacy.Scaler().fit(np.vstack([data[c][h][1] for c in ordered_train_cases for h in [24,48,72]]),fit_ids)
    sv=increment_scale([data[c] for c in ordered_train_cases])
    scale=float(ts.span[0]) if arm=='R' else sv
    coords=np.vstack([cs.transform(data[c][24][0]) for c in ordered_train_cases])
    seq=np.vstack([np.stack([ts.transform(data[c][h][1]) for h in [24,48]],axis=1) for c in ordered_train_cases])
    target=legacy.finite(np.vstack([(data[c][72][1]-data[c][48][1])/scale for c in ordered_train_cases]).astype(np.float32))
    return dict(coords=coords,seq=seq,target=target,cs=cs,ts=ts,scale=scale,s_v=sv,fit_ids=fit_ids)

def decode_increment(pred, last, scale):
    pred=legacy.finite(np.asarray(pred,dtype=np.float32).reshape(-1))
    last=legacy.finite(np.asarray(last,dtype=np.float32).reshape(-1))
    if pred.shape!=last.shape or not np.isfinite(scale) or scale<=0:
        raise ValueError('Invalid decoding shape or scale')
    return legacy.finite(last+np.float32(scale)*pred)

def mse_only(pred, target):
    return torch.mean((pred-target)**2)

def fit_model(coords, seq, target, seed, epochs=20, batch_size=1024, epoch_callback=None):
    legacy.seed_all(seed)
    generator=torch.Generator().manual_seed(seed)
    ds=TensorDataset(torch.from_numpy(coords),torch.from_numpy(seq),torch.from_numpy(target))
    loader=DataLoader(ds,batch_size=batch_size,shuffle=True,num_workers=0,drop_last=False,generator=generator,worker_init_fn=legacy.seed_worker)
    model=legacy.Model()
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    digest=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    curve=[];max_grad=0.
    for ep in range(1,epochs+1):
        start=time.monotonic();model.train();total=0.;n=0
        for xb,sb,yb in loader:
            optimizer.zero_grad(set_to_none=True)
            pred=model(xb,sb);loss=mse_only(pred,yb)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            max_grad=max(max_grad,float(norm));optimizer.step()
            total+=float(loss.detach())*len(xb);n+=len(xb)
        rec=dict(epoch=ep,n=n,train_mse_normalized=total/n,wall_s=time.monotonic()-start)
        curve.append(rec)
        if epoch_callback is not None:epoch_callback(curve)
    return dict(model=model,epochs=curve,initial_digest=digest,max_grad=max_grad,n_train=len(ds))
