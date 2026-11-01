"""V55 CPU-only numerical core; original CFD records remain read-only."""
import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn


def finite(arr):
    if not np.isfinite(arr).all():
        raise ValueError('Nonfinite numerical data')
    return arr


class Scaler:
    def fit(self, data, fit_ids):
        a = finite(np.asarray(data, dtype=np.float32))
        self.minimum = a.min(axis=0)
        self.span = a.max(axis=0) - self.minimum
        self.span[self.span == 0] = 1.
        self.fit_ids = list(fit_ids)
        return self

    def transform(self, data):
        return finite((np.asarray(data, dtype=np.float32) - self.minimum) / self.span)

    def inverse(self, data):
        return finite(np.asarray(data, dtype=np.float32) * self.span + self.minimum)

    def record(self):
        return dict(minimum=self.minimum.tolist(), span=self.span.tolist(), fit_ids=self.fit_ids)


def training_ids(cases, held):
    return [f'{case}/{time}h' for case in cases if case != held for time in (24, 48, 72)]


def metrics(truth, pred):
    y, p = finite(np.asarray(truth, dtype=np.float64).reshape(-1)), finite(np.asarray(pred, dtype=np.float64).reshape(-1))
    if y.shape != p.shape or len(y) < 1:
        raise ValueError('Invalid metric arrays')
    d = p - y
    den = np.sum((y - y.mean()) ** 2)
    return dict(n=len(y), MAE_K=float(np.mean(np.abs(d))), RMSE_K=float(np.sqrt(np.mean(d*d))),
                MaxAE_K=float(np.max(np.abs(d))), R2=float(1 - np.sum(d*d) / den) if den > 0 else None,
                Bias_K=float(np.mean(d)), P95AE_K=float(np.quantile(np.abs(d), .95)))


def coordinate_metrics(coords, truth, pred):
    frame = pd.DataFrame(np.asarray(coords, dtype=np.float64), columns=['X','Y','Z'])
    counts = frame.groupby(['X','Y','Z'], sort=False)['X'].transform('size').to_numpy()
    w = 1. / counts
    groups = float(w.sum())
    w = w / groups
    y = finite(np.asarray(truth,dtype=np.float64).reshape(-1))
    p = finite(np.asarray(pred,dtype=np.float64).reshape(-1))
    d = p-y
    mean = np.sum(w*y)
    den = np.sum(w*(y-mean)**2)
    order = np.argsort(np.abs(d),kind='stable')
    q95 = np.abs(d)[order][min(np.searchsorted(np.cumsum(w[order]),.95),len(d)-1)]
    return dict(n=len(y),n_coordinate_groups=int(round(groups)),MAE_K=float(np.sum(w*np.abs(d))),
                RMSE_K=float(np.sqrt(np.sum(w*d*d))),MaxAE_K=float(np.max(np.abs(d))),
                R2=float(1-np.sum(w*d*d)/den) if den>0 else None,Bias_K=float(np.sum(w*d)),P95AE_K=float(q95))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # No CUDA device used; these declarations also document the non-used path.
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_encoder = nn.Sequential(nn.Linear(3,64),nn.Tanh(),nn.Linear(64,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh())
        self.lstm = nn.LSTM(129,128,num_layers=2,batch_first=True,dropout=.1)
        self.decoder = nn.Sequential(nn.Linear(256,128),nn.Tanh(),nn.Linear(128,64),nn.Tanh(),nn.Linear(64,32),nn.Tanh(),nn.Linear(32,1))

    def forward(self, coords, seq):
        spatial = self.spatial_encoder(coords)
        spatial_exp = spatial.unsqueeze(1).expand(-1,seq.shape[1],-1)
        temporal,_ = self.lstm(torch.cat([spatial_exp,seq],dim=-1))
        return self.decoder(torch.cat([spatial,temporal[:,-1,:]],dim=-1))


def loss_value(pred, target, last, penalty):
    return torch.mean((pred-target)**2) + penalty * torch.mean(torch.abs(pred-last))


def load_csv(path):
    df = pd.read_csv(path, usecols=lambda c: c.strip() in ['X','Y','Z','temperature','Static Temperature'])
    df.columns = df.columns.str.strip()
    tc = 'Static Temperature' if 'Static Temperature' in df.columns else 'temperature'
    # Float64 export coordinates retained for exact audit/grouping; only model inputs cast to float32.
    return finite(df[['X','Y','Z']].to_numpy(dtype=np.float64)), finite(df[[tc]].to_numpy(dtype=np.float64))
