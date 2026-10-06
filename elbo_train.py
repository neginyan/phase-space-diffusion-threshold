"""(a) Training with likelihood (ELBO) weighting.

Continuous-time ELBO term:  1/2 int E[(s_theta - s)^T G (s_theta - s)] dt.  With s = -eps/sigma this is
1/2 int E[(eps_theta - eps)^T W (eps_theta - eps)] dt with the time-independent matrix
W = G / sigma^2 = 2 (r I - Sym B).   For a scalar drift B = b I this reduces to the VDM weight -(log SNR)'.
W has the negative eigenvalue 2(r - kappa) when r < kappa, so the denoising objective is unbounded below.

usage: python elbo_train.py <seed> <ratio>  -> results/elbo_train_s<seed>_r<ratio>.json
Logged every 100 steps: weighted loss, unweighted eps-MSE (true accuracy), and the mean squared error
along the negative eigenvector u of W (u = top eigenvector of Sym B).
"""
import numpy as np, json, sys, os
from psdt import *

seed, rr = int(sys.argv[1]), float(sys.argv[2])
steps, bs, lr = 4000, 512, 2e-3
B = hamiltonian_B(2.0); kap = kappa_of(B); p = Process(B, rr * kap)
W = 2 * (p.r * np.eye(4) - p.SymB)
u = p.V[:, -1]                                            # Sym B eigenvalue +kappa -> W eigenvalue 2(r-kappa)
rng = np.random.default_rng(1000 + seed)
data = two_moons(20000, rng)
m = ScoreModel(p, np.random.default_rng(seed))
log = []
for k in range(steps):
    lr_k = lr * 0.5 * (1 + np.cos(np.pi * k / steps))
    z0 = data[rng.integers(0, len(data), bs)]
    t = rng.uniform(0, p.T, bs)
    Mt = batch_M(p, t); eps = rng.standard_normal((bs, 4))
    zt = np.einsum('nij,nj->ni', Mt, z0) + p.sigma(t)[:, None] * eps
    out = m.net.forward(m.features(zt, t)); diff = out - eps
    wl = (diff * (diff @ W)).sum(1).mean()
    m.net.adam(m.net.backward(2 * diff @ W / bs), lr_k)
    if k % 100 == 0 or k == steps - 1:
        log.append({'step': k, 'weighted_loss': float(wl), 'mse': float((diff**2).sum(1).mean()),
                    'mse_u': float(((diff @ u)**2).mean()), 'finite': bool(np.isfinite(wl))})
    if not np.isfinite(wl): break
os.makedirs('results', exist_ok=True)
json.dump({'seed': seed, 'ratio': rr, 'kappa': kap, 'W_eigs': np.linalg.eigvalsh(W).tolist(), 'log': log},
          open(f'results/elbo_train_s{seed}_r{rr}.json', 'w'), indent=1)
print(seed, rr, 'final weighted', round(log[-1]['weighted_loss'], 3), 'mse', round(log[-1]['mse'], 3),
      'mse_u', round(log[-1]['mse_u'], 3), flush=True)
