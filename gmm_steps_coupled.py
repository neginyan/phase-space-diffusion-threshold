"""Step-size study with coupled Brownian paths (exact GMM score; noise seeds 0, 1, 2).

All step counts K in {4000, 8000, 16000} use the SAME Brownian path: fine increments are generated on the
16000-step grid from counter-based seeds, and a coarse increment is the sum of the fine increments it covers.
The prior draw is also shared. This removes most, but not all, of the Monte Carlo noise from differences
between K (the dynamics amplify small path differences).  usage: python gmm_steps_coupled.py <dyn: ham|ctrl> <ratio> <noise_seed>
-> results/gmm_steps_<dyn>_r<ratio>_n<noise_seed>.json
"""
import numpy as np, json, sys, os
from psdt import *
from gmm_exact import fit_gmm, gmm_sample, ExactScore, system, gkl

KF, N = 16000, 4000
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 0

def sample_coupled(score_fn, proc, z, mode, K):
    ts = np.linspace(proc.T, 0, K + 1); m = KF // K; hf = proc.T / KF; B = proc.B
    for k in range(K):
        t, t2 = ts[k], ts[k + 1]; h = t - t2
        s = score_fn(z, t); G = proc.G(t); Gp = proc.Gplus(t)
        drift = z @ B.T - (s @ Gp.T if mode == 'naive' else 0.5 * s @ (G + Gp).T)
        dW = sum(np.random.default_rng([20 + NS, j]).standard_normal(z.shape) for j in range(m * k, m * k + m)) * np.sqrt(hf)
        z = z - h * drift + dW @ proc.sqrtGplus(t).T
    return z

if __name__ == '__main__':
    dyn, rr = sys.argv[1], float(sys.argv[2])
    w, mu, S = fit_gmm()
    B, r = system(dyn, rr); p = Process(B, r); sc = ExactScore(p, w, mu, S)
    data = gmm_sample(w, mu, S, 20000, np.random.default_rng(1000 + NS))
    z0 = prior_sample(p, data, N, np.random.default_rng(10 + NS))
    truth = gmm_sample(w, mu, S, 400000, np.random.default_rng(777)); mt, Ct = truth.mean(0), np.cov(truth.T)
    st = lambda a: (a.mean(0), np.cov(a.T))
    out = {'dyn': dyn, 'ratio': rr, 'N': N, 'noise_seed': NS, 'coupled': True}
    for K in [4000, 8000, 16000]:
        smp = {md: sample_coupled(sc, p, z0.copy(), md, K) for md in ['fixed', 'naive']}
        out[f'K{K}'] = {
            'paired_KL': gkl(*st(smp['naive']), *st(smp['fixed'])),
            'log_vr': float(np.log(np.trace(np.cov(smp['naive'].T)) / np.trace(np.cov(smp['fixed'].T)))),
            'log_vr_to_truth_fixed': float(np.log(np.trace(np.cov(smp['fixed'].T)) / np.trace(Ct))),
            'KL_to_truth_fixed': gkl(*st(smp['fixed']), mt, Ct)}
        print(dyn, rr, NS, K, {k: round(v, 5) for k, v in out[f'K{K}'].items()}, flush=True)
        json.dump(out, open(f'results/gmm_steps_{dyn}_r{rr}_n{NS}.json', 'w'), indent=1)
