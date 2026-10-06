"""(b) Mechanism: two-component mixture with FIXED mean and covariance, varying separation.

x1 ~ 1/2 N(-a, s^2) + 1/2 N(+a, s^2) with a^2 + s^2 = 1 (Var x1 = 1 for every a); x2 ~ N(0, 0.5);
momenta N(0, 0.25^2).  Since mean and covariance do not depend on a, the Gaussian-moment theory gives
the same prediction for every a.  The exact-score clipped-vs-corrected deviation is computed for each a.
Hypothesis: the clipped sampler acts like extra attraction to high density; it shrinks spread within
modes but not the distance between modes, so the variance shrinkage falls as the within-mode share
of the variance s^2 decreases.

usage: python gmm_mechanism.py <a> <noise_seed> -> results/mech_a<a>_n<seed>.json
"""
import numpy as np, json, sys, os
from psdt import *
from gmm_exact import ExactScore, gkl, gmm_sample

def mixture(a):
    s2 = 1 - a**2
    w = np.array([0.5, 0.5]); mu = np.zeros((2, 4)); mu[0, 0], mu[1, 0] = -a, a
    S = np.zeros((2, 4, 4))
    for k in range(2):
        S[k] = np.diag([s2, 0.25**2, 0.5, 0.25**2])
    return w, mu, S

if __name__ == '__main__':
    a, ns = float(sys.argv[1]), int(sys.argv[2])
    rr, N, K = 0.6, 8000, 4000
    w, mu, S = mixture(a)
    B = hamiltonian_B(2.0); p = Process(B, rr * kappa_of(B)); sc = ExactScore(p, w, mu, S)
    data = gmm_sample(w, mu, S, 20000, np.random.default_rng(1000 + ns))
    smp = {}
    for mode in ['fixed', 'naive']:
        z = prior_sample(p, data, N, np.random.default_rng(10 + ns))
        smp[mode] = sample(sc, p, z, mode, np.random.default_rng(20 + ns), K=K)
    st = lambda x: (x.mean(0), np.cov(x.T))
    Cn, Cf = np.cov(smp['naive'].T), np.cov(smp['fixed'].T)
    out = {'a': a, 'within_share_x1': 1 - a**2, 'noise_seed': ns,
           'paired_KL': gkl(*st(smp['naive']), *st(smp['fixed'])),
           'log_vr': float(np.log(np.trace(Cn) / np.trace(Cf))),
           'log_var_ratio_x1': float(np.log(Cn[0, 0] / Cf[0, 0])),
           'log_var_ratio_x2': float(np.log(Cn[2, 2] / Cf[2, 2]))}
    os.makedirs('results', exist_ok=True); json.dump(out, open(f'results/mech_a{a}_n{ns}.json', 'w'), indent=1)
    print({k: round(v, 5) for k, v in out.items()}, flush=True)
