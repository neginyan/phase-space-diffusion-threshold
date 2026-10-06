"""(b) Exact-score reference for the clipped-SDE deviation, with Gaussian-mixture data.

A Gaussian mixture fitted to the two-moons phase-space data stands in for the data.  Its forward
marginals are mixtures with components N(M mu_k, M S_k M^T + sigma^2 I), so the score is exact.
The same three samplers as for the network are run with this exact score (same Gaussian prior, same
step counts, paired noise), giving the deviation of the clipped sampler free of network error.

usage: python gmm_exact.py <dyn: ham|ctrl> <ratio> <noise_seed> [K_sde]
  -> results/gmm_<dyn>_r<ratio>_n<seed>.json
"""
import numpy as np, json, sys, os
from sklearn.mixture import GaussianMixture
from psdt import *

N_COMP = 24
def fit_gmm(seed=0, n=200000, cache='results/gmm_fit.npz'):
    """GMM over the 4-D phase space: positions fitted to two moons, momenta N(0, 0.25^2) independent."""
    if os.path.exists(cache):
        f = np.load(cache); return f['w'], f['mu'], f['S']
    z = two_moons(n, np.random.default_rng(seed))
    gm = GaussianMixture(N_COMP, covariance_type='full', random_state=seed, max_iter=500, tol=1e-6).fit(z[:, 0::2])
    w = gm.weights_; mu = np.zeros((N_COMP, 4)); S = np.zeros((N_COMP, 4, 4))
    mu[:, 0::2] = gm.means_
    for k in range(N_COMP):
        S[k][np.ix_([0, 2], [0, 2])] = gm.covariances_[k]
        S[k][1, 1] = S[k][3, 3] = 0.25**2
    os.makedirs(os.path.dirname(cache), exist_ok=True); np.savez(cache, w=w, mu=mu, S=S)
    return w, mu, S

def gmm_sample(w, mu, S, n, rng):
    k = rng.choice(len(w), n, p=w)
    L = np.linalg.cholesky(S)
    return mu[k] + np.einsum('nij,nj->ni', L[k], rng.standard_normal((n, 4)))

class ExactScore:
    def __init__(self, proc, w, mu, S):
        self.p, self.w, self.mu, self.S = proc, w, mu, S
        self.cache = {}
    def params(self, t):
        key = round(float(t), 12)
        if key not in self.cache:
            M = self.p.M(t); m = self.mu @ M.T
            C = np.einsum('ij,kjl,ml->kim', M, self.S, M) + self.p.sigma(t)**2 * np.eye(4)
            P = np.linalg.inv(C); ld = np.linalg.slogdet(C)[1]
            self.cache[key] = (m, P, np.log(self.w) - 0.5 * ld)
        return self.cache[key]
    def __call__(self, z, t):
        m, P, c = self.params(t)
        d = z[:, None, :] - m[None]                               # (N, K, 4)
        Pd = np.einsum('kij,nkj->nki', P, d)
        logp = c[None] - 0.5 * (d * Pd).sum(-1)                   # (N, K)
        logp -= logp.max(1, keepdims=True); r = np.exp(logp); r /= r.sum(1, keepdims=True)
        return -(r[..., None] * Pd).sum(1)

def system(dyn, rr):
    if dyn == 'ham':
        B = hamiltonian_B(2.0); return B, rr * kappa_of(B)
    k = 0.9 / rr; return hamiltonian_B(np.sqrt(1 + 2 * k)), 0.9

def gkl(m1, C1, m2, C2):
    iC2 = np.linalg.inv(C2); d = m2 - m1
    return 0.5 * (np.trace(iC2 @ C1) + d @ iC2 @ d - len(m1) + np.log(np.linalg.det(C2) / np.linalg.det(C1)))

if __name__ == '__main__':
    dyn, rr, ns = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    K_sde = int(sys.argv[4]) if len(sys.argv) > 4 else 4000
    N = 8000
    w, mu, S = fit_gmm()
    B, r = system(dyn, rr); p = Process(B, r)
    sc = ExactScore(p, w, mu, S)
    data = gmm_sample(w, mu, S, 20000, np.random.default_rng(1000 + ns))   # only for the Gaussian prior moments
    out = {'dyn': dyn, 'ratio': rr, 'noise_seed': ns, 'K_sde': K_sde, 'N': N}
    smp = {}
    for mode in ['fixed', 'naive']:          # ODE not needed for the paired deviation
        if mode == 'naive' and rr >= 1.0:
            smp[mode] = smp['fixed']; continue
        z = prior_sample(p, data, N, np.random.default_rng(10 + ns))
        smp[mode] = sample(sc, p, z, mode, np.random.default_rng(20 + ns), K=1000 if mode == 'ode' else K_sde)
    truth = gmm_sample(w, mu, S, 400000, np.random.default_rng(777))
    mt, Ct = truth.mean(0), np.cov(truth.T)
    st = lambda a: (a.mean(0), np.cov(a.T))
    out['paired_KL'] = gkl(*st(smp['naive']), *st(smp['fixed']))
    out['log_vr'] = float(np.log(np.trace(np.cov(smp['naive'].T)) / np.trace(np.cov(smp['fixed'].T))))
    for md in smp:
        out[f'KL_to_truth_{md}'] = gkl(*st(smp[md]), mt, Ct)
        out[f'log_vr_to_truth_{md}'] = float(np.log(np.trace(np.cov(smp[md].T)) / np.trace(Ct)))
    os.makedirs('results', exist_ok=True)
    json.dump(out, open(f'results/gmm_{dyn}_r{rr}_n{ns}' + ('' if K_sde == 4000 else f'_K{K_sde}') + '.json', 'w'), indent=1)
    np.save(f'results/gmm_{dyn}_r{rr}_n{ns}' + ('' if K_sde == 4000 else f'_K{K_sde}') + '_naive.npy', smp['naive'][:4000].astype(np.float32))
    print(dyn, rr, ns, K_sde, {k: round(v, 5) for k, v in out.items() if isinstance(v, float)}, flush=True)
