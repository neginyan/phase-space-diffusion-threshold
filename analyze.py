"""Day-2 analysis (5 seeds; main sweep 'ham' and fixed-r/fixed-T control 'ctrl').

Paired statistics, clipped Anderson SDE ('naive') vs corrected SDE ('fixed'): same network,
same prior draw, same Brownian increments, same step count.
  paired_KL  = KL( N(naive fit) || N(fixed fit) )            (>= 0)
  log_vr     = log( tr Cov[naive] / tr Cov[fixed] )         (signed; theory predicts < 0 below kappa)
Uncertainty: bootstrap over sample indices (paired resampling, 300 draws) and spread over seeds.
Theory: Gaussian-moment prediction of the clipped sampler with the two-moons mean/covariance.
Writes results/summary_day2.json and figures/threshold_day2.png
"""
import numpy as np, json, glob, os
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from psdt import *

ref = two_moons(400000, np.random.default_rng(1)); m0 = ref.mean(0); S0 = np.cov(ref.T); I4 = np.eye(4)

def gkl(m1, C1, m2, C2):
    iC2 = np.linalg.inv(C2); d = m2 - m1
    return 0.5 * (np.trace(iC2 @ C1) + d @ iC2 @ d - len(m1) + np.log(np.linalg.det(C2) / np.linalg.det(C1)))

def system(dyn, rr):
    if dyn == 'ham':
        B = hamiltonian_B(2.0); return B, rr * kappa_of(B)
    if dyn == 'ctrl':
        k = 0.9 / rr; return hamiltonian_B(np.sqrt(1 + 2 * k)), 0.9
    B = cld_B(); return B, rr * kappa_of(B)

def theory(dyn, rr):
    B, r = system(dyn, rr); p = Process(B, r)
    def mom(t):
        M = expm(B * t); return M @ m0, M @ S0 @ M.T + p.sigma(t)**2 * I4
    def rhs(tau, y):
        t = p.T - tau; mu = y[:4]; Sg = y[4:].reshape(4, 4); m, C = mom(t); P = np.linalg.inv(C)
        Gp = p.Gplus(t); A = -B - Gp @ P
        return np.concatenate([A @ mu + Gp @ P @ m, (A @ Sg + Sg @ A.T + Gp).ravel()])
    m, C = mom(p.T)
    y = solve_ivp(rhs, [0, p.T], np.concatenate([m, C.ravel()]), rtol=1e-9, atol=1e-12, method='DOP853').y[:, -1]
    Sg = y[4:].reshape(4, 4); Sg = (Sg + Sg.T) / 2; mt, Ct = mom(0)
    return gkl(y[:4], Sg, mt, Ct), float(np.log(np.trace(Sg) / np.trace(Ct)))

def paired(zn, zf, rng, nb=300):
    def stats(a, b):
        return gkl(a.mean(0), np.cov(a.T), b.mean(0), np.cov(b.T)), np.log(np.trace(np.cov(a.T)) / np.trace(np.cov(b.T)))
    kl0, lv0 = stats(zn, zf)
    bs = np.array([stats(zn[i], zf[i]) for i in (rng.integers(0, len(zn), len(zn)) for _ in range(nb))])
    return kl0, lv0, np.percentile(bs[:, 0], [2.5, 97.5]), np.percentile(bs[:, 1], [2.5, 97.5])

def load(dyn):
    rows = []
    files = glob.glob(f'results/{dyn}_s*_r*.json')
    if dyn == 'ctrl':                       # ctrl at ratio 0.6 is literally the ham run at ratio 0.6
        files += glob.glob('results/ham_s*_r0.6.json')
    rng = np.random.default_rng(0)
    for f in sorted(files):
        J = json.load(open(f)); src = J['dyn']; s, rr = J['seed'], J['ratio']
        z = {md: np.load(f'results/{src}_s{s}_r{rr}_{md}.npy').astype(float) for md in ['ode', 'naive', 'fixed']}
        kl0, lv0, klci, lvci = paired(z['naive'], z['fixed'], rng)
        rows.append({'seed': s, 'ratio': rr, 'T': J['T'], 'omega': J.get('omega'),
                     'paired_KL': kl0, 'paired_KL_ci': klci.tolist(),
                     'log_vr': lv0, 'log_vr_ci': lvci.tolist(),
                     'mmd4': {md: J[md]['mmd4'] for md in z}, 'sw4': {md: J[md]['sw4'] for md in z}})
    return rows

summary = {}
for dyn in ['ham', 'ctrl']:
    rows = load(dyn)
    if not rows: continue
    ratios = sorted(set(r['ratio'] for r in rows))
    th = {float(g): theory(dyn, g) for g in np.round(np.linspace(0.5, 1.4, 19), 3)}
    summary[dyn] = {'rows': rows, 'theory': {str(k): {'KL': v[0], 'log_var_ratio': v[1]} for k, v in th.items()}}
    print(f'\n=== {dyn} ===')
    print('r/k   n  T     paired_KL mean±sd   theory   log_vr mean±sd     theory   seeds<0  CI excl.0   MMD ode/naive/fixed (mean)')
    for rr in ratios:
        sub = [r for r in rows if r['ratio'] == rr]
        kl = np.array([r['paired_KL'] for r in sub]); lv = np.array([r['log_vr'] for r in sub])
        neg = sum(v < 0 for v in lv); excl = sum(r['log_vr_ci'][1] < 0 for r in sub)
        t = theory(dyn, rr)
        mm = [np.mean([r['mmd4'][md] for r in sub]) for md in ['ode', 'naive', 'fixed']]
        print(f"{rr:4.2f}  {len(sub)}  {sub[0]['T']:5.2f}  {kl.mean():.4f}±{kl.std():.4f}   {t[0]:.4f}   "
              f"{lv.mean():+.4f}±{lv.std():.4f}   {t[1]:+.4f}   {neg}/{len(sub)}     {excl}/{len(sub)}      "
              f"{mm[0]:.5f} {mm[1]:.5f} {mm[2]:.5f}")
json.dump(summary, open('results/summary_day2.json', 'w'), indent=1)

# ---------------------------------------------------------------- figure
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
os.makedirs('figures', exist_ok=True)
dyns = [d for d in ['ham', 'ctrl'] if d in summary]
fig, ax = plt.subplots(2, len(dyns), figsize=(5.2 * len(dyns), 7.2), squeeze=False)
title = {'ham': r'Main sweep ($\omega=2$, $\kappa=1.5$, $r$ varies, $T$ varies)',
         'ctrl': r'Control ($r=0.9$, $T=9.22$ fixed; $\omega$ varies)'}
cols = {'ode': '#2c3e50', 'naive': '#c0392b', 'fixed': '#2980b9'}
labs = {'ode': 'probability-flow ODE', 'naive': 'clipped Anderson SDE', 'fixed': r'corrected SDE ($D=G_+$)'}
for j, dyn in enumerate(dyns):
    rows = summary[dyn]['rows']; ratios = sorted(set(r['ratio'] for r in rows))
    th = summary[dyn]['theory']; g = sorted(float(k) for k in th)
    a = ax[0, j]
    a.plot(g, [th[str(x)]['KL'] for x in g], '-', color='#888', lw=1.5, label='Gaussian-moment theory')
    for r in rows:
        a.plot(r['ratio'] + np.random.default_rng(r['seed']).uniform(-0.012, 0.012), r['paired_KL'],
               'o', ms=4, color='#c0392b', alpha=0.6)
    a.plot([], [], 'o', color='#c0392b', alpha=0.6, label='trained network (5 seeds)')
    a.axvline(1.0, color='k', ls='--', lw=1); a.set_xlabel(r'$r/\kappa$')
    a.set_ylabel('KL( clipped SDE || corrected SDE )'); a.set_title(title[dyn], fontsize=10); a.legend(fontsize=8)
    a = ax[1, j]
    for md in ['ode', 'naive', 'fixed']:
        mu = [np.mean([r['mmd4'][md] for r in rows if r['ratio'] == rr]) for rr in ratios]
        sd = [np.std([r['mmd4'][md] for r in rows if r['ratio'] == rr]) for rr in ratios]
        a.errorbar(ratios, mu, yerr=sd, fmt='o-', color=cols[md], capsize=3, label=labs[md], ms=4)
    a.axvline(1.0, color='k', ls='--', lw=1); a.set_xlabel(r'$r/\kappa$'); a.set_ylabel(r'MMD$^2$ to data (4D), mean $\pm$ sd')
    a.legend(fontsize=8)
fig.tight_layout(); fig.savefig('figures/threshold_day2.png', dpi=150)
print('saved figures/threshold_day2.png')
