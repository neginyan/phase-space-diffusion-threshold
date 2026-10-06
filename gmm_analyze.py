"""(b) Summary: Gaussian-moment theory vs exact-score GMM reference vs trained network.
Writes results/summary_gmm.json and figures/gmm.png"""
import numpy as np, json, glob, os
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from psdt import *
from gmm_exact import fit_gmm, gmm_sample, system, gkl
from gmm_mechanism import mixture

I4 = np.eye(4)
def theory(B, r, m0, S0):
    """Gaussian-moment prediction for the clipped sampler: (KL to truth, log total-variance ratio)."""
    p = Process(B, r)
    def mom(t):
        M = expm(B * t); return M @ m0, M @ S0 @ M.T + p.sigma(t)**2 * I4
    def rhs(tau, y):
        t = p.T - tau; mu = y[:4]; Sg = y[4:].reshape(4, 4); m, C = mom(t); P = np.linalg.inv(C)
        Gp = p.Gplus(t); A = -B - Gp @ P
        return np.concatenate([A @ mu + Gp @ P @ m, (A @ Sg + Sg @ A.T + Gp).ravel()])
    m, C = mom(p.T)
    y = solve_ivp(rhs, [0, p.T], np.concatenate([m, C.ravel()]), rtol=1e-9, atol=1e-12, method='DOP853').y[:, -1]
    Sg = y[4:].reshape(4, 4); Sg = (Sg + Sg.T) / 2; mt, Ct = mom(0)
    return gkl(y[:4], Sg, mt, Ct), float(np.log(np.trace(Sg) / np.trace(Ct))), Sg, Ct

w, mu, S = fit_gmm()
g = gmm_sample(w, mu, S, 400000, np.random.default_rng(777)); m0g, S0g = g.mean(0), np.cov(g.T)
net = json.load(open('results/summary_day2.json'))
rows = []
for dyn in ['ham', 'ctrl']:
    for rr in [0.6, 0.8, 0.9]:
        src = 'ham' if (dyn == 'ctrl' and rr == 0.6) else dyn
        ex = [json.load(open(f)) for f in glob.glob(f'results/gmm_{src}_r{rr}_n*.json') if '_K' not in f]
        nt = [r for r in net[dyn]['rows'] if r['ratio'] == rr]
        B, r = system(dyn, rr); th = theory(B, r, m0g, S0g)
        rows.append({'dyn': dyn, 'ratio': rr,
                     'theory_KL': th[0], 'theory_log_vr': th[1],
                     'exact_KL': [e['paired_KL'] for e in ex], 'exact_log_vr': [e['log_vr'] for e in ex],
                     'net_KL': [x['paired_KL'] for x in nt], 'net_log_vr': [x['log_vr'] for x in nt]})
mech = [json.load(open(f)) for f in sorted(glob.glob('results/mech_a*_n*.json'))]
wm, mm, Sm = mixture(0.0); B = hamiltonian_B(2.0)
_, mech_th_lvr, Sg_m, Ct_m = theory(B, 0.6 * kappa_of(B), (wm[:, None] * mm).sum(0), Sm[0])
mech_th_x1, mech_th_x2 = float(np.log(Sg_m[0, 0] / Ct_m[0, 0])), float(np.log(Sg_m[2, 2] / Ct_m[2, 2]))
summ = {'rows': rows,
        'mechanism': mech, 'mechanism_gauss_theory': {'log_vr': mech_th_lvr, 'x1': mech_th_x1, 'x2': mech_th_x2}}
json.dump(summ, open('results/summary_gmm.json', 'w'), indent=1)

f = lambda v: f'{np.mean(v):.4f}±{np.std(v):.4f}'
print('dyn   r/k  | paired KL: theory   exact-GMM        network         | log var ratio: theory  exact-GMM        network')
for r in rows:
    print(f"{r['dyn']:5s} {r['ratio']:.1f}  |  {r['theory_KL']:.4f}   {f(r['exact_KL'])}  {f(r['net_KL'])} |  "
          f"{r['theory_log_vr']:+.4f}  {np.mean(r['exact_log_vr']):+.4f}±{np.std(r['exact_log_vr']):.4f}  "
          f"{np.mean(r['net_log_vr']):+.4f}±{np.std(r['net_log_vr']):.4f}")
print(f'mechanism Gaussian theory (a=0): x1 {mech_th_x1:+.4f}, x2 {mech_th_x2:+.4f}, total {mech_th_lvr:+.4f}')
for a in sorted(set(m['a'] for m in mech)):
    sub = [m for m in mech if m['a'] == a]
    print(f"a={a:.2f} within-share={1-a*a:.3f}: x1 {np.mean([m['log_var_ratio_x1'] for m in sub]):+.4f}  "
          f"x2 {np.mean([m['log_var_ratio_x2'] for m in sub]):+.4f}  KL {np.mean([m['paired_KL'] for m in sub]):.4f}")

# ---------------------------------------------------------------- figure
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
cols = {'theory': '#888888', 'exact': '#27ae60', 'net': '#c0392b'}
for j, (key, lab) in enumerate([('KL', 'paired KL (clipped || corrected)'), ('log_vr', 'log total-variance ratio (clipped / corrected)')]):
    a = ax[j]
    for dyn, mk, off in [('ham', 'o', -0.012), ('ctrl', 's', 0.012)]:
        sub = [r for r in rows if r['dyn'] == dyn]; x = np.array([r['ratio'] for r in sub]) + off
        a.plot(x, [r[f'theory_{key}'] for r in sub], mk, color=cols['theory'], mfc='none', ms=8,
               label='Gaussian-moment theory' if dyn == 'ham' else None)
        a.errorbar(x, [np.mean(r[f'exact_{key}']) for r in sub], yerr=[np.std(r[f'exact_{key}']) for r in sub],
                   fmt=mk, color=cols['exact'], capsize=3, label='exact score, GMM data (3 noise seeds)' if dyn == 'ham' else None)
        a.errorbar(x, [np.mean(r[f'net_{key}']) for r in sub], yerr=[np.std(r[f'net_{key}']) for r in sub],
                   fmt=mk, color=cols['net'], capsize=3, mfc='none', label='trained network (5 seeds)' if dyn == 'ham' else None)
    a.axhline(0, color='k', lw=0.6); a.set_xlabel(r'$r/\kappa$   (o: main sweep, s: control)'); a.set_ylabel(lab, fontsize=9)
    a.legend(fontsize=7)
ax[0].set_title('Deviation of the clipped sampler', fontsize=10)
ax[1].set_title('Variance shrinkage', fontsize=10)
a = ax[2]
sh = sorted(set(1 - m['a']**2 for m in mech))
for key, c, lab in [('log_var_ratio_x1', '#8e44ad', r'$x_1$ (bimodal coordinate)'), ('log_var_ratio_x2', '#16a085', r'$x_2$ (Gaussian coordinate)')]:
    a.plot(sh, [np.mean([m[key] for m in mech if abs(1 - m['a']**2 - s) < 1e-9]) for s in sh], 'o-', color=c, label=lab)
a.axhline(mech_th_x1, color='#888', ls='--', lw=1, label='Gaussian-moment theory (same for all separations)')
a.set_xscale('log'); a.set_xlabel(r'within-mode share of Var$\,x_1$  ($1-a^2$)'); a.set_ylabel('log variance ratio (clipped / corrected)', fontsize=9)
a.set_title(r'Mechanism: fixed covariance, varying separation ($r/\kappa=0.6$)', fontsize=10); a.legend(fontsize=7, loc='lower left', bbox_to_anchor=(0.0, 0.1))
fig.tight_layout(); os.makedirs('figures', exist_ok=True); fig.savefig('figures/gmm.png', dpi=150)
print('saved figures/gmm.png')
