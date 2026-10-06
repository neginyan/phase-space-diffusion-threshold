"""One training+sampling job.

usage: python run_sweep.py <dyn> <seed> <ratio>
  dyn = ham  : Hamiltonian omega = 2 (kappa = 1.5), r = ratio * kappa  (T varies with r)
  dyn = ctrl : control with fixed r = 0.9 and fixed T; omega chosen so that kappa = r / ratio,
               i.e. omega = sqrt(1 + 2 kappa)  (kappa moves across r, everything else fixed)
  dyn = cld  : critically damped CLD (beta = 1, M = 1/4, Gamma = 1), r = ratio * kappa
writes results/<dyn>_s<seed>_r<ratio>.json and the generated samples (.npy).
For ratio >= 1 the clipped and corrected SDE are the same algorithm (G_+ = G); the naive run is
then copied from the corrected run instead of being recomputed.
"""
import numpy as np, json, sys, time, os
from psdt import *

dyn, seed, rr = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
steps, N, K_ode, K_sde = 12000, 4000, 1000, 4000
R_CTRL = 0.9

if dyn == 'ham':
    omega = 2.0; B = hamiltonian_B(omega); kap = kappa_of(B); r = rr * kap
elif dyn == 'ctrl':
    kap = R_CTRL / rr; omega = float(np.sqrt(1 + 2 * kap)); B = hamiltonian_B(omega); r = R_CTRL
    assert abs(kappa_of(B) - kap) < 1e-9
elif dyn == 'cld':
    omega = None; B = cld_B(); kap = kappa_of(B); r = rr * kap
else:
    raise ValueError(dyn)

rng = np.random.default_rng(1000 + seed)
data = two_moons(20000, rng)
test = two_moons(N, np.random.default_rng(99))          # fixed reference set
os.makedirs('results', exist_ok=True)
t0 = time.time()
p = Process(B, r)
m = ScoreModel(p, np.random.default_rng(seed))
losses = []
train(m, data, rng, steps=steps, log=lambda k, l: losses.append((k, float(l))))
rec = {'dyn': dyn, 'seed': seed, 'ratio': rr, 'kappa': kap, 'omega': omega, 'r': r, 'T': p.T,
       'loss': losses, 'Gmin_eig_over_2sigma2': float((r - p.w).min())}
smps = {}
for mode in ['ode', 'fixed', 'naive']:
    if mode == 'naive' and rr >= 1.0:
        smp = smps['fixed']                               # identical algorithm when G_+ = G
    else:
        z = prior_sample(p, data, N, np.random.default_rng(10 + seed))
        smp = sample(m.score, p, z, mode, np.random.default_rng(20 + seed),
                     K=K_ode if mode == 'ode' else K_sde)
    smps[mode] = smp
    np.save(f'results/{dyn}_s{seed}_r{rr}_{mode}.npy', smp.astype(np.float32))
    rec[mode] = {'sw4': sliced_w2(smp, test, np.random.default_rng(5)),
                 'sw2': sliced_w2(smp[:, 0::2], test[:, 0::2], np.random.default_rng(5)),
                 'mmd4': mmd2(smp, test), 'mmd2': mmd2(smp[:, 0::2], test[:, 0::2])}
rec['time_s'] = time.time() - t0
json.dump(rec, open(f'results/{dyn}_s{seed}_r{rr}.json', 'w'), indent=1)
print(dyn, seed, rr, 'omega', omega, {k: round(rec[k]['mmd4'], 5) for k in ['ode', 'naive', 'fixed']},
      round(rec['time_s']), flush=True)
