"""Step-size study with coupled Brownian paths (exact GMM score, noise seeds 0, 1, 2, 4000 particles).

Euler-Maruyama is first order, so X(K) ~ X_inf + c/K and the Richardson extrapolation is
X_inf ~ 2 X(16000) - X(8000), computed separately for each noise seed. Reported: mean and standard deviation over
the three seeds of X(4000), X_inf and the relative change X_inf/X(4000) - 1, and whether X(K) is monotone in K.
Writes results/summary_steps.json and paper_snippets/table_steps.tex."""
import json, numpy as np, os

SEEDS = [0, 1, 2]
CONF = [('ham', 0.6), ('ham', 0.8), ('ham', 0.9), ('ctrl', 0.8), ('ctrl', 0.9)]
KEYS = ['paired_KL', 'log_vr', 'log_vr_to_truth_fixed']
rows = []
for dyn, rr in CONF:
    rec = {'dyn': dyn, 'ratio': rr, 'per_seed': {}}
    for ns in SEEDS:
        J = json.load(open(f'results/gmm_steps_{dyn}_r{rr}_n{ns}.json'))
        d = {}
        for k in KEYS:
            a, b, c = (J[f'K{K}'][k] for K in [4000, 8000, 16000])
            d[k] = {'K4000': a, 'K8000': b, 'K16000': c, 'inf': 2 * c - b, 'rel': (2 * c - b) / a - 1,
                    'monotone': bool((a - b) * (b - c) > 0)}
        rec['per_seed'][ns] = d
    for k in KEYS:
        v = lambda f: np.array([rec['per_seed'][ns][k][f] for ns in SEEDS], float)
        rec[k] = {f: {'mean': float(v(f).mean()), 'sd': float(v(f).std())} for f in ['K4000', 'K8000', 'K16000', 'inf', 'rel']}
        rec[k]['n_monotone'] = int(sum(rec['per_seed'][ns][k]['monotone'] for ns in SEEDS))
    rows.append(rec)
json.dump(rows, open('results/summary_steps.json', 'w'), indent=1)

print('dyn  r/k | paired KL: K4000  K->inf  rel.change  [monotone]       | log-var ratio: K4000  K->inf  rel.change [monotone] | corrected error K4000 K8000 K16000')
for r in rows:
    f = lambda k, f_, n: f"{r[k][f_]['mean']:.{n}f}±{r[k][f_]['sd']:.{n}f}"
    print(f"{r['dyn']:4s} {r['ratio']} | {f('paired_KL','K4000',4)} {f('paired_KL','inf',4)} "
          f"{100*r['paired_KL']['rel']['mean']:+.1f}±{100*r['paired_KL']['rel']['sd']:.1f}% [{r['paired_KL']['n_monotone']}/3] | "
          f"{f('log_vr','K4000',4)} {f('log_vr','inf',4)} {100*r['log_vr']['rel']['mean']:+.1f}±{100*r['log_vr']['rel']['sd']:.1f}% "
          f"[{r['log_vr']['n_monotone']}/3] | " + ' '.join(f"{r['log_vr_to_truth_fixed'][K]['mean']:.3f}" for K in ['K4000', 'K8000', 'K16000']))
