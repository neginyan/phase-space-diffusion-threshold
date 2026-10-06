"""Figure and summary for (a): figures/elbo.png, results/summary_elbo.json"""
import numpy as np, json, glob, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

G = json.load(open('results/elbo_gauss.json'))
runs = [json.load(open(f)) for f in sorted(glob.glob('results/elbo_train_s*_r*.json'))]
ratios = sorted(set(r['ratio'] for r in runs))
summ = {'gauss_above_cases': len(G['above']),
        'gauss_above_all_KL_le_J': all(x['KL'] <= x['J'] for x in G['above']),
        'gauss_above_max_KL_over_J': max(x['KL'] / x['J'] for x in G['above']),
        'train': {}}
for rr in ratios:
    sub = [r for r in runs if r['ratio'] == rr]
    first_neg = [next((e['step'] for e in r['log'] if e['weighted_loss'] < 0), None) for r in sub]
    summ['train'][str(rr)] = {
        'n': len(sub), 'W_min_eig': sub[0]['W_eigs'][0],
        'final_weighted_loss': [r['log'][-1]['weighted_loss'] for r in sub],
        'final_mse': [r['log'][-1]['mse'] for r in sub], 'final_mse_u': [r['log'][-1]['mse_u'] for r in sub],
        'first_step_loss_negative': first_neg}
json.dump(summ, open('results/summary_elbo.json', 'w'), indent=1)
for rr in ratios:
    t = summ['train'][str(rr)]
    print(f"r/k={rr}: W_min={t['W_min_eig']:+.2f}  final loss {np.median(t['final_weighted_loss']):.3g} (median)  "
          f"mse {np.median(t['final_mse']):.3g}  mse_u {np.median(t['final_mse_u']):.3g}  first loss<0 at {t['first_step_loss_negative']}")

fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
cm = {0.8: '#c0392b', 0.9: '#e67e22', 1.0: '#7f8c8d', 1.1: '#2980b9', 1.3: '#2c3e50'}
a = ax[0]
ud = G['u_direction']
for rr, c in [(0.6, '#8e2c20'), (0.8, '#c0392b'), (0.9, '#e67e22'), (1.0, '#7f8c8d'), (1.1, '#2980b9'), (1.3, '#2c3e50')]:
    pts = sorted([x for x in ud if x['ratio'] == rr], key=lambda x: x['eps'])
    a.plot([x['eps'] for x in pts], [x['J'] for x in pts], 'o-', color=c, label=fr'$r/\kappa={rr}$')
a.axhline(0, color='k', lw=0.8); a.set_xscale('log'); a.set_yscale('symlog', linthresh=1e-2)
a.set_xlabel(r'score-error size $\epsilon$ along $u$'); a.set_ylabel(r'$J$ (weight $G$)')
a.set_title('Exact Gaussian: $J$ for an error along $u$', fontsize=10); a.legend(fontsize=7, ncol=2)
for j, key, ttl in [(1, 'weighted_loss', 'Likelihood-weighted training loss'),
                    (2, 'mse', 'Unweighted score error (true accuracy)')]:
    a = ax[j]
    for r in runs:
        st = [e['step'] for e in r['log']]; v = [e[key] for e in r['log']]
        a.plot(st, v, color=cm[r['ratio']], alpha=0.5, lw=1,
               label=fr"$r/\kappa={r['ratio']}$" if r['seed'] == 0 else None)
    a.set_yscale('symlog' if key == 'weighted_loss' else 'log', **({'linthresh': 1} if key == 'weighted_loss' else {}))
    if key == 'weighted_loss':
        a.set_yticks([-1e21, -1e15, -1e9, -1e3, 10])
    a.set_xlabel('training step'); a.set_title(ttl + ' (5 seeds)', fontsize=10); a.legend(fontsize=7)
fig.tight_layout(); os.makedirs('figures', exist_ok=True); fig.savefig('figures/elbo.png', dpi=150)
print('saved figures/elbo.png')
