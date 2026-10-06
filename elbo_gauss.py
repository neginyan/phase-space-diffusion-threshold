"""(a) Likelihood weighting = G, exact Gaussian computation.

Data: Gaussian with the mean/covariance of the two-moons phase-space data.  Hamiltonian omega = 2.
Model score: s_theta = s + e(z,t),  e = -eps * g(t) A (z - m_t) + eps * h(t) b,
with g = 1/(1 + sigma^2), h = 1/sqrt(1 + sigma^2).
  J      = 1/2 int_0^T E_{p_t}[ e^T G e ] dt       (continuous-time ELBO / Girsanov term, weight G)
  KL     = KL( p_0 || q_0 ), q_0 = law at t=0 of the Anderson reverse SDE (noise sqrt(G)) driven by
           s_theta and started from the exact p_T (so KL(p_T||prior) = 0).  Exact via moment ODEs.
Claims checked:
  r >= kappa : KL <= J for every perturbation (the bound with weight G holds).
  r <  kappa : (the Anderson reverse SDE does not exist; the model is the probability-flow ODE)
               J < 0 for perturbations along the negative eigenvector of G, J(eps) = eps^2 J(1) -> -inf,
               so J cannot upper-bound any KL (KL >= 0).
"""
import numpy as np, json, os
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from psdt import hamiltonian_B, kappa_of, Process, two_moons

ref = two_moons(400000, np.random.default_rng(1)); M0 = ref.mean(0); S0 = np.cov(ref.T); I4 = np.eye(4)
B = hamiltonian_B(2.0); KAP = kappa_of(B)

def gkl(m1, C1, m2, C2):
    iC2 = np.linalg.inv(C2); d = m2 - m1
    return 0.5 * (np.trace(iC2 @ C1) + d @ iC2 @ d - len(m1) + np.log(np.linalg.det(C2) / np.linalg.det(C1)))

class Setup:
    def __init__(self, rr):
        self.p = Process(B, rr * KAP)
    def mom(self, t):
        M = expm(B * t); return M @ M0, M @ S0 @ M.T + self.p.sigma(t)**2 * I4
    def gh(self, t):
        s2 = self.p.sigma(t)**2; return 1 / (1 + s2), 1 / np.sqrt(1 + s2)

def J_value(S, A, b, eps=1.0, n=4001):
    ts = np.linspace(0, S.p.T, n); vals = []
    for t in ts:
        _, C = S.mom(t); g, h = S.gh(t); G = S.p.G(t)
        D = eps * g * A; d = eps * h * b
        vals.append(np.trace(D @ G @ D @ C) + d @ G @ d)      # E[e^T G e], A symmetric
    vals = np.array(vals)
    return 0.5 * np.trapezoid(vals, ts)

def KL_value(S, A, b, eps=1.0, model='anderson'):
    p = S.p
    def rhs(tau, y):
        t = p.T - tau; mu = y[:4]; Sg = y[4:].reshape(4, 4)
        m, C = S.mom(t); P = np.linalg.inv(C); g, h = S.gh(t); G = p.G(t)
        # s_theta(z) = -(P + eps g A)(z - m) + eps h b ; reverse drift dz/dtau = -Bz + G s_theta
        K = P + eps * g * A
        c = 1.0 if model == 'anderson' else 0.5           # probability-flow ODE: dz/dtau = -Bz + G s/2
        D = G if model == 'anderson' else np.zeros((4, 4))
        Am = -B - c * G @ K
        return np.concatenate([Am @ mu + c * G @ (K @ m + eps * h * b), (Am @ Sg + Sg @ Am.T + D).ravel()])
    m, C = S.mom(p.T)
    y = solve_ivp(rhs, [0, p.T], np.concatenate([m, C.ravel()]), rtol=1e-10, atol=1e-12, method='DOP853').y[:, -1]
    mt, Ct = S.mom(0); Sg = y[4:].reshape(4, 4)
    return gkl(mt, Ct, y[:4], (Sg + Sg.T) / 2)               # KL(p_0 || q_0)

def random_pert(rng):
    A = rng.standard_normal((4, 4)); A = (A + A.T) / 2
    return A / np.linalg.norm(A, 2), rng.standard_normal(4) / 2

def run(n_random=40, out='results/elbo_gauss.json'):
    rng = np.random.default_rng(0)
    res = {'kappa': KAP, 'above': [], 'below': []}
    for rr in [1.0, 1.1, 1.3]:
        S = Setup(rr)
        for i in range(n_random):
            A, b = random_pert(rng)
            for eps in [0.1, 0.5]:
                J = J_value(S, A, b, eps); KL = KL_value(S, A, b, eps)
                res['above'].append({'ratio': rr, 'eps': eps, 'J': J, 'KL': KL})
    for rr in [0.6, 0.8, 0.9]:
        S = Setup(rr)
        w, V = np.linalg.eigh(S.p.SymB); u = V[:, -1]            # eigenvector of Sym B with eigenvalue +kappa
        A, b = np.outer(u, u), u                                  # = negative eigen-direction of G
        for eps in [0.1, 0.3, 1.0, 3.0]:
            J = J_value(S, A, b, eps); KL = KL_value(S, A, b, eps, 'ode')
            res['below'].append({'ratio': rr, 'eps': eps, 'J': J, 'KL': KL, 'kind': 'negative direction'})
        for i in range(n_random):                                  # generic perturbations, for reference
            A, b = random_pert(rng)
            J = J_value(S, A, b, 0.5); KL = KL_value(S, A, b, 0.5, 'ode')
            res['below'].append({'ratio': rr, 'eps': 0.5, 'J': J, 'KL': KL, 'kind': 'random'})
    # J along the eigenvector u of Sym B (eigenvalue kappa), on both sides of the threshold (Fig. 2, left)
    res['u_direction'] = []
    for rr in [0.6, 0.8, 0.9, 1.0, 1.1, 1.3]:
        S = Setup(rr)
        u = np.linalg.eigh(S.p.SymB)[1][:, -1]
        for eps in [0.1, 0.3, 1.0, 3.0]:
            res['u_direction'].append({'ratio': rr, 'eps': eps, 'J': J_value(S, np.outer(u, u), u, eps)})
    # control: the probability-flow ODE likelihood is not bounded by J even when G >= 0 (known; Song et al. 2021),
    # so random-perturbation violations of KL_ODE <= J below kappa are NOT evidence for the threshold.
    res['ode_control_above'] = []
    for rr in [1.0, 1.3]:
        S = Setup(rr)
        for i in range(n_random):
            A, b = random_pert(rng)
            res['ode_control_above'].append({'ratio': rr, 'eps': 0.5, 'J': J_value(S, A, b, 0.5),
                                             'KL': KL_value(S, A, b, 0.5, 'ode')})
    os.makedirs('results', exist_ok=True); json.dump(res, open(out, 'w'), indent=1)
    return res

if __name__ == '__main__':
    res = run()
    ab = res['above']
    print('r >= kappa: %d cases, max KL/J = %.3f, min (J - KL) = %.2e, all KL <= J: %s'
          % (len(ab), max(x['KL'] / x['J'] for x in ab), min(x['J'] - x['KL'] for x in ab),
             all(x['KL'] <= x['J'] + 1e-12 for x in ab)))
    for rr in [0.6, 0.8, 0.9]:
        neg = [x for x in res['below'] if x['ratio'] == rr and x['kind'] == 'negative direction']
        rnd = [x for x in res['below'] if x['ratio'] == rr and x['kind'] == 'random']
        print(f'r/k={rr}: negative direction ' + ', '.join(f"eps={x['eps']}: J={x['J']:.3g}, KL={x['KL']:.3g}" for x in neg))
        print(f'         random perturbations: violations of KL<=J {sum(x["KL"] > x["J"] for x in rnd)}/{len(rnd)}, '
              f'J<0 in {sum(x["J"] < 0 for x in rnd)}/{len(rnd)}')
    oc = res['ode_control_above']
    print('control (ODE model, r >= kappa): KL > J in %d/%d -> random violations are not threshold evidence'
          % (sum(x['KL'] > x['J'] for x in oc), len(oc)))
