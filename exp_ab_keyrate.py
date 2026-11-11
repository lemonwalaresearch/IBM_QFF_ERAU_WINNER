"""
Experiment A: qubit (d=2) vs qudit (d=4, 8) BB84 under a depolarizing channel.
Experiment B: tolerance to adjacent-mode crosstalk; positive-key region in (lambda, eps).

Every simulated point is a real Qiskit Aer run of the full protocol (state prep,
channel noise instructions, measurement, sifting) and is overlaid on closed-form theory.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

import style
from qudit_bb84 import QuditBB84, theory_qber_depol, key_rate, threshold_qber

style.apply()
DS = (2, 4, 8)
N_PHOTONS = 4000
results = {"A": {}, "B": {}}

# ------------------------------------------------------------------ A
lams = np.round(np.arange(0, 0.401, 0.025), 3)
lam_fine = np.linspace(0, 0.45, 300)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for d in DS:
    sim = QuditBB84(d, seed=7)
    q_sim, r_sim = [], []
    for lam in lams:
        s = sim.run_session(N_PHOTONS, lam=lam)
        q_sim.append((s["Q_Z"] + s["Q_F"]) / 2)
        r_sim.append(s["rate"])
    q_th = [theory_qber_depol(l, d) for l in lam_fine]
    r_th = [key_rate(q, q, d) for q in q_th]
    thr = threshold_qber(d)
    lam_max = thr * d / (d - 1)
    results["A"][d] = {"lambda": lams.tolist(), "qber_sim": q_sim, "rate_sim": r_sim,
                       "qber_threshold": thr, "max_depolarizing": lam_max,
                       "rate_at_lam_0.10": key_rate(theory_qber_depol(.1, d), theory_qber_depol(.1, d), d)}
    c = style.D_COLOR[d]
    ax1.plot(lam_fine, q_th, color=c)
    ax1.plot(lams, q_sim, "o", color=c, mec=style.SURFACE, mew=1.5)
    ax1.plot(lam_max, thr, "D", color=c, ms=7)
    ax1.annotate(f"d={d}: abort at {thr:.1%}", (lam_max, thr), xytext=(6, -12),
                 textcoords="offset points", fontsize=8.5, color=style.INK2)
    ax2.plot(lam_fine, r_th, color=c, label=f"d = {d}  ({int(np.log2(d))} qubit{'s' if d > 2 else ''})")
    ax2.plot(lams, r_sim, "o", color=c, mec=style.SURFACE, mew=1.5)

ax1.set(title="Error rate vs channel noise", xlabel="Channel depolarization λ",
        ylabel="Sifted error rate Q (QBER)", xlim=(0, 0.45), ylim=(0, 0.42))
ax2.set(title="Secret key per sifted photon", xlabel="Channel depolarization λ",
        ylabel="Secret key (bits / photon)", xlim=(0, 0.45), ylim=(0, 3.1))
ax2.legend(loc="upper right")
fig.text(0.01, -0.03, "Lines: closed-form theory.  Dots: full protocol simulated in Qiskit Aer, "
         f"{N_PHOTONS} sifted photons per point.  Diamonds: security threshold (key rate reaches 0).",
         fontsize=8, color=style.INK2)
fig.savefig("figs/A_qubit_vs_qudit.png")
plt.close(fig)

# ------------------------------------------------------------------ B
# Crosstalk then depolarization:  Q_Z = (1-λ)ε + λ(d-1)/d ,  Q_F = λ(d-1)/d
def rates(lam, eps, d):
    qz = (1 - lam) * eps + lam * (d - 1) / d
    qf = lam * (d - 1) / d
    return key_rate(qz, qf, d), key_rate(max(qz, qf), max(qz, qf), d)  # per-basis, naive symmetric

LAM_B = 0.05
eps_fine = np.linspace(0, 0.5, 501)
eps_pts = np.round(np.arange(0, 0.501, 0.05), 2)
fig, ax = plt.subplots(figsize=(7, 4.4))
for d in DS:
    c = style.D_COLOR[d]
    rp = np.array([rates(LAM_B, e, d)[0] for e in eps_fine])
    rn = np.array([rates(LAM_B, e, d)[1] for e in eps_fine])
    first_zero = lambda r: float(eps_fine[np.argmax(r <= 0)]) if (r <= 0).any() else None
    ax.plot(eps_fine, rp, color=c, label=f"d = {d}, per-basis accounting")
    ax.plot(eps_fine, rn, color=c, ls=(0, (3, 2)), lw=1.5, label=f"d = {d}, single averaged QBER")
    sim = QuditBB84(d, seed=11)
    r_sim = []
    for e in eps_pts:
        s = sim.run_session(N_PHOTONS, lam=LAM_B, eps=e)
        r_sim.append(s["rate"])
    ax.plot(eps_pts, r_sim, "o", color=c, mec=style.SURFACE, mew=1.5)
    results["B"][d] = {"lambda": LAM_B, "eps": eps_pts.tolist(), "rate_sim": r_sim,
                       "eps_budget_per_basis": first_zero(rp), "eps_budget_naive": first_zero(rn)}
ax.set(title=f"Key rate vs mode crosstalk (channel λ = {LAM_B})", xlabel="Adjacent-mode crosstalk ε",
       ylabel="Secret key (bits / photon)", xlim=(0, 0.5), ylim=(0, 2.7))
ax.legend(fontsize=7.5, loc="upper right")
fig.text(0.01, -0.03, "Crosstalk only corrupts the mode (Z) basis. Accounting for each basis separately "
         "keeps far more key than one averaged error rate.  Dots: Qiskit Aer.", fontsize=8, color=style.INK2)
fig.savefig("figs/B_crosstalk_budget.png")
plt.close(fig)

json.dump(results, open("results/ab.json", "w"), indent=1, default=float)
for d in DS:
    a = results["A"][d]
    print(f"d={d}: threshold Q={a['qber_threshold']:.3f}, max λ={a['max_depolarizing']:.3f}, "
          f"rate@λ=.1={a['rate_at_lam_0.10']:.2f} | B budget per-basis={results['B'][d]['eps_budget_per_basis']} naive={results['B'][d]['eps_budget_naive']}")
