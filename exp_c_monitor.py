"""
Experiment C: attack-aware link monitor for d = 4 qudit BB84.

Each "session" is one short key-exchange block (N sifted photons) simulated in
Qiskit Aer. From its error statistics alone, can a monitor tell
    0 nominal            small channel noise + small crosstalk
    1 hardware fault     elevated adjacent-mode crosstalk (drifted / damaged optics)
    2 symmetric excess   EITHER an intercept-resend eavesdropper OR a turbulent channel
and, inside class 2, can it tell the eavesdropper from turbulence?

Methods compared on identical features and folds:
  QBER rule        standard practice: one averaged error rate vs a threshold
  physics rule     two thresholds: basis asymmetry (Q_Z - Q_F) and average QBER
  logistic regression, random forest, MLP   (classical ML)
  quantum-kernel SVM   ZZ feature map on 4 qubits, fidelity kernel |<phi(x)|phi(y)>|^2
"""
import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC

import style
from qudit_bb84 import QuditBB84

style.apply()
D, N_PHOTONS, PER_GROUP, SEED = 4, 400, 200, 2026
FEATURES = ["Q_Z", "Q_F", "adj_Z", "adj_F"]
CACHE = Path("results/monitor_sessions.npz")

# ------------------------------------------------------------------ data
def make_sessions():
    rng = np.random.default_rng(SEED)
    sim = QuditBB84(D, seed=SEED)
    X, y, src, params = [], [], [], []
    def add(label, source, lam, eps, f):
        s = sim.run_session(N_PHOTONS, lam=lam, eps=eps, f_eve=f)
        X.append([s[k] for k in FEATURES]); y.append(label); src.append(source)
        params.append([lam, eps, f])
    for _ in range(PER_GROUP):   # nominal
        add(0, "nominal", rng.uniform(0, .04), rng.uniform(0, .04), 0)
    for _ in range(PER_GROUP):   # hardware fault: crosstalk
        add(1, "crosstalk", rng.uniform(0, .04), rng.uniform(.06, .25), 0)
    for _ in range(PER_GROUP):   # eavesdropper, intercept fraction f  (adds Q = f*(d-1)/(2d))
        add(2, "eve", rng.uniform(0, .04), rng.uniform(0, .04), rng.uniform(.15, .6))
    for _ in range(PER_GROUP):   # turbulence with the SAME error-rate range (λ_extra = f/2)
        add(2, "turbulence", rng.uniform(0, .04) + rng.uniform(.075, .3), rng.uniform(0, .04), 0)
    return np.array(X), np.array(y), np.array(src), np.array(params)


if CACHE.exists():
    z = np.load(CACHE); X, y, src, params = z["X"], z["y"], z["src"], z["params"]
else:
    t = time.time()
    X, y, src, params = make_sessions()
    np.savez(CACHE, X=X, y=y, src=src, params=params)
    print(f"simulated {len(y)} sessions in {time.time() - t:.0f}s")

# ------------------------------------------------------------------ models
class QBERRule(BaseEstimator, ClassifierMixin):
    """Standard practice: nominal if averaged QBER < t, else 'problem' (class 2)."""
    def fit(self, X, y):
        q = X[:, :2].mean(1)
        cands = np.quantile(q, np.linspace(.05, .95, 91))
        self.t_ = max(cands, key=lambda t: f1_score(y, self._p(q, t), average="macro"))
        self.classes_ = np.unique(y); return self
    def _p(self, q, t): return np.where(q < t, 0, 2)
    def predict(self, X): return self._p(X[:, :2].mean(1), self.t_)


class PhysicsRule(BaseEstimator, ClassifierMixin):
    """Crosstalk only corrupts Z: large Q_Z - Q_F => hardware; else large mean QBER => symmetric."""
    def fit(self, X, y):
        a, q = X[:, 0] - X[:, 1], X[:, :2].mean(1)
        ta = np.quantile(a, np.linspace(.05, .95, 37)); tq = np.quantile(q, np.linspace(.05, .95, 37))
        best = max(((f1_score(y, self._p(a, q, i, j), average="macro"), i, j) for i in ta for j in tq))
        _, self.ta_, self.tq_ = best; self.classes_ = np.unique(y); return self
    def _p(self, a, q, ta, tq): return np.where(a > ta, 1, np.where(q > tq, 2, 0))
    def predict(self, X): return self._p(X[:, 0] - X[:, 1], X[:, :2].mean(1), self.ta_, self.tq_)


class QuantumKernelSVC(BaseEstimator, ClassifierMixin):
    """Fidelity quantum kernel K(x,x') = |<phi(x)|phi(x')>|^2 with a ZZ feature map,
    evaluated exactly with Qiskit Statevector, fed to a precomputed-kernel SVM."""
    def __init__(self, reps=2, C=1.0, bandwidth=1.0):
        # bandwidth scales features into [0, bandwidth*pi]; small values avoid the
        # exponential kernel concentration of full-range ZZ maps (Shaydulin & Wild, PRA 2022)
        self.reps, self.C, self.bandwidth = reps, C, bandwidth
    def _states(self, X):
        Xs = self.scaler_.transform(X)
        return np.array([Statevector(self.fmap_.assign_parameters(x)).data for x in Xs])
    def fit(self, X, y):
        self.scaler_ = MinMaxScaler((0, self.bandwidth * np.pi)).fit(X)
        self.fmap_ = zz_feature_map(X.shape[1], reps=self.reps)
        self.S_ = self._states(X)
        K = np.abs(self.S_.conj() @ self.S_.T) ** 2
        self.svc_ = SVC(kernel="precomputed", C=self.C).fit(K, y)
        self.classes_ = self.svc_.classes_; return self
    def predict(self, X):
        K = np.abs(self._states(X).conj() @ self.S_.T) ** 2
        return self.svc_.predict(K)


MODELS = {
    "QBER rule\n(standard)": lambda: QBERRule(),
    "Physics rule\n(2 thresholds)": lambda: PhysicsRule(),
    "Logistic\nregression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Random\nforest": lambda: RandomForestClassifier(300, random_state=0),
    "MLP": lambda: make_pipeline(StandardScaler(), MLPClassifier((32, 16), max_iter=3000, random_state=0)),
    "RBF-kernel SVM\n(classical, tuned)": lambda: GridSearchCV(
        make_pipeline(StandardScaler(), SVC()), {"svc__C": [1, 10, 100], "svc__gamma": ["scale", .1, 1]},
        cv=3, scoring="f1_macro"),
    "Quantum-kernel\nSVM (untuned)": lambda: QuantumKernelSVC(reps=2, C=1.0),
    "Quantum-kernel\nSVM (tuned)": lambda: GridSearchCV(
        QuantumKernelSVC(reps=2), {"C": [1, 10, 100], "bandwidth": [.1, .25, .5, 1.0]},
        cv=3, scoring="f1_macro"),
}


def cv_eval(X, y, folds=5):
    skf = StratifiedKFold(folds, shuffle=True, random_state=0)
    out = {}
    for name, mk in MODELS.items():
        f1s, accs, cms = [], [], np.zeros((len(np.unique(y)),) * 2, int)
        for tr, te in skf.split(X, y):
            m = mk().fit(X[tr], y[tr]); p = m.predict(X[te])
            f1s.append(f1_score(y[te], p, average="macro")); accs.append(accuracy_score(y[te], p))
            cms += confusion_matrix(y[te], p, labels=np.unique(y))
        out[name] = {"f1": float(np.mean(f1s)), "f1_sd": float(np.std(f1s)),
                     "acc": float(np.mean(accs)), "acc_sd": float(np.std(accs)), "cm": cms.tolist()}
    return out


# Task 1: 3-class (balance class 2 by taking half eve, half turbulence -> 400 each? use all, macro-F1)
res3 = cv_eval(X, y)
# Task 2: eavesdropper vs turbulence, symmetric sessions only
m2 = np.isin(src, ["eve", "turbulence"])
y2 = (src[m2] == "eve").astype(int)
res2 = cv_eval(X[m2], y2)

# ------------------------------------------------------------------ figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
names = list(MODELS)
SHORT = ["QBER\nrule", "Physics\nrule", "Logistic\nreg.", "Random\nforest", "MLP",
         "RBF SVM\n(tuned)", "Q-kernel\n(untuned)", "Q-kernel\n(tuned)"]
xs = np.arange(len(names))
cols = [style.MUTED, style.MUTED] + [style.SERIES[0]] * 4 + [style.SERIES[1]] * 2
for ax, res, title, chance in [(ax1, res3, "Diagnose the link: nominal / hardware fault / symmetric excess", 1 / 3),
                               (ax2, res2, "Eavesdropper vs turbulence (same error rate)", 0.5)]:
    v = [res[n]["f1"] if ax is ax1 else res[n]["acc"] for n in names]
    e = [res[n]["f1_sd"] if ax is ax1 else res[n]["acc_sd"] for n in names]
    ax.bar(xs, v, width=0.62, color=cols, edgecolor=style.SURFACE, linewidth=2, zorder=3)
    ax.errorbar(xs, v, yerr=e, fmt="none", ecolor=style.INK2, elinewidth=1, capsize=3, zorder=4)
    for xi, vi, ei in zip(xs, v, e):
        ax.text(xi, vi + ei + 0.025, f"{vi:.2f}", ha="center", fontsize=8.5, color=style.INK)
    ax.axhline(chance, color=style.INK2, lw=1, ls=(0, (4, 3)), zorder=5)
    ax.text(-0.45, chance + 0.015, "chance", ha="left", fontsize=8, color=style.INK2)
    ax.set_xticks(xs, SHORT, fontsize=8); ax.set_ylim(0, 1.12); ax.set_title(title, fontsize=10.5)
    ax.grid(axis="x", visible=False)
ax1.set_ylabel("Macro F1 (5-fold CV)"); ax2.set_ylabel("Accuracy (5-fold CV)")
fig.text(0.01, -0.04, f"d = 4, {N_PHOTONS} sifted photons per session, {PER_GROUP} sessions per source, "
         "all simulated in Qiskit Aer.  Gray: rules.  Blue: classical ML.  Orange: quantum kernel.",
         fontsize=8, color=style.INK2)
fig.savefig("figs/C_monitor.png"); plt.close(fig)

# feature-space view: why task 2 is impossible
fig, ax = plt.subplots(figsize=(6.4, 4.4))
lab = {"nominal": ("Nominal", style.SERIES[0]), "crosstalk": ("Hardware: crosstalk", style.SERIES[1]),
       "eve": ("Eavesdropper", style.SERIES[2]), "turbulence": ("Turbulence", style.INK2)}
for s, (name, c) in lab.items():
    m = src == s
    ax.scatter(X[m, 1], X[m, 0], s=14, color=c, alpha=.75, edgecolor=style.SURFACE, linewidth=.6,
               label=name, marker="o" if s != "turbulence" else "x")
ax.plot([0, .35], [0, .35], color=style.MUTED, lw=1, ls=(0, (4, 3)))
ax.text(.25, .27, "Q_Z = Q_F", fontsize=8, color=style.INK2, rotation=33)
ax.set(xlabel="Error rate in Fourier basis  Q_F", ylabel="Error rate in mode basis  Q_Z",
       title="Each session in error-rate space", xlim=(0, .3), ylim=(0, .35))
ax.legend(loc="lower right", fontsize=8)
fig.savefig("figs/C_feature_space.png"); plt.close(fig)

summary = {"three_class": res3, "eve_vs_turbulence": res2,
           "setup": {"d": D, "photons_per_session": N_PHOTONS, "sessions_per_source": PER_GROUP}}
json.dump(summary, open("results/monitor.json", "w"), indent=1)
for task, res in [("3-class macro-F1", res3), ("eve-vs-turbulence acc", res2)]:
    print(task)
    for n, r in res.items():
        k = "f1" if "F1" in task else "acc"
        print(f"   {n.replace(chr(10), ' '):32s} {r[k]:.3f} ± {r[k + '_sd']:.3f}")
