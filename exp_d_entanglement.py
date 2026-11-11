"""
Experiment D (simulation side): certified entanglement dimension vs local noise.
Each qubit gets depolarizing noise p after state preparation (Qiskit Aer).
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit import transpile

import style
from entanglement import witness_circuits, p_equal, certify, check_bound

style.apply()
SHOTS = 8000
ps = np.round(np.arange(0, 0.301, 0.02), 3)
res = {"bound_check_max_eig": {d: check_bound(d) for d in (2, 4, 8)}}


def joint_from_aer(counts):
    j = {}
    for bits, c in counts.items():
        mb, ma = bits.split()
        j[(int(ma, 2), int(mb, 2))] = j.get((int(ma, 2), int(mb, 2)), 0) + c
    return j


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for d in (2, 4, 8):
    circs = witness_circuits(d)
    fl, ks = [], []
    for p in ps:
        nm = NoiseModel()
        if p > 0:  # noise on the entangling step and on every single-qubit op
            nm.add_all_qubit_quantum_error(depolarizing_error(p, 2), ["cx"])
            nm.add_all_qubit_quantum_error(depolarizing_error(p / 10, 1), ["h", "u", "rz", "sx", "x"])
        sim = AerSimulator(noise_model=nm)
        pz = p_equal(joint_from_aer(sim.run(transpile(circs["ZZ"], sim), shots=SHOTS).result().get_counts()))
        pf = p_equal(joint_from_aer(sim.run(transpile(circs["FF"], sim), shots=SHOTS).result().get_counts()))
        c = certify(pz, pf, d)
        fl.append(c["fidelity_lower_bound"]); ks.append(c["certified_schmidt_number"])
    res[d] = {"p": ps.tolist(), "fidelity_lb": fl, "certified": ks}
    col = style.D_COLOR[d]
    ax1.plot(ps, fl, "o-", color=col, mec=style.SURFACE, mew=1.2, label=f"d = {d}  ({2 * int(np.log2(d))} qubits)")
    ax2.step(ps, ks, where="mid", color=col, label=f"d = {d}")
ax1.axhline(0.25, color=style.MUTED, lw=1, ls=(0, (4, 3)))
ax1.text(0.295, 0.265, "F > 1/4: d = 4 is entangled", ha="right", fontsize=8, color=style.INK2)
ax1.set(title="Fidelity lower bound from 2 settings", xlabel="Two-qubit gate depolarization p",
        ylabel="F ≥ P_Z + P_F − 1", xlim=(0, .3), ylim=(-0.05, 1.05))
ax1.legend(loc="upper right")
ax2.set(title="Certified entanglement dimension", xlabel="Two-qubit gate depolarization p",
        ylabel="Schmidt number certified ≥", xlim=(0, .3), ylim=(0.5, 8.5), yticks=range(1, 9))
ax2.legend(loc="upper right")
fig.text(0.01, -0.03, f"Qiskit Aer, {SHOTS} shots per setting. A Schmidt number of k means the photons "
         "share at least k-dimensional entanglement — no qubit pair can exceed 2.", fontsize=8, color=style.INK2)
fig.savefig("figs/D_entanglement_sim.png"); plt.close(fig)
json.dump(res, open("results/entanglement_sim.json", "w"), indent=1, default=float)
print("bound check (should be <= 0):", res["bound_check_max_eig"])
for d in (2, 4, 8):
    print(d, "p=0:", round(res[d]["fidelity_lb"][0], 3), res[d]["certified"][0],
          "| p=.1:", round(res[d]["fidelity_lb"][5], 3), res[d]["certified"][5])
