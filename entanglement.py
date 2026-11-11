"""
High-dimensional entanglement certification with two measurement settings.

State: |Phi_d> = (1/sqrt d) sum_k |k>_A |k>_B   on n + n qubits (d = 2**n)
       (= n Bell pairs (a_i, b_i), prepared with H + CNOT).

Settings:  ZZ   measure both qudits in the computational basis, P_Z = Pr[a == b]
           FF   A in Fourier basis (F^dag), B in conjugate Fourier basis (F^T),
                P_F = Pr[a == b]  (since (F ⊗ F*)|Phi> = |Phi>)

Bound:     fidelity  F(rho, Phi) >= P_Z + P_F - 1
           (operator inequality Pi_Z + Pi_F - I <= |Phi><Phi|, checked numerically in
           check_bound()).  Any state of Schmidt number <= k has F <= k/d, so
           F_lower > k/d certifies entanglement dimension >= k+1.
"""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import UnitaryGate

from qudit_bb84 import fourier


def witness_circuits(d: int):
    n = int(np.log2(d))
    F = fourier(d)
    out = {}
    for setting in ("ZZ", "FF"):
        a, b = QuantumRegister(n, "A"), QuantumRegister(n, "B")
        ca, cb = ClassicalRegister(n, "ma"), ClassicalRegister(n, "mb")
        qc = QuantumCircuit(a, b, ca, cb, name=f"wit_d{d}_{setting}")
        for i in range(n):
            qc.h(a[i]); qc.cx(a[i], b[i])
        if setting == "FF":
            qc.append(UnitaryGate(F.conj().T, label="F†"), a)
            qc.append(UnitaryGate(F.T, label="Fᵀ"), b)
        qc.measure(a, ca); qc.measure(b, cb)
        out[setting] = qc
    return out


def p_equal(joint: dict) -> float:
    """joint: {(a, b): count} -> Pr[a == b]."""
    tot = sum(joint.values())
    return sum(c for (x, y), c in joint.items() if x == y) / tot


def certify(p_z: float, p_f: float, d: int) -> dict:
    f_lb = p_z + p_f - 1
    k = int(np.floor(f_lb * d - 1e-12)) + 1 if f_lb > 1 / d else 1  # smallest k with f_lb <= k/d, +1
    k = min(max(k, 1), d)
    return {"P_Z": p_z, "P_F": p_f, "fidelity_lower_bound": f_lb,
            "certified_schmidt_number": k, "thresholds": [j / d for j in range(1, d)]}


def check_bound(d: int) -> float:
    F, I = fourier(d), np.eye(d)
    kk = [np.kron(I[k], I[k]) for k in range(d)]
    PZ = sum(np.outer(v, v.conj()) for v in kk)
    U = np.kron(F, F.conj())
    PF = U @ PZ @ U.conj().T
    phi = sum(kk) / np.sqrt(d)
    return float(np.linalg.eigvalsh(PZ + PF - np.eye(d * d) - np.outer(phi, phi.conj())).max())
