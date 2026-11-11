"""
High-dimensional BB84 (qudit QKD) in Qiskit.

A qudit of dimension d = 2**n is carried by n qubits.
  * Basis Z ("OAM mode" / time-bin index):  |k>
  * Basis F (Fourier / superposition basis): QFT|k>
Z and F are mutually unbiased for every d.

Channel models (all applied as real Qiskit noise instructions inside the circuit):
  * depolarizing(lam)  - generic loss of coherence; affects both bases equally
  * crosstalk(eps)     - adjacent-mode leakage |k> -> |k+-1>, the dominant
                         imperfection of mode sorters (e.g. orbital-angular-momentum optics).
                         It is a cyclic shift, so it is diagonal in F: it corrupts
                         Z only. That asymmetry is a hardware fingerprint.
  * intercept-resend   - Eve measures a fraction f of photons in a random basis
                         (mid-circuit measurement) and re-prepares what she saw.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit_aer.noise import depolarizing_error, mixed_unitary_error


# ---------------------------------------------------------------- operators
def fourier(d: int) -> np.ndarray:
    w = np.exp(2j * np.pi / d)
    return np.array([[w ** (j * k) for k in range(d)] for j in range(d)]) / np.sqrt(d)


def shift(d: int, s: int = 1) -> np.ndarray:
    return np.roll(np.eye(d), s, axis=0)  # |k> -> |k+s mod d>


def theory_qber_depol(lam: float, d: int) -> float:
    return lam * (d - 1) / d


def theory_qber_ir(f: float, d: int) -> float:
    """Intercept-resend on fraction f: Eve guesses wrong basis half the time,
    then Bob's outcome is uniform -> error (d-1)/d."""
    return f * 0.5 * (d - 1) / d


def h_d(q: float, d: int) -> float:
    if q <= 0:
        return 0.0
    if q >= 1:
        return np.log2(d - 1)
    return -q * np.log2(q / (d - 1)) - (1 - q) * np.log2(1 - q)


def key_rate(q_z: float, q_f: float, d: int) -> float:
    """Asymptotic secret fraction per sifted photon, asymmetric-error form:
    r = log2 d - h_d(Q_Z) - h_d(Q_F)  (reduces to log2 d - 2 h_d(Q) when equal)."""
    return max(0.0, np.log2(d) - h_d(q_z, d) - h_d(q_f, d))


def threshold_qber(d: int) -> float:
    from scipy.optimize import brentq
    return brentq(lambda q: np.log2(d) - 2 * h_d(q, d), 1e-6, (d - 1) / d - 1e-6)


# ---------------------------------------------------------------- circuits
class QuditBB84:
    def __init__(self, d: int, backend=None, seed: int | None = None):
        assert d >= 2 and (d & (d - 1)) == 0, "d must be a power of two"
        self.d, self.n = d, int(np.log2(d))
        self.F = UnitaryGate(fourier(d), label="F")
        self.Fdg = UnitaryGate(fourier(d).conj().T, label="F†")
        # note: no fixed seed_simulator -- a fixed seed would reuse one random
        # stream for every circuit and correlate the errors across circuits.
        self.backend = backend or AerSimulator()
        self.rng = np.random.default_rng(seed)

    def _prepare(self, qc, qr, k, basis):
        for i in range(self.n):
            if (k >> i) & 1:
                qc.x(qr[i])
        if basis == "F":
            qc.append(self.F, qr)

    def _measure(self, qc, qr, cr, basis):
        if basis == "F":
            qc.append(self.Fdg, qr)
        qc.measure(qr, cr)

    def circuit(self, k: int, basis: str, lam=0.0, eps=0.0, eve_basis: str | None = None):
        """One sifted photon: Alice sends symbol k in `basis`, Bob measures in `basis`."""
        qr = QuantumRegister(self.n, "photon")
        cb = ClassicalRegister(self.n, "bob")
        regs = [qr, cb]
        if eve_basis:
            ce = ClassicalRegister(self.n, "eve")
            regs.append(ce)
        qc = QuantumCircuit(*regs)
        self._prepare(qc, qr, k, basis)
        qc.barrier(label="Alice→")

        if eve_basis:  # intercept-resend: measure, collapse, re-prepare in her basis
            self._measure(qc, qr, ce, eve_basis)
            if eve_basis == "F":
                qc.append(self.F, qr)
            qc.barrier(label="Eve")

        if eps > 0:  # mode-sorter crosstalk: adjacent-mode leakage
            S = shift(self.d, 1)
            qc.append(mixed_unitary_error(
                [(np.eye(self.d), 1 - eps), (S, eps / 2), (S.conj().T, eps / 2)]), qr)
        if lam > 0:  # channel depolarization (background light, turbulence)
            qc.append(depolarizing_error(lam, self.n), qr)
        qc.barrier(label="→Bob")
        self._measure(qc, qr, cb, basis)
        return qc

    # ------------------------------------------------------------ a session
    def run_session(self, n_photons=2000, lam=0.0, eps=0.0, f_eve=0.0):
        """Simulate one key-exchange block of n sifted photons.
        Returns per-basis error counts and error-offset histograms."""
        d, rng = self.d, self.rng
        bases = rng.choice(["Z", "F"], n_photons)
        syms = rng.integers(0, d, n_photons)
        eve = rng.random(n_photons) < f_eve
        eve_b = rng.choice(["Z", "F"], n_photons)

        # group identical photons into one circuit with `shots` = multiplicity
        groups: dict[tuple, int] = {}
        for b, k, e, eb in zip(bases, syms, eve, eve_b):
            key = (int(k), b, eb if e else None)
            groups[key] = groups.get(key, 0) + 1

        keys = list(groups)
        circs = [self.circuit(k, b, lam, eps, eb) for (k, b, eb) in keys]
        # Aer runs noise instructions natively; shots differ per circuit
        out = {"Z": np.zeros(d, int), "F": np.zeros(d, int)}  # histogram of (bob-alice) mod d
        for (k, b, eb), qc in zip(keys, circs):
            res = self.backend.run(qc, shots=groups[(k, b, eb)],
                                   seed_simulator=int(rng.integers(2**31))).result().get_counts()
            for bits, c in res.items():
                bob = int(bits.split()[-1], 2)  # last register printed = first added (bob)
                out[b][(bob - k) % d] += c
        return summarize(out, d)


def summarize(hist: dict, d: int) -> dict:
    s = {}
    for b in ("Z", "F"):
        tot = hist[b].sum()
        err = tot - hist[b][0]
        s[f"Q_{b}"] = err / tot if tot else 0.0
        adj = hist[b][1] + hist[b][-1] if d > 2 else hist[b][1]
        s[f"adj_{b}"] = adj / err if err else 0.0  # share of errors that are ±1 mode slips
        s[f"n_{b}"] = int(tot)
    s["hist"] = hist
    s["rate"] = key_rate(s["Q_Z"], s["Q_F"], d)
    return s
