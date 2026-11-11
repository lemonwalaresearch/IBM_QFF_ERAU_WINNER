# Qudit Link Bench
**How many dimensions should a tactical quantum link use — and can it tell an attacker from a broken component?**

🏆 **Winner, Florida Qiskit Fall Fest 2026 (Embry-Riddle site)** · Project type: **benchmark**
Author: Murtuza Lemonwala · Built with Qiskit 2.6, Qiskit Aer 0.18, Qiskit IBM Runtime 0.51

---

## 1. Problem
Field units, drones and relay nodes need encryption keys that stay secret even against future computers. Quantum key distribution (QKD) gets its security from physics: a measurement disturbs the photon, so an eavesdropper leaves errors behind. But tactical links are noisy — background light, turbulence, vibration — and the textbook protocol (BB84 on qubits) aborts once the error rate passes **11%**.

Encoding each photon as a **qudit** (one of *d* states instead of 2) carries log₂*d* bits per photon and raises the abort threshold. Two practical questions follow:
1. How much does dimension actually buy, and where does it break?
2. When errors rise, is it an **attacker**, the **weather**, or **failing hardware**? An operator needs a different response to each.

## 2. Why quantum (and what quantum does *not* do here)
- **Security is quantum by construction.** Mutually unbiased bases (Z and Fourier) make any measurement by an eavesdropper visible as errors. No classical scheme has this property.
- **Higher dimension helps for a physical reason:** an eavesdropper who guesses the wrong basis scrambles a *d*-level state over *d* outcomes, so the tolerable error rate grows with *d*.
- **Where we tested quantum machine learning, we found parity, not advantage.** A tuned quantum-kernel classifier matches classical models on the link-diagnosis task. We report this as it is.
- **Today's limits:** on a gate-based processor a *d* = 8 Fourier transform costs up to ~36 two-qubit gates, so the chip itself becomes the noise source. In optics the same operation is a passive lens. Our hardware runs test the protocol and the mitigation, not a real fiber or free-space link.

## 3. What we built (all in Qiskit)
| Piece | Qiskit used |
|---|---|
| Qudit BB84 for *d* = 2, 4, 8 on 1–3 qubits. Fourier basis as a `UnitaryGate`; channel noise as `depolarizing_error` / `mixed_unitary_error` instructions | circuits, Aer noise |
| Intercept-resend eavesdropper as a **mid-circuit measurement** plus re-preparation | dynamic circuits |
| Link monitor: two rules, logistic regression, random forest, MLP, tuned RBF SVM, and a **quantum-kernel SVM** (ZZ feature map, fidelity kernel via `Statevector`) | `zz_feature_map`, `quantum_info` |
| Two-setting **high-dimensional entanglement witness** on 2, 4 and 6 qubits | circuits |
| Hardware pipeline: preset pass manager (opt. level 3), `SamplerV2` in job mode, dynamical decoupling, Pauli twirling, tensored readout-error correction from calibration circuits | transpiler, Runtime primitives, error suppression/mitigation |

## 4. Results (simulation, validated against theory)

**A · Qubit vs qudit** (`figs/A_qubit_vs_qudit.png`). The full protocol simulated in Aer matches closed-form theory (mean bias < 0.3% QBER).

| *d* | qubits | abort threshold | max channel depolarization | key at λ = 0.10 |
|---|---|---|---|---|
| 2 | 1 | 11.0% | 0.22 | 0.43 bits/photon |
| 4 | 2 | 18.9% | 0.25 | 0.99 bits/photon |
| 8 | 3 | 24.7% | 0.28 | 1.65 bits/photon |

**B · Component crosstalk** (`figs/B_crosstalk_budget.png`). Adjacent-mode leakage corrupts only the mode (Z) basis; the Fourier basis stays clean. Accounting for each basis separately, instead of using one averaged error rate, raises the tolerable crosstalk at λ = 0.05 from **16% to 44%** for *d* = 4, and from **21% to beyond 50%** for *d* = 8. This gives a concrete acceptance spec for a component.

**C · Link monitor** (`figs/C_monitor.png`, `figs/C_feature_space.png`). *d* = 4, 400-photon sessions, 800 sessions, 5-fold cross-validation.

| Method | Nominal / hardware fault / symmetric excess (macro F1) | Eavesdropper vs turbulence (accuracy) |
|---|---|---|
| Standard QBER threshold | 0.58 | 0.46 |
| Physics rule (2 thresholds) | 0.93 | 0.44 |
| Logistic regression | **0.96** | 0.50 |
| Random forest | 0.95 | 0.46 |
| MLP | 0.96 | 0.48 |
| RBF SVM, tuned | 0.96 | 0.52 |
| Quantum-kernel SVM, untuned | 0.80 | 0.54 |
| Quantum-kernel SVM, tuned bandwidth | 0.95 | 0.52 |

- The standard single-QBER rule cannot separate hardware faults from everything else; the basis-asymmetry feature can.
- **No method beats chance at eavesdropper vs turbulence, and none can.** Intercept-resend on a fraction *f* of photons produces exactly the sifted statistics of a depolarizing channel with λ = *f*/2. Error statistics alone cannot reveal the attacker; side information (timing, detector telemetry, decoy states) is required.
- Quantum kernel: the untuned ZZ map suffers from kernel concentration. With bandwidth tuning it reaches parity with classical models — no advantage.

**D · Entanglement dimension** (`figs/D_entanglement_sim.png`). Two measurement settings give fidelity F ≥ P_Z + P_F − 1 (operator inequality, checked numerically). F > k/*d* certifies Schmidt number ≥ k+1. Noise-free, the circuits certify the full dimension 2, 4 and 8. The *d* = 8 certificate is the first to collapse as two-qubit gate noise rises.

## 5. Real hardware: IBM Kingston (`run_hardware.py`, `analyze_hardware.py`, `figs/E_hardware.png`)
Two jobs on `ibm_kingston` (156-qubit Heron), Qiskit Runtime `SamplerV2`, 7 Nov 2026. **Raw job:** 92 circuits, 116,000 shots, **33 s of QPU time** (job `dbnicfoou4as73b4l1ig`). **Mitigated job:** 36 circuits, 60,000 shots, run 27 min later (job `dbnip00qtr0c73ad5hig`). Readout correction uses calibration circuits measured on the same 21 physical qubits inside each job.

| *d* | QBER: raw → readout fix → DD + twirl + readout | key, bits/photon (raw → readout fix) | witness F ≥ (raw → readout fix) | certified dimension |
|---|---|---|---|---|
| 2 | 0.6% → 0.25% → 0.15% | 0.89 → 0.95 | 0.969 → 0.984 | 2 of 2 |
| 4 | 2.9% → 2.1% → 1.9% | 1.55 → 1.69 | 0.885 → 0.922 | 4 of 4 |
| 8 | 5.8% → 4.0% → 5.2% | 2.09 → 2.37 | **0.727 ± 0.007 → 0.806 ± 0.006** | **≥ 6 raw (13.6σ) → ≥ 7 corrected (8.8σ)** |

What we observed:
- **Six-dimensional entanglement, certified from raw data.** F = 0.727 exceeds the 5/8 bound for any state of Schmidt number ≤ 5 by 13.6 standard deviations (binomial shot noise only). A qubit pair can never exceed 2.
- **Higher dimension still wins on a noisy chip.** At *d* = 8 the raw link keeps 2.09 bits per photon against 0.89 for qubits, far below every abort threshold.
- **The eavesdropper is caught every time.** Intercept-resend errors of 25.3%, 39.2% and 46.0% for *d* = 2, 4, 8 against theory of 25.0%, 37.5% and 43.8%. The excess grows with *d* because her measure-and-resend adds a second Fourier transform (up to 72 two-qubit gates) of its own noise.
- **The chip's noise has a basis fingerprint.** Q_F ≫ Q_Z (10.0% vs 1.6% at *d* = 8) because only the Fourier basis needs two-qubit gates — the same asymmetry experiment B uses to flag hardware faults.
- **The calibrated noise model was pessimistic for Kingston at *d* = 8** (predicted F = 0.692 from the Torino model; measured 0.727). No Kingston rehearsal was run: its noise model was too slow to simulate locally.

**Noise analysis: what each mitigation layer bought.**
- **Readout correction gave the largest, most consistent gain** at every *d* (d = 8: F 0.727 → 0.806; QBER 5.8% → 4.0%).
- **Dynamical decoupling + Pauli twirling helped the shallow circuits but not the deep ones.** At *d* = 2 it lowered QBER further (0.25% → 0.15%) and raised F to 0.994. At *d* = 8 it did worse than readout correction alone (F 0.775 vs 0.806; without readout correction, 0.698 vs 0.727 raw). Twirling turns coherent gate errors into stochastic ones, which can remove partial cancellations in a fixed compiled circuit; the inserted decoupling pulses add operations of their own. **The jobs ran 27 minutes apart, so calibration drift is confounded with the technique**; a same-job interleaved comparison is needed to separate them.

## 6. What these results cannot support
- **The chip is not a photonic link.** Gate and readout noise on a superconducting processor stand in for the channel. The hardware run validates the protocol, the eavesdropper circuit and the mitigation, not the performance of a real link.
- **Key rates are asymptotic and per sifted photon.** No finite-key effects, photon loss, detector efficiency or decoy states are modeled, so these are upper bounds, not throughput.
- **Qudit cost on qubits is an artifact.** The growth of *d* = 8 errors comes from compiling the Fourier transform into two-qubit gates; photonic qudits do not pay this cost. Conversely, real optics have losses we do not model.
- **The crosstalk model is idealized** (cyclic leakage to adjacent modes only).
- **Monitor results are on synthetic sessions** with classes we defined. The impossibility result covers intercept-resend vs depolarizing noise only; other attacks (e.g. detector blinding) leave signatures we did not model.
- **The quantum kernel was evaluated exactly in simulation**, not estimated on hardware. Hardware estimation would add shot noise and likely lower performance. We claim no quantum advantage.
- **The entanglement witness trusts the measurements** (not device-independent). Readout-corrected certificates depend on the readout model; raw values are reported alongside.
- **One run per configuration.** Error bars cover shot noise only, not drift between jobs or calibrations; the DD + twirl comparison is confounded with a 27-minute calibration gap.
- **Noise-model predictions are approximate.** The Torino model stood in for Kingston; it was pessimistic here, but noise models can err in either direction.

## 7. Next steps
- Train the monitor on hardware-derived sessions and add side-channel telemetry (timing, detector click patterns) so it can separate attacks from turbulence.
- Finite-key and decoy-state analysis; loss and detector models.
- Estimate the quantum kernel on hardware and measure how shot noise affects parity.
- A hybrid time-bin/mode encoding that shifts information to whichever basis is cleaner.

## 8. Run it
```bash
pip install qiskit qiskit-aer qiskit-ibm-runtime scikit-learn scipy matplotlib
python exp_ab_keyrate.py          # A, B      (~15 s)
python exp_c_monitor.py           # C         (~5 min; sessions cached in results/)
python exp_d_entanglement.py      # D         (~10 s)
python run_hardware.py --mode fake --backend ibm_torino --job both   # free rehearsal
python run_hardware.py --mode real --backend ibm_kingston --job both   # ~1 min QPU
python recover_job.py --job-id <ID> --job raw                         # if disconnected
python analyze_hardware.py --backend ibm_kingston --fake fake_torino
```
Save your IBM credentials locally with `QiskitRuntimeService.save_account(...)`. Never commit a token.

## License
MIT. See `LICENSE`.

## Files
`qudit_bb84.py` protocol and channels · `entanglement.py` witness · `exp_*.py` experiments · `run_hardware.py` / `recover_job.py` / `analyze_hardware.py` hardware pipeline · `style.py` plotting · `figs/` · `results/`

## References
- Cerf, Bourennane, Karlsson, Gisin, "Security of quantum key distribution using d-level systems," *Phys. Rev. Lett.* 88, 127902 (2002).
- Bavaresco *et al.*, "Measurements in two bases are sufficient for certifying high-dimensional entanglement," *Nature Physics* 14, 1032 (2018).
- Shaydulin & Wild, "Importance of kernel bandwidth in quantum machine learning," *Phys. Rev. A* 106, 042407 (2022).
- Havlíček *et al.*, "Supervised learning with quantum-enhanced feature spaces," *Nature* 567, 209 (2019).

_AI assistance: code and analysis were developed with help from Claude (Anthropic); the author reviewed, ran and is responsible for all results._
