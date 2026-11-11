"""
Run the qudit-link benchmark on IBM Quantum hardware (or a calibrated fake backend).

Two jobs, Qiskit Runtime SamplerV2 in job mode (works on the Open Plan):
  raw        QKD d=2,4,8 x {no Eve, Eve-Z, Eve-F} (84 circuits) + entanglement witness
             d=2,4,8 (6 circuits) + 2 readout-calibration circuits.  No error suppression.
  mitigated  Same no-Eve QKD circuits + witness + calibration, with dynamical decoupling
             and Pauli gate twirling.  Readout correction is applied afterwards in analysis
             from the calibration circuits (to both jobs, so its effect is separable).

Usage
  python run_hardware.py --mode fake --backend ibm_torino --job both       # rehearsal, free
  python run_hardware.py --mode real --backend ibm_torino --job raw --dry-run
  python run_hardware.py --mode real --backend ibm_torino --job raw        # spends QPU time

Credentials: save your IBM token once on YOUR machine with
  from qiskit_ibm_runtime import QiskitRuntimeService
  QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="...", instance="...")
Never paste the token into chat, code, or the repo.
"""
import argparse
import json
import time
from datetime import datetime, timezone

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2

from qudit_bb84 import QuditBB84
from entanglement import witness_circuits

QKD_SHOTS, WIT_SHOTS, CAL_SHOTS = 1000, 4000, 4000
FAKES = {"ibm_torino": "FakeTorino", "ibm_fez": "FakeFez", "ibm_marrakesh": "FakeMarrakesh",
         "ibm_kingston": "FakeKingston", "ibm_aachen": "FakeAachen", "ibm_brisbane": "FakeBrisbane"}


def logical_circuits():
    """(name, circuit, shots, meta) for every experiment circuit, before transpilation."""
    out = []
    for d in (2, 4, 8):
        q = QuditBB84(d)
        for eve in (None, "Z", "F"):
            for basis in ("Z", "F"):
                for k in range(d):
                    qc = q.circuit(k, basis, eve_basis=eve)
                    qc.name = f"qkd_d{d}_{basis}_k{k}_eve{eve}"
                    out.append((qc.name, qc, QKD_SHOTS,
                                {"kind": "qkd", "d": d, "basis": basis, "k": k, "eve": eve}))
    for d in (2, 4, 8):
        for setting, qc in witness_circuits(d).items():
            out.append((qc.name, qc, WIT_SHOTS, {"kind": "witness", "d": d, "setting": setting}))
    return out


def measured_physical(tqc):
    """{register name: [physical qubit of bit 0, bit 1, ...]} -- last measurement wins."""
    m = {}
    for inst in tqc.data:
        if inst.operation.name == "measure":
            pq = tqc.find_bit(inst.qubits[0]).index
            for reg, idx in tqc.find_bit(inst.clbits[0]).registers:
                m.setdefault(reg.name, {})[idx] = pq
    return {r: [v[i] for i in sorted(v)] for r, v in m.items()}


def calibration_circuits(backend, phys):
    cals = []
    for state in (0, 1):
        qc = QuantumCircuit(backend.num_qubits)
        cr = ClassicalRegister(len(phys), "cal"); qc.add_register(cr)
        for i, p in enumerate(phys):
            if state: qc.x(p)
            qc.measure(p, cr[i])
        qc.name = f"cal_{state}"
        cals.append((qc.name, qc, CAL_SHOTS, {"kind": "cal", "state": state, "physical": phys}))
    return cals


def build(backend, seed=11):
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=seed)
    items = []
    for name, qc, shots, meta in logical_circuits():
        t = pm.run(qc)
        ops = t.count_ops()
        meta = dict(meta, measured=measured_physical(t), two_q=int(ops.get("cz", 0) + ops.get("ecr", 0)
                    + ops.get("cx", 0)), depth=t.depth())
        items.append((name, t, shots, meta))
    phys = sorted({p for _, _, _, m in items for reg in m["measured"].values() for p in reg})
    return items + calibration_circuits(backend, phys)


def select(items, job):
    if job == "raw":
        return items
    return [it for it in items if it[3]["kind"] != "qkd" or it[3]["eve"] is None]


def extract(pub_result, meta):
    data = pub_result.data
    if meta["kind"] == "qkd":
        return {"bob": data.bob.get_counts()}
    if meta["kind"] == "witness":
        a, b = data.ma.get_bitstrings(), data.mb.get_bitstrings()
        joint = {}
        for x, y in zip(a, b):
            joint[f"{int(x, 2)},{int(y, 2)}"] = joint.get(f"{int(x, 2)},{int(y, 2)}", 0) + 1
        return {"joint": joint}
    return {"cal": data.cal.get_counts()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fake", "real"], required=True)
    ap.add_argument("--backend", default="ibm_torino")
    ap.add_argument("--job", choices=["raw", "mitigated", "both"], default="both")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.mode == "fake":
        import qiskit_ibm_runtime.fake_provider as fp
        backend = getattr(fp, FAKES[args.backend])()
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        backend = QiskitRuntimeService().backend(args.backend)
    print(f"backend: {backend.name}  ({backend.num_qubits} qubits)")

    t0 = time.time()
    items = build(backend)
    print(f"transpiled {len(items)} circuits in {time.time() - t0:.0f}s")
    for d in (2, 4, 8):
        q = [m for _, _, _, m in items if m["kind"] == "qkd" and m["d"] == d and m["eve"] is None]
        w = [m for _, _, _, m in items if m["kind"] == "witness" and m["d"] == d]
        print(f"  d={d}: QKD 2q gates {min(m['two_q'] for m in q)}-{max(m['two_q'] for m in q)}, "
              f"witness 2q gates {[m['two_q'] for m in w]}")

    jobs = ["raw", "mitigated"] if args.job == "both" else [args.job]
    for job in jobs:
        sel = select(items, job)
        shots = sum(s for _, _, s, _ in sel)
        print(f"[{job}] {len(sel)} circuits, {shots:,} shots, rough QPU estimate "
              f"{shots * 3.5e-4 + 0.05 * len(sel):.0f}s (the job's reported usage is the real number)")
        if args.dry_run:
            continue
        sampler = SamplerV2(mode=backend)
        if job == "mitigated":
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = "XpXm"
            sampler.options.twirling.enable_gates = True
            sampler.options.twirling.enable_measure = False
        t1 = time.time()
        jb = sampler.run([(c, None, s) for _, c, s, _ in sel])
        print(f"  submitted job {jb.job_id()}; waiting ...")
        result = jb.result()
        usage = None
        try:
            usage = jb.metrics().get("usage", {}).get("quantum_seconds")
        except Exception:
            pass
        rows = [{"name": n, "meta": m, "shots": s, **extract(r, m)}
                for (n, _, s, m), r in zip(sel, result)]
        out = {"backend": backend.name, "mode": args.mode, "job": job, "job_id": jb.job_id(),
               "quantum_seconds": usage, "wall_seconds": time.time() - t1,
               "utc": datetime.now(timezone.utc).isoformat(), "rows": rows}
        path = f"results/hw_{args.mode}_{backend.name}_{job}.json"
        json.dump(out, open(path, "w"), indent=1, default=str)
        print(f"  saved {path}  (QPU seconds reported: {usage})")


if __name__ == "__main__":
    main()
