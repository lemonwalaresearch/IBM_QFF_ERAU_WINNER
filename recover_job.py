"""
Recover a finished IBM Runtime job by its ID and save it exactly like run_hardware.py does.
Use this if the laptop disconnected while run_hardware.py was waiting.

  python recover_job.py --job-id <ID> --job raw
  python recover_job.py --job-id <ID> --job mitigated

Circuit metadata (which qudit, basis, symbol, eavesdropper, physical qubits) is rebuilt
from the circuits stored with the job itself, so it matches what actually ran.
"""
import argparse
import json
import re
from datetime import datetime, timezone

from qiskit_ibm_runtime import QiskitRuntimeService

from run_hardware import measured_physical, extract


def meta_from_circuit(qc):
    name = qc.name
    m = re.fullmatch(r"qkd_d(\d+)_([ZF])_k(\d+)_eve(None|Z|F)", name)
    if m:
        d, basis, k, eve = m.groups()
        meta = {"kind": "qkd", "d": int(d), "basis": basis, "k": int(k), "eve": None if eve == "None" else eve}
    elif (m := re.fullmatch(r"wit_d(\d+)_(ZZ|FF)", name)):
        meta = {"kind": "witness", "d": int(m.group(1)), "setting": m.group(2)}
    elif (m := re.fullmatch(r"cal_([01])", name)):
        phys = [qc.find_bit(i.qubits[0]).index for i in qc.data if i.operation.name == "measure"]
        return {"kind": "cal", "state": int(m.group(1)), "physical": phys}
    else:
        raise ValueError(f"unrecognised circuit name {name!r}")
    ops = qc.count_ops()
    meta.update(measured=measured_physical(qc), depth=qc.depth(),
                two_q=int(ops.get("cz", 0) + ops.get("ecr", 0) + ops.get("cx", 0)))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--job", choices=["raw", "mitigated"], required=True)
    args = ap.parse_args()

    service = QiskitRuntimeService()
    jb = service.job(args.job_id)
    print(f"job {args.job_id}: status {jb.status()}")
    result = jb.result()  # waits if still running
    pubs = jb.inputs["pubs"]
    rows = []
    for pub, res in zip(pubs, result):
        qc = pub[0]
        shots = pub[2] if len(pub) > 2 and pub[2] is not None else res.data[next(iter(res.data))].num_shots
        meta = meta_from_circuit(qc)
        rows.append({"name": qc.name, "meta": meta, "shots": shots, **extract(res, meta)})
    usage = None
    try:
        usage = jb.metrics().get("usage", {}).get("quantum_seconds")
    except Exception:
        pass
    backend = jb.backend().name
    out = {"backend": backend, "mode": "real", "job": args.job, "job_id": args.job_id,
           "quantum_seconds": usage, "recovered": True,
           "utc": datetime.now(timezone.utc).isoformat(), "rows": rows}
    path = f"results/hw_real_{backend}_{args.job}.json"
    json.dump(out, open(path, "w"), indent=1, default=str)
    print(f"saved {path} with {len(rows)} circuits (QPU seconds: {usage})")


if __name__ == "__main__":
    main()
