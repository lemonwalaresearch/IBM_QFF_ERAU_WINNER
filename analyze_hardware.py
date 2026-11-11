"""
Analyze hardware (or fake-backend) result files from run_hardware.py.

Layers compared (whichever files exist):
  Calibrated-noise prediction   results/hw_fake_<fake>_raw.json
  Hardware, raw                 results/hw_real_<backend>_raw.json
  Hardware, + readout fix       same file, tensored readout-error inversion
  Hardware, DD + twirl + readout  results/hw_real_<backend>_mitigated.json

Outputs figs/E_hardware.png and results/hardware_summary.json.
Usage: python analyze_hardware.py --backend ibm_torino
"""
import argparse
import json
from functools import reduce
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import style
from qudit_bb84 import key_rate, threshold_qber, theory_qber_ir
from entanglement import certify

style.apply()
FAKE = {"ibm_torino": "fake_torino", "ibm_fez": "fake_fez", "ibm_marrakesh": "fake_marrakesh",
        "ibm_kingston": "fake_kingston", "ibm_aachen": "fake_aachen", "ibm_brisbane": "fake_brisbane"}


# ------------------------------------------------------------- readout correction
def readout_model(rows):
    cal = {r["meta"]["state"]: r for r in rows if r["meta"]["kind"] == "cal"}
    phys = cal[0]["meta"]["physical"]
    flips = {}
    for i, p in enumerate(phys):
        e = []
        for s in (0, 1):
            cnt = cal[s]["cal"]; tot = sum(cnt.values())
            wrong = sum(c for b, c in cnt.items() if int(b[::-1][i]) != s)
            e.append(wrong / tot)
        flips[p] = e  # [P(1|0), P(0|1)]
    return flips


def correct(probs: np.ndarray, phys_bits: list, flips: dict) -> np.ndarray:
    """probs indexed by integer x = sum b_i 2^i, bit i measured on phys_bits[i]."""
    mats = [np.array([[1 - flips[p][0], flips[p][1]], [flips[p][0], 1 - flips[p][1]]]) for p in phys_bits]
    M = reduce(np.kron, mats[::-1])
    q = np.linalg.solve(M, probs)
    q = np.clip(q, 0, None)
    return q / q.sum()


# ------------------------------------------------------------- per-layer summary
def summarize(rows, use_ro):
    flips = readout_model(rows) if use_ro else None
    out = {"qkd": {}, "witness": {}, "gates": {}}
    for d in (2, 4, 8):
        for eve in (None, "Z", "F"):
            hist = {"Z": np.zeros(d), "F": np.zeros(d)}
            sel = [r for r in rows if r["meta"]["kind"] == "qkd" and r["meta"]["d"] == d
                   and r["meta"]["eve"] == eve]
            if not sel:
                continue
            for r in sel:
                p = np.zeros(d)
                for b, c in r["bob"].items():
                    p[int(b, 2)] += c
                p /= p.sum()
                if use_ro:
                    p = correct(p, r["meta"]["measured"]["bob"], flips)
                k = r["meta"]["k"]
                hist[r["meta"]["basis"]] += np.roll(p, -k)  # index = (bob - k) mod d
            qz, qf = 1 - hist["Z"][0] / hist["Z"].sum(), 1 - hist["F"][0] / hist["F"].sum()
            out["qkd"][f"d{d}_eve{eve}"] = {"Q_Z": qz, "Q_F": qf, "rate": key_rate(qz, qf, d),
                                            "lambda_eff": (qz + qf) / 2 * d / (d - 1)}
        twoq = [r["meta"]["two_q"] for r in rows if r["meta"]["kind"] == "qkd" and r["meta"]["d"] == d
                and r["meta"]["eve"] is None]
        out["gates"][d] = {"qkd_2q_mean": float(np.mean(twoq))}
        ps = {}
        for r in rows:
            m = r["meta"]
            if m["kind"] != "witness" or m["d"] != d:
                continue
            P = np.zeros(d * d)
            for ab, c in r["joint"].items():
                a, b = map(int, ab.split(","))
                P[a + d * b] += c
            P /= P.sum()
            if use_ro:
                P = correct(P, m["measured"]["ma"] + m["measured"]["mb"], flips)
            ps[m["setting"]] = sum(P[a + d * a] for a in range(d))
            out["gates"][d][f"witness_{m['setting']}_2q"] = m["two_q"]
        if ps:
            out["witness"][d] = certify(ps["ZZ"], ps["FF"], d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ibm_torino")
    ap.add_argument("--fake", default=None, help="fake backend name for the prediction layer, e.g. fake_torino")
    args = ap.parse_args()
    fake = args.fake or FAKE[args.backend]
    files = {
        f"Prediction\n({fake.replace('fake_', '')} noise model)": (f"results/hw_fake_{fake}_raw.json", False),
        f"{args.backend},\nraw": (f"results/hw_real_{args.backend}_raw.json", False),
        f"{args.backend},\n+ readout fix": (f"results/hw_real_{args.backend}_raw.json", True),
        f"{args.backend}, DD +\ntwirl + readout": (f"results/hw_real_{args.backend}_mitigated.json", True),
    }
    layers = {}
    for label, (path, ro) in files.items():
        if Path(path).exists():
            layers[label] = summarize(json.load(open(path))["rows"], ro)
    if not any(k.startswith(args.backend) for k in layers):  # rehearsal: show the fake pipeline twice
        p = f"results/hw_fake_{fake}_raw.json"
        layers["Calibrated-noise sim\n+ readout fix"] = summarize(json.load(open(p))["rows"], True)
        p = f"results/hw_fake_{fake}_mitigated.json"
        if Path(p).exists():
            layers["Calibrated-noise sim,\nmitigation pipeline"] = summarize(json.load(open(p))["rows"], True)
    names = list(layers)
    palette = [style.MUTED, style.SERIES[0], style.SERIES[2], style.SERIES[1]][:len(names)]

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    ds = np.array([2, 4, 8]); w = 0.8 / len(names)
    for i, (lab, L) in enumerate(layers.items()):
        x = np.arange(3) + (i - (len(names) - 1) / 2) * w
        q = [(L["qkd"][f"d{d}_eveNone"]["Q_Z"] + L["qkd"][f"d{d}_eveNone"]["Q_F"]) / 2 for d in ds]
        r = [L["qkd"][f"d{d}_eveNone"]["rate"] for d in ds]
        k = [L["witness"][d]["fidelity_lower_bound"] if d in L["witness"] else np.nan for d in ds]
        for ax, v in zip(axs, (q, r, k)):
            ax.bar(x, v, w, color=palette[i], edgecolor=style.SURFACE, linewidth=2, label=lab, zorder=3)
    for j, d in enumerate(ds):
        axs[0].plot([j - .45, j + .45], [threshold_qber(d)] * 2, color=style.CRITICAL, lw=1.5, zorder=6)
        axs[2].plot([j - .45, j + .45], [1 / d] * 2, color=style.CRITICAL, lw=1.5, zorder=6)
        axs[1].plot([j - .45, j + .45], [np.log2(d)] * 2, color=style.INK2, lw=1, ls=(0, (3, 2)), zorder=6)
    axs[0].text(2.47, threshold_qber(8) + .005, "abort\nthreshold", fontsize=7.5, color=style.INK2, ha="right")
    axs[2].text(2.47, 1 / 8 + .02, "entanglement\nthreshold 1/d", fontsize=7.5, color=style.INK2, ha="right")
    axs[1].text(2.47, 3.03, "ideal log₂d", fontsize=7.5, color=style.INK2, ha="right")
    for ax, t, yl in zip(axs, ("Error rate on the chip-as-channel", "Secret key per sifted photon",
                               "Entanglement fidelity lower bound"),
                         ("Average QBER", "bits / photon", "F ≥ P_Z + P_F − 1")):
        ax.set_xticks(range(3), [f"d = {d}" for d in ds]); ax.set_title(t, fontsize=11); ax.set_ylabel(yl)
        ax.grid(axis="x", visible=False)
    axs[1].set_ylim(0, 3.2); axs[2].set_ylim(-.05, 1.05)
    axs[1].legend(fontsize=7.5, loc="upper left")
    fig.savefig("figs/E_hardware.png"); plt.close(fig)

    json.dump({k.replace("\n", " "): v for k, v in layers.items()},
              open("results/hardware_summary.json", "w"), indent=1, default=float)
    for lab, L in layers.items():
        print(lab.replace("\n", " "))
        for d in ds:
            n = L["qkd"][f"d{d}_eveNone"]; e = L["qkd"].get(f"d{d}_eveZ")
            wv = L["witness"].get(d, {})
            print(f"  d={d}: QBER Z/F {n['Q_Z']:.3f}/{n['Q_F']:.3f}  rate {n['rate']:.2f}  "
                  f"λ_eff {n['lambda_eff']:.3f}  | Eve QBER {'%.3f' % ((e['Q_Z'] + e['Q_F']) / 2) if e else '-'}"
                  f" (theory+ {theory_qber_ir(1, d):.3f}) | witness F≥{wv.get('fidelity_lower_bound', float('nan')):.3f}"
                  f" → dim ≥ {wv.get('certified_schmidt_number', '-')}  | 2q {L['gates'][d]}")


if __name__ == "__main__":
    main()
