# EAFCode/toy/run_toy.py
"""
Toy model runner (baseline + sweep) per la verifica della condizione di stabilità.
Genera:
- results/normalizers.json
- results/baseline_V.csv
- results/nu_sweep.csv
- results/run_nu_<value>.csv
"""

import argparse
import csv
import json
import os
import numpy as np

from .model import Params, simulate
from .metrics import V_series, composite_S

RESULTS_DIR = "results"


# ---------------------- util ----------------------

def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_normalizers(traj, T_list=(64,), q=0.95):
    """
    Calcola normalizzatori M_rad, M_path (quantili su A0-val).
    """
    rad_vals, path_vals = [], []
    a1 = traj[:, 0, :]  # layer 1
    for T in T_list:
        for t in range(0, len(a1) - T):
            u = np.linalg.norm(a1[t])
            v = np.linalg.norm(a1[t + T])
            rad_vals.append(abs(np.log(max(v, 1e-12)) - np.log(max(u, 1e-12))))
            path_vals.append(np.linalg.norm(a1[t + T] - a1[t]))
    M_rad = float(np.quantile(np.array(rad_vals), q)) if rad_vals else 1.0
    M_path = float(np.quantile(np.array(path_vals), q)) if path_vals else 1.0
    return M_rad, M_path


def empirical_stability(V: np.ndarray,
                        last_frac: float = 0.33,
                        vmin: float = 1e-9,
                        stable_slope_tol: float = -1e-10) -> float:
    """
    Restituisce la pendenza di log V sugli ultimi step.
    Se il tail è al di sotto del floor numerico, ritorna un piccolo valore negativo
    per contare come 'stabile' (contrattivo).
    """
    n0 = int(len(V) * (1.0 - last_frac))
    tail = V[n0:]
    if len(tail) < 5:
        return stable_slope_tol
    # tail numericamente "piatto" al floor => contrazione già esaurita
    if np.median(tail) <= vmin or tail[-1] <= vmin:
        return stable_slope_tol
    mask = tail > vmin
    if mask.sum() < 5:
        return stable_slope_tol
    x = np.arange(n0, len(V))[mask]
    y = np.log(tail[mask])
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def spectral_radius_stable(eps_tuple, nu, gamma):
    """
    Test spettrale per il toy lineare 3x3:
    stabile sse rho(M) < 1.
    """
    e1, e2, e3 = map(float, eps_tuple)
    M = np.array([[1.0 - e1, nu * gamma, 0.0],
                  [nu * gamma, 1.0 - e2, nu * gamma],
                  [0.0,        nu * gamma, 1.0 - e3]], dtype=float)
    vals = np.linalg.eigvals(M)
    rho = float(np.max(np.abs(vals)))
    return rho, (rho < 1.0)


# ---------------------- main ----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "sweep"], default="sweep")
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--eps", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--nu", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--noise_std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--T", type=int, default=64)
    args = parser.parse_args()

    ensure_dirs()

    if args.mode == "baseline":
        # A0: nu = 0 per normalizzatori
        p = Params(m=args.m, d=args.d, eps=tuple(args.eps), nu=0.0,
                   gamma=args.gamma, steps=args.steps,
                   noise_std=args.noise_std, seed=args.seed)
        out = simulate(p)
        traj = out["traj"]
        V = V_series(traj)

        # normalizzatori (A0-val)
        M_rad, M_path = compute_normalizers(traj, T_list=(args.T,))
        with open(os.path.join(RESULTS_DIR, "normalizers.json"), "w") as f:
            json.dump({"M_rad": M_rad, "M_path": M_path, "T": args.T}, f, indent=2)

        # serie V
        with open(os.path.join(RESULTS_DIR, "baseline_V.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "V"])
            for t, v in enumerate(V):
                w.writerow([t, float(v)])

        print(f"[baseline] M_rad={M_rad:.4f}, M_path={M_path:.4f}")
        return

    # --------- sweep ----------
    # carica o computa normalizzatori
    norm_path = os.path.join(RESULTS_DIR, "normalizers.json")
    if os.path.exists(norm_path):
        with open(norm_path) as f:
            norm = json.load(f)
        M_rad, M_path = norm["M_rad"], norm["M_path"]
    else:
        p0 = Params(m=args.m, d=args.d, eps=tuple(args.eps), nu=0.0,
                    gamma=args.gamma, steps=args.steps,
                    noise_std=args.noise_std, seed=args.seed)
        out0 = simulate(p0)
        M_rad, M_path = compute_normalizers(out0["traj"], T_list=(args.T,))

    # griglia nu
    nu_grid = np.linspace(0.0, 0.05, 11)  # 0.000, 0.005, ..., 0.050
    eps_star = float(min(args.eps))
    threshold = eps_star / (4.0 * args.gamma)

    sweep_csv = os.path.join(RESULTS_DIR, "nu_sweep.csv")
    with open(sweep_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["nu", "gamma", "eps_star",
                    "pred_stable", "emp_slope", "emp_stable",
                    "S_T", "rho", "spec_stable"])
        for nu in nu_grid:
            p = Params(m=args.m, d=args.d, eps=tuple(args.eps), nu=float(nu),
                       gamma=args.gamma, steps=args.steps,
                       noise_std=args.noise_std, seed=args.seed)
            out = simulate(p)
            traj = out["traj"]
            V = V_series(traj)

            # empirico
            slope = empirical_stability(V)
            emp_stable = (slope < 0.0) or (V[-1] <= 1e-9)

            # predizione del teorema (condizione sufficiente)
            pred_stable = (nu <= threshold)

            # composita S(T)
            S_T = composite_S(traj, args.T, M_rad, M_path)

            # test spettrale "verità a terra" per il toy lineare
            rho, spec_ok = spectral_radius_stable(tuple(args.eps), float(nu), args.gamma)

            # salva serie V per ogni nu
            run_csv = os.path.join(RESULTS_DIR, f"run_nu_{nu:.3f}.csv")
            with open(run_csv, "w", newline="") as fr:
                wr = csv.writer(fr)
                wr.writerow(["t", "V"])
                for t, v in enumerate(V):
                    wr.writerow([t, float(v)])

            # riga riepilogo
            w.writerow([f"{nu:.3f}", args.gamma, eps_star,
                        int(pred_stable), f"{slope:.6f}", int(emp_stable),
                        f"{S_T:.6f}", f"{rho:.6f}", int(spec_ok)])

    print(f"[sweep] wrote {sweep_csv} and per-run V series in results/")
    print(f"[theory] threshold nu* <= eps*/(4*gamma) = {threshold:.6f}")


if __name__ == "__main__":
    main()
