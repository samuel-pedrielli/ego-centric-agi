"""
CLI for running the toy model:
- Baseline run to compute normalizers M_rad, M_path (95th percentiles)
- Sweep over nu and test predicted stability threshold vs empirical slope of log V
Writes:
- results/nu_sweep.csv
- results/run_nu_<value>.csv (per-run V series)
"""

import argparse, csv, json, os
import numpy as np
from .model import Params, simulate
from .metrics import V_series, composite_S, S_angle, S_rad_norm, S_path_norm

RESULTS_DIR = "results"

def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_normalizers(traj, T_list=(32,64), q=0.95):
    # Gather raw radial/path deltas to compute quantiles (A0 validation analogue)
    rad_vals, path_vals = [], []
    a1 = traj[:,0,:]
    for T in T_list:
        for t in range(0, len(a1)-T):
            u = np.linalg.norm(a1[t]); v = np.linalg.norm(a1[t+T])
            rad_vals.append(abs(np.log(max(v,1e-12)) - np.log(max(u,1e-12))))
            path_vals.append(np.linalg.norm(a1[t+T]-a1[t]))
    M_rad = float(np.quantile(np.array(rad_vals), q)) if rad_vals else 1.0
    M_path = float(np.quantile(np.array(path_vals), q)) if path_vals else 1.0
    return M_rad, M_path

def empirical_stability(V: np.ndarray, last_frac=0.33) -> float:
    # Return slope of log V over last third of the trajectory (negative = contracting)
    n0 = int(len(V) * (1.0 - last_frac))
    x = np.arange(n0, len(V))
    y = np.log(np.maximum(V[n0:], 1e-12))
    if len(x) < 5:
        return 0.0
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline","sweep"], default="sweep")
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--eps", type=float, nargs="+", default=[0.1,0.2,0.3])
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--nu", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--noise_std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--T", type=int, default=64)
    args = parser.parse_args()

    ensure_dirs()

    if args.mode == "baseline":
        p = Params(m=args.m, d=args.d, eps=tuple(args.eps), nu=0.0, gamma=args.gamma,
                   steps=args.steps, noise_std=args.noise_std, seed=args.seed)
        out = simulate(p)
        V = V_series(out["traj"])
        M_rad, M_path = compute_normalizers(out["traj"], T_list=(args.T,))
        with open(os.path.join(RESULTS_DIR, "normalizers.json"), "w") as f:
            json.dump({"M_rad": M_rad, "M_path": M_path, "T": args.T}, f, indent=2)
        with open(os.path.join(RESULTS_DIR, "baseline_V.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["t","V"])
            for t, v in enumerate(V): w.writerow([t, float(v)])
        print(f"[baseline] M_rad={M_rad:.4f}, M_path={M_path:.4f}")
        return

    # sweep mode
    # Load or compute normalizers
    norm_path = os.path.join(RESULTS_DIR, "normalizers.json")
    if os.path.exists(norm_path):
        with open(norm_path) as f:
            norm = json.load(f)
        M_rad, M_path = norm["M_rad"], norm["M_path"]
    else:
        # compute from a quick baseline
        p0 = Params(m=args.m, d=args.d, eps=tuple(args.eps), nu=0.0, gamma=args.gamma,
                    steps=args.steps, noise_std=args.noise_std, seed=args.seed)
        out0 = simulate(p0)
        M_rad, M_path = compute_normalizers(out0["traj"], T_list=(args.T,))

    # sweep nu values
    nu_grid = np.linspace(0.0, 0.05, 11)  # 0.00, 0.005, ..., 0.05
    eps_star = float(min(args.eps))
    threshold = eps_star / (4.0 * args.gamma)

    sweep_csv = os.path.join(RESULTS_DIR, "nu_sweep.csv")
    with open(sweep_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["nu","gamma","eps_star","pred_stable","emp_slope","emp_stable","S_T"])
        for nu in nu_grid:
            p = Params(m=args.m, d=args.d, eps=tuple(args.eps), nu=float(nu),
                       gamma=args.gamma, steps=args.steps, noise_std=args.noise_std, seed=args.seed)
            out = simulate(p)
            traj = out["traj"]
            V = V_series(traj)
            slope = empirical_stability(V)
            emp_stable = slope < 0.0
            pred_stable = (nu <= threshold)
            # simple composite S using already-computed normalizers
            S_T = composite_S(traj, args.T, M_rad, M_path)
            # write per-run V
            run_csv = os.path.join(RESULTS_DIR, f"run_nu_{nu:.3f}.csv")
            with open(run_csv, "w", newline="") as fr:
                wr = csv.writer(fr); wr.writerow(["t","V"])
                for t, v in enumerate(V): wr.writerow([t, float(v)])
            w.writerow([f"{nu:.3f}", args.gamma, eps_star, int(pred_stable), f"{slope:.6f}", int(emp_stable), f"{S_T:.6f}"])

    print(f"[sweep] wrote {sweep_csv} and per-run V series in results/")
    print(f"[theory] threshold nu* <= eps*/(4*gamma) = {threshold:.6f}")

if __name__ == "__main__":
    main()
