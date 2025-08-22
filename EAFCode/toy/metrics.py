"""
Metrics for the toy model:
- V(a_t) = sum_j ||a_t^{(j)}||^2
- S_angle(T): 1 - cosine(a^{(1)}_t, a^{(1)}_{t+T}) averaged over valid t
- S_rad_norm, S_path_norm with normalizers M_rad and M_path
"""

import numpy as np

def V_series(traj: np.ndarray) -> np.ndarray:
    # traj: (steps+1, m, d)
    return np.sum(traj**2, axis=(1,2))

def cosine(u: np.ndarray, v: np.ndarray) -> float:
    du = np.linalg.norm(u); dv = np.linalg.norm(v)
    if du == 0 or dv == 0:
        return 0.0
    return float(np.dot(u, v) / (du * dv))

def S_angle(traj: np.ndarray, T: int) -> float:
    # average (1 - cosine) across t
    a1 = traj[:, 0, :]  # layer 1
    vals = []
    for t in range(0, len(a1) - T):
        vals.append(1.0 - cosine(a1[t], a1[t+T]))
    return float(np.mean(vals)) if vals else 0.0

def S_rad_norm(traj: np.ndarray, T: int, M_rad: float) -> float:
    a1 = traj[:, 0, :]
    vals = []
    for t in range(0, len(a1) - T):
        u = np.linalg.norm(a1[t]); v = np.linalg.norm(a1[t+T])
        val = min(1.0, abs(np.log(max(v,1e-12)) - np.log(max(u,1e-12))) / max(M_rad, 1e-12))
        vals.append(val)
    return float(np.mean(vals)) if vals else 0.0

def S_path_norm(traj: np.ndarray, T: int, M_path: float) -> float:
    a1 = traj[:, 0, :]
    vals = []
    for t in range(0, len(a1) - T):
        dist = np.linalg.norm(a1[t+T] - a1[t])
        val = min(1.0, dist / max(M_path, 1e-12))
        vals.append(val)
    return float(np.mean(vals)) if vals else 0.0

def composite_S(traj: np.ndarray, T: int, M_rad: float, M_path: float,
                w_angle=0.5, w_rad=0.25, w_path=0.25) -> float:
    sA = S_angle(traj, T)
    sR = S_rad_norm(traj, T, M_rad)
    sP = S_path_norm(traj, T, M_path)
    return w_angle*(1.0 - (1.0 - sA)) + w_rad*sR + w_path*sP  # equivalent to: w_angle*sA + ...
