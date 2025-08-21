"""
Toy model for the ego-centric stability dynamics.
- m layers, each of dimension d
- Intra-layer contraction: a_{t+1}^{(j)} += (1 - eps_j) * a_t^{(j)}
- Cross-layer coupling:   a_{t+1}^{(j)} += nu * gamma * (a_t^{(j-1)} + a_t^{(j+1)})
- Add small Gaussian noise (optional) to avoid degenerate trajectories.
Anchors set to 0 for simplicity: V(a) = sum_j ||a^{(j)}||^2.
"""

from dataclasses import dataclass
import numpy as np
from typing import Dict, Tuple

@dataclass
class Params:
    m: int = 3
    d: int = 128
    eps: Tuple[float, float, float] = (0.1, 0.2, 0.3)  # epsilon_j per layer
    nu: float = 0.02
    gamma: float = 1.0
    steps: int = 300
    noise_std: float = 0.0
    seed: int = 1

def simulate(p: Params) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(p.seed)

    # state a^{(j)}_t for j=1..m: shape (m, d)
    a = rng.normal(0.0, 1.0, size=(p.m, p.d))
    traj = np.zeros((p.steps+1, p.m, p.d))
    traj[0] = a

    eps = np.asarray(p.eps, dtype=float)
    assert len(eps) == p.m, "len(eps) must equal m"

    for t in range(p.steps):
        a_next = np.zeros_like(a)
        for j in range(p.m):
            # intra-layer contraction
            a_next[j] += (1.0 - eps[j]) * a[j]
            # cross-layer coupling
            if j-1 >= 0:
                a_next[j] += p.nu * p.gamma * a[j-1]
            if j+1 < p.m:
                a_next[j] += p.nu * p.gamma * a[j+1]
        # noise
        if p.noise_std > 0.0:
            a_next += rng.normal(0.0, p.noise_std, size=a.shape)

        a = a_next
        traj[t+1] = a

    return {"traj": traj, "eps": eps, "nu": p.nu, "gamma": p.gamma}
