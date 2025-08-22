# tests/test_toy.py
import numpy as np
from EAFCode.toy.model import Params, simulate
from EAFCode.toy.run_toy import spectral_radius_stable, empirical_stability

def test_spectral_matches_empirical_stable():
    p = Params(m=3, d=64, eps=(0.1, 0.2, 0.3), nu=0.03, gamma=1.0, steps=180, seed=1)
    out = simulate(p)
    V = np.sum(out["traj"]**2, axis=(1,2))
    slope = empirical_stability(V, last_frac=0.33)
    emp_ok = (slope < 0.0) or (V[-1] <= 1e-9)
    rho, spec_ok = spectral_radius_stable(p.eps, p.nu, p.gamma)
    assert spec_ok, f"spectral says unstable: rho={rho}"
    assert emp_ok,  f"empirical says unstable: slope={slope}, tail V={V[-1]}"

def test_spectral_detects_instability():
    p = Params(m=3, d=64, eps=(0.1, 0.2, 0.3), nu=0.20, gamma=1.0, steps=120, seed=2)
    out = simulate(p)
    V = np.sum(out["traj"]**2, axis=(1,2))
    slope = empirical_stability(V, last_frac=0.33)
    emp_ok = (slope < 0.0) or (V[-1] <= 1e-9)
    rho, spec_ok = spectral_radius_stable(p.eps, p.nu, p.gamma)
    assert not spec_ok, f"spectral says stable unexpectedly: rho={rho}"
    assert not emp_ok,  f"empirical says stable unexpectedly: slope={slope}, tail V={V[-1]}"

def test_threshold_is_conservative():
    eps_star, gamma = 0.1, 1.0
    threshold = eps_star / (4.0 * gamma)  # 0.025
    rho, spec_ok = spectral_radius_stable((0.1, 0.2, 0.3), 0.03, 1.0)
    assert 0.03 > threshold, "nu should be above sufficient threshold"
    assert spec_ok, "system should still be spectrally stable above the sufficient bound"
