import numpy as np


def sample_points(rng: np.random.Generator, n_points: int) -> np.ndarray:
    pts = rng.uniform(-1.0, 1.0, size=(n_points, 3)).astype(np.float32)
    return pts


def sample_params(rng: np.random.Generator) -> np.ndarray:
    a = rng.uniform(0.35, 0.85)
    b = rng.uniform(0.35, 0.85)
    c = rng.uniform(0.35, 0.85)
    e1 = rng.uniform(0.25, 1.5)
    e2 = rng.uniform(0.25, 1.5)
    return np.array([a, b, c, e1, e2], dtype=np.float32)

def superellipsoid_implicit(xyz: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    f(x,y,z) = ( (|x/a|^(2/e2) + |y/b|^(2/e2))^(e2/e1) + |z/c|^(2/e1) ) - 1
    f < 0  => inside
    f = 0  => on surface
    f > 0  => outside
    Returns:
        f: (N,) float32
    """
    a, b, c, e1, e2 = params.astype(np.float32)

    # Avoid division by zero if params ever get extremely small
    a = max(float(a), 1e-6)
    b = max(float(b), 1e-6)
    c = max(float(c), 1e-6)

    x = xyz[:, 0] / a
    y = xyz[:, 1] / b
    z = xyz[:, 2] / c

    term_xy = (np.abs(x) ** (2.0 / e2) + np.abs(y) ** (2.0 / e2)) ** (e2 / e1)
    term_z = (np.abs(z) ** (2.0 / e1))
    f = term_xy + term_z - 1.0
    return f.astype(np.float32)

def compute_sdf(xyz: np.ndarray, params: np.ndarray) -> np.ndarray:
    f = superellipsoid_implicit(xyz, params)
    a, b, c, _, _ = params.astype(np.float32)
    scale = float(min(a, b, c))
    sdf = (f * scale).astype(np.float32)

    return np.clip(sdf, -0.2, 0.2).astype(np.float32)