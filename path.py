import numpy as np

def sin_curve() -> np.ndarray:
    x = np.linspace(0, 2*np.pi, 500)
    y = np.sin(x)
    path = np.c_[x, y]
    return path
