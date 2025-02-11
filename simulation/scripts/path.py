import numpy as np

def sin_curve() -> np.ndarray:
    x = np.linspace(0, 2*np.pi, 500)
    y = np.sin(x)
    path = np.c_[x, y]
    return path

def step_curve() -> np.ndarray:
    # section 1
    x1 = np.linspace(0, 3, 300)
    y1 = np.zeros_like(x1)
    # section 2
    y2 = np.linspace(0, 3, 300)
    x2 = np.ones_like(y2) * 3.0
    # section 3
    x3 = np.linspace(3, 6, 300)
    y3 = np.ones_like(x3) * 3.0
    
    x = np.concatenate([x1, x2, x3])
    y = np.concatenate([y1, y2, y3])
    path = np.c_[x, y]
    
    return path

def z_curve() -> np.ndarray:
    # section 1
    x1 = np.linspace(0, 3, 300)
    y1 = np.zeros_like(x1)
    # section 2
    y2 = np.linspace(0, 3, 300)
    x2 = np.ones_like(y2) * 3.0
    # section 3
    x3 = np.linspace(3, 6, 300)
    y3 = np.ones_like(x3) * 3.0
    
    x = np.concatenate([x1, x2, x3])
    y = np.concatenate([y1, y2, y3])
    path = np.c_[x, y]
    
    return path
