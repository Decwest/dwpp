import numpy as np
import math

def sin_curves() -> list:
    paths = []
    
    x = np.linspace(0, 2*np.pi, 500)
    
    # y = a*sin(bx)
    a_list = [0.5, 1.0, 1.5]
    b_list = [1.0, 2.0 ,3.0]
    
    for a in a_list:
        for b in b_list:
            y = a * np.sin(b * x)
            path = np.c_[x, y]
            paths.append(path)
            
    return paths

def step_curves() -> list:
    
    paths = []
    
    theta_list = [np.pi/4, np.pi/2, 3*np.pi/4]
    l_list = [2.0, 3.0, 4.0]
    
    for theta in theta_list:
        for l in l_list:
            # section 1
            x1 = np.linspace(0, 1, 100)
            y1 = np.zeros_like(x1)
            
            # section 2
            x2 = np.linspace(1.0, 1.0+l*math.cos(theta), 100)
            y2 = np.linspace(0.0, l*math.sin(theta), 100)
            
            # section 3
            x3 = np.linspace(1.0+l*math.cos(theta), 4.0+l*math.cos(theta), 100)
            y3 = np.ones_like(x3) * l * math.sin(theta)
            
            x = np.concatenate([x1, x2, x3])
            y = np.concatenate([y1, y2, y3])
            path = np.c_[x, y]
            paths.append(path)
    
    return paths

