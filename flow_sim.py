import numpy as np

def potential_flow(X, Y, gamma=1.0, x0=0.5, y0=0.0, U_inf=1.0):
    """
    Compute potential flow around a point vortex.
    Returns velocity components, velocity magnitude, and Cp.
    """
    u = U_inf + gamma/(2*np.pi) * (Y - y0)/((X - x0)**2 + (Y - y0)**2)
    v = - gamma/(2*np.pi) * (X - x0)/((X - x0)**2 + (Y - y0)**2)
    V = np.sqrt(u**2 + v**2)
    Cp = 1 - (V/U_inf)**2
    return u, v, V, Cp

def panel_flow(X, Y, xu, yu, xl, yl, U_inf=1.0, gamma=1.0):
    """
    Simple vortex panels along the airfoil for visual effect.
    Returns u, v, V, Cp.
    """
    u = U_inf * np.ones_like(X)
    v = np.zeros_like(X)

    # Combine upper and lower surfaces
    x_points = np.concatenate([xu, xl[::-1]])
    y_points = np.concatenate([yu, yl[::-1]])

    # Each panel adds a small vortex
    for x0, y0 in zip(x_points, y_points):
        dx = X - x0
        dy = Y - y0
        r2 = dx**2 + dy**2 + 1e-5  # avoid divide by zero
        u += gamma/(2*np.pi) * dy / r2
        v += -gamma/(2*np.pi) * dx / r2

    V = np.sqrt(u**2 + v**2)
    Cp = 1 - (V / U_inf)**2
    return u, v, V, Cp