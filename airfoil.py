import numpy as np

def naca4_airfoil(code, n_points=200, alpha=0.0):
    """
    Generate coordinates for a NACA 4-digit airfoil.
    Returns rotated upper and lower surfaces.
    """
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0

    beta = np.linspace(0, np.pi, n_points)
    x = (1 - np.cos(beta)) / 2

    yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)

    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    for i in range(n_points):
        if x[i] < p and p != 0:
            yc[i] = m / (p**2) * (2*p*x[i] - x[i]**2)
            dyc_dx[i] = 2*m / (p**2) * (p - x[i])
        elif p != 0:
            yc[i] = m / ((1 - p)**2) * ((1 - 2*p) + 2*p*x[i] - x[i]**2)
            dyc_dx[i] = 2*m / ((1 - p)**2) * (p - x[i])
        else:
            yc[i] = 0
            dyc_dx[i] = 0

    theta = np.arctan(dyc_dx)
    xu = x - yt*np.sin(theta)
    yu = yc + yt*np.cos(theta)
    xl = x + yt*np.sin(theta)
    yl = yc - yt*np.cos(theta)

    # Rotate for angle of attack
    alpha_rad = np.radians(alpha)
    xu_rot = xu*np.cos(alpha_rad) - yu*np.sin(alpha_rad)
    yu_rot = xu*np.sin(alpha_rad) + yu*np.cos(alpha_rad)
    xl_rot = xl*np.cos(alpha_rad) - yl*np.sin(alpha_rad)
    yl_rot = xl*np.sin(alpha_rad) + yl*np.cos(alpha_rad)

    return xu_rot, yu_rot, xl_rot, yl_rot


def blunt_airfoil(alpha=0.0, style="rounded"):
    """
    Generate a fully rounded blunt wing (capsule-shaped) and rotate by alpha.
    Only style "rounded" is used here.
    """
    x = np.linspace(0, 1, 100)
    # Parameters
    width = 0.05  # max half-thickness
    nose_radius = 0.1
    tail_radius = 0.1

    y_upper = np.zeros_like(x)

    # Leading edge: half-circle
    for i, xi in enumerate(x):
        if xi <= nose_radius:
            y_upper[i] = np.sqrt(nose_radius ** 2 - (xi - nose_radius) ** 2)
        elif xi >= 1 - tail_radius:
            # Trailing edge: half-circle down to 0 at end
            y_upper[i] = np.sqrt(tail_radius ** 2 - (xi - (1 - tail_radius)) ** 2)
        else:
            # Midsection: flat
            y_upper[i] = width

    y_lower = -y_upper

    # Rotation
    alpha_rad = np.radians(alpha)
    xu_rot = x * np.cos(alpha_rad) - y_upper * np.sin(alpha_rad)
    yu_rot = x * np.sin(alpha_rad) + y_upper * np.cos(alpha_rad)
    xl_rot = x * np.cos(alpha_rad) - y_lower * np.sin(alpha_rad)
    yl_rot = x * np.sin(alpha_rad) + y_lower * np.cos(alpha_rad)

    return xu_rot, yu_rot, xl_rot, yl_rot



