import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from airfoil import naca4_airfoil, blunt_airfoil
from flow_sim import panel_flow

# --- Main Window ---
root = tk.Tk()
root.title("Airflow Optimization Mini-Game")
root.geometry("900x600")

# --- Matplotlib Figure ---
fig, ax = plt.subplots(figsize=(6, 4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# --- Controls Frame ---
control_frame = tk.Frame(root)
control_frame.pack(side=tk.RIGHT, fill=tk.Y)

# --- Airfoil selection ---
tk.Label(control_frame, text="Select Airfoil:").pack(pady=5)
airfoil_var = tk.StringVar(value="NACA 2412")
ttk.Combobox(control_frame, textvariable=airfoil_var,
             values=["NACA 0012", "NACA 2412", "NACA 4412", "Blunt"]).pack(pady=5)

# --- Blunt wing style selection ---
tk.Label(control_frame, text="Blunt Wing Style:").pack(pady=5)
blunt_style_var = tk.StringVar(value="rectangle")  # default style
ttk.Combobox(control_frame, textvariable=blunt_style_var,
             values=["rectangle", "rounded"]).pack(pady=5)

# Angle of Attack
tk.Label(control_frame, text="Angle of Attack (deg):").pack(pady=5)
aoa_var = tk.DoubleVar(value=0.0)
tk.Scale(control_frame, variable=aoa_var, from_=-10, to=20, orient=tk.HORIZONTAL, length=200).pack(pady=5)

# Cp toggle
show_cp_var = tk.BooleanVar()
tk.Checkbutton(control_frame, text="Show Cp Plot", variable=show_cp_var).pack(pady=5)


# --- Update Plot Function ---
# Keep a global colorbar reference
cb = None

def update_plot():
    global cb
    ax.clear()
    code = airfoil_var.get()
    alpha = aoa_var.get()

    # Generate airfoil coordinates
    if "Blunt" in code:
        xu, yu, xl, yl = blunt_airfoil(alpha, style=blunt_style_var.get())
    else:
        naca_code = code.split()[1]
        xu, yu, xl, yl = naca4_airfoil(naca_code, alpha=alpha)

    # Grid
    X, Y = np.meshgrid(np.linspace(-1, 2, 200), np.linspace(-1, 1, 100))
    u, v, V, Cp = panel_flow(X, Y, xu, yu, xl, yl, U_inf=1.0, gamma=0.05)

    # AoA effect on velocity
    alpha_rad = np.radians(alpha)
    V_mod = V * (1 + 0.15 * np.sin(alpha_rad))

    # Remove previous colorbar if it exists
    if cb:
        cb.remove()
        cb = None

    if show_cp_var.get():
        cp_plot = ax.contourf(X, Y, Cp, levels=50, cmap='coolwarm')
        ax.fill_between(xu, yu - 0.001, yu + 0.001, color='k', alpha=0.7)
        ax.fill_between(xl, yl - 0.001, yl + 0.001, color='k', alpha=0.7)
        ax.set_title(f"{code} - Cp Plot")
        cb = fig.colorbar(cp_plot, ax=ax)
    else:
        ax.streamplot(X, Y, u, v, density=1.5, color=V_mod, cmap='plasma', linewidth=2)
        ax.fill_between(xu, yu - 0.001, yu + 0.001, color='k', alpha=0.7)
        ax.fill_between(xl, yl - 0.001, yl + 0.001, color='k', alpha=0.7)
        ax.set_title(f"{code} - Streamlines (AoA effect)")

    canvas.draw()



tk.Button(control_frame, text="Update", command=update_plot).pack(pady=20)

root.mainloop()
