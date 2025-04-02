import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyval.val import val

def Asin(X,Y, A=1, w = 3):
    return A*np.sin(w*X)

alturas = np.linspace(0.01, 1, 20)
all_data = []
vmin, vmax = np.inf, -np.inf

for  altura in alturas: 
    X,Y, h,Ii, values,I0 = val(Asin, k = 0,A = altura, centrado_si = True)
    data = values - h.T/np.max(h.T)
    all_data.append(data)
    vmin = min(vmin, np.min(data))
    vmax = max(vmax, np.max(data))

N = 500

fila_corte = N // 2  # Índice de la fila central

cmap = cm.gnuplot 
norm = mcolors.Normalize(vmin=np.min(alturas), vmax=np.max(alturas))  # Normalización

fig, ax = plt.subplots()
for altura, data in zip(alturas, all_data):
    x_vals = np.linspace(0, np.max(X), N)  # Eje X
    y_vals = data[fila_corte, :]  # Corte en la fila central
    color = cmap(norm(altura))  # Color según altura

    ax.plot(x_vals, y_vals, color=color, label=f"H = {altura:.2f}")

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Altura")

ax.set_xlabel("x")
ax.set_ylabel("Intensidad")
ax.set_title("Cortes en la fila central para distintas alturas")
ax.grid(linestyle = '--', alpha = 0.5)
ax.plot()
