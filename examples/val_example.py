import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyval.val import val

def step(X,a= 0.02,w = 400):
    x0 = len(X)//2
    return 1 / (1 + np.exp(-a * (X - x0 + w/2))) * 1 / (1 + np.exp(a * (X - x0 - w/2)))
def gauss_sin(X,Y, A = 100, w = 0.05):
    return step(X)*step(Y)*A*np.sin(w*(X+Y))
def gauss(X,Y,A = 100, sigma=1024/6):
    N = np.max(X)
    return (A*np.exp(-(X-(N//2))**2/(2*sigma**2))*np.exp(-(Y-(N//2))**2/(2*sigma**2)))

# elegir la función deseada, y si se quiere cambiar los parámetros:
    # el integrador (k = 0 o k = 1)
    # la frecuencia de puntos N
    # la altura efectiva H
    # el espaciado de los puntos del patrón n
# en este caso, está todo definido para que el factor de calibración devuelto sea 1. Simplemente para la simulación.
    
X,Y, h,Ii, values,I0, calibration_factor = val(0, func = gauss_sin, centrado_si = False, mascara_si = True, X0 = 512, Y0 = 512, radio = 100)

fig, ax = plt.subplots(1,3, figsize = (10,4))

im0 = ax[0].imshow(Ii, origin='lower')
cbar = fig.colorbar(im0, ax=ax[0], orientation='horizontal')
ax[0].set_title('Patrón deformado')

im1 = ax[1].imshow(values, origin='lower')
cbar = fig.colorbar(im1, ax=ax[1], orientation='horizontal')
ax[1].set_title('Mapa devuelto por la FCD')



im2 = ax[2].imshow(values-h, origin='lower')
cbar = fig.colorbar(im2, ax=ax[2], orientation='horizontal')
ax[2].set_title(r'Resta de alturas: <' + f'{np.max(np.abs(values-h))*100/(np.max(np.abs(values))):.2f}%')

#%%
N = 1024
x = np.linspace(0, N, N, endpoint = False)
resta = values-h

# Crear figura con 3 subplots verticales
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axs[0].plot(x, Ii[512, :])
axs[0].set_title("Ii")

axs[1].plot(x, values[512,:])
axs[1].set_title("values")

axs[2].plot(x, resta[512, :])
axs[2].set_title("resta = values - h")

for ax in axs:
    ax.set_ylabel("x")

axs[-1].set_xlabel("índice horizontal")  # Solo el último eje tiene label x
fig.suptitle("Corte transversal con mascara", fontsize=16)

plt.tight_layout()
plt.show()







