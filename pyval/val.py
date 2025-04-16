'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pyfcd.fcd import compute_height_map
from pyfcd.fourier_space import wavenumber_meshgrid

def h_grad(h,x,y,k=0):  
    if k == 0:
        kx, ky = wavenumber_meshgrid(h.shape)
        h = h-np.mean(h)
        h_hat = np.fft.fft2(h)
        h_x_hat = 1j * kx * h_hat
        h_y_hat = 1j * ky * h_hat
        h_x = np.fft.ifft2(h_x_hat).real
        h_y = np.fft.ifft2(h_y_hat).real
        grad = np.stack((h_x, h_y), axis=-1)
    elif k ==1: 
        h_x, h_y = np.gradient(h, x, y, axis=(0, 1))
        grad = np.stack((h_x, h_y), axis=-1)
    else: 
        raise ValueError('Parámetro k entre 0 (pseudoespectral) y 1 (diferencias finitas)')
    return grad

def val(k,func = None,h = None, N = 1024,  H = 1, n = 60, centrado_si = False, mascara_si = False, X0=None, Y0=None, radio=None, *args, **kwargs):
    '''
    '''
    Parámetros necesarios:
        'func': función usada. Necesariamente tiene que ser función de X e Y, con estructura 'func(X,Y,*kwargs*)'
        'k': integrador a usar. k = 0 pseudoespectral, k = 1 diferencias finitas
        '**kwargs': parámetros necesarios de 'func', como amplitud, fase, frecuencia, o lo necesario.
    Parámetros opcionales (llamarlos con otro valor si se quiere modificarlos):
        'N = 1024': Grillado X e Y
        'H = 1': altura efectiva del agua. 
        'n = 60': número de cíclos del patrón de fondo
        'centrado_si = False': Centrar y normalizar los valores devueltos por la FCD
    '''
    '''
    square_size = N/(2*n)       #para querer que el factor de calibración sea 1, este es el tamaño del cuadrado
    
    x = np.linspace(0, N, N, endpoint = False)
    y = np.linspace(0, N, N, endpoint = False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    if func is not None:
        if h is None:
            h = func(X,Y,*args, **kwargs)
        else:
            raise Warning("Provide either h or func, not both.")
    else: 
        if h is None:
            raise Warning("Provide either h or func, not empty.")
        else: 
            h = h
    if mascara_si == True:
        if X0 is None or Y0 is None or radio is None:
            raise ValueError("Provide X0, Y0 and masks radius")
        mascara = (X - X0) ** 2 + (Y - Y0) ** 2 <= radio ** 2
        h[mascara] = 0 

    u = -H*h_grad(h,x,y,k=k)
    if mascara_si == True:
        if X0 is None or Y0 is None or radio is None:
            raise ValueError("Provide X0, Y0 and masks radius")
        mascara = (X - X0) ** 2 + (Y - Y0) ** 2 <= radio ** 2
        u[mascara] = 0 
    
    kx = 2 * np.pi * n / N
    ky = 2 * np.pi * n / N
    I0 = 0.5 + (np.sin(X *kx) * np.sin(Y * ky))/2
    r = np.stack((X, Y), axis=-1)
    r_prim = (r - u)  
    r_prim[..., 0] = np.clip(r_prim[..., 0], x.min(), x.max())
    r_prim[..., 1] = np.clip(r_prim[..., 1], y.min(), y.max())

    interp_I0 = RegularGridInterpolator((x, y), I0, bounds_error=False, fill_value=0)
    if mascara_si == True:
        if X0 is None or Y0 is None or radio is None:
            raise ValueError("Provide X0, Y0 and masks radius")
        mascara = (X - X0) ** 2 + (Y - Y0) ** 2 <= radio ** 2
        I0[mascara] = 0 
    I = interp_I0(r_prim.reshape(-1, 2)).reshape(N, N)
    # Aplicar máscara si está activada
    mascara = np.ones_like(I, dtype=bool)
    if mascara_si == True:
        if X0 is None or Y0 is None or radio is None:
            raise ValueError("Provide X0, Y0 and masks radius")
        mascara = (X - X0) ** 2 + (Y - Y0) ** 2 <= (radio + 15) ** 2
        I[mascara] = 0 
    #I = 0.5 + (np.cos(r_prim[..., 0] * kx) * np.cos(r_prim[..., 1] * ky)) / 2    #interpolo directamente evaluando en el patron
    values = compute_height_map(I0, I, square_size=square_size, height=H)    #tuple = (height_map, phases, calibration_factor)
    calibration_factor = values[2]
    if mascara_si == True:
        if X0 is None or Y0 is None or radio is None:
            raise ValueError("Provide X0, Y0 and masks radius")
        mascara = (X - X0) ** 2 + (Y - Y0) ** 2 <= (radio + 15) ** 2
        values[0][mascara] = 0 
    if centrado_si == True:
        values[0] = centrado(values[0])
    else:
        pass
    return X,Y,h, I, values[0], I0, calibration_factor

def centrado(v):
    meanh = (np.max(v) + np.min(v))*0.5
    difh = (np.max(v) - np.min(v))*0.5
    return (v-meanh)/difh

'''


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from pyfcd.fcd import compute_height_map
from pyfcd.fourier_space import wavenumber_meshgrid

def h_grad(h,x,y,k=0):  
    if k == 0:
        kx, ky = wavenumber_meshgrid(h.shape)
        h = h - np.mean(h)
        h_hat = np.fft.fft2(h)
        h_x_hat = 1j * kx * h_hat
        h_y_hat = 1j * ky * h_hat
        h_x = np.fft.ifft2(h_x_hat).real
        h_y = np.fft.ifft2(h_y_hat).real
        grad = np.stack((h_x, h_y), axis=-1)
    elif k == 1: 
        h_x, h_y = np.gradient(h, x, y, axis=(0, 1))
        grad = np.stack((h_x, h_y), axis=-1)
    else: 
        raise ValueError('Parámetro k entre 0 (pseudoespectral) y 1 (diferencias finitas)')
    return grad

def soft_mask(X, Y, X0, Y0, radio, sigma=10):
    """Returns a soft-edged mask via Gaussian blur of a binary circular mask."""
    binary_mask = ((X - X0) ** 2 + (Y - Y0) ** 2 <= radio ** 2).astype(float)
    blurred_mask = gaussian_filter(binary_mask, sigma=sigma)
    return blurred_mask

def centrado(v):
    meanh = (np.max(v) + np.min(v)) * 0.5
    difh = (np.max(v) - np.min(v)) * 0.5
    return (v - meanh) / difh

def val(k, func=None, h=None, N=1024, H=1, n=60, centrado_si=False, mascara_si=False, X0=None, Y0=None, radio=None, sigma_mask=15, *args, **kwargs):
    square_size = N / (2 * n)
    x = np.linspace(0, N, N, endpoint=False)
    y = np.linspace(0, N, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Initial field generation
    if func is not None:
        if h is None:
            h = func(X, Y, *args, **kwargs)
        else:
            raise Warning("Provide either h or func, not both.")
    else:
        if h is None:
            raise Warning("Provide either h or func, not empty.")

    # Create pattern
    kx = 2 * np.pi * n / N
    ky = 2 * np.pi * n / Ns
    I0 = 0.5 + (np.sin(X * kx) * np.sin(Y * ky)) / 2

    # Apply soft mask
    if mascara_si:
        if X0 is None or Y0 is None or radio is None:
            raise ValueError("Provide X0, Y0 and mask radius")

        mask = soft_mask(X, Y, X0, Y0, radio, sigma=sigma_mask)
        
        # Fade out h and pattern
        h = h * (1 - mask)
        I0 = I0 * (1 - mask)

    # Compute displacement
    u = -H * h_grad(h, x, y, k=k)

    if mascara_si:
        u = u * (1 - mask[..., np.newaxis])  # apply soft mask to each component

    # Apply displacement
    r = np.stack((X, Y), axis=-1)
    r_prim = r - u
    r_prim[..., 0] = np.clip(r_prim[..., 0], x.min(), x.max())
    r_prim[..., 1] = np.clip(r_prim[..., 1], y.min(), y.max())

    interp_I0 = RegularGridInterpolator((x, y), I0, bounds_error=False, fill_value=0)
    I = interp_I0(r_prim.reshape(-1, 2)).reshape(N, N)

    if mascara_si:
        # Optionally: fill in I with original pattern in masked areas (occlusion handling)
        # I = I * (1 - mask) + I0 * mask
        I = I * (1 - mask)  # or just fade the edges smoothly
        
    # Compute FCD
    values = compute_height_map(I0, I, square_size=square_size, height=H)
    height_map, _, calibration_factor = values

    if mascara_si:
        height_map = height_map * (1 - mask)

    if centrado_si:
        height_map = centrado(height_map)

    return X, Y, h, I, height_map, I0, calibration_factor






























