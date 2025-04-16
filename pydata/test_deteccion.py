import numpy as np
#!pip install circle_fit
import matplotlib.pyplot as plt
plt.ion()
import circle_fit
from skimage import feature
from skimage.io import imread
from skimage import filters
from skimage.measure import regionprops, label
from circle_fit import taubinSVD

#%%
#image = imread('structure.bmp').astype('double')

#%%

#image = imread('/Users/mateoeljatib/Documents/structure.bmp').astype(np.double)


#%%

def deteccion(image):

    imt = image < filters.threshold_otsu(image)/2


    # Compute the Canny filter for two values of sigma
    edges = feature.canny(imt, sigma=5)
    L = label(edges)

    props = regionprops(L)

    # Calcular función de costo y guardar (evitando divisiones por cero)
    props_with_cost = []
    for region in props:
        major = region.major_axis_length
        minor = region.minor_axis_length
        if major > 0:  # evitar división por cero
            cost = region.area * (minor / major)
            props_with_cost.append((region.label, cost))

    # Ordenar por costo de mayor a menor
    props_sorted_by_cost = sorted(props_with_cost, key=lambda x: x[1], reverse=True)

    # Obtener los dos con mayor costo
    top_two_labels = [item[0] for item in props_sorted_by_cost[:2]]

    XYma = props[top_two_labels[0]-1].coords
    XYmi = props[top_two_labels[1]-1].coords

    # plt.figure()
    # plt.imshow(image)
    # plt.plot(XYma[:,1], XYma[:,0], 'r.')
    # plt.plot(XYmi[:,1], XYmi[:,0], 'r.')
    # plt.colorbar()



    ycma, xcma, rma, sigmama = taubinSVD(XYma)
    ycmi, xcmi, rmi, sigmimi = taubinSVD(XYmi)
    ###
    ny, nx = image.shape
    Y, X = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')

    # Distancia de cada píxel al centro de cada círculo
    dist_to_big = np.sqrt((Y - ycma)**2 + (X - xcma)**2)
    dist_to_small = np.sqrt((Y - ycmi)**2 + (X - xcmi)**2)

    # Máscara para quedarte solo con lo que está:
    # Fuera del círculo grande o dentro del chico
    mask = (dist_to_big > rma) | (dist_to_small < rmi)

    # Aplicar la máscara a la imagen
    masked_image = image.copy()
    masked_image[~mask] = 0  # Apagás lo que está entre ambos
###
    return masked_image, ycma, xcma, rma, ycmi, xcmi, rmi



#%%
'''
masked_image, ycma, xcma, rma, ycmi, xcmi, rmi = deteccion(image)


theta = np.linspace(0, 2*np.pi, 1000)

plt.figure()
plt.imshow(image)
plt.plot(xcma+rma*np.cos(theta), ycma+rma*np.sin(theta), 'r.-')
plt.plot(xcmi+rmi*np.cos(theta), ycmi+rmi*np.sin(theta), 'r.-')

#%%

ny, nx = image.shape
Y, X = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')

# Distancia de cada píxel al centro de cada círculo
dist_to_big = np.sqrt((Y - ycma)**2 + (X - xcma)**2)
dist_to_small = np.sqrt((Y - ycmi)**2 + (X - xcmi)**2)

# Máscara para quedarte solo con lo que está:
# Fuera del círculo grande o dentro del chico
mask = (dist_to_big > rma) | (dist_to_small < rmi)

# Aplicar la máscara a la imagen
masked_image = image.copy()
masked_image[~mask] = 0  # Apagás lo que está entre ambos

# Visualizar
plt.figure()
plt.imshow(masked_image, cmap='gray')
plt.title('Zona fuera de ambos círculos')
plt.show()

'''


