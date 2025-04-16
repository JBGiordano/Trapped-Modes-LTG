import matplotlib.pyplot as plt
import sys
import os
from skimage.io import imread
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
# Ahora podés importar la función deteccion
from pydata.test_deteccion import deteccion
from pyfcd.fcd import compute_height_map
#%%
image_path1 = os.path.join(BASE_DIR, 'structure.bmp')
image_raw = imread(image_path1).astype('double')

image_path2 = os.path.join(BASE_DIR, 'prueba1.tif')
reference = imread(image_path2).astype('double')
#%%
#reference = 

# TODO: esto es, en orden desde abajo hasta la cámara, altura e índice del medio
layers = [[5.7e-2,1.0003], [ 1.2e-2,1.48899], [4.3e-2,1.34], [ 80e-2 ,1.0003]]

masked_image = deteccion(image_raw)[0]

square_size = 0.022

out = compute_height_map(reference, masked_image, square_size, layers)

#%%
plt.figure()
plt.imshow(out[0])
plt.colorbar()
plt.title('prueba1')