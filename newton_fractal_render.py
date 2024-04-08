#Leonardo Perrini SSAS 2023

'''Questo programma Python utilizza la libreria Tensorflow per il calcolo parallelo su GPU per creare
e renderizzare immagini di frattali di Newton. Le immagini sono colorate tramite dei semplici 
algoritmi di coloring RGB o HSV interi'''

import tensorflow as tf
import numpy as np
import PIL.Image
from tqdm import tqdm

n = int(input("inserisci il grado del polinomio: "))
coeff = np.zeros(n + 1)
for cc in range(n + 1):
    print("inserisci il coefficiente del termine di grado ", cc)
    x = float(input(">"))
    coeff[n - cc] = x  # coefficienti polinomiali in grado decrescente

p1 = np.poly1d(coeff)  #creo un polinomio con i coefficienti acquisiti

roots = np.roots(coeff)  #trovo le radici (complesse) del polinomio 

pder = np.polyder(p1, 1)  #calcolo la derivata del polinomio


def render_hsv(x):
    x_norm = (x*0.4*np.pi/n).reshape(list(x.shape) + [1])
    img = np.concatenate([360 * np.cos(x_norm), 255 * np.ones_like(x_norm), 255 * np.ones_like(x_norm)], 2)
    img = np.uint8(np.mod(img, 360))
    return PIL.Image.fromarray(img, 'HSV')

def render_hsv2(x):
    x_norm = (x * 250/n).reshape(list(x.shape) + [1])
    img = np.concatenate([x_norm * np.ones_like(x_norm), 255 * np.ones_like(x_norm), 255 * np.ones_like(x_norm)], 2)
    img = np.uint8(np.mod(img, 360))
    return PIL.Image.fromarray(img, 'HSV')


def render_rgb(x):
    x_norm = (x).reshape(list(x.shape) + [1])
    img = np.concatenate(
        [255 * x_norm * (x_norm - 2.0) * (x_norm - 3.0), 255 * x_norm * (1.0 - x_norm) * (x_norm - 3.0),
         255 * x_norm * (x_norm - 2.0) * (x_norm - 1.0)], 2)
    img = np.uint8(np.clip(img, 0, 255))
    return PIL.Image.fromarray(img, 'RGB')


def newton(render_size, center, zoom, cycles):
    f = zoom / render_size[0]
    real_start = center[0] - (render_size[0] / 2) * f
    real_end = real_start + render_size[0] * f
    imag_start = center[1] - (render_size[1] / 2) * f
    imag_end = imag_start + render_size[1] * f
    real_range = tf.range(real_start, real_end, f, dtype=tf.float64)
    imag_range = tf.range(imag_start, imag_end, f, dtype=tf.float64)
    real, imag = tf.meshgrid(real_range, imag_range)
    grid_c = tf.constant(tf.complex(real, imag))
    z = tf.Variable(grid_c)

    # definisco gli array 2D con le varie radici del polinomio complesso, ROOTS è un array 3D (ogni elemento è una matrice)
    ROOTS = np.zeros((n, render_size[1], render_size[0]), dtype=np.complex128)
    for i in range(n):
        ROOTS[i] = roots[i] * np.ones_like(grid_c.numpy(), dtype=np.complex128)

    screen = np.ones((render_size[1], render_size[0])) 

    # iterazione che calcola le radici del polinomio assegnato tramite il metodo di Newton usando come seed iniziale ogni punto z del piano complesso
    for i in range(cycles):
        z = z - tf.math.divide_no_nan(np.polyval(p1, z), np.polyval(pder, z))

    # definisco la distanza fra z e le varie n radici (array 3D)
    D = np.zeros((n, render_size[1], render_size[0]), dtype=np.float64)
    for j in range(n):
        D[j] = tf.abs(z - ROOTS[j]).numpy()

    # per ogni z in C trovo la radice di P(z) da cui ha la minima distanza e assegno un valore da 1 a deg(p(z)) agli elementi della matrice screen
    for j in range(render_size[0]):
        for k in range(render_size[1]):
            counter = 0
            for ii in range(n):
                if D[ii][k][j]<D[counter][k][j]:
                    counter = ii
            screen[k][j] = counter + 1

    return screen


zoom = 4.0

for i in tqdm(range(3000)):
    counts = newton((1920, 1080), (0, 0), zoom, 200)
    zoom *= 0.99
    img = render_hsv2(counts)
    img = img.convert('RGB')  # per l'HSV
    img.save(f"newton-{i}.png")

print("zoom totale: ",zoom)
