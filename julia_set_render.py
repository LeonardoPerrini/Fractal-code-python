#Leonardo Perrini SSAS 2023

'''Questo programma Python utilizza la libreria Tensorflow per il calcolo parallelo su GPU per creare
e renderizzare immagini di insiemi di Julia. Le immagini sono colorate tramite dei semplici 
algoritmi di coloring RGB o HSV sinusoidali'''

import tensorflow as tf
import numpy as np
import PIL.Image
from tqdm import tqdm


def render_hsv(x):
  x_norm = (x*0.005).reshape(list(x.shape)+[1]) 
  img = np.concatenate([360*np.sin(x_norm), 255*np.ones_like(x_norm), 200*np.ones_like(x_norm)], 2) 
  img = np.uint8(np.mod(img, 360))     
  return PIL.Image.fromarray(img,'HSV')


def render_rgb(x):
  x_norm = (x*0.02).reshape(list(x.shape)+[1])     
  img = np.concatenate([255*np.cos(3*x_norm), 255*np.sin(3*x_norm), 255*np.cos(5*x_norm)], 2)
  img = np.uint8(np.clip(img, 0, 255))     
  return PIL.Image.fromarray(img, 'RGB')


def julia(seed,render_size,center,zoom,cycles,escape_radius):
  f = zoom/render_size[0]     
  real_start = center[0] - (render_size[0]/2)*f     
  real_end = real_start + render_size[0]*f     
  imag_start = center[1] - (render_size[1]/2)*f    
  imag_end = imag_start + render_size[1]*f     
  real_range = tf.range(real_start,real_end,f,dtype=tf.float64)     
  imag_range = tf.range(imag_start,imag_end,f,dtype=tf.float64)     
  real, imag = tf.meshgrid(real_range,imag_range)     
  z = tf.Variable(tf.complex(real, imag))        
  counts = tf.Variable(tf.zeros_like(z, tf.float64))     
  z_bis = tf.Variable(z)     
  matrix_diverged = tf.zeros_like(z, tf.float64)     
  
  for i in range(cycles):
      
    z = z*z + seed     #z(n+1)=z(n)^2+c solo che qui z0 non parte da 0 come per il mandelbrot ma z sono tutti i numeri complessi sul piano mentre c qui è fissata per tutti i punti
      
    not_diverged = (tf.abs(z) <= escape_radius)     
    not_diverged_ = tf.cast(not_diverged, tf.float64)
    counts = counts + not_diverged_     
      
    z_bis = z_bis*z_bis + seed     #z(n+1)=z(n)^2+c un'altra volta
      
    diverged = (tf.abs(z_bis) > escape_radius)
    diverged = tf.cast(diverged , tf.float64)     
      
    n_diverged = (tf.abs(z_bis) <= escape_radius)     
    n_diverged = tf.cast(n_diverged , tf.float64)
      
    matrix_diverged = matrix_diverged + tf.math.multiply_no_nan(tf.abs(z_bis), diverged) 
      
    matrix_not_diverged_complex = tf.cast(n_diverged, tf.complex128)
    z_bis = tf.math.multiply_no_nan(z_bis, matrix_not_diverged_complex)     
  matrix_diverged = matrix_diverged + not_diverged_*escape_radius    
  return (counts.numpy() + np.log(np.log(escape_radius))/np.log(2) - np.log(np.log(matrix_diverged.numpy()))/np.log(2))   


zoom = 4.0


#valori interessanti per i seed:  (-0.70176-0.3842j) , (0.285+0.01j)
for i in tqdm(range(3000)):
  counts = julia((-0.70176 -0.3842j),(3840,2160),(0,0),zoom,500,10000000000000000000)    
  zoom *= 0.99
  img = render_rgb(counts)
  #img = img.convert('RGB')     #per l'HSV
  img.save(f"julia-{i}.png")

print("zoom totale: ",zoom)
