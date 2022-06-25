import tensorflow as tf
import numpy as np
import PIL.Image
from io import BytesIO
from IPython.display import Image, display
def render_hsv(a):
  a_cyclic = (a*0.01).reshape(list(a.shape)+[1]) #trasformo la matrice schermo in un tensore di profondità 1, valori belli per il coefficiente moltiplicativo di a sono: 0.02, 0.015, 0.009
  img = np.concatenate([360*np.sin(a_cyclic), 255*np.ones_like(a_cyclic), 255*np.ones_like(a_cyclic)], 2) #creo il tensore di profondità 3 concatenandone 3 sull'asse 2 (profondità)
  img[a==a.max()] = 0     #i valori massimi che corrispondono ai punti sicuramente dentro al mandelbrot vengono posti a 0 colorandoli di nero
  a = img
  a = np.uint8(np.mod(a, 360))     #riduce i valori della hue modulo 360
  f = BytesIO()
  return PIL.Image.fromarray(a,'HSV')
def render_rgb(a):
  a_cyclic = (a*0.07).reshape(list(a.shape)+[1])     #giocare con il valore per cui è moltiplicato a, anche sotto con i coseni e i seni
  img = np.concatenate([10+20*np.cos(a_cyclic), 30+50*np.sin(a_cyclic), 155-80*np.cos(a_cyclic)], 2)
  img[a==a.max()] = 0     #i valori massimi che corrispondono ai punti sicuramente dentro al mandelbrot vengono posti a 0 colorandoli di nero
  a = img
  a = np.uint8(np.clip(a, 0, 255))     #clippa i valori rgb maggiori di 255 a 255
  f = BytesIO()
  return PIL.Image.fromarray(a, 'RGB')
def mandelbrot(render_size,center,zoom,cycles,escape_radius):
  f = zoom/render_size[0]     #dimensione del pixel nel nostro mondo
  real_start = center[0]-(render_size[0]/2)*f     #inizio a sinistra dell'asse x
  real_end = real_start + render_size[0]*f     #fine a destra dell'asse x
  imag_start = center[1]-(render_size[1]/2)*f     #inizio in basso dell'asse y
  imag_end = imag_start + render_size[1]*f     #fine in alto dell'asse y
  real_range = tf.range(real_start,real_end,f,dtype=tf.float64)     #asse x diviso in step di f (pixel riscalato)
  imag_range = tf.range(imag_start,imag_end,f,dtype=tf.float64)     #asse y diviso in step di f (pixel riscalato)
  real, imag = tf.meshgrid(real_range,imag_range)     #piano cartesiano generato dagli assi nel quale l'unità base è un pixel quadrato di lato f
  grid_c = tf.constant(tf.complex(real, imag))     #converto il mio piano cartesiano in un piano complesso (valori costanti immutabili, è la c nell'iterazione)
  current_values = tf.Variable(tf.zeros_like(grid_c))     #questa sarà la mia z, pongo z0=0
  counts = tf.Variable(tf.zeros_like(grid_c, tf.float32))     #questa è la matrice che tiene conto delle iterazioni per ciascun pixel, inizia a 0
  z_values=tf.Variable(tf.zeros_like(grid_c))     #uguale a current_values ma come un altra variabile
  matrix_diverged=tf.zeros_like(grid_c, tf.float32)     #matrice di zeri in float non complessa
  for i in range(cycles):
      z = current_values*current_values + grid_c     #z(n+1)=z(n)^2+c
      not_diverged = tf.abs(z) <= escape_radius     #vincolo per cui un valore di zn non divergerà, è una matrice di variabili booleane (True, False)
      current_values.assign(z)     #assegno a z(n) il nuovo valore che è z(n+1) per proseguire l'iterazione
      counts.assign_add(tf.cast(not_diverged, tf.float32))     #aggiunge alla matrice di 0 counts, la matrice not_diverged convertita da booleana a float in modo che True=1 e False=0 (tiene così conto del numero intero di iterazioni)
      z_bis=z_values*z_values + grid_c     #z(n+1)=z(n)^2+c un'altra volta
      diverged=tf.cast(tf.abs(z_bis)>escape_radius , tf.float32)     #vincolo per cui un valore di zn divergerà superando in modulo l'escape radius stabilito in input, è una matrice di variabili booleane trasformate a float (True=1, False=0)
      n_diverged=tf.cast(tf.abs(z_bis)<=escape_radius , tf.float32)     #vincolo per cui un valore di zn non divergerà superando in modulo l'escape radius stabilito in input, è una matrice di variabili booleane trasformate a float (True=1, False=0)
      matrix_diverged=matrix_diverged+tf.cast(tf.abs(z_bis), tf.float32)*diverged     #inserisco in questa matrice che aggiornerò ogni ciclo i valori di abs(zn) che hanno superato l'escape radius
      z_bis=z_bis*tf.cast(n_diverged, tf.complex128)     #assegno a zn+1 il nuovo valore in modo da eliminare gli elementi che sono ora in matrix_diverged
      z_values.assign(z_bis)     #assegno a zn, zn+1 per ripetere l'iterazione
  matrix_diverged_final=matrix_diverged+tf.cast(tf.abs(z_bis), tf.float32)     #aggiungo a matrix_diverged i valori mancanti (i moduli di zn che non hanno superato l'escape radius)
  intermedio=tf.cast(matrix_diverged_final<escape_radius, tf.float32)*escape_radius     #una matrice intermedia di valore escape radius dove il modulo di zn era minore di escape radius e 0 altrove
  matrix_abs_final=matrix_diverged_final*tf.cast(matrix_diverged_final>escape_radius, tf.float32)+intermedio     #matrice finale contenente i valori del modulo di zn maggiori dell'escape radius e il valore dell'escape radius dove il modulo di zn era inferiore a quest'ultimo
  return counts.numpy()-np.log(np.log(matrix_abs_final)/np.log(escape_radius))/np.log(2)     #numero di iterazioni normalizzato (per rendere la colorazione smooth)
from tqdm import tqdm
zoom = 1.0
for i in tqdm(range(3000)):
  counts = mandelbrot((3840,2160),(-0.7746806106269039,-0.1374168856037867),zoom,2000,1000000000000000000)     #il numero massimo per l'escape radius è 1000000000000000000 dopodichè numpy dà -inf, punti interessanti: (-0.74453986035590838012,0.12172377389442482241), (-0.7746806106269039,-0.1374168856037867)
  zoom *=0.99
  img = render_hsv(counts)
  img=img.convert('RGB')     #per l'HSV
  img.save(f"output.png-{i}.png")
print(zoom)
