#Leonardo Perrini SSAS 2023

'''Questo programma Python utilizza la libreria Tensorflow per il calcolo parallelo su GPU per creare
e renderizzare immagini di insiemi di Mandelbrot. Le immagini sono colorate tramite dei semplici 
algoritmi di coloring RGB o HSV sinusoidali'''

import tensorflow as tf
import numpy as np
import PIL.Image
from tqdm import tqdm


def render_hsv(x):
    val = x.max()
    x_norm = (x/val).reshape(list(x.shape)+[1])   #trasformo la matrice schermo in un tensore di profondità (rank) 1, valori visivamente accattivanti per il coefficiente moltiplicativo di a sono: 0.02, 0.015, 0.009 o in genere valori minori di 0.1
    img = np.concatenate([2000*x_norm, 255*np.ones_like(x_norm), 255*np.ones_like(x_norm)], 2)   #creo il tensore di profondità 3 concatenandone 3 sull'asse z (profondità)
    img[x==x.max()] = 0     #i valori massimi che corrispondono ai punti sicuramente dentro al mandelbrot vengono posti a 0 colorandoli di nero
    img = np.uint8(np.mod(img, 360))     #riduce i valori della hue modulo 360
    #a = np.uint8(np.clip(a, 0, 360))
    return PIL.Image.fromarray(img,'HSV')


def render_rgb(x):
    x_norm = (x*0.02).reshape(list(x.shape)+[1])     
    img = np.concatenate([255*np.cos(3*x_norm), 255*np.sin(3*x_norm), 255*np.cos(5*x_norm)], 2)  #una sorta di gradiente RGB "continuo e liscio" in modo da evitare bande di colore
    img[x==x.max()] = 0     #i valori massimi che corrispondono ai punti sicuramente dentro al mandelbrot vengono posti a 0 colorandoli di nero
    img = np.uint8(np.clip(img, 0, 255))     #clippa i valori rgb maggiori di 255 a 255
    return PIL.Image.fromarray(img, 'RGB')


def render_rgb2(x):
    val = x.max()
    x_norm = (x/val).reshape(list(x.shape)+[1])     
    img = np.concatenate([255*x_norm, 255*x_norm-30, 255*x_norm-57], 2)  #una sorta di gradiente RGB "continuo e liscio" in modo da evitare bande di colore
    img[x==x.max()] = 0     #i valori massimi che corrispondono ai punti sicuramente dentro al mandelbrot vengono posti a 0 colorandoli di nero
    img = np.uint8(np.clip(img, 0, 255))     #clippa i valori rgb maggiori di 255 a 255
    return PIL.Image.fromarray(img, 'RGB')


def mandelbrot(render_size,center,zoom,cycles,escape_radius):
    f = zoom/render_size[0]     #dimensione del pixel nel sistema
    real_start = center[0] - (render_size[0]/2)*f     #inizio a sinistra sull'asse Re(z)
    real_end = real_start + render_size[0]*f     #fine a destra sull'asse Re(z)
    imag_start = center[1] - (render_size[1]/2)*f     #inizio in basso sull'asse Im(z)
    imag_end = imag_start + render_size[1]*f     #fine in alto sull'asse Im(z)
    real_range = tf.range(real_start,real_end,f,dtype=tf.float64)     #asse Re diviso in step di f (pixel riscalato)
    imag_range = tf.range(imag_start,imag_end,f,dtype=tf.float64)     #asse Im diviso in step di f (pixel riscalato)
    real, imag = tf.meshgrid(real_range,imag_range)     #piano cartesiano generato dagli assi nel quale l'unità base è un pixel quadrato di lato f
    grid_c = tf.constant(tf.complex(real, imag, tf.complex128))     #converto il mio piano cartesiano in un piano complesso (valori costanti immutabili, è la c nell'iterazione)
    z = tf.Variable(tf.zeros_like(grid_c, tf.complex128))     #questa sarà la mia z, pongo z0=0
    counts = tf.Variable(tf.zeros_like(grid_c, tf.float64))     #questa è la matrice che tiene conto delle iterazioni per ciascun pixel, inizia a 0
    z_bis = tf.Variable(tf.zeros_like(grid_c, tf.complex128))     #uguale a z ma come un altra variabile identica e ausiliaria
    matrix_diverged = tf.zeros_like(grid_c, tf.float64)     #matrice di zeri in float64 non complessa
  
    for i in range(cycles):
        z = z*z + grid_c     #z(n+1)=z(n)^2+c iterazione (funzione olomorfa e anlitica) del Mandelbrot
        not_diverged = (tf.abs(z) <= escape_radius)     #vincolo per cui un valore di zn non divergerà, è una matrice di variabili booleane (True, False)
        not_diverged_ = tf.cast(not_diverged, tf.float64)  #converto in reali (0 e 1 per false e true)
        counts = counts + not_diverged_  #update matrice iterazioni
        z_bis = z_bis*z_bis + grid_c     #z(n+1)=z(n)^2+c un'altra volta ausiliaria
      
        diverged = (tf.abs(z_bis) > escape_radius)
        diverged = tf.cast(diverged , tf.float64)     #vincolo per cui un valore di zn divergerà superando in modulo l'escape radius stabilito in input, è una matrice di variabili booleane trasformate a float (True=1, False=0)
      
        n_diverged = (tf.abs(z_bis) <= escape_radius)     #vincolo per cui un valore di zn non divergerà, è una matrice di variabili booleane trasformate a float (True=1, False=0)
        n_diverged = tf.cast(n_diverged , tf.float64)
      
        matrix_diverged = matrix_diverged + tf.math.multiply_no_nan(tf.abs(z_bis), diverged)    #inserisco in questa matrice che aggiornerò ogni ciclo i valori di |zn| che hanno superato l'escape radius
        not_diverged_complex = tf.cast(n_diverged, tf.complex128)  #cast complesso di n_diverged
        z_bis = tf.math.multiply_no_nan(z_bis, not_diverged_complex)     #assegno a zn+1 (bis) il nuovo valore in modo da eliminare gli elementi che sono ora in matrix_diverged e non devono più continuare l'iterazione
  
    matrix_diverged = matrix_diverged + not_diverged_*escape_radius     #aggiungo a matrix_diverged i valori mancanti (i moduli di zn che non hanno superato l'escape radius) normalizzati al valore dell'escape radius
    return (counts.numpy() + np.log(np.log(escape_radius))/np.log(2) - np.log(np.log(matrix_diverged.numpy()))/np.log(2))     #numero di iterazioni frazionario (per rendere la colorazione uniforme e continua senza bande di colore)

zoom = 4.0

for i in tqdm(range(3000)):
    counts = mandelbrot((3840,2160),(-0.7746806106269039,-0.1374168856037867),zoom,500,1000000000000000000)     #il numero massimo per l'escape radius è 1000000000000000000 dopodichè numpy dà -inf, punti interessanti: (-0.74453986035590838012,0.12172377389442482241), (-0.7746806106269039,-0.1374168856037867), (-1.99977406013629036,-0.0000000032900403)
    zoom *= 0.99
    img = render_rgb(counts)
    #img=img.convert('RGB')     #per convertire l'HSV in RGB se uso render_hsv
    img.save(f"immagine_{i}.png")

print("zoom totale: ",zoom)



