import numba
from numba import jit
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import math
import numpy as np
start=timer()
def get_iter(c:complex, tresh, max_steps) -> int:
    z=c
    i=1
    while i<max_steps and (z*z.conjugate()).real<tresh:
        z=z*z+c
        i+=1
    return i
get_iter_jit=jit()(get_iter)
def plotter(n, tresh, max_steps):
    mx=2.48/(n-1)
    my=2.26/(n-1)
    mapper=lambda x,y: (mx*x-2, my*y-1.13)
    img=np.full((n,n),255)
    for x in range(n):
        for y in range (n):
            it=get_iter_jit(complex(*mapper(x,y)), tresh=tresh, max_steps=max_steps)
            img[y][x]=255-it
    return img
plotter_jit=jit()(plotter)
n=3000
img=plotter_jit(n, 4, 100)
dt=timer()-start
print("Mandelbrot creato in %f s" % dt)
plt.imshow(img, cmap="plasma")
plt.axis("off")
plt.show()
