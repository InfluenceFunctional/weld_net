import numpy as np
import matplotlib.pyplot as plt

image_xsize = 64
image_ysize = 64
linewidth = 1
mean_interdendrite_distance = 5
mean_intradendrite_width = 3
nSamples = 1000


images = np.zeros((nSamples,image_ysize,image_xsize),dtype=bool)

for i in range(nSamples):
    for j in range(image_ysize):
        finished = 0
        xind = 0
        while finished == 0:
            interdendrite = int(np.random.randn(1) * linewidth + mean_interdendrite_distance)
            intradendrite = int(np.random.randn(1) * linewidth + mean_intradendrite_width)
            images[i,j,xind:xind+interdendrite] = 0
            xind += mean_interdendrite_distance
            images[i,j,xind:xind+intradendrite] = 1
            xind += mean_intradendrite_width
            if xind >= image_ysize:
                finished = 1