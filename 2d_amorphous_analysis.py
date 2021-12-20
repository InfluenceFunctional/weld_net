import numpy as np
import os
import matplotlib.pyplot as plt
from accuracy_metrics import *
from utils import rolling_mean
import tqdm
import scipy.ndimage as ndimage
import scipy.fftpack as fftpack

# load datasets
datasets = []
datasets.append(np.random.randint(0,2,size=(100,128,128))) # random pixels, 50% coverage

set = np.zeros((2,128,128))# crystal 1
for i in range(set.shape[1]):
    for j in range(set.shape[2]):
        set[0,i,j] = int((i % 2 == 0) and (j % 2 == 0))
        set[1,i,j] = int(((i + 1) % 2 == 0) and ((j + 1) % 2 == 0))
datasets.append(set)

set = np.zeros((2,128,128))# crystal 2
for i in range(set.shape[1]):
    for j in range(set.shape[2]):
        set[0,i,j] = int((i % 4 == 0) and (j % 3 == 0))
        set[1,i,j] = int(((i + 1) % 4 == 0) and ((j + 1) % 3 == 0))
datasets.append(set)

datasets.append(np.load('C:/Users\mikem\OneDrive\McGill_Simine\Finite_Correlations\PyTorch\PixelCNN\data/big_worm_results.npy',allow_pickle=True)[100:600,0,:,:])# amorphous 1
datasets.append(np.load('C:/Users\mikem\OneDrive\McGill_Simine\Finite_Correlations\PyTorch\PixelCNN\data/new_aggregates/aggs3.npy',allow_pickle=True)[:,0,:128,:128])# amorphous 2
#datasets.append(np.load('C:/Users\mikem\OneDrive\McGill_Simine\Finite_Correlations\PyTorch\PixelCNN\data/hot_repulsive.npy',allow_pickle=True)[0:500])# amorphous 3

datasets.append(np.ones((100000,128,128))) # white background - subract this to get de-pixellated result

# compute correlation functions and fourier transforms
corrRad = []
fourierFreq= []
pairCorrelation = []

for i in tqdm.tqdm(range(len(datasets))):
    #im, g2, radius = spatial_correlation(np.expand_dims(datasets[i],1))
    im, g2, radius = spatial_correlation2(torch.Tensor(np.expand_dims(datasets[i],1)))
    corrRad.append(radius)
    pairCorrelation.append(g2)

correctedCorr = []
structureFactor = []
radialFourier = []
sigma = 1
for i in tqdm.tqdm(range(len(datasets)-1)):
    correctedCorr.append(ndimage.gaussian_filter1d(pairCorrelation[i],sigma)/ndimage.gaussian_filter1d(pairCorrelation[-1] + 1e-6,sigma)) # correct for bin density on discrete grid
    #correctedCorr.append(rolling_mean(pairCorrelation[i]/(pairCorrelation[-1] + 1e-6),avgOver))

    rho = np.average(datasets[i])
    k, Sq = computeStructureFactor(corrRad[i],correctedCorr[i], rho)
    #Sq = 1 + np.pi*4*rho*(fftpack.dst(corrRad[i] * (correctedCorr[i] - 1)))
    #Sq = Sq[0:len(Sq)//2]
    structureFactor.append(Sq)
    fourier = fourier_analysis(np.expand_dims(datasets[i],1))
    radfourier = radial_fourier_analysis(fourier)
    if i == 0:
        radialFourier.append(radfourier)
    else:
        radialFourier.append([radfourier[0],ndimage.gaussian_filter1d(radfourier[1],sigma)/ndimage.gaussian_filter1d(radialFourier[0][1]+1e-6,sigma)])



# plot results
plt.figure(1)
plt.clf()
for i in range(len(datasets) - 1):
    plt.subplot(1,2,1)
    plt.plot(corrRad[i],correctedCorr[i]) # subtract the baseline - only works if correlations bins are identical
    plt.subplot(1,2,2)
    plt.plot(structureFactor[i])




'''
from utils import analyse_samples
sout = analyse_samples(torch.Tensor(samples),1)
tout = analyse_samples(torch.Tensor(traindata),1)
zout = analyse_samples(torch.ones_like(torch.Tensor(traindata)),1)

# make figure
title_fontsize = 18
tick_fontsize = 18


plt.subplot(1,3,2)
bins = rolling_mean(sout['correlation bins'][0:1200],20)
plt.plot(bins,rolling_mean(sout['radial correlation'][0:1200],20)/rolling_mean(zout['radial correlation'][0:1200],20),label='Generated')
plt.plot(bins,rolling_mean(tout['radial correlation'][0:1200],20)/rolling_mean(zout['radial correlation'][0:1200],20),label='Training')
plt.ylabel('Radial Pair Correlation',fontsize=tick_fontsize)
plt.xlabel('nm',fontsize=tick_fontsize)
plt.legend(fontsize=18)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)



'''