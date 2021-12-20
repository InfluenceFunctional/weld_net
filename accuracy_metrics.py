import numpy as np
import torch
import torch.nn.functional as F
from Image_Processing_Utils import *
from os import listdir
from os.path import isfile, join
import numpy.linalg as linalg

def compute_density(image, GPU): #deprecated
    # return particle density
    density = torch.sum(image, axis=(1,2,3)) / image.shape[2] / image.shape[3]
    average_density = torch.mean(density)
    return density, average_density


def compute_interactions(image): # compute the number of neighbors of a given occupied pixel

    filter_1 = torch.Tensor(((1,1,1),(1,0,1),(1,1,1))).unsqueeze(0).unsqueeze(0) # this filter computes the number of range-1 interactions of each particle
    interactions = np.zeros((image.shape[0], 1, image.shape[2]-2, image.shape[3]-2))
    delta = min(1000, image.shape[0])
    for i in range(image.shape[0]//delta):
        test_image = image[i*1000:(i+1)*1000].type(torch.float32)
        interactions[i*1000:(i+1)*1000, :, :,:] = F.conv2d(test_image, filter_1, padding = 0) * test_image[:,:,1:-1,1:-1] + test_image[:,:,1:-1,1:-1] # number of interactions per occupied pixel + a marker for the occupied pixel, not padded

    # since we will have large and small outputs, we will build the distribution as a sample of individual pixels
    samples = int(1e6)
    r_numbers = [np.random.randint(interactions.shape[0],size=samples), np.random.randint(interactions.shape[2],size=samples), np.random.randint(interactions.shape[3],size=samples)]
    interactions_sample = interactions[r_numbers[0],0,r_numbers[1],r_numbers[2]]
    interactions_hist = np.histogram(interactions_sample, bins=9, range=(0,9), density=True)  # histogram of interaction strength - zero means empty pixel, 1 means occupied with no neighbors - we eliminate the zeros here
    average_interactions = np.average(interactions)  # average number of interactions per occupied particle

    return average_interactions, interactions_hist


def anisotropy(image, GPU):

    distribution = image.sum(0).squeeze(0)  # sum all the images to find anisotropy
    distribution2 = distribution.unsqueeze(0)
    distribution2 = distribution2.unsqueeze(0)

    if GPU == 1:
        distribution = distribution.cpu().detach().numpy()
    else:
        distribution = distribution.detach().numpy()

    variance = np.var(distribution/np.amax(distribution + 1e-5))  # compute the overall variance (ideally small!)

    pooled_distribution = F.max_pool2d(1/(distribution2+1e-5),distribution2.shape[2]//10,distribution2.shape[3]//10)  # pool according to the inverse - extra sensitive to minima - gaps in the distribution
    pooled_distribution = pooled_distribution.squeeze((0))
    pooled_distribution = pooled_distribution.squeeze((0))

    if GPU == 1:
        pooled_distribution = pooled_distribution.cpu().detach().numpy()
    else:
        pooled_distribution = pooled_distribution.detach().numpy()

    pooled_variance = np.var(pooled_distribution/np.amax(pooled_distribution))

    return distribution/np.amax(distribution), variance, pooled_distribution/np.amax(pooled_distribution), pooled_variance


def fourier_analysis(sample):
    sample = sample[0:1000, 0, :, :] # do the first thousand samples tops
    n_samples = sample.shape[0]
    slice = 1
    batched_image_transform = np.zeros((n_samples // slice, sample.shape[1], sample.shape[2]))
    for i in range(int(np.amin((n_samples // slice, 1000)))): # up to a max of 1000 samples
        batched_image_transform[i, :, :] = np.average(np.abs(np.fft.fftshift(np.fft.fft2(sample[i * slice:(i + 1) * slice, :, :]))), 0)  # fourier transform of the original image

    transform = np.average(batched_image_transform, 0)

    return transform


def radial_fourier_analysis(transform): # convert 2D fourier transform to radial coordinates
    x0, y0 = [transform.shape[-2] // 2 , transform.shape[-1] // 2 ]  # pick a nice central pixel
    transform[y0,x0] = 0
    max_rad = transform.shape[-1] // 2 - 1 # maximum radius
    nbins = max_rad * 10  # set number of bins for sorting
    a, bins = np.histogram(1, bins=nbins, range=(.01, max_rad + 0.01))  # bin the possible radii
    radial_fourier = np.zeros(nbins)  # radial density
    radial_fourier2 = np.zeros(nbins)  # radial pair-correlation

    for i in range(transform.shape[-2]):  # for a box of radius max_rad around x0, y0
        for j in range(transform.shape[-1]):
            if (i != y0) or (j != x0):

                radius = np.sqrt((i - y0) ** 2 + (j - x0) ** 2)

                if radius <= max_rad:  # if we are within the circle drawn over the square
                    bin_num = np.digitize(radius, bins) - 1  # add to bin
                    radial_fourier[bin_num] += transform[i, j]
                    radial_fourier2[bin_num] += transform[i, j] / radius

    bin_rad = np.zeros(len(bins) - 1)

    for i in range(len(bins) - 1):
        bin_rad[i] = (bins[i] + bins[i + 1]) / 2  # assign a radius to each bin

    rolling_mean = np.zeros(len(radial_fourier))
    rolling_mean2 = np.zeros(len(radial_fourier2))
    run = 1#int(nbins // 20 * 1)  # length of rolling mean
    for i in range(run, len(radial_fourier2)):
        rolling_mean[i] = np.average(radial_fourier[i - run:i])
        rolling_mean2[i] = np.average(radial_fourier2[i - run:i])

    return bin_rad, rolling_mean2


def spatial_correlation(image_set):
    xdim = int(image_set.shape[-1]//2)
    ydim = int(image_set.shape[-2]//2)

    n_samples = 10000

    sample = np.zeros((1,1,ydim,xdim))
    attempts = 0
    while (sample.shape[0] < n_samples) and (attempts < 10): # we want a certain number of samples but if the density is too low we'll get zero, so we give it up to 10 times through the dataset
        attempts += 1

        for i in range(image_set.shape[0]):
            sample = np.append(sample, np.expand_dims(sample_augment(image_set[i,0,:,:], ydim, xdim, 0, 0, 0, int(np.ceil(n_samples / image_set.shape[0]))),1), 0)
            x0, y0 = [sample.shape[-2] // 2 - 1, sample.shape[-1] // 2 - 1]  # pick a nice central pixel
            sample = sample[sample[:, 0, y0, x0] != 0]  # delete samples with zero particles at centre

            if sample.shape[0] >= n_samples:
                break

    #preprocess sample
    max_rad = sample.shape[2] // 2 - 1 # the radius to be explored is automatically set to the maximum possible for the sample image
    nbins = max_rad * 20 # set number of bins for sorting
    box_size = 2 * max_rad + 1 # size of box for radial searching
    sample = sample[sample[:,:,y0,x0]!=0] # delete samples with zero particles at centre (a waste, I know, but you can always just feed it more samples, or get rid of this if you don't need a central particle)
    sample = sample[:,y0-max_rad:y0+max_rad+1, x0-max_rad:x0+max_rad+1] # adjust sample size

    # prep radial bins
    a, bins = np.histogram(1, bins = nbins, range = (.01, max_rad + 0.01)) # bin the possible radii
    circle_square_ratio = np.pi/4  # radio of circle to square area with equal radius

    # prep constants
    dr = bins[1]-bins[0] # differential radius
    N_i = sample.shape[0]  # number of samples
    N_tot = np.sum(sample)*circle_square_ratio - N_i # total particle number adjusted for a circular radius and subtracting the centroid
    rho = np.average(sample)  # particle density

    # initialize outputs
    radial_corr = np.zeros(nbins) # radial density
    radial_corr2 = np.zeros(nbins) # radial pair-correlation
    corr = np.zeros((box_size, box_size)) # 2D density
    corr2 = np.zeros((box_size, box_size)) # 2D pair-correlation

    # for each pixel within a square box of the appropriate size, assign a radius, coordinates and check its occupancy
    for i in range(box_size): # for a box of radius max_rad around x0, y0
        for j in range(box_size):
            if (i != y0) or (j != x0):

                radius= np.sqrt((i - y0) **2 + (j - x0) ** 2)
                corr[i, j] = np.sum(sample[:, i, j]) # density distribution
                corr2[i, j] = corr[i, j] / (radius) # pair-correlation

                if radius <= max_rad: # if we are within the circle drawn over the square
                    bin_num = np.digitize(radius, bins) - 1  # add to bin
                    radial_corr2[bin_num] += corr2[i, j]

    bin_rad = np.zeros(len(bins)-1)

    for i in range(len(bins)-1):
        bin_rad[i] = (bins[i] + bins[i+1]) / 2 #assign a radius to each bin

    radial_corr2 = radial_corr2 / (2 * np.pi * dr * rho * N_i) # normalize the pair-correlation function

    #compute rolling means for correlation functions
    rolling_mean = np.zeros(len(radial_corr2))
    #rolling_mean2 = np.zeros(len(radial_corr))
    run = 1#int(nbins // 20 * 1) # length of rolling mean
    for i in range(run,len(radial_corr2)):
        rolling_mean[i] = np.average(radial_corr2[i-run:i])
        #rolling_mean2[i] = np.average(radial_corr[i-run:i])

    # average out the central points for easier graph viewing
    corr[y0,x0] = np.average(corr)
    #corr2[y0,x0] = np.average(corr2)

    return corr, rolling_mean, bin_rad


def spatial_correlation2(image_set): #faster
    xdim = int(image_set.shape[-1]//2)
    ydim = int(image_set.shape[-2]//2)
    desired_samples = 1e3
    n_samples = int(desired_samples / torch.mean(image_set.float())) // 4 # normalize against the mean
    attempts = 0
    sample = torch.zeros((1,ydim,xdim))
    images = image_set.type(torch.ByteTensor)  # can't take any samples which go outside of the image!


    # set of sub-sampled images to analyze
    while (sample.shape[0] < desired_samples) and (attempts < 10):
        attempts += 1
        y_rands = np.random.randint(0, images.shape[-2]-ydim, n_samples)
        x_rands = np.random.randint(0, images.shape[-1]-xdim, n_samples)
        n_rands = np.random.randint(0, images.shape[0], n_samples)
        slice = torch.zeros((n_samples,1,ydim,xdim))
        for i in range(n_samples):
            slice[i,:,:,:] = images[n_rands[i],:,y_rands[i]:y_rands[i] + ydim, x_rands[i]:x_rands[i] + xdim]  # grab a random sample from the image

        x0, y0 = [slice.shape[-2] // 2 - 1, slice.shape[-1] // 2 - 1]  # pick a nice central pixel
        slice = slice[slice[:, :, y0, x0] != 0]  # delete samples with zero particles at centre
        sample = torch.cat((sample,slice.float()),0)

    sample = np.array(sample[1:,:,:])

    #preprocess sample
    max_rad = sample.shape[-1] // 2 - 1 # the radius to be explored is automatically set to the maximum possible for the sample image
    nbins = max_rad * 500 # set number of bins for sorting
    box_size = 2 * max_rad + 1 # size of box for radial searching

    # prep radial bins
    a, bins = np.histogram(1, bins = nbins, range = (.01, max_rad + 0.01)) # bin the possible radii
    circle_square_ratio = np.pi/4  # radio of circle to square area with equal radius

    # prep constants
    dr = bins[1]-bins[0] # differential radius
    N_i = sample.shape[0]  # number of samples
    N_tot = np.sum(sample)*circle_square_ratio - N_i # total particle number adjusted for a circular radius and subtracting the centroid
    rho = np.average(sample)  # particle density

    # initialize outputs
    radial_corr = np.zeros(nbins) # radial density
    radial_corr2 = np.zeros(nbins) # radial pair-correlation
    corr = np.zeros((box_size, box_size)) # 2D density
    corr2 = np.zeros((box_size, box_size)) # 2D pair-correlation

    # for each pixel within a square box of the appropriate size, assign a radius, coordinates and check its occupancy
    for i in range(box_size): # for a box of radius max_rad around x0, y0
        for j in range(box_size):
            if (i != y0) or (j != x0):

                radius= np.sqrt((i - y0) **2 + (j - x0) ** 2)
                corr[i, j] = np.sum(sample[:, i, j]) # density distribution
                corr2[i, j] = corr[i, j] / (radius) # pair-correlation

                if radius <= max_rad: # if we are within the circle drawn over the square
                    bin_num = np.digitize(radius, bins) - 1  # add to bin
                    radial_corr2[bin_num] += corr2[i, j]

    bin_rad = np.zeros(len(bins)-1)

    for i in range(len(bins)-1):
        bin_rad[i] = (bins[i] + bins[i+1]) / 2 #assign a radius to each bin

    radial_corr2 = radial_corr2 / (2 * np.pi * dr * rho * N_i) # normalize the pair-correlation function

    #compute rolling means for correlation functions
    rolling_mean = np.zeros(len(radial_corr2))
    #rolling_mean2 = np.zeros(len(radial_corr))
    run = 1#int(nbins // 20 * 1) # length of rolling mean
    for i in range(run,len(radial_corr2)):
        rolling_mean[i] = np.average(radial_corr2[i-run:i])
        #rolling_mean2[i] = np.average(radial_corr[i-run:i])

    # average out the central points for easier graph viewing
    corr[y0,x0] = np.average(corr)
    #corr2[y0,x0] = np.average(corr2)

    return corr, rolling_mean, bin_rad


def bond_analysis(images, max_range, particle):
    particle = 1
    n_tests = 1e4
    images = images[:,0,:,:]
    max_bond_length = int(max_range / 0.2)  # avg bond length is about 1.4A, grid is about 0.2 A

    empty = np.zeros((images.shape[0], images.shape[-2] + max_bond_length * 2, images.shape[-1] + max_bond_length * 2))
    empty[:, max_bond_length:-max_bond_length, max_bond_length:-max_bond_length] = images
    images = empty

    #initialize outputs
    bond_order = []
    bond_length = []
    bond_angle = []

    if np.average(images) < 0.1: # don't even try for dense samples
        for n in range(images.shape[0]): #search for neighbors
            if len(bond_order) >= n_tests * 10:
                break

            for i in range(max_bond_length, images[n, :, :].shape[-2] - max_bond_length):
                if len(bond_order) >= n_tests * 10:
                    break

                for j in range(max_bond_length, images[n, :, :].shape[-1] - max_bond_length):
                    if images[n, i, j] == particle:  # if we find a particle
                        radius = []
                        neighborx = []
                        neighbory = []
                        for ii in range(i - max_bond_length, i + max_bond_length + 1):
                            for jj in range(j - max_bond_length, j + max_bond_length + 1):  # search in a ring around it of radius (max bond length)
                                if images[n, ii, jj] == particle:
                                    if not ((i == ii) and (j == jj)):  # if we find a particle that is not the original one, store it's location
                                        rad = (np.sqrt((i - ii) ** 2 + (j - jj) ** 2))
                                        if rad <= max_bond_length:
                                            radius.append(rad * 0.2)
                                            neighborx.append(jj)
                                            neighbory.append(ii)

                        bond_order.append(int(len(radius))) # compute bond_order

                        for r in range(int(len(radius))): # compute bond_lengths
                            bond_length.append(radius[r])
                            for q in range(int(len(radius))):# compute bond angles
                                if r != q:
                                    v1 = np.array([i,j]) - np.array([neighbory[r], neighborx[r]])
                                    v2 = np.array([i,j]) - np.array([neighbory[q], neighborx[q]])
                                    c = np.dot(v1,v2) / linalg.norm(v1) / linalg.norm(v2)
                                    bond_angle.append(np.arccos(np.clip(c,-1,1)))


            # compute distributions
            bond_order_dist = np.histogram(np.array(bond_order), bins=7, range=(0, 7), density = True)
            bond_length_dist = np.histogram(np.array(bond_length), bins=100, range =(1, max_range), density = True)
            bond_angle_dist = np.histogram(np.array(bond_angle), bins=200, range=(0, np.pi), density = True)
            avg_bond_order = np.average(np.array(bond_order))
            avg_bond_length = np.average(np.array(bond_length))
            avg_bond_angle = np.average(np.array(bond_angle))
    else:
        # compute distributions
        bond_order_dist = np.histogram(np.array(0), bins=7, range=(0, 7), density = True)
        bond_length_dist = np.histogram(np.array(0), bins=100, range =(1, max_range), density = True)
        bond_angle_dist = np.histogram(np.array(0), bins=200, range=(0, np.pi), density = True)
        avg_bond_order = np.average(np.array(0))
        avg_bond_length = np.average(np.array(0))
        avg_bond_angle = np.average(np.array(0))

    return avg_bond_order, bond_order_dist, avg_bond_length, avg_bond_angle, bond_length_dist, bond_angle_dist


def probability_analysis(model, g_sample, t_sample, conv_field):
    g_sample = (g_sample.float() + 1) / 2
    # compute pixel probabilities
    t_sample_pad = F.pad(t_sample,(conv_field,conv_field,conv_field + 1, 0),value=0)
    g_sample_pad = F.pad(g_sample,(conv_field,conv_field,conv_field+1,0),value=0)
    with torch.no_grad():
        t_probs = F.softmax(model(t_sample_pad.float())[:,1:],dim=1).cpu().detach().numpy()#model(t_sample_pad.float())[:,1:].cpu().detach().numpy()#
        g_probs = F.softmax(model(g_sample_pad.float())[:,1:],dim=1).cpu().detach().numpy()#model(g_sample_pad.float())[:,1:].cpu().detach().numpy()#

    full_probs = [t_probs.copy(),g_probs.copy()]

    t_probs = t_probs[:,:,conv_field:-conv_field,conv_field:-conv_field]
    g_probs = g_probs[:,:,conv_field:-conv_field,conv_field:-conv_field]
    #flatten images
    t_prob_flat = t_probs.reshape(len(t_probs), t_probs.shape[1], t_probs.shape[-2] * t_probs.shape[-1])
    g_prob_flat = g_probs.reshape(len(g_probs), g_probs.shape[1], g_probs.shape[-2] * g_probs.shape[-1])

    t_sample_cort = t_sample[:,:,conv_field:-conv_field,conv_field:-conv_field].cpu().detach().numpy()
    g_sample_cort = g_sample[:,:,conv_field:-conv_field,conv_field:-conv_field].cpu().detach().numpy()
    t_sample_flat = t_sample_cort.reshape(len(t_sample),t_sample_cort.shape[-2]*t_sample_cort.shape[-1])
    g_sample_flat = g_sample_cort.reshape(len(g_sample),g_sample_cort.shape[-2]*g_sample_cort.shape[-1])
    t_sample_flat = (t_sample_flat * 2 - 1).astype(int)
    g_sample_flat = (g_sample_flat).astype(int)

    # for each image, assign pixel probabilities
    t_prob_imag = []
    g_prob_imag = []
    for i in range(len(t_prob_flat)):
        t_prob_imag.append(np.transpose(t_prob_flat[i, t_sample_flat[i, :], np.arange(t_prob_flat.shape[-1])]))

    for i in range(len(g_prob_flat)):
        g_prob_imag.append(np.transpose(g_prob_flat[i, g_sample_flat[i, :], np.arange(g_prob_flat.shape[-1])]))

    t_prob_imag = np.asarray(t_prob_imag)
    g_prob_imag = np.asarray(g_prob_imag)

    t_prob_imag = t_prob_imag.reshape(len(t_prob_imag), t_sample_cort.shape[-2], t_sample_cort.shape[-1])
    g_prob_imag = g_prob_imag.reshape(len(g_prob_imag), g_sample_cort.shape[-2], g_sample_cort.shape[-1])

    t_avg_prob = np.mean(np.log10(t_prob_imag))
    g_avg_prob = np.mean(np.log10(g_prob_imag))

    t_prob_dist = np.histogram(np.ndarray.flatten(t_prob_imag), bins=100, range=(0,1), density=True)  # histogram of interaction strength - zero means empty pixel, 1 means occupied with no neighbors - we eliminate the zeros here
    g_prob_dist = np.histogram(np.ndarray.flatten(g_prob_imag), bins=100, range=(0,1), density=True)  # histogram of interaction strength - zero means empty pixel, 1 means occupied with no neighbors - we eliminate the zeros here

    return t_avg_prob, t_prob_dist, g_avg_prob, g_prob_dist


def state_analysis(model, sample, conv_field):
    t_sample = (sample.float() + 1) / 2
    # compute pixel probabilities
    t_sample_pad = F.pad(t_sample,(conv_field,conv_field,conv_field + 1, 0),value=0)
    with torch.no_grad():
        t_probs = F.softmax(model(t_sample_pad.float())[:,1:],dim=1).cpu().detach().numpy()

    #flatten images
    t_prob_flat = t_probs.reshape(len(t_probs),t_probs.shape[1],t_probs.shape[-2]*t_probs.shape[-1])

    t_sample = t_sample.cpu().detach().numpy()
    t_sample_flat = t_sample.reshape(len(t_sample),t_sample.shape[-2]*t_sample.shape[-1])
    t_sample_flat = (t_sample_flat * 2 - 1).astype(int)

    # for each image, assign pixel probabilities
    t_prob_imag = []
    for i in range(len(t_prob_flat)):
        t_prob_imag.append(np.transpose(t_prob_flat[i, t_sample_flat[i, :], np.arange(t_prob_flat.shape[-1])]))

    t_prob_imag = np.asarray(t_prob_imag)

    t_avg_prob = np.mean(np.log10(t_prob_imag))

    t_prob_dist = np.histogram(np.array.flatten(t_prob_imag), bins=100, range=(0,1), density=True)  # histogram of interaction strength - zero means empty pixel, 1 means occupied with no neighbors - we eliminate the zeros here


    return t_avg_prob, t_prob_dist


def computeStructureFactor(rads, g2, rho):
    '''
    compute the structure factor given radial distances and radial distribution function
    ref: https://old.vscht.cz/fch/en/tools/kolafa/simen08.8.pdf
    :param rads:
    :param g2:
    :return:
    '''
    dr = np.diff(rads)[0]
    k = np.fft.rfftfreq(len(rads),dr)
    Sk = np.zeros(len(k))

    for i in range(len(k)):
        Sk[i] = 1 + 4 * np.pi * rho / k[i] * np.sum(rads * (g2-1) * np.sin(k[i]*rads) * dr)

    #Sk = 1 + rho * np.fft.rfft(g2 - 1)

    return k, Sk