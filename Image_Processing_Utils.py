import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils
import torch

def scramble_images(images, noise, den_var, GPU): # noise parameter from 0 to 1, determines the maximum proportion of pixels will be scrambled in the batch
    #this probably needs to be done in batches with varying noise densities
    if GPU == 1:
        images = images.cpu()
    images = images.float()
    delta = 1
    slices = np.ceil(images.shape[0]/delta)
    noise_rands = np.arange(slices - 1) / (slices - 1) * noise # proportion of pixels to be resampled in each image
    np.random.shuffle(noise_rands)

    for i in range(int(slices) - 1): # -1 to account for remainder, which will go unmodified
        images_subset = images[i*delta:(i+1)*delta, :, :, :]
        #possible_values = np.unique(images_subset)
        n_samples = int(images_subset.shape[0] * images_subset.shape[2] * images_subset.shape[3] * noise_rands[i]) # number of pixels we have to resample

        image_mean = torch.mean((images_subset.float()- .5) * 2) # pre-existing occupation mean

        den_var_randns = torch.normal(mean = torch.ones(n_samples) * image_mean, std = torch.ones(n_samples) * den_var)  # degree of density variation in the noise
        rand_ind, rand_y, rand_x = [torch.randint(0, images_subset.shape[0], size = (n_samples,)), torch.randint(0,images_subset.shape[2], size = (n_samples,)), torch.randint(0,images_subset.shape[3], size = (n_samples,))]
        sample_rand = (torch.rand(n_samples) < den_var_randns).int()
        #sample_rand = torch.Tensor(np.random.randint(1,len(possible_values)+1, size=n_samples)).float()

        #images_subset[rand_ind, 0, rand_y, rand_x] = (sample_rand - .5) * 2
        images_subset[rand_ind, 0, rand_y, rand_x] = (sample_rand.float() + 1) / 2# / len(possible_values)
        images[i*delta:(i+1)*delta, :, :, :] = images_subset

    if GPU ==1:
        images = images.cuda()

    return images

def random_padding(images, noise_mean, den_var, conv_field, GPU):
    if GPU == 1:
        images = images.cpu()

    xdim=images.shape[-1]
    ydim=images.shape[-2]
    images = np.pad(images,((0,0),(0,0),(conv_field,conv_field),(conv_field,conv_field)),'constant',constant_values=0)
    delta = 10

    for i in range(images.shape[0]//delta): # in a series of slices, add a bound normally distributed stuff of size conv_field
        out = np.random.normal(noise_mean, den_var, size=(delta, ydim + conv_field * 2, xdim + conv_field * 2))
        out[:,conv_field:-conv_field,conv_field:-conv_field] = images[i*delta:(i+1)*delta,0,conv_field:-conv_field,conv_field:-conv_field]
        images[i*delta:(i+1)*delta,0,:,:] = out

    if GPU == 1:
        images = torch.Tensor(images).cuda()
    else:
        images = torch.Tensor(images)

    return images

#target = 'C48_170nm_m01.tif'
#target = 'data/Test_Image.tif'
def process_image(target):
    # import image
    image = plt.imread(target)[:,:,0] # tif input is rgb + 1

    # cut off the bottom - it's black
    black = 1
    i = 1
    while black == 1:
        if not np.sum(image[-i - 3: - i, 0] == [0, 0, 0]): # if the the last three pixels are not black
            black = 0
            image = image[0:-i, :]
        else:
            i += 1

    # balance brightness / contrast - some of them have a weird square in the middle


    return image

#image = process_image(target)

def sample_images(image, xdim, ydim, brightness): # if brigthness = 0 it will not be normalized, if 1 it will be, if 2 it will be binarized about the median
    # compute average brightness
    if brightness == 1:
        avg_brightness = np.average(image)
    else:
        avg_brightness = 1


    # break our image up into lots of little samples of a certain size
    image_size = np.array((image.shape[0], image.shape[1]))
    grid_size = np.array((image_size[0] // xdim, image_size[1] // ydim))
    sample_size = np.array((image_size[0]//grid_size[0], image_size[1]//grid_size[1]))
    n_samples = grid_size[0] * grid_size[1]
    samples = np.zeros((n_samples, sample_size[0], sample_size[1]))

    ind = 0
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            slice = image[i * sample_size[0]:(i + 1) * sample_size[0], j * sample_size[1]:(j + 1) * sample_size[1]]
            if brightness == 1:  #normalize brightness
                slice = slice * avg_brightness // np.average(slice) # correct to the average
            elif brightness == 2:  #binarize by the median
                slice = slice > np.median(slice)

            samples[ind, :, :] = np.array(slice)
            ind += 1

    image_reconstruction = utils.make_grid(torch.Tensor(np.expand_dims(samples, 1)), nrow=image.shape[1] // ydim, padding=0)[0]

    return np.array(image_reconstruction)


def sample_augment(image, xdim, ydim, normalize, binarize, rotflip, n_samples): # if brigthness = 0 it will not be normalized, if 1 it will be, if 2 it will be binarized about the median
    samples = np.zeros((n_samples, xdim, ydim))
    if normalize == 1: # batch-normalize brightness
        avg_brightness = np.average(image)
    else:
        avg_brightness = 1

    # potential top-left pixels to choose from
    pot_samples = image[:-ydim, :-xdim]  # can't take any samples which go outside of the image!
    y_rands = np.random.randint(0, image.shape[0]-ydim, n_samples)
    x_rands = np.random.randint(0, image.shape[1]-xdim, n_samples)
    if rotflip == 1:
        rot_rands = np.random.randint(0, 4, n_samples)
        flip_rands = np.random.randint(0, 2, n_samples)
    else:
        rot_rands = np.zeros(n_samples)
        flip_rands = np.zeros(n_samples)

    for i in range(n_samples):
        slice = image[y_rands[i]:y_rands[i]+ydim, x_rands[i]:x_rands[i]+xdim] # grab a random sample from the image
        # orient it randomly
        if flip_rands[i]:
            slice = np.fliplr(slice)

        slice = np.rot90(slice, rot_rands[i])


        if normalize == 1:  #normalize brightness
            slice = slice * avg_brightness // np.average(slice) # correct to the average
            samples[i, :, :] = np.array(slice).astype('uint8')
        elif binarize == 1:  #binarize by the median
            slice = slice > np.median(slice)
            samples[i, :, :] = np.array(slice).astype('bool')
        else: # leave as-is
            samples[i, :, :] = np.array(slice)

    return samples



def super_augment(image, xdim, ydim, normalize, binarize, rotflip, n_samples): # if brigthness = 0 it will not be normalized, if 1 it will be, if 2 it will be binarized about the median
    samples = np.zeros((n_samples, xdim, ydim))
    if normalize == 1: # batch-normalize brightness
        avg_brightness = np.average(image)
    else:
        avg_brightness = 1

    # potential top-left pixels to choose from
    pot_samples = image[:-ydim, :-xdim]  # can't take any samples which go outside of the image!
    y_rands = np.random.randint(0, image.shape[0]-ydim, n_samples)
    x_rands = np.random.randint(0, image.shape[1]-xdim, n_samples)
    if rotflip == 1:
        rot_rands = np.random.randint(0, 4, n_samples)
        flip_rands = np.random.randint(0, 2, n_samples)
    else:
        rot_rands = np.zeros(n_samples)
        flip_rands = np.zeros(n_samples)

    for i in range(n_samples):
        slice = image[y_rands[i]:y_rands[i]+ydim, x_rands[i]:x_rands[i]+xdim] # grab a random sample from the image
        # orient it randomly
        if flip_rands[i]:
            slice = np.fliplr(slice)

        slice = np.rot90(slice, rot_rands[i])


        if normalize == 1:  #normalize brightness
            slice = slice * avg_brightness // np.average(slice) # correct to the average
            samples[i, :, :] = np.array(slice).astype('uint8')
        elif binarize == 1:  #binarize by the median
            slice = slice > np.median(slice)
            samples[i, :, :] = np.array(slice).astype('bool')
        else: # leave as-is
            samples[i, :, :] = np.array(slice)

    return samples


def correct_brightness(image,xdim,ydim):
    ## correct brightness
    avg_brightness = np.average(image)  #

    # break our image up into lots of little samples of a certain size
    image_size = np.array((image.shape[0], image.shape[1]))
    grid_size = np.array((image_size[0] // xdim, image_size[1] // ydim))
    sample_size = np.array((image_size[0] // grid_size[0], image_size[1] // grid_size[1]))
    n_samples = grid_size[0] * grid_size[1]
    samples = np.zeros((n_samples, sample_size[0], sample_size[1]))

    ind = 0
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            slice = image[i * sample_size[0]:(i + 1) * sample_size[0], j * sample_size[1]:(j + 1) * sample_size[1]]

            slice = slice * avg_brightness // np.average(slice)  # correct to the average

            samples[ind, :, :] = np.array(slice).astype('float32')
            ind += 1

    image_reconstruction = utils.make_grid(torch.Tensor(np.expand_dims(samples, 1)), nrow=image.shape[1] // ydim, padding=0)[0]
    return image_reconstruction

def single_pixel_subsample(images, conv_field, GPU, noise, padding):
    if noise == 99:
        model = 1 # square input
    elif noise == 98:
        model = 2 # gated - no bottom on input
    n_samples = images.shape[0]
    if padding == 1:
        y_rands = np.random.randint(conv_field + (model == 2), images.shape[-2]+conv_field, n_samples) #remove +1 for discriminator
        x_rands = np.random.randint(conv_field, images.shape[-1]+conv_field, n_samples)
    elif padding == 0:
        y_rands = np.random.randint(conv_field + (model == 2), images.shape[-2]-conv_field, n_samples)
        x_rands = np.random.randint(conv_field, images.shape[-1]-conv_field, n_samples)

    if GPU == 1:
        images = images.cpu().detach().numpy()
    if padding == 1:
        images = np.pad(images,((0,0),(0,0),(conv_field,conv_field),(conv_field,conv_field)),'constant')

    if model == 1:
        sample = np.asarray([images[[i], 0, y_rands[i] - conv_field:y_rands[i] + conv_field + 1, x_rands[i] - conv_field:x_rands[i] + conv_field + 1] for i in range(n_samples)])
    elif model == 2:
        sample = np.asarray([images[[i], 0, y_rands[i] - conv_field - 1:y_rands[i] + 1, x_rands[i] - conv_field:x_rands[i] + conv_field + 1] for i in range(n_samples)])
    #sample = images[:, 0, y_rands - conv_field - 1:y_rands + 1, x_rands - conv_field:x_rands + conv_field + 1]  # for discriminator

    sample = torch.Tensor(sample)
    if GPU == 1:
        sample = sample.cuda()

#    target = sample[:,:,-1,conv_field]
#    target = target * (3 - 1)  # switch from training to output space
#    if torch.sum(target==0) > 0:
#        print('stop!')

    return sample

def single_pixel_subsample2(images, conv_field, GPU, padding): # same pixel for each image
    y_rands = np.random.randint(0, images.shape[-2])
    x_rands = np.random.randint(0, images.shape[-1])
    y_rands += conv_field + 1
    x_rands += conv_field

    if GPU == 1:
        images = images.cpu().detach().numpy()

    if padding == 1:
        images = np.pad(images,((0,0),(0,0),(conv_field,conv_field),(conv_field,conv_field)),'constant')

    sample = images[:,:,y_rands - conv_field - 1:y_rands + 1, x_rands - conv_field:x_rands + conv_field + 1]  # grab a random sample from each image

    sample = torch.Tensor(sample)
    if GPU == 1:
        sample = sample.cuda()

    return sample