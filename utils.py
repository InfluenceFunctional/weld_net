# metrics to determine the performance of our learning algorithm
from comet_ml import Experiment
import numpy as np
import torch.nn.functional as F
import os
from torch import nn, optim, cuda, backends
import torch
from torch.utils import data
import time
from torch.utils.data import Dataset
import pickle
import sys
import tqdm
from tqdm import tqdm as barthing
from accuracy_metrics import *
from models import *
from Image_Processing_Utils import *
import h5py


class build_dataset(Dataset):
    def __init__(self, configs):
        np.random.seed(configs.dataset_seed)
        if configs.training_dataset == 'fake welds':
            self.samples = np.load('data/fake_welds.npy', allow_pickle=True).astype('bool')
            self.samples = np.expand_dims(self.samples, axis=1)
        elif configs.training_dataset == 'welds 1':
            file = 'C:/Users\mikem\Desktop\weld_data/480-450-dx0.8_G230000-245000_vp0.0022-0.0023_database.h5'
            self.samples = []
            self.conditions = []
            with h5py.File(file, "r") as f:
                #print("Keys: %s" % f.keys())
                group_keys = list(f.keys())
                self.samples = [list(f[group_keys[i]]) for i in range(len(group_keys))]
                self.conditions = np.asarray([[.23,.22],[.23,.225],[.23,.23],[.2375,.22],[.2375,.225],[.2375,.23],[.245,.22],[.245,.225],[.245,.23]])


            self.conditional_mean = np.mean(self.conditions)
            self.conditional_std = np.sqrt(np.var(self.conditions))
            self.conditions = standardize(self.conditions)
            samples_per_condition = np.asarray([len(samples) for samples in self.samples])

            self.samples = np.asarray(self.samples) # append conditions to samples
            self.samples = self.samples.reshape(self.samples.shape[0] * self.samples.shape[1],self.samples.shape[-2],self.samples.shape[-1])
            self.samples = np.expand_dims(self.samples, axis=1) # image tensors should be 4D (num, channels, height, width)

            # shuffle dataset
            np.random.shuffle(self.samples)
            # cap dataset
            self.samples = self.samples[:configs.max_dataset_size]

            # quantize
            self.samples = np.digitize(self.samples,bins=np.linspace(0.1,0.7,10))#quantiles) # for some reason it really hates quantiles - maybe the ranges are too nonlinear
            self.samples = self.samples - np.amin(self.samples)


            if configs.do_conditioning:
                condition_tensor = torch.zeros((len(self.samples), len(self.conditions[0]), self.samples.shape[-2], self.samples.shape[-1]))
                ind = 0
                for i in range(len(self.conditions)): # for each sample
                    for j in range(len(self.conditions[0])): # for each condition
                        condition_tensor[ind:ind+samples_per_condition[i],j,:,:] = self.conditions[i,j]
                    ind += samples_per_condition[i]

                self.samples = np.concatenate((self.samples,condition_tensor),axis=1) # append conditions to pixel channel


            # augment by horizontal flips
            flipped = np.flip(self.samples.copy(),axis=3)
            self.samples = np.concatenate((self.samples,flipped),axis=0)

            # pre-processing
            self.samples = self.samples[:,:,60:,:] # cut the top
            self.samples = self.samples[:,:,::3,::3] # cut resolution by factor of ::x
            #quantiles = np.quantile(self.samples.flatten(), np.arange(11)/11)

            self.samples = np.rot90(self.samples,-1,axes=(2,3))
            #self.samples = self.samples > 0.3 # coarsening

        self.num_conditioning_variables = self.samples.shape[1] - 1
        assert self.samples.ndim == 4

        self.dataDims = {
            'classes' : len(np.unique(self.samples)),
            'input x dim' : self.samples.shape[-1],
            'input y dim' : self.samples.shape[-2],
            'channels' : 1, # hardcode as one so we don't get confused with conditioning variables
            'dataset length' : len(self.samples),
            'sample x dim' : self.samples.shape[-1],# * configs.sample_outpaint_ratio,
            'sample y dim' : self.samples.shape[-2] * configs.sample_outpaint_ratio,
            'num conditioning variables' : self.num_conditioning_variables,
            'conv field' : configs.conv_layers + configs.conv_size // 2,
            'conditional mean' : self.conditional_mean,
            'conditional std' : self.conditional_std
        }

        # normalize pixel inputs
        self.samples[:,0,:,:] = np.array((self.samples[:,0] + 1)/(self.dataDims['classes'])) # normalize inputs on 0--1


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def get_dir_name(model, training_data, filters, layers, dilation, filter_size, noise, den_var, dataset_size):
    dir_name = "model=%d_dataset=%d_dataset_size=%d_filters=%d_layers=%d_dilation=%d_filter_size=%d_noise=%.1f_denvar=%.1f" % (model, training_data, dataset_size, filters, layers, dilation, filter_size, noise, den_var)  # directory where tensorboard logfiles will be saved

    return dir_name


def get_model(configs, dataDims):
    if configs.model == 'gated1':
        model = GatedPixelCNN(configs, dataDims) # gated, without blind spot
    else:
        sys.exit()

    return model


    #def init_weights(m):
    #    if (type(m) == nn.Conv2d) or (type(m) == MaskedConv2d):
    #        #torch.nn.init.xavier_uniform_(m.weight)
    #        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity = 'relu')

    #model.apply(init_weights) # apply xavier weights to 1x1 and 3x3 convolutions


def get_dataloaders(configs):
    dataset = build_dataset(configs)  # get data
    dataDims = dataset.dataDims
    train_size = int(0.8 * len(dataset))  # split data into training and test sets
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.Subset(dataset, [range(train_size),range(train_size,test_size + train_size)])  # split it the same way every time
    tr = data.DataLoader(train_dataset, batch_size=configs.training_batch_size, shuffle=True, num_workers= 0, pin_memory=True)  # build dataloaders
    te = data.DataLoader(test_dataset, batch_size=configs.training_batch_size, shuffle=False, num_workers= 0, pin_memory=True)

    return tr, te, dataDims


def initialize_training(configs):
    tr, te, dataDims = get_dataloaders(configs)
    model = get_model(configs, dataDims)

    optimizer = optim.AdamW(model.parameters(), amsgrad=True) #optim.SGD(net.parameters(),lr=1e-4, momentum=0.9, nesterov=True)#

    return model, optimizer, dataDims


def compute_loss(output, target):
    target = target[:,:1]
    lossi = []
    lossi.append(F.cross_entropy(output, target.squeeze(1).long()))
    return torch.sum(torch.stack(lossi))


def get_training_batch_size(configs, model):
    finished = 0
    training_batch_0 = 1 * configs.training_batch_size
    #  test various batch sizes to see what we can store in memory
    dataset = build_dataset(configs)
    dataDims = dataset.dataDims
    train_size = int(0.8 * len(dataset))  # split data into training and test sets
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.Subset(dataset, [range(train_size),range(train_size,test_size + train_size)])  # split it the same way every time
    optimizer = optim.AdamW(model.parameters(),amsgrad=True) #optim.SGD(net.parameters(),lr=1e-4, momentum=0.9, nesterov=True)#

    while (configs.training_batch_size > 1) & (finished == 0):
        try:
            test_dataloader = data.DataLoader(test_dataset, batch_size=configs.training_batch_size, shuffle=False, num_workers=0, pin_memory=True)
            model_epoch(configs, dataDims = dataDims, trainData = test_dataloader, model = model, optimizer = optimizer, update_gradients = True, iteration_override = 2)
            finished = 1
        except RuntimeError: # if we get an OOM, try again with smaller batch
            configs.training_batch_size = int(np.ceil(configs.training_batch_size * 0.8)) - 1
            print('Training batch sized reduced to {}'.format(configs.training_batch_size))

    return max(int(configs.training_batch_size * 0.25),1), int(configs.training_batch_size != training_batch_0)


def model_epoch(configs, dataDims = None, trainData = None, model=None, optimizer=None, update_gradients = True, iteration_override = 0):
    if configs.CUDA:
        cuda.synchronize()  # synchronize for timing purposes
    time_tr = time.time()

    err = []
    if update_gradients:
        model.train(True)
    else:
        model.eval()

    for i, input in enumerate(trainData):
        if configs.subsample_images:
            image_depth = 80
            randind = np.random.randint(0,input.shape[-2]-max(image_depth,dataDims['conv field']))
            input = input[:,:,randind:randind+image_depth,:] # try randomly subsampling in the transverse direction as a form of regularization
        if configs.CUDA:
            input = input.cuda(non_blocking=True)

        target = input * dataDims['classes']

        output = model(input.float()) # reshape output from flat filters to channels * filters per channel
        loss = compute_loss(output, target)

        err.append(loss.data)  # record loss

        if update_gradients:
            optimizer.zero_grad()  # reset gradients from previous passes
            loss.backward()  # back-propagation
            optimizer.step()  # update parameters

        if iteration_override != 0:
            if i > iteration_override:
                break


    if configs.CUDA:
        cuda.synchronize()
    time_tr = time.time() - time_tr

    return err, time_tr


def auto_convergence(configs, epoch, tr_err_hist, te_err_hist):
    # set convergence criteria
    # if the test error has increased on average for the last x epochs
    # or if the training error has decreased by less than 1% for the last x epochs
    #train_margin = .000001  # relative change over past x runs
    # or if the training error is diverging from the test error by more than 20%
    test_margin = 10 # max divergence between training and test losses
    # configs.convergence_moving_average_window - the time over which we will average loss in order to determine convergence
    converged = 0
    if epoch > configs.convergence_moving_average_window:
        window = configs.convergence_moving_average_window
        tr_mean, te_mean = [torch.mean(torch.stack(tr_err_hist[-configs.convergence_moving_average_window:])), torch.mean(torch.stack(te_err_hist[-configs.convergence_moving_average_window:]))]
        if (torch.abs((tr_mean - tr_err_hist[-window]) / tr_mean) < configs.convergence_margin) \
                or ((torch.abs(te_mean - tr_mean) / tr_mean) > configs.convergence_margin) \
                or (epoch == configs.max_epochs)\
                or (te_mean > te_err_hist[-window]):

            converged = 1
            print('Learning converged at epoch {}'.format(epoch - window))  # print a nice message  # consider also using an accuracy metric

    return converged


def generate_samples_gated(configs, dataDims, model):
    if configs.sample_generation_mode == 'serial':
        if configs.CUDA:
            cuda.synchronize()
        time_ge = time.time()

        sample_x_padded = dataDims['sample x dim'] + 2 * dataDims['conv field'] * configs.boundary_layers
        sample_y_padded = dataDims['sample y dim'] + dataDims['conv field'] * configs.boundary_layers  # don't need to pad the bottom
        sample_conditions = dataDims['num conditioning variables']

        batches = int(np.ceil(configs.n_samples/configs.sample_batch_size))
        #n_samples = sample_batch_size * batches
        sample = torch.zeros(configs.n_samples, dataDims['channels'], dataDims['sample y dim'], dataDims['sample x dim'])  # sample placeholder
        print('Generating {} Samples'.format(configs.n_samples))

        for batch in range(batches):  # can't do these all at once so we do it in batches
            print('Batch {} of {} batches'.format(batch + 1, batches))
            sample_batch = torch.FloatTensor(configs.sample_batch_size, dataDims['channels'] + sample_conditions, sample_y_padded + 2 * dataDims['conv field'] + 1 - dataDims['conv field'] * 1, sample_x_padded + 2 * dataDims['conv field'])  # needs to be explicitly padded by the convolutional field
            sample_batch.fill_(0)  # initialize with minimum value

            if configs.do_conditioning: # assign conditions so the model knows what we want
                for i in range(len(configs.generation_conditions)):
                    sample_batch[:,1+i,:,:] = (configs.generation_conditions[i] - dataDims['conditional mean']) / dataDims['conditional std']

            if configs.CUDA:
                sample_batch = sample_batch.cuda()

            #generator.train(False)
            model.eval()
            with torch.no_grad():  # we will not be updating weights
                for i in tqdm.tqdm(range(dataDims['conv field'] + 1, sample_y_padded + dataDims['conv field'] + 1)):  # for each pixel
                    for j in range(dataDims['conv field'], sample_x_padded + dataDims['conv field']):
                        for k in range(dataDims['channels']): # should only ever be 1
                            #out = generator(sample_batch.float())
                            out = model(sample_batch[:, :, i - dataDims['conv field'] - 1:i + dataDims['conv field'] * (1 - 1) + 1, j - dataDims['conv field']:j + dataDims['conv field'] + 1].float())
                            out = torch.reshape(out, (out.shape[0], dataDims['classes'] + 1, dataDims['channels'], out.shape[-2], out.shape[-1])) # reshape to select channels
                            probs = F.softmax(out[:, 1:, k, -1, dataDims['conv field']]/1, dim=1).data # the remove the lowest element (boundary)
                            sample_batch[:, k, i, j] = (torch.multinomial(probs, 1).float() + 1).squeeze(1) / dataDims['classes']  # convert output back to training space

                            del out, probs

            for k in range(dataDims['channels']):
                sample[batch * configs.sample_batch_size:(batch + 1) * configs.sample_batch_size, k, :, :] = sample_batch[:, k, (configs.boundary_layers + 1) * dataDims['conv field'] + 1:, (configs.boundary_layers + 1) * dataDims['conv field']:-((configs.boundary_layers + 1) * dataDims['conv field'])] * dataDims['classes'] - 1  # convert back to input space

        if configs.CUDA:
            cuda.synchronize()
        time_ge = time.time() - time_ge
    elif configs.sample_generation_mode == 'parallel':
        if configs.CUDA:
            cuda.synchronize()
        time_ge = time.time()

        sample_x_padded = dataDims['sample x dim'] + 2 * dataDims['conv field'] * configs.boundary_layers
        sample_y_padded = dataDims['sample y dim'] + dataDims['conv field'] * configs.boundary_layers  # don't need to pad the bottom
        sample_conditions = dataDims['num conditioning variables']

        sample = torch.ByteTensor(configs.n_samples, dataDims['channels'], dataDims['sample y dim'], dataDims['sample x dim'])  # sample placeholder
        print('Generating {} Samples'.format(configs.n_samples))

        for image in range(configs.n_samples):  # can't do these all at once so we do it in batches
            print('Image {} of {} images'.format(image + 1, configs.n_samples))
            sample_batch = torch.FloatTensor(1, dataDims['channels'] + sample_conditions, sample_y_padded + 2 * dataDims['conv field'] + 1 - dataDims['conv field'] * 1, sample_x_padded + 2 * dataDims['conv field'])  # needs to be explicitly padded by the convolutional field
            sample_batch.fill_(0)  # initialize with minimum value

            if configs.do_conditioning: # assign conditions so the model knows what we want
                for i in range(len(configs.generation_conditions)):
                    sample_batch[:,1+i,:,:] = (configs.generation_conditions[i] - dataDims['conditional mean']) / dataDims['conditional std']

            if configs.CUDA:
                sample_batch = sample_batch.cuda()

            '''
            the key to this speedup is that workers which are separated by at least conv_field CAN work in parallel on different rows
            we can have a maximum of samble_batch_size workers, as workers reach j > conv_field, the next row (i+1) becomes available for another worker
            we will distribute workers as if they were all working on separate images, even though they are just working on different parts of the same image
            a list of rows which are available to start working on, and assign workers based on capacity
            this is all accomplished by recasting 'sample_batch' at each iteration (j) of the generator
            '''

            model.train(False)
            model.eval()
            with torch.no_grad():  # we will not be updating weights
                finished_rows = 0
                available_rows = [dataDims['conv field'] + 1]
                active_rows = []
                available_workers = configs.sample_batch_size
                row_indices = (np.zeros(sample_y_padded + 2 * dataDims['conv field'] + 1 - dataDims['conv field'] * (1-1)) + dataDims['conv field']).astype(int)
                initialized = 0
                # record = []
                pbar = barthing(total=sample_y_padded)
                while finished_rows < (sample_y_padded):  # generate row-by-row
                    # check if we have spare rows and spare workers
                    if initialized == 0:  # first row - initialization
                        # initialize a row
                        row = available_rows[0]
                        sample_bundle = sample_batch[:, :, row - dataDims['conv field']  - 1:row +dataDims['conv field'] * (1-1) + 1, int(row_indices[row] - dataDims['conv field']):int(row_indices[row] + dataDims['conv field'] + 1)] * 1

                        if configs.CUDA:
                            sample_bundle = sample_bundle.cuda()

                        available_rows = available_rows[1:]  # eliminate first element
                        active_rows.append(row)
                        available_workers -= 1
                        initialized = 1

                    elif (available_rows != []) and (available_workers > 0):
                        if active_rows[-1] < (sample_y_padded + dataDims['conv field'] - 1 + 1):  # unless we are already on the final row
                            # initialize a row
                            row = available_rows[0]
                            sample_bundle = torch.cat((sample_bundle, sample_batch[:, :, row - dataDims['conv field'] - 1:row + dataDims['conv field'] *(1-1) + 1, int(row_indices[row] - dataDims['conv field']):int(row_indices[row] + dataDims['conv field'] + 1)]) * 1, 0)
                            active_rows.append(row)
                            available_rows = available_rows[1:]  # eliminate first element
                            available_workers -= 1

                    for k in range(dataDims['channels']):  # actually do the generation
                        out = model(sample_bundle.float())  # query the network about only area within the receptive field
                        out = torch.reshape(out, (out.shape[0], dataDims['classes'] + 1, dataDims['channels'], out.shape[-2], out.shape[-1]))  # reshape to select channels

                        probs = F.softmax(out[:, 1:, k, -1, dataDims['conv field']] / 1, dim=1).data  # the remove the lowest element (boundary)
                        logits = (torch.multinomial(probs, 1).float() + 1).squeeze(1) / dataDims['classes']
                        for dep in range(sample_bundle.shape[0]):  # assign the new outputs in the right spot
                            sample_batch[:, k, active_rows[dep], row_indices[active_rows[dep]]] = logits[dep].data

                        # record.append(sample_batch[0,0,:,:].cpu().detach().numpy() * 1)

                    # check if we finished a row
                    if row_indices[active_rows[0]] == (sample_x_padded + dataDims['conv field'] - 1):
                        # only one should be possible at a time
                        # delete this row from active list and add to the finished rows
                        active_rows = active_rows[1:]  # the earliest row must be the one which has finished
                        sample_bundle = sample_bundle[1:, :, :, :]
                        finished_rows += 1
                        pbar.update(1)
                        available_workers += 1  # free up a worker

                    # check if any rows have been freed up
                    if active_rows != []:
                        if row_indices[active_rows[-1]] == (2 * dataDims['conv field'] + 1):  # if the bottom worker is more than dataDims['conv field'] from the bound (which has dataDims['conv field'] added as padding), this row comes available
                            # in fact, when working with a blind spot, we can do even better - initial_filter_size + 1 is enough
                            available_rows.append(active_rows[-1] + 1)

                    # update sample_bundle by one pixel to the right
                    for dep in range(sample_bundle.shape[0]):
                        row = active_rows[dep]
                        # update sample bundle_
                        sample_bundle[dep, :, :, :] = sample_batch[:, :, row - dataDims['conv field'] - 1: row + dataDims['conv field'] * ( 1- 1) + 1, int(row_indices[row] - dataDims['conv field'] + 1):int(row_indices[row] + dataDims['conv field'] + 1 + 1)]
                        # update row indices
                        row_indices[row] += 1

                pbar.close()

            for k in range(dataDims['channels']):
                sample[image, k, :, :] = sample_batch[:, k, (configs.boundary_layers + 1) * dataDims['conv field'] + 1:, (configs.boundary_layers + 1) * dataDims['conv field']:-((configs.boundary_layers + 1) * dataDims['conv field'])] * dataDims['classes'] - 1  # convert back to input space, +1 in y dim to get rid of first row

        if configs.CUDA:
            cuda.synchronize()
        time_ge = time.time() - time_ge
    return sample, time_ge


def generation(configs, dataDims, model, input_analysis):
    #err_te, time_te = test_net(model, te)  # clean run net

    sample, time_ge = generate_samples_gated(configs, dataDims, model)  # generate samples

    np.save('samples/run_{}_samples'.format(configs.run_num), sample)

    if len(sample) != 0:
        print('Generated samples')

        output_analysis = analyse_samples(sample)

        agreements = compute_accuracy(configs, dataDims, input_analysis, output_analysis)
        total_agreement = 0
        for i, j, in enumerate(agreements.values()):
            if np.isnan(j) != 1: # kill NaNs
                total_agreement += float(j)

        total_agreement /= len(agreements)

        print('tot = {:.4f}; den={:.2f};time_ge={:.1f}s'.format(total_agreement, agreements['density'], time_ge))
        return sample, time_ge, agreements, output_analysis

    else:
        print('Sample Generation Failed!')
        return 0, 0, 0, 0


def analyse_inputs(configs, dataDims):
    dataset = torch.Tensor(build_dataset(configs))  # get data
    dataset = dataset * (dataDims['classes'])
    input_analysis = analyse_samples(dataset)
    input_analysis['training samples'] = dataset[0:10,0]

    return input_analysis


def analyse_samples(sample):
    sample = sample[:,:1,:,:] # only analyze the first channel
    particles = int(torch.median(sample))
    sample = sample==particles # analyze binary space
    avg_density = torch.mean((sample).type(torch.float32)) # for A
    sum = torch.sum(sample)

    correlation2d, radial_correlation, correlation_bins = spatial_correlation2(sample)
    fourier2d = fourier_analysis(torch.Tensor((sample).float()))
    fourier_bins, radial_fourier = radial_fourier_analysis(fourier2d)

    sample_analysis = {}
    sample_analysis['density'] = avg_density
    sample_analysis['sum'] = sum
    sample_analysis['correlation2d'] = correlation2d
    sample_analysis['radial correlation'] = radial_correlation
    sample_analysis['correlation bins'] = correlation_bins
    sample_analysis['fourier2d'] = fourier2d
    sample_analysis['radial fourier'] = radial_fourier
    sample_analysis['fourier bins'] = fourier_bins

    return sample_analysis


def compute_accuracy(configs, dataDims, input_analysis, output_analysis):

    input_xdim, input_ydim, sample_xdim, sample_ydim = [input_analysis['fourier2d'].shape[-1], input_analysis['fourier2d'].shape[-2], output_analysis['fourier2d'].shape[-1], output_analysis['fourier2d'].shape[-2]]

    input_xdim, input_ydim, sample_xdim, sample_ydim = [input_analysis['correlation2d'].shape[-1], input_analysis['correlation2d'].shape[-2], output_analysis['correlation2d'].shape[-1], output_analysis['correlation2d'].shape[-2]]
    if configs.sample_outpaint_ratio > 1: # shrink inputs to meet outputs or vice-versa
        x_difference = sample_xdim-input_xdim
        y_difference = sample_ydim-input_ydim
        output_analysis['correlation2d'] = output_analysis['correlation2d'][y_difference//2:-y_difference//2, x_difference//2:-x_difference//2]
    elif configs.sample_outpaint_ratio < 1:
        x_difference = input_xdim - sample_xdim
        y_difference = input_ydim- sample_ydim
        input_analysis['correlation2d'] = input_analysis['correlation2d'][y_difference // 2:-y_difference // 2, x_difference // 2:-x_difference // 2]

    agreements = {}
    agreements['density'] = np.amax((1 - np.abs(input_analysis['density'] - output_analysis['density']) / input_analysis['density'],0))
    #agreements['correlation'] = np.amax((1 - np.sum(np.abs(input_analysis['correlation2d'] - output_analysis['correlation2d'])) / (np.sum(input_analysis['correlation2d']) + 1e-8),0))

    return agreements


def save_ckpt(epoch, net, optimizer, dir_name):
    torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, 'ckpts/' + dir_name[:])

def load_all_pickles(path):
    outputs = []
    print('loading all .pkl files from',path)
    files = [ f for f in listdir(path) if isfile(join(path,f)) ]
    for f in files:
        if f[-4:] in ('.pkl'):
            name = f[:-4]+'_'+f[-3:]
            print('loading', f, 'as', name)
            with open(path + '/' + f, 'rb') as f:
                outputs.append(pickle.load(f))

    return outputs

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def find_dir():
    found = 0
    ii = 0
    while found == 0:
        ii += 1
        if not os.path.exists('logfiles/run_%d' % ii):
            found = 1

    return ii

def rolling_mean(input, run):
    output = np.zeros(len(input))
    for i in range(len(output)):
        if i < run:
            output[i] = np.average(input[0:i])
        else:
            output[i] = np.average(input[i - run:i])

    return output


def get_comet_experiment(configs):
    if configs.comet:
        # Create an experiment with your api key
        experiment = Experiment(
            project_name="weld_net",
            workspace="mkilgour",
        )
        experiment.set_name(configs.experiment_name + str(configs.run_num))
        experiment.log_metrics(configs.__dict__)
        experiment.log_others(configs.__dict__)
        if configs.experiment_name[-1] == '_':
            tag = configs.experiment_name[:-1]
        else:
            tag = configs.experiment_name
        experiment.add_tag(tag)
    else:
        experiment = None

    return experiment


def superscale_image(image, f = 1):
    f = 2
    hi, wi = image.shape
    ny = hi // 2
    nx = wi // 2
    tmp = np.reshape(image, (ny, f, nx, f))
    tmp = np.repeat(tmp, f, axis=1)
    tmp = np.repeat(tmp, f, axis=3)
    tmp = np.reshape(tmp, (hi * f, wi * f))

    return tmp

def log_generation_stats(configs, epoch, experiment, sample, agreements, output_analysis):
    if configs.comet:
        for i in range(len(sample)):
            experiment.log_image(np.rot90(sample[i, 0]), name='epoch_{}_sample_{}'.format(epoch, i), image_scale=4, image_colormap='hot')
        experiment.log_metrics(agreements, epoch=epoch)

def log_input_stats(configs, experiment, input_analysis):
    if configs.comet:
        for i in range(len(input_analysis['training samples'])):
            experiment.log_image(np.rot90(input_analysis['training samples'][i]), name = 'training example {}'.format(i), image_scale=4, image_colormap='hot')


def standardize(data):
    return (data - np.mean(data)) / np.sqrt(np.var(data))