# metrics to determine the performance of our learning algorithm
import numpy as np
import torch.nn.functional as F
import os
from torch import nn, optim, cuda, backends
import torch
from torch.utils import data
import time
import pickle
from torch.utils.data import Dataset
import numpy as np
import pickle
from os import listdir
from os.path import isfile, join
import sys
import tqdm
from tqdm import tqdm as barthing
from accuracy_metrics import *
from models import *
from Image_Processing_Utils import *
import argparse



def get_input():
    parser = argparse.ArgumentParser()  # parse run index so we can parallelize submission
    parser.add_argument('--run_num', type=int, default = -1)
    cmd_line_input = parser.parse_args()
    run = cmd_line_input.run_num

    return run

class build_dataset(Dataset):
    def __init__(self, training_data, dataset_size):
        if training_data == 1:
            self.samples = np.load('data/repulsive_redo_configs2.npy', allow_pickle=True).astype('bool')
            self.samples = np.expand_dims(self.samples, axis=1)
        elif training_data == 2:
            self.samples = np.load('data/annealment_redo_configs.npy', allow_pickle=True).astype('uint8')
            self.samples = np.transpose(self.samples, [2,1,0])
            self.samples = np.expand_dims(self.samples, axis=1)
        elif training_data == 3:
            #self.samples = np.load('data/sparse_64x64_configs.npy',allow_pickle=True).astype('uint8')
            #self.samples = np.expand_dims(self.samples, axis=1)
            self.samples = np.load('Finite_T_Sample.npy',allow_pickle=True).astype('uint8')
        elif training_data == 4:
            self.samples = np.load('data/new_brains.npy',allow_pickle=True).astype('uint8')
        elif training_data == 5:
            self.samples = np.load('Augmented_Brain_Sample2.npy',allow_pickle=True).astype('uint8')
        elif training_data == 6:
            self.samples = np.load('data/drying_sample_1.npy', allow_pickle=True)
        elif training_data == 7:
            self.samples = np.load('data/drying_sample_-1.npy', allow_pickle=True)
        elif training_data == 8:
            self.samples = np.load('data/big_worm_results.npy', allow_pickle=True).astype('uint8')
            self.samples = self.samples[5:,:,:,:]
        elif training_data == 9:
            self.samples = (np.load('data/MAC3/MAC_no_tris_982.npy',allow_pickle=True))

        out_maps = len(np.unique(self.samples[0,:,:,:])) + 1
        self.samples = np.array((self.samples[0:dataset_size] + 1)/(out_maps - 1)) # normalize inputs on 0,1,2...

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def get_dir_name(model, training_data, filters, layers, dilation, filter_size, noise, den_var, dataset_size):
    dir_name = "model=%d_dataset=%d_dataset_size=%d_filters=%d_layers=%d_dilation=%d_filter_size=%d_noise=%.1f_denvar=%.1f" % (model, training_data, dataset_size, filters, layers, dilation, filter_size, noise, den_var)  # directory where tensorboard logfiles will be saved

    return dir_name


def get_model(model, filters, filter_size, layers, dilation, out_maps, channels, den_var):
    if model == 1:
        net = PixelCNN2(filters,filter_size, dilation, layers, out_maps,1,channels) # gated, without blind spot
        conv_field = int(np.sum(net.dilations)) + (filter_size - 1) // 2 #(+1 in vertical direction for top block!!)


    def init_weights(m):
        if (type(m) == nn.Conv2d) or (type(m) == MaskedConv2d):
            #torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity = 'relu')

    net.apply(init_weights) # apply xavier weights to 1x1 and 3x3 convolutions

    return net, conv_field

def get_dataloaders(training_data, training_batch, dataset_size):
    dataset = build_dataset(training_data, dataset_size)  # get data
    train_size = int(0.8 * len(dataset))  # split data into training and test sets
    test_size = len(dataset) - train_size
    #train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])  # randomly split the data into training and test sets
    train_dataset, test_dataset = torch.utils.data.Subset(dataset, [range(train_size),range(train_size,test_size + train_size)])  # split it the same way every time
    tr = data.DataLoader(train_dataset, batch_size=training_batch, shuffle=True, num_workers= 0, pin_memory=True)  # build dataloaders
    te = data.DataLoader(test_dataset, batch_size=training_batch, shuffle=False, num_workers= 0, pin_memory=True)

    return tr, te

def initialize_training(model, filters, filter_size, layers, dilation, den_var, training_data, outpaint_ratio, dataset_size):
    tr, te = get_dataloaders(training_data, 4, dataset_size)
    sample_0 = next(iter(tr))
    out_maps = len(np.unique(sample_0)) + 1
    channels = sample_0.shape[1]
    net, conv_field = get_model(model, filters, filter_size, layers, dilation, out_maps, channels, den_var)
    optimizer = optim.Adam(net.parameters(), amsgrad=True) #optim.SGD(net.parameters(),lr=1e-4, momentum=0.9, nesterov=True)#
    input_x_dim, input_y_dim = sample_0.shape[-1], sample_0.shape[-2]  # set input and output dimensions
    sample_x_dim, sample_y_dim = int(input_x_dim * outpaint_ratio), int(input_y_dim * outpaint_ratio)

    return net, conv_field, optimizer, sample_0, input_x_dim, input_y_dim, sample_x_dim, sample_y_dim

def load_checkpoint(net, optimizer, dir_name, GPU, prev_epoch):
    if os.path.exists('ckpts/' + dir_name[:]):  #reload model
        checkpoint = torch.load('ckpts/' + dir_name[:])

        if list(checkpoint['model_state_dict'])[0][0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
            for i in list(checkpoint['model_state_dict']):
                checkpoint['model_state_dict'][i[7:]] = checkpoint['model_state_dict'].pop(i)

        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        prev_epoch = checkpoint['epoch']

        if GPU == 1:
            net.cuda()  # move net to GPU
            for state in optimizer.state.values():  # move optimizer to GPU
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

        net.eval()
        print('Reloaded model: ', dir_name[:])
    else:
        print('New model: ', dir_name[:])

    return net, optimizer, prev_epoch

def compute_loss(output, target, GPU):
    lossi = []
    lossi.append(F.cross_entropy(output, target.long()))
    return torch.sum(torch.stack(lossi))

def get_training_batch_size(training_data, training_batch, model, filters, filter_size, layers, dilation, out_maps, channels, den_var, dataset_size, GPU):
    finished = 0
    training_batch_0 = 1 * training_batch
    #  test various batch sizes to see what we can store in memory
    test_dataset = build_dataset(training_data, dataset_size)
    while (training_batch > 1) & (finished == 0):
        try:
            test_batch(test_dataset, training_batch, model, filters, filter_size, layers, dilation, out_maps, channels, den_var, GPU)
            finished = 1
        except RuntimeError: # if we get an OOM, try again with smaller batch
            training_batch = int(np.ceil(training_batch * 0.8)) - 1

    return int(training_batch * 0.8), int(training_batch != training_batch_0)

def test_batch(test_dataset, training_batch, model, filters, filter_size, layers, dilation, out_maps, channels, den_var, GPU):
    net, conv_field = get_model(model, filters, filter_size, layers, dilation, out_maps, channels, den_var)
    if GPU == 1:
        net = nn.DataParallel(net)
        net.to(torch.device("cuda:0"))

    optimizer = optim.Adam(net.parameters(),amsgrad=True) #optim.SGD(net.parameters(),lr=1e-4, momentum=0.9, nesterov=True)#
    for i in range(3):
        net.train(True)
        test_dataloader = data.DataLoader(test_dataset, batch_size=training_batch, shuffle=False, num_workers=0, pin_memory=True)
        input = next(iter(test_dataloader))
        if GPU == 1:
            input = input.cuda()

        target = input.data * (out_maps - 1)  # switch from training to output space
        channels = target.shape[-3]

        output = net(input.float())  # reshape output from flat filters to channels * filters per channel
        output = torch.reshape(output, (output.shape[0], out_maps, channels, output.shape[-2], output.shape[-1]))

        loss = compute_loss(output, target, GPU)
        optimizer.zero_grad()  # reset gradients from previous passes
        loss.backward()  # back-propagation
        optimizer.step()  # update parameters

def train_net(net, optimizer, writer, tr, epoch, out_maps, noise, den_var, conv_field, GPU, cuda):
    if GPU == 1:
        cuda.synchronize()  # synchronize for timing purposes
    time_tr = time.time()

    err_tr = []
    net.train(True)
    for i, input in enumerate(tr):
        if GPU == 1:
            input = input.cuda(non_blocking=True)

        target = input.data * (out_maps - 1)  # switch from training to output space
        channels = target.shape[-3]

        if noise != 0:
            input = scramble_images(input, noise, den_var, GPU) # introduce uniform noise to training samples (second term controls magnitude), not setup for multi-channel

        output = net(input.float()) # reshape output from flat filters to channels * filters per channel
        output = torch.reshape(output, (output.shape[0], out_maps, channels, output.shape[-2], output.shape[-1]))

        loss = compute_loss(output, target, GPU)

        err_tr.append(loss.data)  # record loss
        optimizer.zero_grad()  # reset gradients from previous passes
        loss.backward()  # back-propagation
        optimizer.step()  # update parameters

        if i % 10 == 0:  # log loss to tensorboard
            writer.add_scalar('training_loss', loss.data, epoch * len(tr) + i)

    if GPU == 1:
        cuda.synchronize()
    time_tr = time.time() - time_tr

    return err_tr, time_tr

def test_net(net, writer, te, out_maps, noise, den_var, epoch, conv_field, GPU, cuda):
    if GPU == 1:
        cuda.synchronize()

    time_te = time.time()
    err_te = []
    net.eval() #train(False)
    with torch.no_grad():  # we're just computing the test set error so we won't be updating the gradients or weights
        for i, input in enumerate(te):
            if GPU == 1:
                input = input.cuda()

            target = input.data * (out_maps - 1)  # switch from training to output space
            channels = target.shape[-3]

            if noise != 0:
                input = scramble_images(input, noise, den_var, GPU) #NOT SETUP FOR MULTI-CHANNEL

            output = net(input.float())  # reshape output from flat filters to channels * filters per channel
            output = torch.reshape(output, (output.shape[0], out_maps, channels, output.shape[-2], output.shape[-1]))

            loss = compute_loss(output, target, GPU)
            err_te.append(loss.data)

            if i % 10 == 0:  # log loss to tensorboard
                writer.add_scalar('test_loss', loss.data, epoch * len(te))  # writer.add_histogram('conv1_weight', net[0].weight[0], epoch)  # if you want to watch the evolution of the filters  # writer.add_histogram('conv1_grad', net[0].weight.grad[0], epoch)

    if GPU == 1:
        cuda.synchronize()
    time_te = time.time() - time_te

    return err_te, time_te

def auto_convergence(train_margin, average_over, epoch, prev_epoch, net, optimizer, dir_name, tr_err_hist, te_err_hist, max_epochs):
    # set convergence criteria
    # if the test error has increased on average for the last x epochs
    # or if the training error has decreased by less than 1% for the last x epochs
    #train_margin = .000001  # relative change over past x runs
    # or if the training error is diverging from the test error by more than 20%
    test_margin = 10 # max divergence between training and test losses
    # average_over - the time over which we will average loss in order to determine convergence
    converged = 0
    if (epoch - prev_epoch) <= average_over:  # early checkpointing
        save_ckpt(epoch, net, optimizer, dir_name[:] +'_ckpt_-{}'.format(average_over - (epoch - prev_epoch)))

    if (epoch - prev_epoch) > average_over:
        os.remove('ckpts/'+dir_name[:]+'_ckpt_-{}'.format(average_over - 1))  # delete trailing checkpoint
        for i in range(average_over - 2, -1, -1):  # move all previous checkpoints
            os.rename('ckpts/'+dir_name[:]+'_ckpt_-{}'.format(i), 'ckpts/'+dir_name[:]+'_ckpt_-{}'.format(i + 1))
        save_ckpt(epoch, net, optimizer, dir_name[:]+'_ckpt_-{}'.format(0))  # save new checkpoint

        tr_mean, te_mean = [torch.mean(torch.stack(tr_err_hist[-average_over:])), torch.mean(torch.stack(te_err_hist[-average_over:]))]
        if (torch.abs((tr_mean - tr_err_hist[-average_over]) / tr_mean) < train_margin) or ((torch.abs(te_mean - tr_mean) / tr_mean) > test_margin) or ((epoch - prev_epoch) == max_epochs):# or (te_mean > te_err_hist[-average_over]):
            converged = 1
            if os.path.exists('ckpts/'+dir_name[:]) & (epoch > 1) & (epoch-prev_epoch < average_over): #can't happen on first epoch
                print('Previously converged this result at epoch {}!'.format(epoch - 1))
            else:
                if os.path.exists('ckpts/'+dir_name[:]): # delete any existing final checkpoint
                    os.remove('ckpts/'+dir_name[:])

                os.rename('ckpts/'+dir_name[:]+'_ckpt_-{}'.format(average_over - 1), 'ckpts/'+dir_name[:])  # save the epoch we converged at
                print('Learning converged at epoch {}'.format(epoch - average_over))  # print a nice message  # consider also using an accuracy metric

    return converged

def get_generator(model, filters, filter_size, layers, out_maps, channels, padding, GPU, net):
    return net

def build_boundary(sample_batch, sample_batch_size, training_data, conv_field, generator, bound_type, out_maps, noise, den_var, dataset_size, GPU): # 0 = empty, 1 = seed in top left only, 2 = seed + random noise with appropriate density, 3 = seed + generated

    if bound_type > 0:  # requires samples are at least as large as the convolutional receptive field, and
        tr, te = get_dataloaders(training_data, int(sample_batch_size/.2), dataset_size) # requires a sufficiently large training set or we won't saturate the seeds
        seeds = next(iter(tr))  # get seeds from training set

        if (bound_type == 1) or (bound_type == 3):
            sample_batch[:, :, 0:conv_field, 0:seeds.shape[3]] = seeds[0:sample_batch_size, :, 0:conv_field, 0:np.amin((seeds.shape[3], sample_batch.shape[3]))]  # seed from the top-left
            sample_batch[:, :, 0:seeds.shape[2], 0:conv_field] = seeds[0:sample_batch_size, :, 0:np.amin((seeds.shape[2], sample_batch.shape[2])), 0:conv_field]  # seed from the top-left

    elif bound_type == 5: # perfect graphene seeds of size up to 1400
        seeds = (torch.Tensor(np.load('data/MAC/big_perfect_graphene.npy',allow_pickle=True) == 2) + 1)/(out_maps - 1) # get seeds from training set
        sample_batch[:, :, 0:seeds.shape[2], 0:seeds.shape[3]] = seeds[0:sample_batch_size, :, 0:np.amin((seeds.shape[2], sample_batch.shape[2])), 0:np.amin((seeds.shape[3], sample_batch.shape[3]))]  # seed from the top-left
        sample_batch[:, :, 0:seeds.shape[2], 0:seeds.shape[3]] = seeds[0:sample_batch_size, :, 0:np.amin((seeds.shape[2], sample_batch.shape[2])), 0:np.amin((seeds.shape[3], sample_batch.shape[3]))]  # seed from the top-left
        sample_batch[:, :, conv_field:, conv_field:-conv_field] = 0

    return sample_batch

def get_sample_batch_size(sample_batch_size, generator, sample_x_dim, sample_y_dim, conv_field, channels, out_maps, model, GPU):
    # dynamically set sample batch size
    finished = 0
    sample_batch_size_0 = 1 * sample_batch_size
    #  test various batch sizes to see what we can store in memory
    while (sample_batch_size > 1) & (finished == 0):
        try:
            input = torch.Tensor(sample_batch_size, channels, sample_y_dim + 2 * conv_field + int(model == 1) - conv_field * int(model == 1), sample_x_dim + 2 * conv_field)
            if GPU == 1:
                input = input.cuda()

            temperature = 1
            out = generator(input[:, :, conv_field - conv_field - int(model == 1) + 1:conv_field + conv_field * (1 - int(model == 1)) + 1, conv_field - conv_field:conv_field + conv_field + 1].float())  # query the network about only area within the receptive field
            out = torch.reshape(out, (out.shape[0], out_maps, channels, out.shape[-2], out.shape[-1]))  # reshape to select channels
            if temperature == 0:
                normed_temp = 1
            else:
                normed_temp = torch.mean(torch.abs(out[:, 1:, 0, 0, 0])) * (temperature)  # + np.exp(- i/conv_field/2)) # normalize temperature, graded against the boundary
            probs = F.softmax(out[:, 1:, 0, 0, 0] / normed_temp, dim=1).data  # the remove the lowest element (boundary)
            aaaa = (torch.multinomial(probs, 1).float() + 1).squeeze(1) / (out_maps - 1)
            finished = 1
        except RuntimeError:  # if we get an OOM, try again with smaller batch
            sample_batch_size = int(np.ceil(sample_batch_size * 0.9)) - 1

    return int(np.ceil(sample_batch_size)), int(sample_batch_size != sample_batch_size_0)

def generate_samples_gated(generation_type, n_samples, sample_batch_size, sample_x_dim, sample_y_dim, conv_field, generator, bound_type, GPU, cuda, training_data, out_maps, boundary_layers, noise, den_var, channels, temperature, dataset_size, model):
    if generation_type == 1:
        if GPU == 1:
            cuda.synchronize()
        time_ge = time.time()

        sample_x_padded = sample_x_dim + 2 * conv_field * boundary_layers
        sample_y_padded = sample_y_dim + conv_field * boundary_layers  # don't need to pad the bottom

        sample_batch_size, changed = get_sample_batch_size(sample_batch_size, generator, sample_x_padded, sample_y_padded, conv_field, channels, out_maps, model, GPU) # add extra padding by conv_field in both x-directions, and in the + y direction, which we will remove later
        sample_batch_size = np.int(np.ceil(sample_batch_size // 2))  # gated generation is twice as expensive!
        if changed:
            print('Sample batch size changed to {}'.format(sample_batch_size))


        batches = int(np.ceil(n_samples/sample_batch_size))
        #n_samples = sample_batch_size * batches
        sample = torch.ByteTensor(n_samples, channels, sample_y_dim, sample_x_dim)  # sample placeholder
        print('Generating {} Samples'.format(n_samples))

        for batch in range(batches):  # can't do these all at once so we do it in batches
            print('Batch {} of {} batches'.format(batch + 1, batches))
            sample_batch = torch.FloatTensor(sample_batch_size, channels, sample_y_padded + 2 * conv_field + int(model == 1) - conv_field * int(model == 1), sample_x_padded + 2 * conv_field)  # needs to be explicitly padded by the convolutional field
            sample_batch.fill_(0)  # initialize with minimum value

            if bound_type > 0:
                sample_batch = build_boundary(sample_batch, sample_batch_size, training_data, conv_field, generator, bound_type, out_maps, noise, den_var, dataset_size, GPU)

            if GPU == 1:
                sample_batch = sample_batch.cuda()

            #generator.train(False)
            generator.eval()
            with torch.no_grad():  # we will not be updating weights
                for i in tqdm.tqdm(range(conv_field + int(model == 1), sample_y_padded + conv_field + int(model == 1))):  # for each pixel
                    for j in range(conv_field, sample_x_padded + conv_field):
                        for k in range(channels):
                            #out = generator(sample_batch.float())
                            out = generator(sample_batch[:, :, i - conv_field - int(model == 1):i + conv_field * (1-int(model == 1)) + 1, j - conv_field:j + conv_field + 1].float())
                            out = torch.reshape(out, (out.shape[0], out_maps, channels, out.shape[-2], out.shape[-1])) # reshape to select channels
                            if temperature == 0:
                                normed_temp = 1
                            else:
                                normed_temp = torch.mean(torch.abs(out[:, 1:, k, -1, conv_field])) * (temperature)# + np.exp(- i/conv_field/2)) # normalize temperature, graded against the boundary
                            probs = F.softmax(out[:, 1:, k, -1, conv_field]/normed_temp, dim=1).data # the remove the lowest element (boundary)
                            sample_batch[:, k, i, j] = (torch.multinomial(probs, 1).float() + 1).squeeze(1) / (out_maps -1)  # convert output back to training space

                            del out, probs

            for k in range(channels):
                sample[batch * sample_batch_size:(batch + 1) * sample_batch_size, k, :, :] = sample_batch[:, k, (boundary_layers + 1) * conv_field + int(model == 1):, (boundary_layers + 1) * conv_field:-((boundary_layers + 1) * conv_field)] * (out_maps - 1) - 1  # convert back to input space

        if GPU == 1:
            cuda.synchronize()
        time_ge = time.time() - time_ge
    elif generation_type == 2:
        if GPU == 1:
            cuda.synchronize()
        time_ge = time.time()

        sample_x_padded = sample_x_dim + 2 * conv_field * boundary_layers
        sample_y_padded = sample_y_dim + conv_field * boundary_layers  # don't need to pad the bottom

        sample_batch_size, changed = get_sample_batch_size(sample_batch_size, generator, sample_x_padded, sample_y_padded, conv_field, channels, out_maps, model, GPU)  # add extra padding by conv_field in both x-directions, and in the + y direction, which we will remove later
        print('{} workers available for generation'.format(sample_batch_size))

        sample = torch.ByteTensor(n_samples, channels, sample_y_dim, sample_x_dim)  # sample placeholder
        print('Generating {} Samples'.format(n_samples))

        for image in range(n_samples):  # can't do these all at once so we do it in batches
            print('Image {} of {} images'.format(image + 1, n_samples))
            sample_batch = torch.FloatTensor(1, channels, sample_y_padded + 2 * conv_field + int(model == 1) - conv_field * int(model == 1), sample_x_padded + 2 * conv_field)  # needs to be explicitly padded by the convolutional field
            sample_batch.fill_(0)  # initialize with minimum value

            if bound_type > 0:
                sample_batch = build_boundary(sample_batch, 1, training_data, conv_field, generator, bound_type, out_maps, noise, den_var, dataset_size, GPU)

            if GPU == 1:
                sample_batch = sample_batch.cuda()

            '''
            the key to this speedup is that workers which are separated by at least conv_field CAN work in parallel on different rows
            we can have a maximum of samble_batch_size workers, as workers reach j > conv_field, the next row (i+1) becomes available for another worker
            we will distribute workers as if they were all working on separate images, even though they are just working on different parts of the same image
            a list of rows which are available to start working on, and assign workers based on capacity
            this is all accomplished by recasting 'sample_batch' at each iteration (j) iteration of the generator
            '''

            generator.train(False)
            generator.eval()
            with torch.no_grad():  # we will not be updating weights
                finished_rows = 0
                available_rows = [conv_field + int(model == 1)]
                active_rows = []
                available_workers = sample_batch_size
                row_indices = (np.zeros(sample_y_padded + 2 * conv_field + int(model == 1) - conv_field * (1-int(model == 1))) + conv_field).astype(int)
                initialized = 0
                # record = []
                pbar = barthing(total=sample_y_padded)
                while finished_rows < (sample_y_padded):  # generate row-by-row
                    # check if we have spare rows and spare workers
                    if initialized == 0:  # first row - initialization
                        # initialize a row
                        row = available_rows[0]
                        sample_bundle = sample_batch[:, :, row - conv_field  - int(model == 1):row +conv_field * (1-int(model == 1)) + 1, int(row_indices[row] - conv_field):int(row_indices[row] + conv_field + 1)] * 1

                        if GPU == 1:
                            sample_bundle = sample_bundle.cuda()

                        available_rows = available_rows[1:]  # eliminate first element
                        active_rows.append(row)
                        available_workers -= 1
                        initialized = 1

                    elif (available_rows != []) and (available_workers > 0):
                        if active_rows[-1] < (sample_y_padded + conv_field - 1 + int(model == 1)):  # unless we are already on the final row
                            # initialize a row
                            row = available_rows[0]
                            sample_bundle = torch.cat((sample_bundle, sample_batch[:, :, row - conv_field - int(model == 1):row + conv_field *(1-int(model == 1)) + 1, int(row_indices[row] - conv_field):int(row_indices[row] + conv_field + 1)]) * 1, 0)
                            active_rows.append(row)
                            available_rows = available_rows[1:]  # eliminate first element
                            available_workers -= 1

                    for k in range(channels):  # actually do the generation
                        out = generator(sample_bundle.float())  # query the network about only area within the receptive field
                        out = torch.reshape(out, (out.shape[0], out_maps, channels, out.shape[-2], out.shape[-1]))  # reshape to select channels
                        if temperature == 0:
                            normed_temp = 1
                        else:
                            normed_temp = torch.mean(torch.abs(out[:, 1:, k, -1, conv_field])) * (temperature)  # + np.exp(- i/conv_field/2)) # normalize temperature, graded against the boundary
                        probs = F.softmax(out[:, 1:, k, -1, conv_field] / normed_temp, dim=1).data  # the remove the lowest element (boundary)
                        logits = (torch.multinomial(probs, 1).float() + 1).squeeze(1) / (out_maps - 1)
                        for dep in range(sample_bundle.shape[0]):  # assign the new outputs in the right spot
                            sample_batch[:, k, active_rows[dep], row_indices[active_rows[dep]]] = logits[dep].data

                        # record.append(sample_batch[0,0,:,:].cpu().detach().numpy() * 1)

                    # check if we finished a row
                    if row_indices[active_rows[0]] == (sample_x_padded + conv_field - 1):
                        # only one should be possible at a time
                        # delete this row from active list and add to the finished rows
                        active_rows = active_rows[1:]  # the earliest row must be the one which has finished
                        sample_bundle = sample_bundle[1:, :, :, :]
                        finished_rows += 1
                        pbar.update(1)
                        available_workers += 1  # free up a worker

                    # check if any rows have been freed up
                    if active_rows != []:
                        if row_indices[active_rows[-1]] == (2 * conv_field + 1):  # if the bottom worker is more than conv_field from the bound (which has conv_field added as padding), this row comes available
                            # in fact, when working with a blind spot, we can do even better - initial_filter_size + 1 is enough
                            available_rows.append(active_rows[-1] + 1)

                    # update sample_bundle by one pixel to the right
                    for dep in range(sample_bundle.shape[0]):
                        row = active_rows[dep]
                        # update sample bundle_
                        sample_bundle[dep, :, :, :] = sample_batch[:, :, row - conv_field - int(model == 1): row + conv_field * ( 1- int(model == 1)) + 1, int(row_indices[row] - conv_field + 1):int(row_indices[row] + conv_field + 1 + 1)]
                        # update row indices
                        row_indices[row] += 1

                pbar.close()

            for k in range(channels):
                sample[image, k, :, :] = sample_batch[:, k, (boundary_layers + 1) * conv_field + int(model == 1):, (boundary_layers + 1) * conv_field:-((boundary_layers + 1) * conv_field)] * (out_maps - 1) - 1  # convert back to input space, +1 in y dim to get rid of first row

        if GPU == 1:
            cuda.synchronize()
        time_ge = time.time() - time_ge
    return sample, time_ge, sample_batch_size, n_samples

def generation(generation_type, dir_name, input_analysis, outpaint_ratio, epoch, model, filters, filter_size, layers, net, writer, te, out_maps, noise, den_var, conv_field, sample_x_dim, sample_y_dim, n_samples, sample_batch_size, bound_type, training_data, boundary_layers, channels, softmax_temperature, dataset_size, GPU, cuda, TB):
    err_te, time_te = test_net(net, writer, te, out_maps, noise, den_var, epoch, conv_field, GPU, cuda)  # clean run net

    net = get_generator(model, filters, filter_size, layers, out_maps, channels, 0, GPU, net)
    sample, time_ge, sample_batch_size, n_samples = generate_samples_gated(generation_type, n_samples, sample_batch_size, sample_x_dim, sample_y_dim, conv_field, net, bound_type, GPU, cuda, training_data, out_maps, boundary_layers, noise, den_var, channels, softmax_temperature, dataset_size, model)  # generate samples



    np.save('samples/' + dir_name[:] + '_T=%.3f'%softmax_temperature, sample)
    if n_samples != 0:
        print('Generated samples')

        output_analysis = analyse_samples(sample, training_data)
        save_outputs(dir_name, output_analysis, sample, softmax_temperature, epoch, TB)

        agreements = compute_accuracy(input_analysis, output_analysis, outpaint_ratio, training_data)
        total_agreement = 0
        for i, j, in enumerate(agreements.values()):
            if np.isnan(j) != 1: # kill NaNs
                total_agreement += float(j)

        total_agreement /= len(agreements)

        if training_data >= 9:#(training_data == 10 or training_data == 9):
            print('tot = {:.4f}; den={:.2f}; b_order={:.2f}; b_length={:.2f}; b_angle={:.2f}; corr={:.2f}; fourier={:.2f}; time_ge={:.1f}s'.format(total_agreement, agreements['density'], agreements['order'], agreements['bond'], agreements['angle'], agreements['correlation'], agreements['fourier'], time_ge))
        else:
            print('tot = {:.4f}; den={:.2f}; en={:.2f}; corr={:.2f}; fourier={:.2f}; time_ge={:.1f}s'.format(total_agreement, agreements['density'], agreements['energy'], agreements['correlation'], agreements['fourier'], time_ge))

    return sample, time_ge, n_samples, agreements, output_analysis

def analyse_inputs(training_data, out_maps, dataset_size):
    dataset = torch.Tensor(build_dataset(training_data, dataset_size))  # get data
    dataset = dataset * (out_maps - 1) - 1
    input_analysis = analyse_samples(dataset, training_data)

    return input_analysis

def analyse_samples(sample, training_data):
    sample = sample[:,0,:,:].unsqueeze(1) # for now only analyze the first dimension
    particles = int(torch.max(sample))
    sample = sample==particles
    avg_density = torch.mean((sample).type(torch.float32)) # for A
    sum = torch.sum(sample[:,0,:,:],0)
    variance = torch.var(sum/torch.mean(sum + 1e-5))
    correlation2d, radial_correlation, correlation_bins = spatial_correlation2(sample)
    fourier2d = fourier_analysis(torch.Tensor((sample).float()))
    fourier_bins, radial_fourier = radial_fourier_analysis(fourier2d)

    if training_data >= 9: # carbon-based data  #(training_data == 10 or training_data == 9):
        avg_bond_order, bond_order_dist, avg_bond_length, avg_bond_angle, bond_length_dist, bond_angle_dist = bond_analysis(sample, 1.7, particles)
    else:
        avg_interactions, en_dist = compute_interactions(sample)

    sample_analysis = {}
    sample_analysis['density'] = avg_density
    sample_analysis['sum'] = sum
    sample_analysis['variance'] = variance
    sample_analysis['correlation2d'] = correlation2d
    sample_analysis['radial correlation'] = radial_correlation
    sample_analysis['correlation bins'] = correlation_bins
    sample_analysis['fourier2d'] = fourier2d
    sample_analysis['radial fourier'] = radial_fourier
    sample_analysis['fourier bins'] = fourier_bins
    if training_data >= 9:#(training_data == 10 or training_data == 9):
        sample_analysis['average bond order'] = avg_bond_order
        sample_analysis['bond order dist'] = bond_order_dist
        sample_analysis['average bond length'] = avg_bond_length
        sample_analysis['bond length dist'] = bond_length_dist
        sample_analysis['average bond angle'] = avg_bond_angle
        sample_analysis['bond angle dist'] = bond_angle_dist
    else:
        sample_analysis['average interactions'] = avg_interactions
        sample_analysis['interactions dist'] = en_dist

    return sample_analysis

def compute_accuracy(input_analysis, output_analysis, outpaint_ratio, training_data):
    input_xdim, input_ydim, sample_xdim, sample_ydim = [input_analysis['fourier2d'].shape[-1], input_analysis['fourier2d'].shape[-2], output_analysis['fourier2d'].shape[-1], output_analysis['fourier2d'].shape[-2]]

    if outpaint_ratio > 1: # shrink inputs to meet outputs or vice-versa
        x_difference = sample_xdim-input_xdim
        y_difference = sample_ydim-input_ydim
        output_analysis['fourier2d'] = output_analysis['fourier2d'][y_difference//2:-y_difference//2, x_difference//2:-x_difference//2]
    elif outpaint_ratio < 1:
        x_difference = input_xdim - sample_xdim
        y_difference = input_ydim- sample_ydim
        input_analysis['fourier2d'] = input_analysis['fourier2d'][y_difference // 2:-y_difference // 2, x_difference // 2:-x_difference // 2]

    input_xdim, input_ydim, sample_xdim, sample_ydim = [input_analysis['correlation2d'].shape[-1], input_analysis['correlation2d'].shape[-2], output_analysis['correlation2d'].shape[-1], output_analysis['correlation2d'].shape[-2]]
    if outpaint_ratio > 1: # shrink inputs to meet outputs or vice-versa
        x_difference = sample_xdim-input_xdim
        y_difference = sample_ydim-input_ydim
        output_analysis['correlation2d'] = output_analysis['correlation2d'][y_difference//2:-y_difference//2, x_difference//2:-x_difference//2]
    elif outpaint_ratio < 1:
        x_difference = input_xdim - sample_xdim
        y_difference = input_ydim- sample_ydim
        input_analysis['correlation2d'] = input_analysis['correlation2d'][y_difference // 2:-y_difference // 2, x_difference // 2:-x_difference // 2]

    agreements = {}
    agreements['density'] = np.amax((1 - np.abs(input_analysis['density'] - output_analysis['density']) / input_analysis['density'],0))
    agreements['fourier'] = np.amax((1 - np.sum(np.abs(input_analysis['fourier2d'] - output_analysis['fourier2d'])) / (np.sum(input_analysis['fourier2d']) + 1e-8),0))
    agreements['correlation'] = np.amax((1 - np.sum(np.abs(input_analysis['correlation2d'] - output_analysis['correlation2d'])) / (np.sum(input_analysis['correlation2d']) + 1e-8),0))

    if training_data >= 9:#(training_data == 10 or training_data == 9):
        agreements['order'] = np.amax((1 - np.average(np.abs(input_analysis['bond order dist'][0] - output_analysis['bond order dist'][0])) / np.average(input_analysis['bond order dist'][0]), 0))
        agreements['bond'] = np.amax((1 - np.average(np.abs(input_analysis['bond length dist'][0] - output_analysis['bond length dist'][0])) / np.average(input_analysis['bond length dist'][0]), 0))
        agreements['angle'] = np.amax((1 - np.average(np.abs(input_analysis['bond angle dist'][0] - output_analysis['bond angle dist'][0])) / np.average(input_analysis['bond angle dist'][0]), 0))
    else:
        agreements['energy'] = np.amax((1 - np.average(np.abs(input_analysis['interactions dist'][0] - output_analysis['interactions dist'][0])) / np.average(input_analysis['interactions dist'][0]), 0))

    return agreements

def write_inputs(layers, filters, max_epochs, n_samples, filter_size, writer):
    writer.add_text('layers', '%d' % layers)
    writer.add_text('feature maps', '%d' % filters)
    writer.add_text('epochs', '%d' % max_epochs)
    writer.add_text('# samples', '%d' % n_samples)
    writer.add_text('filter size', '%d' % filter_size)

def save_outputs(dir_name, sample_analysis, sample, softmax_temperature, epoch, TB):
    # save to file
    output = {}
    output['sample analysis'] = sample_analysis
    output['sample'] = sample

    with open('outputs/'+dir_name[:] + '_T=%.3f'%softmax_temperature +'_epoch=%d'%epoch+'.pkl', 'wb') as f:
        pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)

    # to load
    '''
    #with open('outputs/' + dir_name[:] + '.pkl', 'rb') as f:
        outputs = pickle.load(f)
    '''

    if TB == 1:  # save to tensorboard #DEPRECATED
        '''
        #writer.add_scalar('pooled spatial variance', pooled_variance, epoch)
        #writer.add_scalar('spatial variance', sum_variance, epoch)
        writer.add_scalar('fourier overlap', fourier_overlap, epoch)
        #writer.add_scalar('average interactions', avg_interactions, epoch)
        #writer.add_scalar('average density', avg_density, epoch)
        writer.add_scalar('density overlap', density_agreement, epoch)
        writer.add_scalar('interactions overlap', interactions_overlap, epoch)
        writer.add_scalar('variance overlap', variance_overlap, epoch)
        writer.add_image('fourier transform', np.log(image_transform)/np.log(image_transform.max()), epoch, dataformats='HW')
        writer.add_image('pooled distribution', pooled_dist, epoch, dataformats='HW')
        writer.add_image('spatial distribution', sum_dist, epoch, dataformats='HW')
        writer.add_images('samples', sample[0:64,:,:,:], epoch)
        '''

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
