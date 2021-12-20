import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.special as special
from itertools import combinations

class MaskedConv2d_h(nn.Conv2d):  # add a mask to the regular Conv2D function, so that it cannot learn about the future
    def __init__(self, mask_type, channels, *args, **kwargs):
        super(MaskedConv2d_h, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}  # mask A is for the first convolutional layer only, all deeper layers use mask B
        _, _, kH, kW = self.weight.size()  # get the size of the convolutional filter
        self.register_buffer('mask', self.weight.data.clone())  # initialize mask
        self.mask.fill_(1)  # start building the masks

        # spatial masking - prevent information from neighbours
        if mask_type=='A':
            self.mask[:, :, :, -1] = 0  # mask type B allows access to the 'present' pixel, mask A does not

        if channels > 1:
            # channel masking - block information from nearby color channels - ONLY 2 CHANNELS
            ''' 
            filters will be stacked as x1,x2,x3,x1,x2,x3,... , therefore, we will mask such that 
            e.g. filter 2 serving x2 can see previous outputs from x3, but not x1
            we will achieve this by building a connections graph, which will zero-out all elements from given channels 
            '''
            # mask A only allows information from lower channels
            Cin = self.mask.shape[1] # input filters
            Cout = self.mask.shape[0] # output filters
            def channel_mask(i_out, i_in): # a map which defines who is allowed to see what
                cout_idx = np.expand_dims(np.arange(Cout) % 2 == i_out, 1)
                cin_idx = np.expand_dims(np.arange(Cin) % 2 == i_in, 0)
                a1, a2 = np.broadcast_arrays(cout_idx, cin_idx)
                return a1 * a2

            mask = np.array(self.mask)
            for c in range(2): # mask B allows information from current and lower channels
                mask[channel_mask(c, c), kH // 2, kW // 2] = 0.0 if mask_type == 'A' else 1.0

            mask[channel_mask(0, 1), kH // 2, kW // 2] = 0.0
            self.mask = torch.from_numpy(mask)
    def forward(self, x):
        self.weight.data *= self.mask  # at each forward pass, apply the mask to all the filters (zero-out information about the future)
        return super(MaskedConv2d_h, self).forward(x)

class MaskedConv2d_v(nn.Conv2d):  # purposely blind the v-stack at selected intervals
    def __init__(self, *args, **kwargs):
        super(MaskedConv2d_v, self).__init__(*args, **kwargs)
        _, _, kH, kW = self.weight.size()  # get the size of the convolutional filter
        self.register_buffer('mask', self.weight.data.clone())  # initialize mask
        self.mask.fill_(1)  # start building the masks

        # spatial masking - prevent information from neighbours
        self.mask[:, :, -1, kW // 2 + 1:] = 0  # since this is a v-stack convolution, we only blind the final row


    def forward(self, x):
        self.weight.data *= self.mask  # at each forward pass, apply the mask to all the filters (zero-out information about the future)
        return super(MaskedConv2d_v, self).forward(x)

class DoubleMaskedConv2d(nn.Conv2d):  # adds to regular masked conv2d by masking also the input in subsequent layers (densenet only)
    def __init__(self, mask_type, *args, **kwargs):
        super(DoubleMaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}  # mask A is for the first convolutional layer only, all deeper layers use mask B
        _, _, self.kH, self.kW = self.weight.size()  # get the size of the convolutional filter
        self.register_buffer('mask', self.weight.data.clone())  # initialize mask
        self.mask.fill_(1)  # start building the masks
        self.mask[:, :, self.kH // 2, self.kW // 2 + (mask_type == 'B'):] = 0  # mask type B allows access to the 'present' pixel, mask A does not
        self.mask[:, :, self.kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask  # at each forward pass, apply the mask to all the filters (zero-out information about the future)
        self.weight.data[0,:, self.kH//2, self.kW//2] *=0 # mask the central pixel of the first filter (which will always be the input in a densent)
        return super(DoubleMaskedConv2d, self).forward(x)

class MaskedPointwiseConv2d(nn.Conv2d):  # adds to regular masked conv2d by masking also the input in subsequent layers (densenet only)
    def __init__(self, *args, **kwargs):
        super(MaskedPointwiseConv2d, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data[:,0, 0, 0] *=0 # mask the entirety of the first filter (which will always be the input in a densenet)
        return super(MaskedPointwiseConv2d, self).forward(x)

def gated_activation(input):
    # implement gated activation from Conditional Generation with PixelCNN Encoders
    assert (input.shape[1] % 2) == 0
    a, b = torch.chunk(input, 2, 1) # split input into two equal parts - only works for even number of filters
    a = torch.tanh(a)
    b = torch.sigmoid(b)

    return torch.mul(a,b) # return element-wise (sigmoid-gated) product

class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return gated_activation(input)



class param_activation(nn.Module): # a better (pytorch-friendly) implementation of activation as a linear combination of basis functions
    def __init__(self, n_basis, span, channels, *args, **kwargs):
        super(param_activation, self).__init__(*args, **kwargs)

        self.channels, self.n_basis = channels, n_basis
        # define the space of basis functions
        self.register_buffer('dict', torch.linspace(-span, span, n_basis))# + 1)) # positive and negative values for Dirichlet Kernel
        #self.register_buffer('dict', torch.linspace(1, span, n_basis + 1)) # positive values for ReLU

        # define module to learn parameters
        # 1d convolutions allow for grouping of terms, unlike nn.linear which is always fully-connected.
        # #This way should be fast and efficient, and play nice with pytorch optim
        self.linear = nn.Conv1d(channels * (n_basis), channels, kernel_size=(1,1), groups=int(channels), bias=False)

        #nn.init.normal(self.linear.weight.data, std=0.1)

        self.eps = 1e-8

    def kernel(self, x):
        # x has dimention batch, features, y, x
        # must return object of dimension batch, features, y, x, basis
        x = x.unsqueeze(4)
        #return (torch.sin(x * (self.dict+0.5))+self.eps)/(2 * np.pi * torch.sin(x/2) + self.eps) # dirichlet kernel
        #return F.relu(x) / (self.dict)
        return torch.exp(-(x-self.dict)**2)

    def forward(self, x):
        x = self.kernel(x) # run activation, output shape batch, features, y, x, basis
        x = x.permute(0,1,4,2,3).reshape(x.shape[0],x.shape[1]*x.shape[-1],x.shape[2],x.shape[3]) # concatenate basis functions with filters
        x = self.linear(x) # apply linear coefficients and sum

        #y = torch.zeros((x.shape[0], self.channels, x.shape[-2], x.shape[-1])).cuda() #initialize output
        #for i in range(self.channels):
        #    y[:,i,:,:] = self.linear[i](x[:,i,:,:,:]).squeeze(-1) # multiply coefficients channel-wise (probably slow)

        return x


class Activation(nn.Module):
    def __init__(self, activation_func, filters, *args, **kwargs):
        super().__init__()
        if activation_func == 'gated':
            self.activation = gated_activation
        elif activation_func == 'relu':
            self.activation = F.relu
        elif activation_func == 'dirichlet':
            self.activation = param_activation(n_basis=12, span=4, channels=filters)

    def forward(self, input):
        return self.activation(input)


class MaskedConv2d(nn.Conv2d):  # add a mask to the regular Conv2D function, so that it cannot learn about the future
    def __init__(self, mask_type, channels, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}  # mask A is for the first convolutional layer only, all deeper layers use mask B
        _, _, kH, kW = self.weight.size()  # get the size of the convolutional filter
        self.register_buffer('mask', self.weight.data.clone())  # initialize mask
        self.mask.fill_(1)  # start building the masks

        # spatial masking - prevent information from neighbours
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0  # mask type B allows access to the 'present' pixel, mask A does not
        self.mask[:, :, kH // 2 + 1:] = 0

        if channels > 1:
            # channel masking - block information from nearby color channels - ONLY 2 CHANNELS
            ''' 
            filters will be stacked as x1,x2,x3,x1,x2,x3,... , therefore, we will mask such that 
            e.g. filter 2 serving x2 can see previous outputs from x3, but not x1
            we will achieve this by building a connections graph, which will zero-out all elements from given channels 
            '''
            # mask A only allows information from lower channels
            Cin = self.mask.shape[1] # input filters
            Cout = self.mask.shape[0] # output filters
            def channel_mask(i_out, i_in): # a map which defines who is allowed to see what
                cout_idx = np.expand_dims(np.arange(Cout) % 2 == i_out, 1)
                cin_idx = np.expand_dims(np.arange(Cin) % 2 == i_in, 0)
                a1, a2 = np.broadcast_arrays(cout_idx, cin_idx)
                return a1 * a2

            mask = np.array(self.mask)
            for c in range(2): # mask B allows information from current and lower channels
                mask[channel_mask(c, c), kH // 2, kW // 2] = 0.0 if mask_type == 'A' else 1.0

            mask[channel_mask(0, 1), kH // 2, kW // 2] = 0.0
            self.mask = torch.from_numpy(mask)
    def forward(self, x):
        self.weight.data *= self.mask  # at each forward pass, apply the mask to all the filters (zero-out information about the future)
        return super(MaskedConv2d, self).forward(x)


class StackedConvolution(nn.Module):
    def __init__(self, f_in, f_out, kernel_size, padding, dilation, activation, *args, **kwargs):
        super(StackedConvolution, self).__init__(*args, **kwargs)

        self.padding = padding
        self.act_func = activation
        self.pad = dilation * (kernel_size // 2)
        self.v_activation = Activation(self.act_func, f_out) # for ReLU, must change number of filters as gated approach halves filters on each application
        self.h_activation = Activation(self.act_func, f_out)
        if activation == 'gated': # filter ratio - need to double a bunch of filters for gated activation
            f_rat = 2
        else:
            f_rat = 1
        self.v_Conv2d = nn.Conv2d(f_in, f_rat * f_out, (kernel_size//2 + 1, kernel_size), 1, (padding * (self.pad), padding * self.pad), dilation, bias=False, padding_mode='zeros')
        self.v_to_h_fc = nn.Conv2d(f_rat * f_out, f_rat * f_out, 1, bias=False)
        #self.h_Conv2d = nn.Conv2d(f_in, f_rat * f_out, (1, kernel_size // 2 + 1), 1, (0, padding * self.pad), dilation, bias=True, padding_mode='zeros')
        self.h_Conv2d = nn.Conv2d(f_in, f_rat * f_out, (1, kernel_size // 2 + 1), 1, (0, padding * self.pad), dilation, bias=False, padding_mode='zeros')
        self.h_to_skip = nn.Conv2d(f_out, f_out, 1, bias=False)
        self.h_to_h = nn.Conv2d(f_out, f_out, 1, bias=False)

    def forward(self, v_in, h_in):
        residue = h_in.clone() # residual track

        if self.padding == 0:
            v_in = self.v_Conv2d(v_in) # remove extra padding
            v_to_h = self.v_to_h_fc(v_in)#[:,:,:-1,:] # align v stack to h
            h_in = self.h_Conv2d(h_in)[:, :, (self.pad):, :-self.pad]  # unpad by 1 on rhs
            residue = residue[:,:,self.pad:,self.pad:-self.pad]
        else:
            v_in = self.v_Conv2d(v_in)[:, :, :-(self.pad), :]  # remove extra padding
            v_to_h = self.v_to_h_fc(v_in)#[:,:,:-1,:] # align v stack to h
            h_in = self.h_Conv2d(h_in)[:, :, :, :-self.pad]  # unpad by 1 on rhs
        h_out = self.h_activation(torch.add(h_in, v_to_h))
        v_out = self.v_activation(v_in)

        skip = self.h_to_skip(h_out)
        h_out = self.h_to_h(h_out)
        h_out = torch.add(h_out, residue)

        return v_out, h_out, skip


class PixelCNNWIP(nn.Module):  # Dense or residual, gated, blocked, dilated PixelCNN with batchnorm
    def __init__(self, filters, initial_convolution_size, dilation, layers, out_maps, padding, channels, activation):
        super(PixelCNNWIP, self).__init__()

        ### initialize
        if activation == 1:
            self.act_func = 'relu'
        elif activation == 2:
            self.act_func = 'gated'
        elif activation == 3:
            self.act_func = 'dirichlet'

        if self.act_func == 'gated': # filter ratio - need to double a bunch of filters for gated activation
            f_rat = 2
        else:
            f_rat = 1
        self.initial_pad = (initial_convolution_size - 1) // 2
        kernel_size = 3 #initial_convolution_size
        self.padding = padding

        # layer & block structure
        self.blocks = dilation # each block is a different dilation
        self.layers_per_block = layers // self.blocks
        initial_filters = filters
        f_in = (np.ones(layers + 1) * filters).astype(int)
        f_out = (np.ones(layers + 1) * filters).astype(int)
        self.h_init_activation = Activation(self.act_func, initial_filters)
        self.v_init_activation = Activation(self.act_func, initial_filters)


        # apply dilations
        self.dilations = (np.ones((layers // dilation) * dilation)).astype(int) # split to whole number blocks
        for i in range(self.blocks): # apply dilation to each block
            for j in range(self.layers_per_block):
                self.dilations[i * self.layers_per_block + j] = i+1


        # initial layer
        self.v_initial_convolution = nn.Conv2d(channels, f_rat * initial_filters, (self.initial_pad + 1, initial_convolution_size), 1, (padding * (self.initial_pad) + 1, padding * self.initial_pad), padding_mode='zeros', bias=False)
        self.v_to_h_initial = nn.Conv2d(f_rat * initial_filters, f_rat * initial_filters,1, bias=False)
        self.h_initial_convolution = MaskedConv2d_h('A', channels, channels, f_rat * initial_filters, (1, self.initial_pad+1), 1, (0, padding * self.initial_pad), padding_mode='zeros', bias=False)
        self.h_to_skip_initial = nn.Conv2d(initial_filters, initial_filters, 1, bias=False)
        self.h_to_h_initial = nn.Conv2d(initial_filters, initial_filters, 1, bias=False)

        # stack hidden layers in blocks
        self.conv_layer = []
        for j in range(self.blocks):
            self.conv_layer.append([StackedConvolution(f_in[i], f_out[i], kernel_size, padding, self.dilations[i + j * self.layers_per_block], self.act_func) for i in range(self.layers_per_block)]) # stacked convolution (no blind spot)
        for j in range(self.blocks):
            self.conv_layer[j] = nn.ModuleList(self.conv_layer[j])
        self.conv_layer = nn.ModuleList(self.conv_layer)

        #output layers
        fc_filters = 64
        #self.fc_activation = Activation(self.act_func, fc_filters // f_rat)
        self.fc1 = nn.Conv2d(f_out[-1], fc_filters, (1,1), bias=True) # add skip connections
        self.fc2 = nn.Conv2d(fc_filters, out_maps * channels, 1, bias=False) # gated activation cuts filters by 2

        #self.init_conv = nn.Conv2d(1,f_out[-1],(2,3),1,0,1,bias=True)

    def forward(self, input):
        # initial convolution
        v_data = self.v_initial_convolution(input)[:, :, :-2, :]  # align with h-stack
        v_to_h_data = self.v_to_h_initial(v_data)#[:,:,:-1,:] # align with h-stack
        h_data = self.h_initial_convolution(input)[:,:,self.initial_pad:,:-self.initial_pad] # unpad rhs of image
        v_data = self.v_init_activation(v_data)
        #h_data = self.h_init_activation(torch.cat((v_to_h_data, h_data), dim=1))
        h_data = self.h_init_activation(v_to_h_data + h_data)

        skip = self.h_to_skip_initial(h_data)
        h_data = self.h_to_h_initial(h_data)

        # hidden layers
        for i in range(self.blocks):
            for j in range(self.layers_per_block):
                v_data, h_data, skip_i = self.conv_layer[i][j](v_data, h_data) # stacked convolutions fix blind spot
                if self.padding == 1:
                    skip = torch.add(skip,skip_i)
                else:
                    skip = torch.add(skip[:,:,1:,1:-1],skip_i)

        # output convolutions
        #x = self.fc_activation(self.fc1(skip))
        x = F.relu(self.fc1(F.relu(skip)))
        x = self.fc2(x)

        #x = self.init_conv(input)[:,:,1:,:]
        #x = F.relu(self.fc1(x))
        #x = self.fc2(x)

        return x[:,:,1:,:]

class PixelCNN2(nn.Module):  # Dense or residual, gated, blocked, dilated PixelCNN with batchnorm
    def __init__(self, filters, initial_convolution_size, dilation, layers, out_maps, padding, channels, activation):
        super(PixelCNN2, self).__init__()

        ### initialize
        if activation == 1:
            self.act_func = 'relu'
        elif activation == 2:
            self.act_func = 'gated'
        elif activation == 3:
            self.act_func = 'dirichlet'

        if self.act_func == 'gated': # filter ratio - need to double a bunch of filters for gated activation
            f_rat = 2
        else:
            f_rat = 1
        self.initial_pad = (initial_convolution_size - 1) // 2
        kernel_size = 3 #initial_convolution_size

        # layer & block structure
        self.blocks = dilation # each block is a different dilation
        self.layers_per_block = layers // self.blocks
        initial_filters = filters
        f_in = (np.ones(layers + 1) * filters).astype(int)
        f_out = (np.ones(layers + 1) * filters).astype(int)
        self.h_init_activation = Activation(self.act_func, initial_filters)
        self.v_init_activation = Activation(self.act_func, initial_filters)


        # apply dilations
        self.dilations = (np.ones((layers // dilation) * dilation)).astype(int) # split to whole number blocks
        for i in range(self.blocks): # apply dilation to each block
            for j in range(self.layers_per_block):
                self.dilations[i * self.layers_per_block + j] = i+1


        # initial layer
        #self.v_initial_convolution = nn.Conv2d(channels, f_rat * initial_filters, (self.initial_pad + 1, initial_convolution_size), 1, padding * (self.initial_pad + 1, self.initial_pad), padding_mode='zeros', bias=True)
        self.v_initial_convolution = nn.Conv2d(channels, f_rat * initial_filters, (self.initial_pad + 1, initial_convolution_size), 1, padding * (self.initial_pad + 1, self.initial_pad), padding_mode='zeros', bias=False)
        #self.v_to_h_initial = nn.Conv2d(f_rat * initial_filters, f_rat * initial_filters,1, bias=True)
        self.v_to_h_initial = nn.Conv2d(f_rat * initial_filters, f_rat * initial_filters, 1, bias=False)
        #self.h_initial_convolution = MaskedConv2d_h('A', channels, channels, f_rat * initial_filters, (1, self.initial_pad+1), 1, padding * (0, self.initial_pad), padding_mode='zeros', bias=True)
        self.h_initial_convolution = MaskedConv2d_h('A', channels, channels, f_rat * initial_filters, (1, self.initial_pad + 1), 1, padding * (0, self.initial_pad), padding_mode='zeros', bias=False)
        self.h_to_skip_initial = nn.Conv2d(initial_filters, initial_filters, 1, bias=False)
        self.h_to_h_initial = nn.Conv2d(initial_filters, initial_filters, 1, bias=False)

        # stack hidden layers in blocks
        self.conv_layer = []
        for j in range(self.blocks):
            self.conv_layer.append([StackedConvolution(f_in[i], f_out[i], kernel_size, padding, self.dilations[i + j * self.layers_per_block], self.act_func) for i in range(self.layers_per_block)]) # stacked convolution (no blind spot)
        for j in range(self.blocks):
            self.conv_layer[j] = nn.ModuleList(self.conv_layer[j])
        self.conv_layer = nn.ModuleList(self.conv_layer)

        #output layers
        fc_filters = 128
        self.fc_activation = Activation(self.act_func, fc_filters // f_rat)
        #self.fc1 = nn.Conv2d(f_out[-1], fc_filters, (1,1), bias=True) # add skip connections
        self.fc1 = nn.Conv2d(f_out[-1], fc_filters, (1, 1), bias=True)  # add skip connections
        self.fc2 = nn.Conv2d(fc_filters // f_rat, out_maps * channels, 1, bias=False) # gated activation cuts filters by 2

    def forward(self, input):
        # initial convolution
        v_data = self.v_initial_convolution(input)[:, :, :-(self.initial_pad + 2), :]  # remove extra
        v_to_h_data = self.v_to_h_initial(v_data)#[:,:,:-1,:] # align with h-stack
        h_data = self.h_initial_convolution(input)[:,:,:,:-self.initial_pad] # unpad rhs of image
        v_data = self.v_init_activation(v_data)
        if self.act_func == 'gated':
            #h_data = self.h_init_activation(torch.cat((v_to_h_data, h_data), dim=1))
            h_data = self.h_init_activation(v_to_h_data + h_data)
        else:
            h_data = self.h_init_activation(v_to_h_data + h_data)

        skip = self.h_to_skip_initial(h_data)
        h_data = self.h_to_h_initial(h_data)

        # hidden layers
        for i in range(self.blocks):
            for j in range(self.layers_per_block):
                v_data, h_data, skip_i = self.conv_layer[i][j](v_data, h_data) # stacked convolutions fix blind spot
                skip = torch.add(skip,skip_i)

        # output convolutions
        x = self.fc_activation(self.fc1(skip))
        x = self.fc2(x)

        return x

class OldStackedConvolution(nn.Module):
    def __init__(self, f_in, f_out, padding, dilation, *args, **kwargs):
        super(OldStackedConvolution, self).__init__(*args, **kwargs)

        #self.v_norm = nn.BatchNorm2d(f_in)
        self.v_Conv2d = nn.Conv2d(f_in, 2 * f_out, (2, 3), 1, padding * (1,1), dilation, bias=True, padding_mode='zeros')
        self.v_to_h_fc = nn.Conv2d(2 * f_out, f_out, 1, bias=True)
        #self.h_norm = nn.BatchNorm2d(f_in)
        self.h_Conv2d = nn.Conv2d(f_in, f_out, (1, 2), 1, padding * (0,1), dilation, bias=True, padding_mode='zeros')
        self.h_to_h = nn.Conv2d(f_out, f_out, 1, bias=True)
        self.activation = Activation('gated',1) # for ReLU, must change number of filters as gated approach halves filters on each application
        self.unpad = dilation
        self.padding = padding

    def forward(self, v_in, h_in):
        residue = h_in * 1 # residual track

        if self.padding == 0:
            v_in = self.v_Conv2d(v_in)
            v_out = self.activation(v_in)
            v_to_h = self.v_to_h_fc(v_in)
            h_in = self.h_Conv2d(h_in)[:, :, 1:, :-1]
            h_out = self.activation(torch.cat((h_in, v_to_h), 1)) # remove rhs, and top 2 lines to align with v_stack
            h_out = self.h_to_h(h_out) + residue[:,:,self.unpad:,self.unpad:-self.unpad]
        else:
            #v_in = self.v_Conv2d(self.v_norm(v_in))[:,:,:-1,:] # remove extra padding
            v_in = self.v_Conv2d(v_in)[:, :, :-1, :]  # remove extra padding
            v_out = self.activation(v_in)
            v_to_h = self.v_to_h_fc(v_in)[:,:,:-1,:] # align v stack to h
            #h_in = self.h_Conv2d(self.h_norm(h_in))[:, :, :, :-1]  # unpad by 1 on rhs
            h_in = self.h_Conv2d(h_in)[:, :, :, :-1]  # unpad by 1 on rhs
            h_out = self.activation(torch.cat((h_in, v_to_h), 1))
            h_out = self.h_to_h(h_out) + residue

        return v_out, h_out

class OldPixelCNN2(nn.Module):  # Dense or residual, gated, blocked, dilated PixelCNN with batchnorm
    def __init__(self, filters, initial_convolution_size, dilation, layers, out_maps, padding, channels):
        super(OldPixelCNN2, self).__init__()

        blocks = 1
        ### initialize constants
        self.padding = padding
        self.layers_per_block = layers
        self.blocks = blocks
        self.layers = int(self.layers_per_block * blocks)
        self.initial_pad = (initial_convolution_size - 1) // 2
        self.main_pad = 1
        initial_filters = filters
        self.input_depth = 1 #for now just 1 channels
        f_in = (np.ones(self.layers + 1) * filters).astype(int)
        f_out = (np.ones(self.layers + 1) * filters).astype(int)
        self.dilation = (np.ones(self.layers) * dilation).astype(int) # not yet in use

        if self.padding == 0:
            self.unpad = np.zeros(self.layers + 1).astype(int)
            for i in range(1,self.layers):
                self.unpad[i] = dilation[i].astype(int)

            self.unpad [0] = (initial_convolution_size-1)//2
        else:
            self.unpad = np.zeros(self.layers + 1).astype(int)
        ###
        self.activation = Activation('gated',1)

        # initial layer
        self.v_initial_convolution = nn.Conv2d(self.input_depth, 2 * initial_filters, (initial_convolution_size//2 + 1, initial_convolution_size), 1, padding * (self.initial_pad + 1, self.initial_pad), padding_mode='zeros', bias=True)
        self.v_to_h_initial = nn.Conv2d(2 * initial_filters, initial_filters,1, bias=True)
        self.h_initial_convolution = MaskedConv2d_h('A', self.input_depth, self.input_depth, initial_filters, (1, initial_convolution_size//2 + 1), 1, padding * (0, self.initial_pad), padding_mode='zeros', bias=True)
        self.h_to_h_initial = nn.Conv2d(initial_filters, initial_filters, 1, bias=True)
        #self.initial_norm = nn.BatchNorm2d(channels)


        # stack layers in blocks
        self.conv_layer = []
        for j in range(blocks):
            self.conv_layer.append([OldStackedConvolution(f_in[i + j * self.layers_per_block], f_out[i + j * self.layers_per_block], padding, self.dilation[i + j * self.layers_per_block]) for i in range(self.layers_per_block)]) # stacked convolution (no blind spot)
        for j in range(blocks):
            self.conv_layer[j] = nn.ModuleList(self.conv_layer[j])
        self.conv_layer = nn.ModuleList(self.conv_layer)

        #output layers
        self.fc1 = nn.Conv2d(f_out[-1], 256, 1)
        self.fc2 = nn.Conv2d(256 // 2, out_maps * channels, 1) # gated activation cuts filters by 2

    def forward(self, input):
        # initial convolution
        #input = self.initial_norm(input)
        # separate stacks
        if self.padding == 0:
            if self.layers == 0:
                v_data = self.v_initial_convolution(input)[:,:,:-self.initial_pad,:]
            else:
                v_data = self.v_initial_convolution(input)
            v_to_h_data = self.v_to_h_initial(v_data)
            h_data = self.h_initial_convolution(input)
            if self.layers == 0:
                h_data = self.activation(torch.cat((v_to_h_data, h_data[:,:,-1:,:1]), dim=1)) # for final layer
            else:
                h_data = self.activation(torch.cat((v_to_h_data, h_data[:, :, self.initial_pad:, :-self.initial_pad]), dim=1))
            #h_data = self.activation(torch.cat((v_to_h_data[:,:,0,:].unsqueeze(2), h_data[:,:,-1,0].unsqueeze(2).unsqueeze(2)), dim=1))
        else:
            v_data = self.v_initial_convolution(input)[:, :, :-(2 * self.initial_pad), :]  # remove extra
            v_to_h_data = self.v_to_h_initial(v_data)[:,:,:-1,:] # align with h-stack
            h_data = self.h_initial_convolution(input)[:,:,:,:-self.initial_pad] # unpad rhs of image
            h_data = self.activation(torch.cat((v_to_h_data, h_data), dim=1))
        h_data = self.h_to_h_initial(h_data)
        v_data = self.activation(v_data)

        for i in range(self.blocks): # loop over convolutional layers
            for j in range(self.layers_per_block):
               v_data, h_data = self.conv_layer[i][j](v_data, h_data) # stacked convolutions fix blind spot

        if (self.padding == 0) and (self.layers != 0):
            h_data = h_data[:,:,-1:,:]
        # output convolutions
        x = self.activation(self.fc1(h_data))
        x = self.fc2(x)

        return x

class SimpleNet(nn.Module):  # a purpose-build, ULTRA simple PixelCNN to explore functional forms
    def __init__(self, conditional, filters, initial_convolution_size, dilation, layers, out_maps, padding, channels):
        super(SimpleNet, self).__init__()

        ### initialize constants
        self.conditional = conditional
        self.initial_pad = (initial_convolution_size - 1) // 2
        channels = 1 #for now just 1 channels
        initial_filters = filters

        ###
        self.activation_type = 'relu'
        self.activation = Activation(self.activation_type)
        if self.activation_type == 'relu':
            filter_ratio = 1
        elif self.activation_type == 'gated':
            filter_ratio = 2

        ### initial layer
        self.v_initial_convolution = nn.Conv2d(channels, filter_ratio * initial_filters, (initial_convolution_size//2 + 1, initial_convolution_size), 1, padding * (self.initial_pad + 1, self.initial_pad), padding_mode='zeros', bias=False)
        self.v_to_h_initial = nn.Conv2d(filter_ratio * initial_filters, initial_filters,1, bias=True)
        self.h_initial_convolution = MaskedConv2d_h('A', channels, channels, initial_filters, (1, initial_convolution_size//2 + 1), 1, padding * (0, self.initial_pad), padding_mode='zeros', bias=False)
        self.h_to_h_initial = nn.Conv2d(initial_filters, initial_filters, 1, bias=True)
        self.v_initial_conditional = nn.Linear(1, filter_ratio * initial_filters, bias=False)
        self.h_initial_conditional = nn.Linear(1, filter_ratio * initial_filters, bias=False)

        #output layers
        self.fc1 = nn.Conv2d(initial_filters, initial_filters, 1)#nn.Conv2d(filters, 16, (1,1), bias=True)
        self.fc1_conditional = nn.Linear(1, filter_ratio * initial_filters, bias=False)
        self.fc11 = nn.Conv2d(initial_filters, initial_filters, 1)#nn.Conv2d(filters, 16, (1,1), bias=True)
        self.fc11_conditional = nn.Linear(1, filter_ratio * initial_filters, bias=False)
        self.fc12 = nn.Conv2d(initial_filters, initial_filters, 1)#nn.Conv2d(filters, 16, (1,1), bias=True)
        self.fc12_conditional = nn.Linear(1, filter_ratio * initial_filters, bias=False)
        self.fc13 = nn.Conv2d(initial_filters, initial_filters, 1)#nn.Conv2d(filters, 16, (1,1), bias=True)
        self.fc13_conditional = nn.Linear(1, filter_ratio * initial_filters, bias=False)
        self.fc14 = nn.Conv2d(initial_filters, initial_filters, 1)#nn.Conv2d(filters, 16, (1,1), bias=True)
        self.fc14_conditional = nn.Linear(1, filter_ratio * initial_filters, bias=False)

        self.fc2 = nn.Conv2d(initial_filters, out_maps * channels, 1)#nn.Conv2d(int(self.fc1.weight.shape[0]/filter_ratio), out_maps * channels, 1) # gated activation cuts filters by 2


    def forward(self, input, condition):
        # initial convolution
        #input = self.initial_norm(input)
        v_data = self.v_initial_convolution(input)[:, :, :-(2 * self.initial_pad), :]  # remove extra
        v_to_h_data = self.v_to_h_initial(v_data)[:,:,:-1,:] # align with h-stack
        h_data = self.h_initial_convolution(input)[:,:,:,:-self.initial_pad] # unpad rhs of image
        v_conditions = torch.ones(v_data.shape[0], 1, v_data.shape[2], v_data.shape[3]).cuda().permute(0,2,3,1)
        h_conditions = torch.ones((h_data.shape[0], 1 ,h_data.shape[2],h_data.shape[3])).cuda().permute(0,2,3,1)
        for i in range(v_conditions.shape[0]):
            v_conditions[i,:,:,:]=condition[i]
            h_conditions[i,:,:,:]=condition[i]
        #v_data = self.activation(v_data + self.v_initial_conditional(v_conditions).permute(0,3,1,2)) # manual toggle for GPU
        if self.activation_type == 'relu':
            h_data = self.activation(v_to_h_data + h_data + self.h_initial_conditional(h_conditions).permute(0,3,1,2))
        elif self.activation_type == 'gated':
            h_data = self.activation(torch.cat((v_to_h_data, h_data), dim=1) + self.h_initial_conditional(h_conditions).permute(0,3,1,2))

        h_data = self.h_to_h_initial(h_data)

        # output convolutions
        x = self.activation(self.fc1(h_data) + self.fc1_conditional(h_conditions).permute(0,3,1,2))
        x = self.activation(self.fc11(h_data) + self.fc11_conditional(h_conditions).permute(0,3,1,2))
        x = self.activation(self.fc12(h_data) + self.fc12_conditional(h_conditions).permute(0,3,1,2))
        x = self.activation(self.fc13(h_data) + self.fc13_conditional(h_conditions).permute(0,3,1,2))
        x = self.activation(self.fc14(h_data) + self.fc14_conditional(h_conditions).permute(0,3,1,2))

        x = self.fc2(x)

        return x

class PixelCNN_Discriminator(nn.Module):  # Dense or residual, gated, blocked, dilated PixelCNN with batchnorm
    def __init__(self, filters, initial_convolution_size, dilation, layers, out_maps, padding, channels):
        super(PixelCNN_Discriminator, self).__init__()

        padding = 0
        self.initial_convolution = nn.Conv2d(channels, 2*filters, initial_convolution_size, 1, int(padding * (initial_convolution_size - 1) // 2), padding_mode='zeros', bias=False)
        self.hidden_convolutions = nn.ModuleList([nn.Conv2d(filters, filters, 3, 1, padding, padding_mode='zeros', bias=True) for i in range(layers)])
        self.shrink_features = nn.ModuleList([nn.Conv2d(2 * filters, filters, 1) for i in range(layers)])
        self.grow_features = nn.ModuleList([nn.Conv2d(filters, 2 * filters, 1) for i in range(layers)])
        self.conv_field = layers + initial_convolution_size // 2

        self.fc1 = nn.Conv2d(2 * filters, 256, 1)
        self.fc2 = nn.Conv2d(256, out_maps * channels, 1)

    def forward(self, x): # pre-activated residual model
        #x[:,:,x.shape[-2]//2,x.shape[-1]//2]=0 # blind the central pixel (one we are trying to predict)
        x = F.relu(self.initial_convolution(x))

        for i in range(len(self.hidden_convolutions)):
            residue = x
            x = self.shrink_features[i](F.relu(x))
            x = self.hidden_convolutions[i](F.relu(x))
            x = self.grow_features[i](F.relu(x))
            x += residue[:,:,1:-1,1:-1] # unpad residue

        x = self.fc2(F.relu(self.fc1(x)))
        return x

class brutesolver(nn.Module):  # a manually initialized correlation solver
    def __init__(self):
        super(brutesolver, self).__init__()

        # define correlations
        conv_range = 5
        self.corr_order = 5
        kernel_size = (2* conv_range + 1)**2 // 2 # number of unmasked pixels we can correlate with within range conv_range
        perms = 0 # not counting density coefficient
        for i in range(1,self.corr_order + 1):
            perms += special.comb(kernel_size, i, repetition=False)

        perms = int(perms)

        correlations = []
        for i in range(1, self.corr_order + 1): # define all possible correlations in list space
            correlations.append(np.asarray(list(combinations(np.arange(kernel_size),i))))

        filters = np.zeros((perms, kernel_size + conv_range + 1)) # need to 'complete the square' - extra terms in the last row
        self.corr_order_index = torch.zeros(perms) # list correlation order
        self.corr_indices = torch.zeros(self.corr_order + 1).int()
        ind = 0
        for i in range(1,self.corr_order + 1): # define correlations (convolutional filters) in list space
            for j in range(int(special.comb(kernel_size, i, repetition=False))):
                for k in range(i):
                    filters[ind,correlations[i - 1][j,k]] = 1
                    self.corr_order_index[ind] = i
                ind += 1
            self.corr_indices[i] = j + self.corr_indices[i-1] + 1

        # convert filters from list space to pixel space
        self.filters = filters.reshape(perms, conv_range + 1, 2*conv_range + 1)

        # initialize trainable parameters (coefficients)
        self.correlation = nn.Linear(perms, 1, bias=False)
        nn.init.constant_(self.correlation.weight, 0) #initialize filters at zero
        self.den_corr = nn.Linear(1,1,bias=True)
        self.conditional = 0

    def forward(self, x):
        filters = (torch.Tensor(self.filters).cuda()).unsqueeze(1)
        conv_field = filters.shape[-1]//2

        x = torch.tensor(x==1).float().cuda() # kill empties

        conv_out = F.conv2d(x, filters, stride=1, padding=(conv_field), dilation = 1)[:, :, : -conv_field, :]
        for i in range(1, self.corr_order + 1):
            conv_out[:,self.corr_indices[i-1]:self.corr_indices[i],:,:]=conv_out[:,self.corr_indices[i-1]:self.corr_indices[i],:,:]== i

        conv_out = conv_out.permute(0,2,3,1)
        y = self.correlation(conv_out)[:,:,:,0] # sum of all convolution coefficients
        y = y + self.den_corr(torch.zeros(1).cuda()) # linear density coefficient

        out = torch.zeros((y.shape[0],3,y.shape[-2],y.shape[-1])).cuda()
        out[:,2,:,:]=y
        out[:,1,:,:]=-y
        out[:,0,:,:]=-100

        return out




''' version from July 27
class PixelCNN2(nn.Module):  # Dense or residual, gated, blocked, dilated PixelCNN with batchnorm
    def __init__(self, conditional, filters, initial_convolution_size, dilation, layers, out_maps, padding, channels):
        super(PixelCNN2, self).__init__()

        ### initialize constants
        self.dense = 1
        self.conditional = conditional
        self.blocks = dilation # each block is a different dilation
        self.initial_pad = (initial_convolution_size - 1) // 2
        input_depth = 1 #for now just 1 channels
        initial_filters = filters
        self.dilations = (np.ones((layers // dilation) * dilation)).astype(int) # split to whole number blocks
        self.layers_per_block = layers // self.blocks
        if self.dense == 0:
            f_in = (np.ones(layers + 1) * filters).astype(int)
            f_out = (np.ones(layers + 1) * filters).astype(int)
        elif self.dense == 1:
            f_in = ((np.arange(self.layers_per_block) + 1) * filters).astype(int)
            f_out = (np.ones(self.layers_per_block) * filters).astype(int)

        for i in range(self.blocks): # apply dilation to each block
            for j in range(self.layers_per_block):
                self.dilations[i * self.layers_per_block + j] = i+1

        blindness = np.zeros(self.layers_per_block)  # re-blind the network at set intervals to enable quicker generation
        for i in range(len(blindness)):
            if ((i+1) % 2) == 0:
                blindness[i]=0
        ###
        self.activation = Activation('gated')

        # initial layer
        self.v_initial_convolution = nn.Conv2d(input_depth, 2 * initial_filters, (initial_convolution_size//2 + 1, initial_convolution_size), 1, padding * (self.initial_pad + 1, self.initial_pad), padding_mode='zeros', bias=True)
        self.v_to_h_initial = nn.Conv2d(2 * initial_filters, initial_filters,1, bias=True)
        self.h_initial_convolution = MaskedConv2d_h('A', input_depth, input_depth, initial_filters, (1, initial_convolution_size//2 + 1), 1, padding * (0, self.initial_pad), padding_mode='zeros', bias=True)
        self.h_to_h_initial = nn.Conv2d(initial_filters, initial_filters, 1, bias=True)
        if conditional == 1:
            self.v_initial_conditional = nn.Linear(2 * initial_filters, 2 * initial_filters,bias=False)
            self.h_initial_conditional = nn.Linear(2 * initial_filters, 2 * initial_filters,bias=False)
        #self.initial_norm = nn.InstanceNorm2d(channels)

        # stack layers in blocks
        self.conv_layer = []
        for j in range(self.blocks):
            self.conv_layer.append([StackedConvolution(self.conditional, blindness[i], self.dense, f_in[i] + (j * filters)*self.dense, f_out[i], padding, self.dilations[i + j * self.layers_per_block]) for i in range(self.layers_per_block)]) # stacked convolution (no blind spot)
        for j in range(self.blocks):
            self.conv_layer[j] = nn.ModuleList(self.conv_layer[j])
        self.conv_layer = nn.ModuleList(self.conv_layer)

        #output layers
        if self.dense == 0:
            self.fc1 = nn.Conv2d(f_out[-1], 128, (1,1), bias=True)
        elif self.dense == 1:
            self.fc1 = nn.Conv2d((filters * self.blocks)+initial_filters, 128, (1,1), bias=True)
        self.fc2 = nn.Conv2d(64, out_maps * channels, 1) # gated activation cuts filters by 2

    def forward(self, input, condition):
        # initial convolution
        #input = self.initial_norm(input)
        if self.dense == 0:
            v_data = self.v_initial_convolution(input)[:, :, :-(2 * self.initial_pad), :]  # remove extra
            v_to_h_data = self.v_to_h_initial(v_data)[:,:,:-1,:] # align with h-stack
            h_data = self.h_initial_convolution(input)[:,:,:,:-self.initial_pad] # unpad rhs of image
            if self.conditional == 0:
                v_data = self.activation(v_data)
                h_data = self.activation(torch.cat((v_to_h_data, h_data), dim=1))
            elif self.conditional == 1:
                v_conditions = torch.ones(v_data.size()).cuda().permute(0,2,3,1)
                h_conditions = torch.ones((h_data.shape[0],h_data.shape[1]*2,h_data.shape[2],h_data.shape[3])).cuda().permute(0,2,3,1)
                for i in range(v_conditions.shape[0]):
                    v_conditions[i,:,:,:]=condition[i]
                    h_conditions[i,:,:,:]=condition[i]
                v_data = self.activation(v_data + self.v_initial_conditional(v_conditions).permute(0,3,1,2)) # manual toggle for GPU
                h_data = self.activation(torch.cat((v_to_h_data, h_data), dim=1) + self.h_initial_conditional(h_conditions).permute(0,3,1,2))

            h_data = self.h_to_h_initial(h_data)

            for i in range(self.blocks): # loop over convolutional layers
                for j in range(self.layers_per_block):
                   v_data, h_data = self.conv_layer[i][j](v_data, h_data, condition) # stacked convolutions fix blind spot
        elif self.dense == 1:
            h_residues = []
            v_residues = []
            #h_residues.append(input)
            #v_residues.append(input)

            # initial convolution
            v_data = self.v_initial_convolution(input)[:, :, :-(2 * self.initial_pad), :]  # remove extra
            v_to_h_data = self.v_to_h_initial(v_data)[:,:,:-1,:] # align with h-stack
            h_data = self.h_initial_convolution(input)[:,:,:,:-self.initial_pad] # unpad rhs of image
            if self.conditional == 0:
                v_data = self.activation(v_data)
                h_data = self.activation(torch.cat((v_to_h_data, h_data), dim=1))
            elif self.conditional == 1:
                v_conditions = torch.ones(v_data.size()).cuda().permute(0, 2, 3, 1)
                h_conditions = torch.ones((h_data.shape[0], h_data.shape[1] * 2, h_data.shape[2], h_data.shape[3])).cuda().permute(0, 2, 3, 1)
                for i in range(v_conditions.shape[0]):
                    v_conditions[i, :, :, :] = condition[i]
                    h_conditions[i, :, :, :] = condition[i]
                v_data = self.activation(v_data + self.v_initial_conditional(v_conditions).permute(0,3,1,2))
                h_data = self.activation(torch.cat((v_to_h_data, h_data), dim=1) + self.h_initial_conditional(h_conditions).permute(0,3,1,2))

            h_data = self.h_to_h_initial(h_data)

            h_residues.append(h_data)
            v_residues.append(v_data)

            for i in range(self.blocks): # loop over convolutional blocks
                h_block_residues = []
                v_block_residues = []
                h_block_residues.append(torch.cat(h_residues, 1))
                v_block_residues.append(torch.cat(v_residues, 1))
                for j in range(self.layers_per_block): # loop over layers
                    v_data, h_data = self.conv_layer[i][j](torch.cat(v_block_residues, 1), torch.cat(h_block_residues, 1), condition) # stacked convolutions fix blind spot
                    h_block_residues.append(h_data)
                    v_block_residues.append(v_data)

                h_residues.append(h_data)
                v_residues.append(v_data)


        # output convolutions
        if self.dense == 0:
            x = self.activation(self.fc1(h_data))
        elif self.dense == 1:
            x = self.activation(self.fc1(torch.cat(h_residues, 1)))
        x = self.fc2(x)

        return x

'''