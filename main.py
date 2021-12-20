from torch import nn, optim, cuda, backends
from utils import *
from models import *
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


run = get_input()
## parameters
if run == -1:  # user-defined parameters
    # architecture
    model = 5 # 5 is Gated PixelCNN with optional dilation, 6 is an ultra-simple network for exploring time-dependence, 7 is the PixelCNN Discriminator
    filters = 5#20 # number of convolutional filters or feature maps (at initial layer for model ==3)
    filter_size = 3 # initial layer convolution size
    layers = 5#60 # number of hidden convolutional layers (make divisible by dilation)
    dilation = 1 # 1 is no dilation, for >1, splits stack evenly by this number
    bound_type = 0 # type of boundary layer, 0 = empty, 1 = seed in top left only, 5 = large graphene seed, only works for sample batch <= 64 currently #DEPRECATED 3 = generated bound
    boundary_layers = 2 # number of layers of conv_fields between sample and boundary
    softmax_temperature = 0 #ratio to batch mean at which softmax will sample, set to 0 to sample at training temperature
    activation = 2 # 1-relu, 2-gated, 3 - Learnable/parametric

    # training
    training_data = 9 # select training set: 1 - repulsive, 2 - short/medium range agg, 3 - 256x256 new aggregates, 4 - hot repulsive, 5- refined branes, 6 - synthetic drying, age =1, 7 - synthetic drying, age = -1, 8- small sample synthetic drying age = -1, 9-MAC test image, 10 - graphene
    training_batch = 24 #int(2048*4) # size of training and test batches - it will try to run at this size, bu5t if it doesn't fit it will go smaller
    sample_batch_size = 1024  # batch size for sample generator - will auto-adjust to be sm/eq than training_batch
    n_samples = 1  # total samples to be generated when we generate, must not be zero (it may make more if there is available memory)
    run_epochs = 10 # number of incremental epochs which will be trained over - if zero, will run just the generator
    dataset_size = 100#40000 # the maximum number of samples to consider from our dataset
    train_margin = 1e-4 # the convergence criteria for training error
    average_over = 5 # how many epochs to average over to determine convergence
    outpaint_ratio = 2 # sqrt of size of output relative to input
    generation_type = 2 # 1 - fast for many small images, 2 - fast for few, large images, 3 - 'old-school', very slow generation approach
    noise = 0 # training noise intensity, may be deprecated, 99 is a flag for model=7, 98 is a flag for the new superPixelCNN
    den_var = 0 # standard deviation of noise density variation
    GPU = 1  # if 1, runs on GPU (requires CUDA), if 0, runs on CPU (slow!)
    TB = 0  # if 1, save everything to tensorboard as well as to file, if 0, just save outputs to file
else:
    with open('batch_parameters.pkl', 'rb') as f:
        inputs = pickle.load(f)
    # architecture
    model = inputs['model'][run]
    filters = inputs['filters'][run]
    filter_size = inputs['filter_size'][run]
    layers = inputs['layers'][run]
    dilation = inputs['dilation'][run]
    bound_type = inputs['bound_type'][run]
    boundary_layers = inputs['boundary_layers'][run]
    softmax_temperature = inputs['softmax_temperature'][run]
    activation = inputs['activation'][run]


    # training
    training_data = inputs['training_data'][run]
    training_batch = int(inputs['training_batch'][run])
    sample_batch_size = inputs['sample_batch_size'][run]
    n_samples = inputs['n_samples'][run]
    run_epochs = inputs['run_epochs'][run]
    dataset_size = inputs['dataset_size'][run]
    train_margin = inputs['train_margin'][run]
    average_over = int(inputs['average_over'][run])
    outpaint_ratio = inputs['outpaint_ratio'][run]
    generation_type = inputs['generation_type'][run]
    noise = inputs['noise'][run]
    den_var = inputs['den_var'][run]
    GPU = inputs['GPU'][run]
    TB = inputs['TB'][run]

if GPU == 1:
    backends.cudnn.benchmark = True  # auto-optimizes certain backend processes


dir_name = get_dir_name(model, training_data, filters, layers, dilation, filter_size, noise, den_var, activation, dataset_size)  # get directory name for I/O
writer = SummaryWriter('logfiles/'+dir_name[:]+'_T=%.3f'%softmax_temperature)  # initialize tensorboard writer

prev_epoch = 0
if __name__ == '__main__':  # run it!
    net, conv_field, optimizer, sample_0, input_x_dim, input_y_dim, sample_x_dim, sample_y_dim = initialize_training(model, filters, filter_size, layers, dilation, den_var, training_data, outpaint_ratio, dataset_size, activation)
    net, optimizer, prev_epoch = load_checkpoint(net, optimizer, dir_name, GPU, prev_epoch)
    channels = sample_0.shape[1]
    out_maps = len(np.unique(sample_0)) + 1
    input_analysis = analyse_inputs(training_data, out_maps, dataset_size) # analyse inputs to prepare accuracy metrics

    if prev_epoch == 0: # if we are just beginning training, save inputs and relevant analysis
        save_outputs(dir_name, input_analysis, sample_0, softmax_temperature, 1, TB)

    print('Imported and Analyzed Training Dataset {}'.format(training_data))

    if GPU == 1:
        net = nn.DataParallel(net) # go to multi-GPU training
        print("Using", torch.cuda.device_count(), "GPUs")
        net.to(torch.device("cuda:0"))
        print(summary(net, [(channels, conv_field+2 + conv_field, 4*conv_field+1)]))  # doesn't work on CPU, not sure why

    max_epochs = run_epochs + prev_epoch + 1

    ## BEGIN TRAINING/GENERATION
    if run_epochs == 0:  # no training, just samples
        prev_epoch += 1
        epoch = prev_epoch

        # to a test of the net to get it warmed up
        training_batch, changed = get_training_batch_size(training_data, training_batch, model, filters, filter_size, layers, dilation, out_maps, channels, den_var, dataset_size, noise, activation, GPU)  # confirm we can keep on at this batch size
        if changed == 1:  # if the training batch is different, we have to adjust our batch sizes and dataloaders
            tr, te = get_dataloaders(training_data, training_batch, dataset_size)
            print('Training batch set to {}'.format(training_batch))
        else:
            tr, te = get_dataloaders(training_data, training_batch, dataset_size)

        sample, time_ge, n_samples, agreements, output_analysis, total_agreement = generation(generation_type, dir_name, input_analysis, outpaint_ratio, epoch, model, filters, filter_size, layers, net, writer, te, out_maps, noise, den_var, conv_field, sample_x_dim, sample_y_dim, n_samples, sample_batch_size, bound_type, training_data, boundary_layers, channels, softmax_temperature, dataset_size, GPU, cuda, TB)


    else: #train it AND make samples!
        epoch = prev_epoch + 1
        converged = 0
        tr_err_hist = []
        te_err_hist = []

        generation_stats = []

        while (epoch <= (max_epochs + 1)) & (converged == 0):# over a certain number of epochs or until converged
            if (epoch - prev_epoch) < 3: # check batch size over first few epochs
                training_batch, changed = get_training_batch_size(training_data, training_batch, model, filters, filter_size, layers, dilation, out_maps, channels, den_var, dataset_size, noise, activation, GPU)  # confirm we can keep on at this batch size
                if changed == 1: # if the training batch is different, we have to adjust our batch sizes and dataloaders
                    tr, te = get_dataloaders(training_data, training_batch, dataset_size)
                    print('Training batch set to {}'.format(training_batch))
                else:
                    tr, te = get_dataloaders(training_data, training_batch, dataset_size)

            err_tr, time_tr = train_net(net, optimizer, writer, tr, epoch, out_maps, noise, den_var, conv_field, GPU, cuda)  # train & compute loss
            err_te, time_te = test_net(net, writer, te, out_maps, noise, den_var, epoch, conv_field, GPU, cuda)  # compute loss on test set
            tr_err_hist.append(torch.mean(torch.stack(err_tr)))
            te_err_hist.append(torch.mean(torch.stack(err_te)))
            np.save(dir_name[:] + '_training_curve', np.array((tr_err_hist, te_err_hist)))
            print('epoch={}; nll_tr={:.5f}; nll_te={:.5f}; time_tr={:.1f}s; time_te={:.1f}s'.format(epoch, torch.mean(torch.stack(err_tr)), torch.mean(torch.stack(err_te)), time_tr, time_te))
            save_ckpt(epoch, net, optimizer, dir_name[:]) #save checkpoint
            converged = auto_convergence(train_margin, average_over, epoch, prev_epoch, net, optimizer, dir_name, tr_err_hist, te_err_hist, max_epochs)
            epoch += 1



            # every X epochs, generate samples and print analysis
            if (epoch % 4) == 0:
                if model != 7:  # if we're not doing the discrimination
                    with HiddenPrints():
                        sample, time_ge, n_samples, agreements, output_analysis, total_agreement = generation(generation_type, dir_name, input_analysis, outpaint_ratio, epoch, model, filters, filter_size, layers, net, writer, te, out_maps, noise, den_var, conv_field, sample_x_dim, sample_y_dim, n_samples, sample_batch_size, bound_type, training_data, boundary_layers, channels, softmax_temperature,
                                                                                                              dataset_size, GPU, cuda, TB)
                        #np.save('samples/epoch=%d'%epoch +'_'+ dir_name[:] + '_T=%.3f' % softmax_temperature, sample)
                        generation_stats.append([agreements, output_analysis, sample])
                        np.save('samples/generation_'+dir_name[:] + '_T=%.3f' % softmax_temperature, generation_stats)

                    print('epoch = {}; tot={:.2f}; den={:.2f}; corr={:.2f}; prob={:.2f}; time_ge={:.1f}s'.format(epoch, total_agreement, agreements['density'], agreements['correlation'], agreements['probability'], time_ge))

        tr, te = get_dataloaders(training_data, 4, 100)  # get something from the dataset
        example = next(iter(tr)).cuda()  # get seeds from test set
        raw_out = net(example[0:2, :, :, :].float())
        raw_out = F.softmax(raw_out, dim=1)
        raw_out = raw_out[0].unsqueeze(1)
        raw_grid = utils.make_grid(raw_out, nrow=int(out_maps), padding=0)
        raw_grid = raw_grid[0].cpu().detach().numpy()

        np.save('raw_outputs/' + dir_name[:], raw_grid)

        # generate samples
        if model != 7: # if we're not doing the discrimination
            sample, time_ge, n_samples, agreements, output_analysis, total_agreement = generation(generation_type, dir_name, input_analysis, outpaint_ratio, epoch, model, filters, filter_size, layers, net, writer, te, out_maps, noise, den_var, conv_field, sample_x_dim, sample_y_dim, n_samples, sample_batch_size, bound_type, training_data, boundary_layers, channels, softmax_temperature, dataset_size, GPU, cuda, TB)

'''
# show final predictions from training for a specific example
tr, te = get_dataloaders(training_data, 4, 100)  # get something from the dataset
example = next(iter(tr)).cuda()  # get seeds from test set
raw_out = net(example[0:2, :, :, :].float())
raw_out = F.softmax(raw_out, dim=1)
raw_out = raw_out[0].unsqueeze(1)
raw_grid = utils.make_grid(raw_out, nrow=int(out_maps), padding=0)
raw_grid = raw_grid[0].cpu().detach().numpy()
plt.figure()
plt.imshow(raw_grid)

plt.figure()
plt.imshow(example[0,0,:,:].cpu())

# tensorboard command for windows - just make sure the directory is correct
# tensorboard --logdir=logfiles/ --host localhost

radial_fourier = outputs0['radial fourier']
radial_correlation = outputs0['radial correlation']
radial_fourier_out = outputs['radial fourier']
radial_correlation_out = outputs['radial correlation']

correlation = np.zeros(len(radial_correlation))
fourier = np.zeros(len(radial_fourier))
correlation_out = np.zeros(len(radial_correlation))
fourier_out = np.zeros(len(radial_fourier))
run = int(len(radial_correlation)/10)
for i in range(len(radial_fourier)):
    if i < run:
        fourier[i] = np.average(radial_fourier[0:i])
        fourier_out[i] = np.average(radial_fourier_out[0:i])
    else:
        fourier[i] = np.average(radial_fourier[i-run:i])
        fourier_out[i] = np.average(radial_fourier_out[i-run:i])

for i in range(len(radial_correlation)):
    if i < run:
        correlation[i] = np.average(radial_correlation[0:i])
        fourier_out[i] = np.average(radial_correlation_out[0:i])
    else:
        correlation[i] = np.average(radial_correlation[i-run:i])
        correlation_out[i] = np.average(radial_correlation_out[i-run:i])

plt.subplot(2,3,1)
plt.imshow(outputs0['sample transform'])
plt.subplot(2,3,2)
plt.imshow(outputs0['sample transform'])
plt.subplot(2,3,3)
plt.plot(fourier)
plt.plot(fourier_out)
plt.subplot(2,3,4)
plt.imshow(outputs0['density correlation'])
plt.subplot(2,3,5)
plt.imshow(outputs['density correlation'])
plt.subplot(2,3,6)
plt.plot(correlation)
plt.plot(correlation_out)

n_samples = 1
maxrange = 5
bins = np.arange(-maxrange, maxrange,2 * maxrange / 25)
grid_in = np.expand_dims(sample[0,:,:,:].cpu().detach().numpy(),0)
n_particles = np.sum(grid_in != 0)
delta = bins[1]-bins[0]
re_coords = []
for n in range(n_samples):
    re_coords.append([])

for n in range(n_samples):
    for i in range(grid_in.shape[-2]):
        for j in range(grid_in.shape[-1]):
            if grid_in[n,0,i,j] != 0:
                re_coords[n].append((i - grid_in[n,0,i,j] * delta + maxrange, j - grid_in[n,1,i,j] * delta + maxrange))

new_coords2 = np.zeros((n_samples,n_particles,2))
for n in range(n_samples):
    for m in range(len(re_coords[n])):
        new_coords2[n,m,:] = re_coords[n][m] # the reconstructed coordinates
        
# check convolutional field        
xlim = 21
ylim = 25
out = torch.zeros((ylim,xlim))
dry = torch.zeros((ylim,xlim))
for i in range(ylim):
    for j in range(xlim):    
        ini = torch.zeros((1,1,ylim,xlim)).cuda()
        empty = torch.zeros((1,1,ylim,xlim)).cuda()
        condition = torch.zeros(1)
        ini[:,:,i,j] = 100
        out[i,j] = net(ini.float(),condition.float())[0,0,-1,xlim//2]
        dry[i,j] = net(empty.float(),condition.float())[0,0,-1,xlim//2]

plt.imshow(np.abs(out.detach().numpy() - dry.detach().numpy())!=0)


# analyze init activation
a = torch.linspace(1e-4,5,1000).cuda()
b = torch.ones((1,12,1,1)).cuda()
out = torch.zeros((b.shape[1],len(a))).cuda()
for i in range(len(a)):
    b[:,:,:,:]=a[i]
    out[:,i]=net.module.h_init_activation(b)[0,:,0,0]
 
out = out.cpu().detach().numpy()
plt.plot(a.cpu().detach().numpy(),out.transpose())

# BRUTESOLVER filters visualization
plt.clf()

sub_edge = np.sqrt(ngraphs)
filters = net.module.filters
weights = np.abs(net.module.correlation.weight[0].cpu().detach().numpy())
nnz = np.count_nonzero(weights)
nfilters = nnz # we'll display only nonzero contributing filters
ngraphs = np.ceil(np.sqrt(nfilters))**2
nz_ind = np.nonzero(weights)
for i in range(nfilters):
    plt.subplot(sub_edge,sub_edge,i+1)
    plt.imshow(np.log10(filters[nz_ind[0][i]]*weights[nz_ind[0][i]]),vmin=np.log10(np.min(np.unique(weights)[1:])),vmax = np.log10(np.max(np.abs(weights))))
    
    
    
# sequence analysis
tr, te = get_dataloaders(training_data, 100, 100)  # get something from the dataset
example = next(iter(tr))  # get seeds from test set
p_mat, seq_pop, rho_eq, d_equil, max_dens, max_configs = sequence_analysis(net, conv_field, example.cpu().detach().numpy())



# gated
example = example.cuda()
# for single-pixel nets
if net.module.v_initial_convolution.padding[1]==0:
    raw_out = torch.zeros((2,out_maps,example.shape[-2],example.shape[-1]))
    example = torch.constant_pad_nd(example,(conv_field,conv_field,conv_field+1,0))
    for iix in range(conv_field,raw_out.shape[-1]+conv_field):
        for iiy in range(conv_field+1,raw_out.shape[-2] + conv_field + 1):
            raw_out[:,:,iiy-conv_field-1,iix-conv_field] = net(example[0:2,:,iiy - conv_field - 1:iiy+1,iix-conv_field:iix+conv_field+1].float())[:,:,0,0]
else:
    raw_out = net(example[0:2, :, :, :].float())
raw_out = F.softmax(raw_out, dim=1)
raw_out = raw_out[0].unsqueeze(1)
raw_grid = utils.make_grid(raw_out, nrow=int(out_maps), padding=0)
raw_grid = raw_grid[0].cpu().detach().numpy()


# discriminator
example = example.cuda()
# for single-pixel nets
if net.module.v_initial_convolution.padding[1]==0:
    raw_out = torch.zeros((2,out_maps,example.shape[-2],example.shape[-1]))
    example = torch.constant_pad_nd(example,(conv_field,conv_field,conv_field,conv_field))
    for iix in range(conv_field,raw_out.shape[-1]+conv_field):
        for iiy in range(conv_field,raw_out.shape[-2] + conv_field):
            raw_out[:,:,iiy-conv_field,iix-conv_field] = net(example[0:2,:,iiy - conv_field:iiy + conv_field +1,iix-conv_field:iix+conv_field+1].float())[:,:,0,0]
else:
    raw_out = net(example[0:2, :, :, :].float())
raw_out = F.softmax(raw_out, dim=1)
raw_out = raw_out[0].unsqueeze(1)
raw_grid = utils.make_grid(raw_out, nrow=int(out_maps), padding=0)
raw_grid = raw_grid[0].cpu().detach().numpy()



density = []
correlation = []
order = []
length = []
angle = []
probability = []
for i in range(len(aa)):
    density.append(aa[i][0]['density'])
    correlation.append(aa[i][0]['correlation'])
    order.append(aa[i][0]['order'])
    length.append(aa[i][0]['bond'])
    angle.append(aa[i][0]['angle'])
    probability.append(aa[i][0]['probability'])

density = np.asarray(density)
correlation = np.asarray(correlation)
order = np.asarray(order)
length = np.asarray(length)
angle = np.asarray(angle)
probability = np.asarray(probability) 


plt.legend(('density','correlation','order','length','angle','probability'))   
'''