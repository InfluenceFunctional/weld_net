from mainfile import main
import argparse


parser = argparse.ArgumentParser()
def add_bool_arg(parser, name, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action = 'store_true')
    group.add_argument('--no-' + name, dest=name, action = 'store_false')
    parser.set_defaults(**{name:default})

parser.add_argument('--run_num', type = int, default = 0)
parser.add_argument('--experiment_name', type = str, default = 'testing')
# model architecture
parser.add_argument('--model', type = str, default = 'gated1') # model architecture -- 'gated1'
parser.add_argument('--fc_depth', type = int, default = 64) # number of neurons for final fully connected layers
parser.add_argument('--init_conv_size', type=int, default= 7) # size of the initial convolutional window # ODD NUMBER
parser.add_argument('--conv_filters', type = int, default = 32) # number of filters per gated convolutional layer
parser.add_argument('--init_conv_filters', type=int, default = 32) # number of filters for the first convolutional layer  # MUST BE THE SAME AS 'conv_filters'
parser.add_argument('--conv_size', type = int, default = 3) # ODD NUMBER
parser.add_argument('--conv_layers', type = int, default = 10) # number of layers in the convnet - should be larger than the correlation length
parser.add_argument('--dilation', type = int, default = 1) # must be 1 - greater than 1 is deprecated
parser.add_argument('--activation_function', type = str, default = 'gated') # 'gated' is only working option
parser.add_argument('--fc_dropout_probability', type = float, default = 0) # dropout probability on hidden FC layer(s) [0,1)
parser.add_argument('--fc_norm', type = str, default = 'batch') # None or 'batch'
add_bool_arg(parser, 'subsample_images', default = True) # cut training images in transverse direction by a custom amount at runtime

add_bool_arg(parser,'do_conditioning', default = True) # incorporate conditioning variables in model training
parser.add_argument('--init_conditioning_filters', type=int, default=64) # number of filters for optional conditioning layers
parser.add_argument('-l', '--generation_conditions', nargs='+', default=[0.23, 0.22]) # conditions used to generate samples at runtime

# training parameters
parser.add_argument('--training_dataset', type = str, default = 'welds 1') # name of training dataset - 'fake welds', 'welds 1'
parser.add_argument('--training_batch_size', type = int, default = 1000) # maximum training batch size
add_bool_arg(parser, 'auto_training_batch', default = True) # whether to automatically set training batch size to largest value < the max
parser.add_argument('--max_epochs', type = int, default = 500) # number of epochs over which to train
parser.add_argument('--convergence_moving_average_window', type = int, default = 10000) # BROKEN - moving average window used to compute convergence criteria
parser.add_argument('--max_dataset_size', type = int, default = 10000) # maximum dataset size (limited by size of actual dataset) - not sure the dataset is properly shuffled prior to this being applied
parser.add_argument('--convergence_margin', type = float, default = 1e-4) # cutoff which determines when the model has converged
parser.add_argument('--dataset_seed', type = int, default = 0)
parser.add_argument('--model_seed', type = int, default = 0)

# sample generation parameters
parser.add_argument('--bound_type', type = str, default = 'empty') # what is outside the image during training and generation 'empty'
parser.add_argument('--boundary_layers', type = int, default = 0) # number of layers of conv_field between sample and actual image boundary
parser.add_argument('--sample_outpaint_ratio', type = int, default = 4) # size of sample images, relative to the input images
parser.add_argument('--sample_generation_mode', type = str, default = 'parallel') # 'parallel' or 'serial' - serial is currently untested
parser.add_argument('--sample_batch_size', type = int, default = 100) # maximum sample batch size - no automated test but can generally be rather large (1e3),
parser.add_argument('--generation_period', type = int, default = 5) # how often to run (expensive) generation during training
# utility of higher batch sizes for parallel generation is only realized with extremely large samples
parser.add_argument('--n_samples', type = int, default = 1) # number of samples to generate

add_bool_arg(parser, 'CUDA', default=True)
add_bool_arg(parser, 'comet', default=True)

configs = parser.parse_args()

if __name__ == '__main__':  # run it!
    main(configs)
