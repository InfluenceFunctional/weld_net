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
parser.add_argument('--conv_filters', type = int, default = 64)
parser.add_argument('--conv_size', type = int, default = 3)
parser.add_argument('--conv_layers', type = int, default = 15)
parser.add_argument('--dilation', type = int, default = 1) # must be 1 - greater than 1 is deprecated
parser.add_argument('--activation_function', type = str, default = 'gated') # 'gated' is only working option for 'gated1' model

# training parameters
parser.add_argument('--training_dataset', type = str, default = 'fake welds') # name of training dataset - 'fake welds'
parser.add_argument('--training_batch_size', type = int, default = 1000) # maximum training batch size
add_bool_arg(parser, 'auto_training_batch', default = True) # whether to automatically set training batch size to largest value < the max
parser.add_argument('--max_epochs', type = int, default = 100) # number of epochs over which to train
parser.add_argument('--convergence_moving_average_window', type = int, default = 10000) # moving average window used to compute convergence criteria
parser.add_argument('--max_dataset_size', type = int, default = 1000000) # maximum dataset size (limited by size of actual dataset)
parser.add_argument('--convergence_margin', type = float, default = 1e-4) # cutoff which determines when the model has converged
parser.add_argument('--dataset_seed', type = int, default = 0)
parser.add_argument('--model_seed', type = int, default = 0)

# sample generation parameters
parser.add_argument('--bound_type', type = str, default = 'empty') # what is outside the image during training and generation 'empty'
parser.add_argument('--boundary_layers', type = int, default = 2) # number of layers of conv_field between sample and actual image boundary
parser.add_argument('--sample_outpaint_ratio', type = int, default = 4) # size of sample images, relative to the input images
parser.add_argument('--sample_generation_mode', type = str, default = 'parallel') # 'parallel' or 'serial'
parser.add_argument('--sample_batch_size', type = int, default = 1000) # maximum sample batch size
parser.add_argument('--n_samples', type = int, default = 1) # number of samples to generate

add_bool_arg(parser, 'CUDA', default=True)
add_bool_arg(parser, 'comet', default=True)

configs = parser.parse_args()

if __name__ == '__main__':  # run it!
    main(configs)
