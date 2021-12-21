from utils import *
from torch import backends
from torchsummary import summary


def main(configs):
    experiment = get_comet_experiment(configs)
    model, optimizer, dataDims = initialize_training(configs)
    input_analysis = analyse_inputs(configs, dataDims)  # analyse inputs to prepare accuracy metrics
    log_input_stats(configs, experiment, input_analysis)

    print('Imported and Analyzed Training Dataset {}'.format(configs.training_dataset))

    if configs.CUDA:
        backends.cudnn.benchmark = True  # auto-optimizes certain backend processes
        #model = nn.DataParallel(model)  # go to multi-GPU training
        print("Using", torch.cuda.device_count(), "GPUs")
        model.to(torch.device("cuda:0"))
        print(summary(model, [(dataDims['channels'], dataDims['input y dim'], dataDims['input x dim'])]))

    ## BEGIN TRAINING/GENERATION
    if configs.max_epochs == 0:  # no training, just samples
        sample, time_ge, agreements, output_analysis = generation(configs, dataDims, model, input_analysis)
        log_generation_stats(experiment, sample, agreements, output_analysis)

    else:  # train it AND make samples!
        epoch = 1
        converged = 0
        tr_err_hist = []
        te_err_hist = []

        if configs.auto_training_batch:
            configs.training_batch_size, changed = get_training_batch_size(configs, model)  # confirm we can keep on at this batch size
        else:
            changed = 0
        if changed == 1:  # if the training batch is different, we have to adjust our batch sizes and dataloaders
            tr, te, _ = get_dataloaders(configs)
            print('Training batch set to {}'.format(configs.training_batch_size))
        else:
            tr, te, _ = get_dataloaders(configs)

        while (epoch <= (configs.max_epochs + 1)) & (converged == 0):  # over a certain number of epochs or until converged
            err_tr, time_tr = model_epoch(configs, dataDims = dataDims, trainData = tr, model = model, optimizer = optimizer, update_gradients = True)  # train & compute loss
            err_te, time_te = model_epoch(configs, dataDims = dataDims, trainData = te, model = model, update_gradients = False)  # compute loss on test set
            tr_err_hist.append(torch.mean(torch.stack(err_tr)))
            te_err_hist.append(torch.mean(torch.stack(err_te)))
            print('epoch={}; nll_tr={:.5f}; nll_te={:.5f}; time_tr={:.1f}s; time_te={:.1f}s'.format(epoch, torch.mean(torch.stack(err_tr)), torch.mean(torch.stack(err_te)), time_tr, time_te))
            converged = auto_convergence(configs, epoch, tr_err_hist, te_err_hist)

            if configs.comet:
                # get raw images
                sample0 = next(iter(tr))
                out = F.softmax(model(sample0.cuda().float()), dim=1).cpu().detach().numpy()
                experiment.log_metric(name = 'train error', value = tr_err_hist[-1], epoch = epoch)
                experiment.log_metric(name = 'test error', value = te_err_hist[-1], epoch = epoch)
                experiment.log_image(superscale_image(out[0,-1,:,:],1), name = 'raw training output epoch_{}'.format(epoch), image_scale=8)

            if epoch % 5 == 0:
                sample, time_ge, agreements, output_analysis = generation(configs, dataDims, model, input_analysis)
                log_generation_stats(configs, epoch, experiment, sample, agreements, output_analysis)

            epoch += 1

        # generate samples
        sample, time_ge, agreements, output_analysis = generation(configs, dataDims, model, input_analysis)
        log_generation_stats(configs, epoch, experiment, sample, agreements, output_analysis)
