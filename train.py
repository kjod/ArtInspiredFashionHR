import time
import os
import importlib
import torchvision.transforms as transforms
import datetime
from utils.logger import Logger
from options.train_options import TrainOptions
from utils.dataloader import RedbubbleImageDataset
from torch.utils.data import DataLoader
from PIL import Image
#from utils import CreateDataLoader
from models._init_  import create_model
#from util.visualizer import Visualizer

dir_f = lambda x: '.'.join(['models', x, x])

def add_model_specfic_options(model_name):
    model = getattr(importlib.import_module(dir_f(model_name)), model_name)
    model_parser = model.modify_commandline_options()
    return model_parser


def main():
    train_opt = TrainOptions()
    args = train_opt.parse_options(add_model_specfic_options(train_opt.args.model))

    if args.greyscale:
        transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size), ),  # Image.BICUBIC), #Temp
                                        transforms.Grayscale(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((args.mean, args.mean, args.mean),
                                                             (args.sd, args.sd, args.sd))
                                        ])
    else:
        transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size), ),  # Image.BICUBIC), #Temp
                                        transforms.ToTensor(),
                                        transforms.Normalize((args.mean, args.mean, args.mean),
                                                             (args.sd, args.sd, args.sd))
                                        ])

    logger = Logger(os.path.join('models', args.model), args)
    model = create_model(args, logger, dir_f(args.model))
    
    '''TODO: Make data-sets generic'''
    val_dataloader = DataLoader(RedbubbleImageDataset("", transform=transform, train=False, n=args.n_examples, clothing_types=[args.dataset_label]),
                batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)
    data_loader = DataLoader(RedbubbleImageDataset("", transform=transform, train=True, n=args.n_examples, clothing_types=[args.dataset_label]),
               batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)

    model.setup(args)
    total_steps = 0
    batch_average_time = 9999
    epoch_average_time = 99999
    total_epochs = args.niter + args.niter_decay
    total_batches = len(data_loader)

    for epoch in range(args.epoch_count, args.niter + args.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, batch in enumerate(data_loader):
            batch_start_time = time.time()
            total_steps += args.batch_size
            epoch_iter += args.batch_size
            model.set_input(batch, args)
            model.optimize_parameters(total_steps, total_batches - i < args.batch_size * epoch or total_steps % total_batches == args.batch_size)

            iter_data_time = time.time()

            if total_steps % args.print_freq == 0:
                batch_average_time = time.time() - batch_start_time
                logger.log_losses(model.get_losses(), total_steps)

                print('Epoch: %d/%d Batch: %d/%d Time Taken: %s sec  Estimated Remaining: %s sec' %
                      (
                          epoch, total_epochs,
                          i, total_batches,
                          str(datetime.timedelta(seconds=batch_average_time)),
                          str(datetime.timedelta(seconds=epoch_average_time * (args.niter + args.niter_decay - epoch)))
                      )
                      )
                print('')

            if total_steps % args.save_training_sample_freq == 0:
                model.sample_training_images(batch, logger.individual_sample_path, total_steps,
                                               args.n_individual_samples)
                print('Saving training image samples (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                print('')

            if total_steps % args.save_model_freq == 0:
                model.save_models(logger.model_path, total_steps)
                print('Saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                print('')

            if total_steps % args.save_sample_freq == 0:
                model.sample_images(val_dataloader, logger.sample_path, total_steps)
                print('Saving the latest samples (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                print('')

            if total_steps % args.save_individual_sample_freq == 0:
                model.sample_individual_images(val_dataloader, logger.individual_sample_path, total_steps, args.n_individual_samples)
                print('Saving individual image samples (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                print('')

        epoch_average_time = time.time() - epoch_start_time
        model.update_learning_rate()

    model.sample_images(val_dataloader, logger.sample_path, total_steps)
    model.save_models(logger.model_path, total_steps)

if __name__ == '__main__':
    main()
