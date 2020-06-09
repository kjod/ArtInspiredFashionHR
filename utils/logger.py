import json
import os
from datetime import datetime
from torchvision.utils import save_image
from PIL import Image
from tensorboardX import SummaryWriter

class Logger:
    def __init__(self, logging_path, args):
        self.date_of_run = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')
        self.local_path = '%s/logs/%s/%s' % (logging_path, args.dataset, self.date_of_run)
        self.sample_path = os.path.join(self.local_path, 'samples/batched')
        self.individual_sample_path = os.path.join(self.local_path, 'samples/individual')
        self.model_path = os.path.join(self.local_path, 'model')
        self.results_path = os.path.join(self.local_path, 'results')
        os.makedirs(self.local_path, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)
        os.makedirs(self.individual_sample_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.results_path)
        self.args = args
        self.log_hyper_parameters()

        print('Arguments {}', args)
        print('Begining run %s' % self.date_of_run)

    def log_hyper_parameters(self,):
        with open(os.path.join(self.local_path, 'model_params.json'), 'w+') as file:
            file.write(json.dumps(vars(self.args)))  # use `json.loads` to do the reverse

    def log_losses(self, losses, batches_done):
        losses_str = ""       
        for name in losses:
            losses_str += "{}: {:.2f}   ".format(name, losses[name])
            self.writer.add_scalar(name, losses[name], batches_done)
        print(losses_str)

    def log_tensorboard_model_data(self, generator, discriminator, batches_done):

        for name, param in generator.named_parameters():
            self.writer.add_histogram("Generator/" + name, param.detach().data.cpu().numpy(), batches_done, bins='auto')

        for name, param in discriminator.named_parameters():
            self.writer.add_histogram("Discriminator/" + name, param.detach().data.cpu().numpy(), batches_done,
                                      bins='auto')

    def close_writers(self):
        self.writer.close()
