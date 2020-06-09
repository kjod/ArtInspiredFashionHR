import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from utils.dataloader import UnNormalize
from PIL import Image
from collections import OrderedDict
from models.networks import get_scheduler
from utils.logger import Logger

class BaseModel():

    def __init__(self, args, logger):
        self.args = args
        self.gpu_ids = args.gpu_ids
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available and self.gpu_ids != '-1':
            id = 'cuda:{}'.format(self.gpu_ids[0])
            #self.gpu_ids = 'cuda:{}'.format(self.gpu_ids[0])
        else:
            id = 'cpu'
        self.device = torch.device(id)
        #self.save_dir = os.path.join(args.checkpoints_dir, args.name)
        self.loss_names = []
        self.model_names = []
        self.sample_names = []
        self.one_samples = []
        self.logger = logger
        self.tensor = torch.cuda.FloatTensor if self.cuda_available else torch.Tensor
        self.tensor_instance = torch.cuda.FloatTensor() if self.cuda_available else torch.FloatTensor()
        self.check_gpu()
        self.unorm = UnNormalize(mean=(self.args.mean, self.args.mean, self.args.mean), std=(self.args.sd, self.args.sd, self.args.sd))

    # modify parser to add command line options,
    # and also change the default values if needed

    @staticmethod
    def modify_commandline_options():
        pass

    def name(self):
        return 'BaseModel'

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def check_gpu(self):
        print('--------------- Device Stats ---------------')
        if torch.cuda.is_available():
            print('Using GPU')
            print('Current device: ', torch.cuda.current_device())
            print('Fist device: ', torch.cuda.device(0))
            print('Device count: ', torch.cuda.device_count())
            print('First device name: ', torch.cuda.get_device_name(0))
        else:
            print("Using CPU")

    # load and print networks; create schedulers
    def setup(self, args, parser=None):
        if not 'continue_train' in args:
            self.schedulers = [get_scheduler(optimizer, args) for optimizer in self.optimizers]
        else:
            self.load_networks(args.which_epoch)
        self.print_networks(args.verbose)

    # make models eval mode during test time
    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backpropg
    def test(self):
        with torch.no_grad():
            self.forward()

    def optimize_parameters(self):
        pass

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        #print('learning rate = %.7f' % lr)

    # return visualization images. train.py will display these images, and save the images to a html
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    
    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        print('__patch_instance_norm_state_dict')
        print(module)
        print(keys)
        print('gsgsgsgsgghsahdsjgjsgdjgdshgjsdgjsdgdjsgdsj')
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            print('keys, ')
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:

            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    # load models from the disk
    '''def load_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (which_epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, '' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)'''

    def load_weights(self, dir, which_step):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s/%s_%s.pth' % (dir, name, which_step)
                if os.path.exists(load_filename):
                    net = getattr(self, '' + name)
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.module
                   
                    state_dict = torch.load(load_filename, map_location=str(self.device))
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata

                    od = OrderedDict()
                    for key in state_dict.keys():
                        od[key.replace('module.', '')] = state_dict[key]
                        #state_dict.pop(key, None)
                    # patch InstanceNorm checkpoints prior to 0.4
                    #for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    #    key = key.replace('module.', '')
                    #    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                    
                    net.load_state_dict(od)
                else:
                    print('%s file missing.' % load_filename)

    def get_losses(self):
        loss_dict = {}
        for name in self.loss_names:
            loss_dict[name] = float(getattr(self, name))
        return loss_dict


    def save_models(self, model_path, total_steps):
        for name in self.model_names:
            torch.save(getattr(self, name).state_dict(), '%s/%s_%d.pth' % (model_path, name, total_steps))
        

    #For dual GANS
    #Might be too slow
    def sample_images(self, val_dataloader, sample_path, total_steps):
        """Saves a generated sample from the validation set"""
   
        #for i, batch in enumerate(val_dataloader):
        imgs = next(iter(val_dataloader))
        self.set_input(imgs, self.args)
        self.forward()
        
        #Overall
        img_sample = Variable(self.tensor_instance)
        batch_size = getattr(self, self.sample_names[0]).size()

        for i in range(0, batch_size[0]):
            for name in self.sample_names:
                if name in self.one_samples:
                    image = getattr(self, name)[0].data
                else:    
                    image = getattr(self, name)[i].data
                
                if(not image.size()[0] == 3):
                    #image = np.stack((image,)*3, axis=-1)      
                    image = np.resize(image, (3, image.size()[0], image.size()[1]))
                    image = torch.from_numpy(image).to(self.device)
                    #image = np.stack((image,)*3, axis=-1)		
                    image = np.resize(image, (3, image.size()[0], image.size()[1]))
                    image = torch.from_numpy(image).to(self.device)
                img_sample = torch.cat([
                    self.unorm(image),
                    img_sample
                ], 0)

        img_sample = img_sample.view(batch_size[0] * len(self.sample_names), batch_size[1], batch_size[2], batch_size[3])
        save_image(img_sample, '%s/%s.png' % (sample_path, total_steps), nrow=len(self.sample_names), normalize=False)

    #Individual
    def sample_individual_images(self, val_dataloader, sample_path, total_steps, n):
        print('------------------')
        #print(iter(val_dataloader))
        imgs = next(iter(val_dataloader))
        self.set_input(imgs, self.args)
        self.forward()
        
        #Overall
        batch_size = getattr(self, self.sample_names[0]).size()
        limit = batch_size[0] if n > batch_size[0] else n
        for i in range(0, limit):
            for name in self.sample_names:
                if 'real' in name:
                    save_image(self.unorm(getattr(self, name)[i].data), '%s/%s_%s.png' % (sample_path, name, i), normalize=False)
                else:
                    save_image(self.unorm(getattr(self, name)[i].data), '%s/%s_%s_%s.png' % (sample_path, name, total_steps, i), normalize=False)


    def sample_training_images(self, batch, sample_path, total_steps, n):
        batch_size = getattr(self, self.sample_names[0]).size()
        limit = batch_size[0] if n > batch_size[0] else n
        for i in range(0, limit):
            for name in self.sample_names:
                if not 'real' in name:
                    save_image(self.unorm(getattr(self, name)[i].data),
                               '%s/%s_%s_%s.png' % (sample_path, name + '_train_', total_steps, i), normalize=False)

    # print network information
    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
