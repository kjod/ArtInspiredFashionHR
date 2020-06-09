from utils.image_pool import ImagePool, unnorm
import torch
import itertools
from models.base_model import BaseModel
from models import networks
from options.model_specific_options import two_domain_parser_options, weighting_shapeGAN_hyperparam
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class shapeGAN(BaseModel):

    def __init__(self, args, logger):
        super().__init__(args, logger)

        if not 'continue_train' in args:
            self.lambda_cycle_loss = self.args.lambda_cycle_loss
            self.lambda_rec_fake_identity = self.args.lambda_rec_fake_identity
            self.lambda_content_loss = self.args.lambda_content_loss
            self.lambda_style_loss = self.args.lambda_style_loss

            #if self.isTrain:
            channels = 3 if not args.greyscale else 1

            self.RAFN = networks.define_G(channels, channels,
                                            args.ngf, 'RAFN', args.norm, not args.no_dropout, args.init_type, args.init_gain, self.gpu_ids)
            self.STN = networks.define_G(channels, channels,
                                            args.ngf, 'STN', args.norm, not args.no_dropout, args.init_type, args.init_gain, self.gpu_ids)
            self.G_c = networks.define_G(channels, channels,
                                            args.ngf, 'cGen', args.norm, not args.no_dropout, args.init_type, args.init_gain, self.gpu_ids)
            self.G_dec = networks.define_G(channels, channels,
                                            args.ngf, 'decGen', args.norm, not args.no_dropout, args.init_type, args.init_gain, self.gpu_ids)
            self.G_M_dec = networks.define_G(channels, channels,
                                           args.ngf, 'decGen', args.norm, not args.no_dropout, args.init_type, args.init_gain, self.gpu_ids)

            self.D_c = networks.define_D(channels, args.ndf, 'basic',
                                            args.n_layers_D, args.norm, init_type=args.init_type, init_gain=args.init_gain, gpu_ids=self.gpu_ids)
            self.D_dec = networks.define_D(channels, args.ndf, 'cDis',
                                            args.n_layers_D, args.norm, init_type=args.init_type, init_gain=args.init_gain, gpu_ids=self.gpu_ids)

            self.fake_A_pool = ImagePool(args.pool_size)
            self.fake_B_pool = ImagePool(args.pool_size)

            mask = Image.open('data/dress_mask.jpg')
            transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size), ),  # Image.BICUBIC), #Temp
                                           transforms.Grayscale(),
                                           transforms.ToTensor(),
                                           transforms.Normalize((args.mean, args.mean, args.mean),
                                                                (args.sd, args.sd, args.sd))
                                           ])
            self.real_mask = transform(mask).to(self.device)

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not args.no_lsgan).to(self.device)

            self.criterionCycle = torch.nn.L1Loss()
            self.RAFN_L1 = torch.nn.L1Loss()
            self.STN_L1 = torch.nn.L1Loss()
            self.paired_L1 = torch.nn.L1Loss()
            self.mask_L1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizer_RAFN =  torch.optim.Adam(self.RAFN.parameters(),
                                                lr=args.g_lr, betas=(args.beta1, 0.999))
            self.optimizer_STN = torch.optim.Adam(self.STN.parameters(),
                                                lr=args.g_lr, betas=(args.beta1, 0.999))
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_c.parameters(), self.G_dec.parameters(), self.G_M_dec.parameters()),
                                                lr=args.g_lr, betas=(args.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.D_c.parameters(), self.D_dec.parameters()),
                                                lr=args.d_lr, betas=(args.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        # add to logger
        self.loss_names = ['loss_D_c', 'loss_dec', 'loss_G', 'loss_cycle', 'RAFN_loss', 'STN_loss', 'adversial_loss',
         'loss_cycle', 'paired_loss']

        self.regularization_loss_names = ['loss_cycle', 'loss_rec_fake', 'content_loss']
        self.loss_names_lambda = {'loss_cycle': self.lambda_cycle_loss,
                                  'loss_rec_fake': self.lambda_rec_fake_identity,
                                  'content_loss': self.lambda_content_loss}

        self.model_names = ['RAFN', 'G_c', 'G_dec', 'G_M_dec', 'D_c', 'D_dec', 'STN']
        self.one_samples = ['real_mask']
        self.sample_names = ['real_A', 'real_B', 'shape_B',  'input_A','input_B','rec_A', 'rec_B','fake_B']

    def name(self):
        return 'shapeGAN'

    @staticmethod
    def modify_commandline_options():
        parser = weighting_shapeGAN_hyperparam()
        return two_domain_parser_options(parser)

    def set_input(self, input, args):
        AtoB = self.args.which_direction == 'AtoB'
        self.real_A = input[args.A_label if AtoB else args.B_label].to(self.device)
        self.real_B = input[args.B_label if AtoB else args.A_label].to(self.device)

    def forward(self):
        #create array of masks#
        mask_batch = self.real_mask#np.array(self.mask.copy())
        mask_batch = self.real_mask.expand(self.real_A.size()[0], 1, self.real_A.size()[2], self.real_A.size()[3]) # specifies new size
        
        #mask_batch.repeat(self.real_A.size[0], 1, 1)  # specifies number of copies

        self.shape_B, self.mask_B = self.RAFN(self.real_A, mask_batch)
        self.input_A, self.input_B = self.STN(self.shape_B.detach(), self.real_A)

        self.fake_B = self.G_c(self.input_A.detach(), self.input_B.detach())

        self.rec_A, self.rec_B = self.G_dec(self.fake_B.detach())
        self.rec_m_A, self.rec_m_B = self.G_dec(self.fake_B)

    def backward_RAFN(self):
        self.RAFN_loss = self.RAFN_L1(self.shape_B, self.real_A) * self.args.lambda_RAFN
        self.RAFN_loss.backward()

    def backward_STN(self):
        self.STN_loss_A = self.STN_L1(self.input_A, self.real_A)
        self.STN_loss_B = self.STN_L1(self.input_B, self.real_B)
        self.STN_loss = (self.STN_loss_A + self.STN_loss_B) * 0.5
        self.STN_loss.backward()
        self.STN_loss = 0

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_c(self, num_steps):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_c = self.backward_D_basic(self.D_c, self.real_B, fake_B) * self.args.lambda_C

    def backward_D_dec(self, num_steps):

        fake_B = self.fake_B_pool.query(self.fake_B)

        pred_real_A = self.D_dec(self.real_B, self.real_A)
        pred_real_B = self.D_dec(self.real_B, self.real_B)
        loss_D_real_A = self.criterionGAN(pred_real_A, True)
        loss_D_real_B = self.criterionGAN(pred_real_B, True)
        loss_D_real = (loss_D_real_A + loss_D_real_B) * 0.5

        pred_fake_A = self.D_dec(fake_B, self.rec_A)
        pred_fake_B = self.D_dec(fake_B, self.rec_B)
        loss_D_fake = (self.criterionGAN(pred_fake_A, False) + self.criterionGAN(pred_fake_B, False)) * 0.5

        ##Include the mask loss
        #self.rec_m_A =
        #self.rec_m_B = self.G_dec(self.fake_B)

        self.loss_dec = (loss_D_real + loss_D_fake) * 0.5 * self.args.lambda_dec
        # backward
        self.loss_dec.backward()

    def backward_G(self):
        #self.mask_loss = torch.nn.L1Loss()
        self.paired_loss = self.paired_L1(self.fake_B, self.real_B) * self.args.lambda_C

        # Forward cycle loss
        
        self.loss_cycle_A = self.criterionCycle(self.rec_A.detach(), self.input_A.detach())
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B.detach(), self.input_B.detach())
        self.loss_cycle = (self.loss_cycle_A + self.loss_cycle_B)/2 * self.args.lambda_dec

        ###Adversial Loss###
        self.loss_G_c = self.criterionGAN(self.D_c(self.fake_B.detach()), True) * self.args.lambda_C
        self.loss_G_dec_A = self.criterionGAN(self.D_dec(self.fake_B.detach(), self.rec_A.detach()), True)
        self.loss_G_dec_B = self.criterionGAN(self.D_dec(self.fake_B.detach(), self.rec_B.detach()), True)
        self.loss_G_dec = (self.loss_G_dec_A + self.loss_G_dec_B)/2 * self.args.lambda_dec

        self.adversial_loss = (self.loss_G_c + self.loss_G_dec)/2

        self.loss_G = self.adversial_loss + self.loss_cycle + self.paired_loss
        self.loss_G.backward()

    def optimize_parameters(self, num_steps, overwite_gen):
        # forward
        self.forward()
        
        self.set_requires_grad([self.D_c, self.D_dec], False)

        self.optimizer_RAFN.zero_grad()
        self.backward_RAFN()
        self.optimizer_RAFN.step()
        
        self.optimizer_STN.zero_grad()  
        self.backward_STN()
        self.optimizer_STN.step()

        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # D_A and D_B
        self.set_requires_grad([self.D_c, self.D_dec], True)
        self.optimizer_D.zero_grad()
        self.backward_D_c(num_steps)
        self.backward_D_dec(num_steps)
        self.optimizer_D.step()
