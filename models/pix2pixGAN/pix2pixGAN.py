#adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
import torch
from utils.image_pool import ImagePool
from models.base_model import BaseModel
from models import networks
from options.model_specific_options import add_lambda_L1, two_domain_parser_options
from torch.autograd import Variable
from torchvision.utils import save_image

class pix2pixGAN(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    @staticmethod
    def modify_commandline_options():
        parser = two_domain_parser_options()
        return add_lambda_L1(parser)

    def __init__(self, args, logger):
        super().__init__(args, logger)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['loss_G', 'loss_D']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G', 'D']

        self.sample_names = ['fake_B', 'real_A', 'real_B']
        # load/define networks
        self.G = networks.define_G(args.input_nc, args.output_nc, args.ngf,
                                      args.which_model_netG, args.norm, not args.no_dropout, args.init_type, args.init_gain, self.gpu_ids)

        if not 'continue_train' in args:
            use_sigmoid = args.no_lsgan
            self.D = networks.define_D(args.input_nc + args.output_nc, args.ndf,
                                          args.which_model_netD,
                                          args.n_layers_D, args.norm, use_sigmoid, args.init_type, args.init_gain, self.gpu_ids)

            self.fake_AB_pool = ImagePool(args.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not args.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.G.parameters(),
                                                lr=args.g_lr, betas=(args.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.D.parameters(),
                                                lr=args.d_lr, betas=(args.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input, args):
        AtoB = self.args.which_direction == 'AtoB'
        self.real_A = input[args.A_label if AtoB else args.B_label].to(self.device)
        self.real_B = input[args.B_label if AtoB else args.A_label].to(self.device)

    def forward(self):
        self.fake_B = self.G(self.real_A)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        pred_fake = self.D(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.D(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.D(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.args.lambda_L1

        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward()

    def optimize_parameters(self, num_steps, overwite_gen):
        self.forward()
        # update D
        self.set_requires_grad(self.D, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.D, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()