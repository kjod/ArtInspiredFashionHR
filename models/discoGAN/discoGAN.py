from utils.image_pool import ImagePool, unnorm
import torch
import itertools
from models.base_model import BaseModel
from models import networks
from options.model_specific_options import two_domain_parser_options
from torch.autograd import Variable
import torchvision.transforms as transforms

class discoGAN(BaseModel):

    def __init__(self, args, logger):
        super().__init__(args, logger)

        if not 'continue_train' in args:
            self.lambda_cycle_loss = self.args.lambda_cycle_loss
            self.lambda_rec_fake_identity = self.args.lambda_rec_fake_identity
            self.lambda_content_loss = self.args.lambda_content_loss
            self.lambda_style_loss = self.args.lambda_style_loss

            #if self.isTrain:
            channels = 3 if not args.greyscale else 1

            self.G_A = networks.define_G(channels, channels,
                                            args.ngf, args.which_model_netG, args.norm, not args.no_dropout, args.init_type, args.init_gain, self.gpu_ids)
            self.G_B = networks.define_G(channels, channels,
                                            args.ngf, args.which_model_netG, args.norm, not args.no_dropout, args.init_type, args.init_gain, self.gpu_ids)
            self.D_A = networks.define_D(channels, args.ndf, args.which_model_netD,
                                            args.n_layers_D, args.norm, init_type=args.init_type, init_gain=args.init_gain, gpu_ids=self.gpu_ids)
            self.D_B = networks.define_D(channels, args.ndf, args.which_model_netD,
                                            args.n_layers_D, args.norm, init_type=args.init_type, init_gain=args.init_gain, gpu_ids=self.gpu_ids)


            self.fake_A_pool = ImagePool(args.pool_size)
            self.fake_B_pool = ImagePool(args.pool_size)

            # define loss functions
            if self.args.use_wgan:
                self.criterionGAN = networks.WGANLoss(self.cuda_available).to(self.device)
            else:
                self.criterionGAN = networks.GANLoss(use_lsgan=not args.no_lsgan).to(self.device)

            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionRecFake = torch.nn.L1Loss()
            self.style_content_network = networks.Nerual_Style_losses(self.device)

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_A.parameters(), self.G_B.parameters()),
                                                lr=args.g_lr, betas=(args.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
                                                lr=args.d_lr, betas=(args.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        # add to logger
        self.loss_names = ['loss_D_A', 'loss_D_B', 'loss_G_A', 'loss_G_B', 'loss_cycle', 'loss_idt',
                           'loss_rec_fake', 'content_loss', 'style_loss']

        self.regularization_loss_names = ['loss_cycle', 'loss_rec_fake', 'content_loss']
        self.loss_names_lambda = {'loss_cycle': self.lambda_cycle_loss,
                                  'loss_rec_fake': self.lambda_rec_fake_identity,
                                  'content_loss': self.lambda_content_loss}

        self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        self.g_names = ['G_A', 'G_B']
        self.sample_names = ['fake_A', 'fake_B', 'rec_A', 'rec_B', 'real_A', 'real_B']

    def name(self):
        return 'DiscoGAN'

    @staticmethod
    def modify_commandline_options():
        return two_domain_parser_options()

    def set_input(self, input, args):
        AtoB = self.args.which_direction == 'AtoB'
        self.real_A = input[args.A_label if AtoB else args.B_label].to(self.device)
        self.real_B = input[args.B_label if AtoB else args.A_label].to(self.device)

    def forward(self):
        self.fake_B = self.G_A(self.real_A)
        self.rec_A = self.G_B(self.fake_B)

        self.fake_A = self.G_B(self.real_B)
        self.rec_B = self.G_A(self.fake_A)

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

    def backward_D_WGAN(self, netD, real, fake, num_steps):
        # generated_data = self.sample_generator(G, batch_size)
        pred_real = netD(real)
        pred_fake = netD(fake.detach())
        loss_D = self.criterionGAN(real, fake, pred_real, pred_fake, num_steps, netD, self.optimizer_D)
        loss_D.backward(retain_graph=True) #check this!!
        return loss_D

    def backward_D_A(self, num_steps):
        fake_B = self.fake_B_pool.query(self.fake_B)
        if self.args.use_wgan:
            self.loss_D_A = self.backward_D_WGAN(self.D_A, self.real_B, fake_B, num_steps)
        else:
            self.loss_D_A = self.backward_D_basic(self.D_A, self.real_B, fake_B)

    def backward_D_B(self, num_steps):
        fake_A = self.fake_A_pool.query(self.fake_A)
        if self.args.use_wgan:
            self.loss_D_B = self.backward_D_WGAN(self.D_B, self.real_A, fake_A, num_steps)
        else:
            self.loss_D_B = self.backward_D_basic(self.D_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_rec_fake = self.lambda_rec_fake_identity
        lambda_idt = self.args.lambda_identity
        lambda_A = self.args.lambda_A
        lambda_B = self.args.lambda_B

        ### Loss between generated image and real image ###
        if lambda_idt > 0:
            #self.idt_A = self.G_B(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.fake_A, self.real_A) * lambda_A * lambda_idt
            #self.idt_B = self.G_A(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.fake_B, self.real_B) * lambda_B * lambda_idt
            self.loss_idt = (self.loss_idt_A + self.loss_idt_B)/2
        else:
            self.loss_idt = 0

        ### Loss between G_A(G_B(real_b) and real_b
        if lambda_rec_fake > 0:
            tmpA = self.rec_A.clone().detach_()
            tmpB = self.rec_B.clone().detach_()

            _, self.loss_rec_fake_A = self.calculate_style_content_loss(self.fake_A, tmpA)
            _, self.loss_rec_fake_B  = self.calculate_style_content_loss(self.fake_B, tmpB)
            self.loss_rec_fake_A = self.loss_rec_fake_A * lambda_A * lambda_rec_fake
            self.loss_rec_fake_B = self.loss_rec_fake_B * lambda_B * lambda_rec_fake
            self.loss_rec_fake = (self.loss_rec_fake_A + self.loss_rec_fake_B)/2
        else:
            self.loss_rec_fake = 0

        if self.lambda_content_loss > 0 or self.lambda_style_loss > 0:
            self.style_lossA, self.content_lossA = self.calculate_style_content_loss(self.fake_A, self.real_A)
            self.style_lossB, self.content_lossB = self.calculate_style_content_loss(self.fake_B, self.real_B)

            self.style_lossA *= self.args.lambda_style_loss * lambda_A
            self.style_lossB *= self.args.lambda_style_loss * lambda_B
            self.content_lossA *= self.lambda_content_loss * lambda_A
            self.content_lossB *= self.lambda_content_loss * lambda_B

            self.content_loss = (self.content_lossA  + self.content_lossB)/2
            self.style_loss = (self.style_lossA + self.style_lossB)/2
        else:
            self.content_loss = 0
            self.style_loss = 0

        # Forward cycle loss
        _, self.loss_cycle_A = self.calculate_style_content_loss(self.rec_A, self.real_A)
        # self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A)
        self.loss_cycle_A *= lambda_A * self.lambda_cycle_loss
        # Backward cycle loss
        _, self.loss_cycle_B = self.calculate_style_content_loss(self.rec_B, self.real_B)
        # self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B)
        self.loss_cycle_B *= lambda_B * self.lambda_cycle_loss
        self.loss_cycle = (self.loss_cycle_A + self.loss_cycle_B)/2

        if self.args.use_wgan:
            self.loss_G_A = self.D_A(self.fake_B).mean()
            self.loss_G_B = self.D_B(self.fake_A).mean()
            self.adversial_loss = -( self.loss_G_A + self.loss_G_B)/2
        else:
            # GAN loss D_A(G_A(A))
            self.loss_G_A = self.criterionGAN(self.D_A(self.fake_B), True)
            # GAN loss D_B(G_B(B))
            self.loss_G_B = self.criterionGAN(self.D_B(self.fake_A), True)
            self.adversial_loss = (self.loss_G_A + self.loss_G_B)/2

        self.loss_G = self.adversial_loss + self.loss_cycle + self.loss_idt + self.loss_rec_fake + self.content_loss + self.style_loss
        self.loss_G.backward()

    def calculate_style_content_loss(self, img, target):
        style_loss = self.style_content_network.get_style_loss(img, target)
        content_loss = self.style_content_network.get_content_loss(img, target)
        return style_loss, content_loss

    def regulate_losses(self):
        model_losses = self.get_losses()

        for i in self.regularization_loss_names:
            loss_amount =self.loss_names_lambda[i]
            if model_losses[i] < self.args.loss_weighting_threshold and loss_amount < 1:
                print('Changing weighting of %s from %f to %f ' % (i, loss_amount, loss_amount * 10))
                print()
                self.loss_names_lambda[i] *= 10

    def optimize_parameters(self, num_steps, overwite_gen):
        # forward
        if overwite_gen or not self.args.use_wgan or num_steps % self.args.critic_iterations == 0:
            self.forward()
            # G_A and G_B
            self.set_requires_grad([self.D_A, self.D_B], False)
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()

        # D_A and D_B
        self.set_requires_grad([self.D_A, self.D_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A(num_steps)
        self.backward_D_B(num_steps)
        self.optimizer_D.step()
        if self.args.use_loss_weighting_check:
            self.regulate_losses()
