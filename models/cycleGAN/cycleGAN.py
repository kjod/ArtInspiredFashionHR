import torch
import itertools
import pandas as pd
from utils.image_pool import ImagePool
from models.base_model import BaseModel
from options.model_specific_options import two_domain_parser_options
from models import networks
from torch.autograd import Variable
from torchvision.utils import save_image

class cycleGAN(BaseModel):

    def __init__(self, args, logger):
        super().__init__(args, logger)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['loss_D_A', 'loss_D_B', 'loss_G_A', 'loss_G_B', 'loss_cycle_A', 'loss_cycle_B', 'loss_idt_A' , 
                            'loss_idt_B', 'content_loss', 'style_loss', 'loss_rec_fake']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        self.sample_names = ['fake_A', 'fake_B', 'rec_A', 'rec_B', 'real_A', 'real_B']

        use_sigmoid = args.no_lsgan

        if True:
            self.G_A = networks.define_G(args.input_nc, args.output_nc,
                                        args.ngf, args.which_model_netG, args.norm, not args.no_dropout, args.init_type,
                                        args.init_gain, self.gpu_ids)
            self.G_B = networks.define_G(args.output_nc, args.input_nc,
                                        args.ngf, args.which_model_netG, args.norm, not args.no_dropout, args.init_type,
                                        args.init_gain, self.gpu_ids)


            self.D_A = networks.define_D(args.output_nc, args.ndf, args.which_model_netD,
                                            args.n_layers_D, args.norm, use_sigmoid, args.init_type, args.init_gain,
                                            self.gpu_ids)
            self.D_B = networks.define_D(args.input_nc, args.ndf, args.which_model_netD,
                                            args.n_layers_D, args.norm, use_sigmoid, args.init_type, args.init_gain,
                                            self.gpu_ids)

        else:
            print('Todo load model')

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_A.parameters(), self.G_B.parameters()),
                                            lr=args.g_lr, betas=(args.beta1, args.beta2))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
                                            lr=args.d_lr, betas=(args.beta1, args.beta2))
        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        
        self.optimizers.append(self.optimizer_D)
        
        self.fake_A_pool = ImagePool(args.pool_size)
        self.fake_B_pool = ImagePool(args.pool_size)
        # define loss functions
        self.lambda_content_loss = self.args.lambda_content_loss
        self.lambda_style_loss = self.args.lambda_style_loss

        self.criterionGAN = networks.GANLoss(use_lsgan=not args.no_lsgan).to(self.device)
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        self.criterionFakeRec = torch.nn.L1Loss()
        self.style_content_network = networks.Nerual_Style_losses(self.device)
            

    def name(self):
        return 'CycleGAN'

    @staticmethod
    def modify_commandline_options():
        return two_domain_parser_options()

    def set_input(self, input, args):
        AtoB = self.args.which_direction == 'AtoB'
        self.real_A = input[args.A_label if AtoB else args.B_label].to(self.device)
        self.real_B = input[args.B_label if AtoB else args.A_label].to(self.device)
        self.bb = input['Bb'].to(self.device)

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

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.D_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.D_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = self.args.lambda_identity
        lambda_rec_fake = self.args.lambda_rec_fake_identity
        lambda_A = self.args.lambda_A
        lambda_B = self.args.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.G_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.bb * self.idt_A, self.bb *self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.G_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.bb * self.idt_B, self.bb * self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0


        if lambda_rec_fake > 0:
            tmpA = self.rec_A.clone().detach_()
            tmpB = self.rec_B.clone().detach_()

            self.loss_rec_fake_A = self.criterionFakeRec(self.fake_A, tmpA)
            self.loss_rec_fake_B  = self.criterionFakeRec(self.fake_B, tmpB)
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

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.D_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.D_B(self.fake_A), True)
        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.bb * self.rec_A, self.bb * self.real_A) * lambda_A
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.bb *self.rec_B, self.bb * self.real_B) * lambda_B
        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + \
                      self.loss_idt_B + self.content_loss + self.style_loss + self.loss_rec_fake
        self.loss_G.backward()


    def calculate_style_content_loss(self, img, target):
        style_loss = self.style_content_network.get_style_loss(img, target)
        content_loss = self.style_content_network.get_content_loss(img, target)
        return style_loss, content_loss


    def optimize_parameters(self, num_steps, overwite_gen):
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.D_A, self.D_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad([self.D_A, self.D_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()
        
