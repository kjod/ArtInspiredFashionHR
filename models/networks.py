#Taken from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
from PIL import Image
import torchvision.transforms as transforms

###############################################################################
# Helper Functions
###############################################################################

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, args):
    if args.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + args.epoch_count - args.niter) / float(args.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_iters, gamma=0.1)
    elif args.lr_upolicy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if not gpu_ids == '-1':
        assert(torch.cuda.is_available())
        #Loop here
        net.to('cuda:{}'.format(gpu_ids[0]))
        net = torch.nn.DataParallel(net, list(map(int, gpu_ids)))
        print('Data parellel activated')
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, model_type, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if model_type == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif model_type == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif model_type == 'unet_64':
        netG = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif model_type == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif model_type == 'vdsr_128':
        netG = VDSR(input_nc, output_nc)
    elif model_type == 'auto_128':
        netG = Autoencoder(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif model_type == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif model_type == 'unet_512':
        netG = UnetGenerator(input_nc, output_nc, 9, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif model_type == 'STN':
        netG = RSTN()
    elif model_type == 'RAFN':
        netG = RAFN()
    elif model_type == 'cGen':
        netG = cGenerator()
    elif model_type == 'decGen':
        netG = decGenerator()
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % model_type)
    return init_net(netG, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, model_type,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if model_type == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif model_type == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif model_type == 'sim_128':
        netD = SimDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif model_type == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif model_type == 'discoD':
        netD = basicDiscriminator(input_nc)
    elif model_type == 'cDis':
        netD = cDiscriminator()
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  model_type)
    return init_net(netD, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class WGANLoss(nn.Module):
    def __init__(self, cuda_available, gp_weight=10, wp=True):
        super(WGANLoss, self).__init__()
        self.gp_weight = gp_weight
        self.cuda_available = cuda_available

    def __call__(self, real_data, generated_data, d_real, d_generated, num_steps, D, opt):
        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(real_data, generated_data, D)
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        return d_loss

    def _gradient_penalty(self, real_data, generated_data, D):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if self.cuda_available:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        if self.cuda_available:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = D(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda() if self.cuda_available else torch.ones(
                               prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def sample_generator(self, num_samples):
        latent_samples = Variable(self.G.sample_latent(num_samples))
        if self.cuda_available:
            latent_samples = latent_samples.cuda()
        generated_data = self.G(latent_samples)
        return generated_data

    def sample(self, num_samples):
        generated_data = self.sample_generator(num_samples)
        # Remove color channel
        return generated_data.data.cpu().numpy()[:, 0, :, :]

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, mask=None)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, mask=''):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            if mask:
                self.mask = mask
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        #elif self.innermost:
        #    return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class EncodeBlock(nn.Module):
    def __init__(self, inner_nc, outer_nc, use_bias=False, norm_layer=nn.BatchNorm2d, kernel_size=4, stride=2, padding=1):
        super(EncodeBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(inner_nc, outer_nc, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            norm_layer(outer_nc),
        )

    def forward(self, x):
        return self.model(x)

class DecodeBlock(nn.Module):
    def __init__(self, inner_nc, outer_nc, use_bias=False, norm_layer=nn.BatchNorm2d, kernel_size=4, stride=2, padding=1, use_dropout=False):
        super(DecodeBlock, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias),
            nn.LeakyReLU(0.2, True),##Change to Leaky Relu
            norm_layer(outer_nc),
        )

    def forward(self, x):
        return self.model(x)


class decGenerator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(decGenerator, self).__init__()
        self.model = nn.Sequential(
            EncodeBlock(3, 32, stride=1, padding=0),
            EncodeBlock(32, 64, kernel_size=4, stride=2, padding=1),
            EncodeBlock(64, 128, kernel_size=4, stride=2, padding=1),
            EncodeBlock(128, 256, kernel_size=4, stride=2, padding=1),
            EncodeBlock(256, 512, kernel_size=4, stride=2, padding=1),
            EncodeBlock(512, 512, kernel_size=4, stride=1, padding=0),
            
            DecodeBlock(512, 256, kernel_size=4, stride=2, padding=1),
            DecodeBlock(256, 128, kernel_size=4, stride=2, padding=1),
            DecodeBlock(128, 64, kernel_size=4, stride=2, padding=1),
            DecodeBlock(64, 32, kernel_size=4, stride=2, padding=1),
            
            nn.ConvTranspose2d(32, 6, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )


    # forward method
    def forward(self, x):
        x = self.model(x)
        image_1 = x[:, 0:3, :]
        image_2 = x[:, 3:6, :]
        return image_1, image_2

class cGenerator(nn.Module):
    # initializers
    def __init__(self, d=2):
        super(cGenerator, self).__init__()
        self.deconv1_1 = EncodeBlock(3, 64, stride=1, padding=0)
        self.deconv1_2 = EncodeBlock(3, 64, stride=1, padding=0)
        self.model = nn.Sequential(
            EncodeBlock(128, 256),
            EncodeBlock(256, 512),
            EncodeBlock(512, 512),
            DecodeBlock(512, 256, kernel_size=4, stride=2, padding=1),
            DecodeBlock(256, 128, kernel_size=3, stride=2, padding=0),
            DecodeBlock(128, 64, kernel_size=4, stride=2, padding=0),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=1, padding=0),
            nn.Tanh()
        )

    # forward method
    def forward(self, input_A, label):
        x = self.deconv1_1(input_A)
        y = self.deconv1_2(label)
        x = torch.cat([x, y], 1)
        x = self.model(x)
        return x

class cDiscriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(cDiscriminator, self).__init__()

        self.conv1_1 = EncodeBlock(3, 64)

        self.conv1_2 = EncodeBlock(3, 64)

        self.model = nn.Sequential(
            EncodeBlock(128, 256),
            EncodeBlock(256, 512),
            EncodeBlock(512, 1024),
            nn.Conv2d(1024, 1, 4, 1, 0),
            nn.Sigmoid()
        )
        # self.conv4 = nn.Conv2d(d*4, 1, 4, 1, 0)

    # forward method
    # def forward(self, input):
    def forward(self, input_A, label):
        x = self.conv1_1(input_A)
        y = self.conv1_2(label)
        x = torch.cat([x, y], 1)
        x = self.model(x)
        return x


def get_mask(input_size):
    mask = Image.open('data/dress_mask.jpg')
    transform = transforms.Compose([transforms.Resize((128, 128), ),  # Image.BICUBIC), #Temp
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
            ])
    mask  = transform(mask)
    mask = mask.expand(input_size[0], 1, input_size[2], input_size[3])
    return mask

#Relative Appearance Flow Network
class RAFN(nn.Module):
    def __init__(self):
        super(RAFN, self).__init__()

        #concatenate mask
        self.encoder = nn.Sequential(
            EncodeBlock(4, 16, False, nn.BatchNorm2d),
            EncodeBlock(16, 32, False, nn.BatchNorm2d),
            EncodeBlock(32, 64, False, nn.BatchNorm2d),
            EncodeBlock(64, 128, False, nn.BatchNorm2d),
            EncodeBlock(128, 256, False, nn.BatchNorm2d),
            EncodeBlock(256, 512, False, nn.BatchNorm2d, kernel_size=5, stride=1, padding=2)
        )

        self.decoder = nn.Sequential(
            DecodeBlock(512, 256, False, nn.BatchNorm2d),
            DecodeBlock(256, 128, False, nn.BatchNorm2d),
            DecodeBlock(128, 64, False, nn.BatchNorm2d),
            DecodeBlock(64, 32, False, nn.BatchNorm2d),
            DecodeBlock(32, 16, False, nn.BatchNorm2d),
        )

        self.flow_activation = nn.Sequential(
                                    nn.ConvTranspose2d(16, 2, 5, 1, 2),#Used 2 in paper
                                    nn.Tanh())
        self.mask_activation = nn.Sequential(
                                    nn.ConvTranspose2d(16, 1, 5, 1, 2),
                                    nn.Sigmoid())


    def forward(self, x, mask=None):
        output_one = False
        #if not mask:
        #    mask = get_mask(x.size())
        #    output_one = True

        combined_image = torch.cat((x, mask), 1)
        encoded_image = self.encoder(combined_image)
        predicted_mask = self.mask_activation(self.decoder(encoded_image)) ## Not sure what do with this
        grid = self.flow_activation(self.decoder(encoded_image))
        #grid = F.affine_grid(grid, x.size())

        grid = grid.view(x.size()[0], 128, 128, 2)#Allows grid sample to work (b,h,w,c)#
        x =  F.grid_sample(x, grid, mode='bilinear')

        if output_one:
            return x
        else:
            return x, predicted_mask

#Relative Spatial Transformer Network
class RSTN(nn.Module):
    def __init__(self):
        super(RSTN, self).__init__()
        def loc_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(inplace=True)
            )

        self.localization = nn.Sequential(
            loc_block(6, 8),
            loc_block(8, 10),
            loc_block(10, 30),
            loc_block(30, 30),
            loc_block(30, 30),
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(120 * 4, 32),#change this ti 480
            nn.ReLU(True),
            nn.Linear(32, 12),
            nn.ReLU(True), #For theta#
            nn.Linear(12, 6)
        )
        # Spatial transformer network forward function

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 120 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x, y):
        combined_image = torch.cat((x, y), 1)
        x = self.stn(combined_image)
        #split images

        return x[:, 0:3, :, :], x[:, 3:6, :, :]

class Autoencoder(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 input_nc=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Autoencoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d


        self.encoder = nn.Sequential(
            EncodeBlock(inner_nc, 64, use_bias, norm_layer, input_nc, outermost, innermost, use_dropout),
            EncodeBlock(64, 128, use_bias, norm_layer, input_nc, outermost, innermost, use_dropout),
            EncodeBlock(128, 256, use_bias, norm_layer, input_nc, outermost, innermost, use_dropout),
            EncodeBlock(256, 512, use_bias, norm_layer, input_nc, outermost, innermost, use_dropout),
            EncodeBlock(512, 1024, use_bias, norm_layer, input_nc, outermost, innermost, use_dropout),
        )

        self.decoder = nn.Sequential(
            DecodeBlock(1024, 512, use_bias, norm_layer, input_nc, outermost, innermost, use_dropout),
            DecodeBlock(512, 256, use_bias, norm_layer, input_nc, outermost, innermost, use_dropout),
            DecodeBlock(256, 128, use_bias, norm_layer, input_nc, outermost, innermost, use_dropout),
            DecodeBlock(128, 64, use_bias, norm_layer, input_nc, outermost, innermost, use_dropout),
            nn.ConvTranspose2d(64, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)

##############################
#        Discriminator
##############################

class basicDiscriminator(nn.Module):
    def __init__(self, in_channels=3, out_channels=128):
        super(basicDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, out_channels, normalization=False),
            *discriminator_block(out_channels, out_channels * 2),
            *discriminator_block(out_channels * 2, out_channels * 2^2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(out_channels * 2^2, 1, 4, padding=1)
        )

    def forward(self, img):
        # Concatenate image and condition image by channels to produce input
        return self.model(img)


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward(self, x):
        return x + self.main(x)


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


from math import sqrt
class VDSR(nn.Module):
    def __init__(self, input_nc=3, out_nc=3):
        super(VDSR, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=input_nc, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=out_nc, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out, residual)
        return out


class SimDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf, n_layers_D, norm_layer=nn.BatchNorm2d, use_sigmoid=False, in_channels=3, out_channels=128):
        super(SimDiscriminator, self).__init__()

        '''def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, out_channels, normalization=False),
            *discriminator_block(out_channels, out_channels * 2),
            *discriminator_block(out_channels * 2, out_channels * 2^2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(out_channels * 2^2, 1, 4, padding=1)
        )'''

        self.model = nn.Sequential(
            nn.ReflectionPad2d(in_channels),
            nn.Conv2d(in_channels, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),
        )

        self.fc = nn.Sequential(
            nn.Linear(8 * 132 * 132, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 5))

    def run(self, x):
        output = self.model(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, imgA, imgB):
        A = self.run(imgA)
        B = self.run(imgB)
        return A, B

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Based on https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch/blob/master/Siamese-networks-medium.ipynb
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin=2.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class Nerual_Style_losses:
    
    def __init__(self, device, mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])):
        self.device = device
        self.mean = mean.to(device)
        self.std = std.to(device)
        self.content_layers = ['conv_1', 'conv_2', 'conv_4']
        self.style_layers = ['conv_1', 'conv_3', 'conv_4', 'conv_5']
        self.create_models()

    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super().__init__()
            # .view the mean and std to make them [C x 1 x 1] so that they can
            # directly work with image Tensor of shape [B x C x H x W].
            # B is batch size. C is number of channels. H is height and W is width.
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def forward(self, img):
            # normalize img
            return (img - self.mean) / self.std

    class ContentLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.save = True
            self.calculate_loss = False

        def set_mode(self, save, calculate_loss):
            self.save = save
            self.calculate_loss = calculate_loss

        def forward(self, input):
            if self.save:##Input image is content image
                self.content_image = input

            if self.calculate_loss:
                self.loss = F.mse_loss(input, self.content_image)

            return input


    class StyleLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.save = True
            self.calculate_loss = False

        def set_mode(self, save, calculate_loss):
            self.save = save
            self.calculate_loss = calculate_loss

        def forward(self, input_image):
            if self.save:##Input image is style image
                self.G_style = self.gram_matrix(input_image).detach()
            
            if self.calculate_loss:
                G_input = self.gram_matrix(input_image)
                self.loss = F.mse_loss(G_input, self.G_style)
            return input_image

        def gram_matrix(self, input):
            a, b, c, d = input.size()  # a=batch size(=1)
            # b=number of feature maps
            # (c,d)=dimensions of a f. map (N=c*d)

            features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

            G = torch.mm(features, features.t())  # compute the gram product

            # we 'normalize' the values of the gram matrix
            # by dividing by the number of element in each feature maps.
            return G.div(a * b * c * d)


    def get_style_loss(self, fake_image, style_image):
        style_loss = 0

        for sl in self.style_losses:
            sl.set_mode(save=True, calculate_loss=False)

        self.loss_model(style_image)

        for sl in self.style_losses:
            sl.set_mode(save=False, calculate_loss=True)

        self.loss_model(fake_image)

        for sl in self.style_losses:
            style_loss += sl.loss
            sl.set_mode(save=False, calculate_loss=False)

        return style_loss

    def get_content_loss(self, fake_image, content_image):
        content_loss = 0

        for cl in self.content_losses:
            cl.set_mode(save=True, calculate_loss=False)

        self.loss_model(content_image)

        for cl in self.content_losses:
            cl.set_mode(save=False, calculate_loss=True)

        self.loss_model(fake_image)

        for cl in self.content_losses:
            content_loss += cl.loss
            cl.set_mode(save=False, calculate_loss=False)

        return content_loss

    def create_models(self):
        cnn = models.vgg19(pretrained=True).features.to(self.device).eval()
        #self.cnn = copy.deepcopy(self.cnn)

        # normalization module
        normalization = self.Normalization(self.mean, self.std).to(self.device)

        # just in order to have an iterable access to or list of content/syle
        # losses
        self.content_losses = []
        self.style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        loss_model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            loss_model.add_module(name, layer)

            if name in self.content_layers:
                # add content loss:
                content_loss = self.ContentLoss()
                loss_model.add_module("content_loss_{}".format(i), content_loss)
                self.content_losses.append(content_loss)

            if name in self.style_layers:
                # add style loss:
                style_loss = self.StyleLoss()
                loss_model.add_module("style_loss_{}".format(i), style_loss)
                self.style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(loss_model) - 1, -1, -1):
            if isinstance(loss_model[i], self.ContentLoss) or isinstance(loss_model[i], self.StyleLoss):
                break

        self.loss_model = loss_model[:(i + 1)]
