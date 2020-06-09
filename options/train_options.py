from options.base_options import BaseOptions
import argparse

class TrainOptions(BaseOptions):
    def __init__(self):
        super().__init__()

        self.train_parser = argparse.ArgumentParser()

        # -- HYPER PARAMS -- #
        self.train_parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.train_parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.train_parser.add_argument('--epoch_count', type=int, default=1,
                                 help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')

        # -- LOGGING -- #
        self.train_parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing and saving training results on console')
        self.train_parser.add_argument('--save_model_freq', type=int, default=10000,
                                       help='frequency of saving models')
        self.train_parser.add_argument('--save_sample_freq', type=int, default=1000,
                                       help='frequency of saving a sample batch of results')
        self.train_parser.add_argument('--save_individual_sample_freq', type=int, default=1000,
                                       help='frequency of saving individual images')
        self.train_parser.add_argument('--n_individual_samples', type=int, default=5,
                                       help='number of individual images to save')
        self.train_parser.add_argument('--save_training_sample_freq', type=int, default=1000,
                                       help='frequency of saving individual training images')

        # -- MODEL OPTIONS -- #
        self.train_parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        self.train_parser.add_argument('--which_model_netG', type=str, default='resnet_9blocks', help='selects model to use for netG')
        self.train_parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.train_parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.train_parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.train_parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.train_parser.add_argument('--use_wgan', action='store_true',help='use WGAN, if false, use vanilla GAN')
        self.train_parser.add_argument('--critic_iterations', type=int, default=5, help='How many time D gets to iterate before G')
        self.train_parser.add_argument('--use_loss_weighting_check', action='store_true',help='Will begin altering loss weighting until it gets to 1')
        self.train_parser.add_argument('--loss_weighting_threshold', type=int, default=0.1, help='Threshold used for loss check')

        self.train_parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.train_parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.train_parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.train_parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.train_parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')

        # -- TRAINING OPTIONS -- #
        self.train_parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.train_parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.train_parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop',
                                 help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.train_parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.train_parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.train_parser.add_argument('--mean', type=float, default=0.5, help='Mean for transformed images')
        self.train_parser.add_argument('--sd', type=float, default=0.5, help='Standard deviation for transformed images')
        self.train_parser.add_argument('--greyscale', action='store_true', help='Greyscale images')

        # -- OPTIMIZER -- #
        self.train_parser.add_argument('--beta1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
        self.train_parser.add_argument('--beta2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')

        # -- LEARNING RATES --#
        self.train_parser.add_argument('--d_lr', type=float, default=0.0002, help='adam: learning rate for discriminator')
        self.train_parser.add_argument('--g_lr', type=float, default=0.0002, help='adam: learning rate for generator')
        self.train_parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        self.train_parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        self.train_parser.add_argument('--loss_margin', type=int, default=2,
                                       help='Margin for contrastive loss')

        # -- WEIGHT ADJUSTING -- #
        self.train_parser.add_argument('--lambda_identity', type=float, default=0.5,
                                 help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the i'
                                      'dentity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the '
                                      'weight of the reconstruction loss, please set lambda_identity = 0.1')
        self.train_parser.add_argument('--lambda_rec_fake_identity', type=float, default=0.1, help='Loss to make sure recreaction and fake are the same')
        self.train_parser.add_argument('--lambda_style_loss', type=float, default=1, help='Loss for style transfer')
        self.train_parser.add_argument('--lambda_content_loss', type=float, default=1, help='Loss for content in image')
        self.train_parser.add_argument('--lambda_cycle_loss', type=float, default=1, help='Loss for cycle loss')

        # -- PREPROCESSING OPTION USED --#
        self.train_parser.add_argument('--preproc_opt', default='Normal', type=str, help='For logging purposes indicate preprocessing option used.')

    def parse_options(self, model_parser):
        namespace, extra = model_parser.parse_known_args(self.extra, self.args)
        self.args, _ = self.train_parser.parse_known_args(extra, namespace=namespace)
        return self.args

