import argparse
import os
import data

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        model_options = ['enhance_discoGAN', 'discoGAN', 'cycleGAN', 'pix2pixGAN', 'shapeGAN']
        model_datasets = ['fmnist', 'redbubble']

        # -- GENERAL OPTIONS -- #
        self.parser.add_argument('-m', '--model', type=str, default=model_options[0], help='the model to use') #choices=model_optionsz
        self.parser.add_argument('-d', '--dataset', type=str, default=model_datasets[0], choices=model_datasets, help='location of dataset')
        self.parser.add_argument('--batch_size', type=int, default=60, help='size of the batches')
        self.parser.add_argument('--dataset_label', type=str, default='Graphic T-Shirt Dress', help='Label to use in dataset')
        self.parser.add_argument('--img_size', type=int, default=128, help='size of image height')
        self.parser.add_argument('-n', '--n_examples', default='all', help='number of training examples')
        self.parser.add_argument('--n_cpu', type=int, default=2, help='number of cpu threads to use during batch generation')  # might remove
        self.parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--exp_name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        self.args, self.extra = self.parser.parse_known_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
