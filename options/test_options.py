from options.base_options import BaseOptions
import argparse
from analysis.inception import InceptionV3

class TestOptions(BaseOptions):
    def __init__(self):
        super().__init__()

        self.test_parser = argparse.ArgumentParser()

        # --Location of samples-- #
        self.test_parser.add_argument('--location_model_dir', type=str, default='models/', help='Where are the models stored')
        self.test_parser.add_argument('--run', type=str, default='latest', help='Will select latest run')

        # -- Quantitative Tests -- #
        self.test_parser.add_argument('--fid_test', action='store_true', help='Perform inception evaluation')
        self.test_parser.add_argument('--dims', type=int, default=2048,
                            choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                            help=('Dimensionality of Inception features to use. '
                                  'By default, uses pool3 features'))
        self.test_parser.add_argument('--model_iter', type=int, default='1000', help='Number of model iteration'),
        self.test_parser.add_argument('--real_set_label', type=str, default='_2.', help='*_1.jpg for painitng in redbubble, *_2.jpg for dress in redbubble')
        self.test_parser.add_argument('--fake_set_label', type=str, default='Dress', help='fake_B')
        
        # -- Qualitative check -- #/
        self.test_parser.add_argument('--display_batch', action='store_true', help='Displays batch listed by run and model')
        self.test_parser.add_argument('--create_model_training_image', action='store_true', help='Perform inception evaluation')
        self.test_parser.add_argument('--create_models_overview', action='store_true', help='Create figure of models comparsion')
        self.test_parser.add_argument('--create_samples_overview', action='store_true', help='Create samples of images from test set')
        self.test_parser.add_argument('--save_fig', action='store_true', help='to save figure')
        self.test_parser.add_argument('--saved_name', default='model_viz', help='name of figure')
        self.test_parser.add_argument('--type', type=str, default='InputTarget',
                                      help='InputTarget or DomADomB')
        self.test_parser.add_argument('--comparsion_models', nargs='+', help='List of models to compare against')
        self.test_parser.add_argument('--list_comparison_runs', nargs='+', help='List of runs from models for comparision. Should correspond with list comparision runs')
        self.test_parser.add_argument('--n_cols', type=int, default=8, help='Number of columns in visualisation')
        self.test_parser.add_argument('--n_rows', type=int, default=8, help='Number of rows in visualisation')
        self.test_parser.add_argument('--n_samples', type=int, default=8, help='Number of samples to use')
        self.test_parser.add_argument('--figure_heading', type=str, default='Heading', help='Heading for created figure')
        self.test_parser.add_argument('--check_stats', action='store_true', help='Check stats logged from model')
        self.test_parser.add_argument('--example_no', type=int, default=0, help='Dress example to use')

    def parse_options(self):
        self.args, _ = self.test_parser.parse_known_args(self.extra, namespace=self.args)
        return self.args

