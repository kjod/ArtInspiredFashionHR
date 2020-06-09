import argparse

def two_domain_parser_options(parser=None):
    if not parser:
        parser = argparse.ArgumentParser()

    parser.add_argument('--A_label', type=str, default='Dress', help='weight for cycle loss (A -> B -> A)')
    parser.add_argument('--B_label', type=str, default='Painting', help='weight for cycle loss (B -> A -> B)')
    parser.add_argument('--lambda_A', type=float, default=1.0, help='weight for cycle loss (A -> B -> A)')  # cycle GAN and disco GAN make model specfic
    parser.add_argument('--lambda_B', type=float, default=1.0, help='weight for cycle loss (B -> A -> B)')
    return parser

def  add_lambda_L1(parser=None):
    if not parser:
        parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
    return parser 

def weighting_shapeGAN_hyperparam(parser=None):
    if not parser:
        parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_C', type=float, default=50, help='weight for L1 loss')
    parser.add_argument('--lambda_RAFN', type=float, default=100.0, help='weight for L1 loss')
    parser.add_argument('--lambda_dec', type=float, default=1, help='weight for L1 loss')
    return parser
