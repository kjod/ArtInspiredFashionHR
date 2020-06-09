import argparse
import os
import shutil
import pandas as pd
from preprocessing.train_classifier import train
from preprocessing.test_classifier import test
from preprocessing.extract_features import extract, extract_FashionNet_boundingBoxes
from preprocessing.create_dataset import create, apply_mask

def matlab_installed():
    for path in os.environ["PATH"].split(":"):
        if os.path.isfile(os.path.join(path, "matlab")):
            return True
    return False


def setup_dataset_folder(save_dataset_loc, copy_dataset_loc):
    if not os.path.exists(save_dataset_loc):
        os.makedirs(save_dataset_loc)

    if copy_dataset_loc:
        for f in os.listdir(copy_dataset_loc):
            shutil.copy(os.path.join(copy_dataset_loc, f), save_dataset_loc)


def check_FashionNet_options(args):
    if args.extract:
        if not matlab_installed():
            print('Add matlab to PATH.')
            return
        extract_FashionNet_boundingBoxes(args.df_box_file_loc, args.image_dir, args.image_reg_exp)

    if args.create:
        setup_dataset_folder(args.save_dataset_loc, args.copy_dataset_loc)
        create(args.save_dataset_loc, None, args.img_size, args.df_box_file_loc)
        
        #Mask after dataset created
        if args.mask:
            apply_mask(args.save_dataset_loc)


def main():
    parser = argparse.ArgumentParser()
    descriptor_options = ['HOG', 'SIFT']
    clasifier_options = ['LIN_SVM', 'SGD', 'MLP', 'FashionNet']

    parser.add_argument('--img_size', type=int, default=128, help='The img size used')

    # -- EXTRACT -- #
    parser.add_argument('--extract', action='store_true', help='Flag to extract features for training. If FashionNet model is used is enabled then \
        bounding box coordinates will be extracted.')
    parser.add_argument('-d', '--descriptor', type=str, default=descriptor_options[0], choices=descriptor_options, help='The descrtiptor to use.')
    parser.add_argument('-pi', '--pos_image_dir_path', default='data/pos_images/', help='Path to pos images')
    parser.add_argument('-ni', '--neg_image_dir_path', default='data/neg_images/', help='Path to neg images')

    # -- TRAIN -- #
    parser.add_argument('--train', action='store_true', help='Flag to begin training')
    parser.add_argument('--save_model_loc', type=str, default='preprocessing/saved_models/test.clf', help='The location of the model to be saved')
    parser.add_argument('-c', '--classifier', type=str, default=clasifier_options[0], choices=clasifier_options, help='Classifier to be used. Choosing \
                        FashionNet will skip train and test steps as model is pretrained.')

    # -- EXTRACT / TRAIN -- #
    parser.add_argument('-p', '--pos_features_path', default='preprocessing/pos_features/', help='Path to the positive features directory')
    parser.add_argument('-n', '--neg_features_path', default='preprocessing/neg_features/', help='Path to the negative features directory')

    # -- TEST -- #
    parser.add_argument('--test', action='store_true', help='Flag to begin test')
    parser.add_argument('--test_set_loc', default='dataset/test-set/', type=str, help='Location of test set')

    # -- DATA-SET CREATION -- #
    parser.add_argument('--create', action='store_true', help='Flag to start data-set creation')
    parser.add_argument('--mask', action='store_true', help='Apply masking')
    parser.add_argument('--copy_dataset_loc', type=str, help='The location of the data-set to copy')
    parser.add_argument('--save_dataset_loc', type=str, default='data/new_dataset', help='The location of the saved data-set')

    # -- TEST / DATA-SET CREATION -- #
    parser.add_argument('--load_model_loc', type=str, default='preprocessing/saved_models/test.clf', help='The location of the model to be loaded')

    # -- FASHIONNET -- #
    parser.add_argument('--df_box_file_loc', type=str, default='../FN_bb.csv', help='The location of where the bounding box coordinates are stored')
    parser.add_argument('--image_dir', type=str, default='../data/redbubble/images/', help='Image directory for extracting coorindates')
    parser.add_argument('--image_reg_exp', type=str, default='*_2.*', help='Filters images to use model on')

    args = parser.parse_args()

    if args.classifier == 'FashionNet':
        check_FashionNet_options(args)
    else:
        if args.extract:
            extract(args.descriptor, args.pos_features_path, args.neg_features_path, args.pos_image_dir_path, args.neg_image_dir_path, args.img_size)

        if args.train:
            train(args.classifier, args.save_model_loc, args.pos_features_path, args.neg_features_path, args.descriptor, args.img_size)

        if args.test:
            test(args.test_set_loc, args.load_model_loc, args.img_size, args.descriptor)

        if args.create:
            setup_dataset_folder(args.save_dataset_loc, args.copy_dataset_loc)

            if args.mask:
                apply_mask(args.save_dataset_loc)
            create(args.save_dataset_loc, args.load_model_loc, args.img_size, descriptor=args.descriptor)
                ##Resize according to model        

if __name__ == '__main__':
    main()
