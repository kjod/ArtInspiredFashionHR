import argparse as ap
import os
import scipy.misc
from skimage.feature import hog
from sklearn.externals import joblib
from tqdm import tqdm
from PIL import Image
import numpy as np
from preprocessing.config import *
from preprocessing.utils import rgb2gray

def extract(des_type, pos_features_path, neg_features_path, pos_image_dir_path, neg_image_dir_path, img_size):

    if des_type == 'HOG':
        f = extract_HOG_features
    else:
       f = extract_SIFT_features

    # If feature directories don't exist, create them
    if not os.path.exists(pos_features_path):
        os.makedirs(pos_features_path)

    # If feature directories don't exist, create them
    if not os.path.exists(neg_features_path):
        os.makedirs(neg_features_path)

    print('Calculating descriptors for the training samples and saving them')

    print('Positive samples extracting ...')
    f(image_dir_path=pos_image_dir_path, feature_dir_path=pos_features_path, n_samples=POS_SAMPLES, img_size=img_size)

    print('Negative samples extracting ...')
    f(image_dir_path=neg_image_dir_path, feature_dir_path=neg_features_path, n_samples=NEG_SAMPLES, img_size=img_size)

    print('Completed calculating features from training images')


def extract_HOG_features(image_dir_path, feature_dir_path, n_samples, ext='.feat', img_size=128):
    progress_bar = tqdm(total=len(os.listdir(image_dir_path)))
    i = 0
    for image_path in os.listdir(image_dir_path):
        
        image = scipy.misc.imread(os.path.join(image_dir_path, image_path))
        if len(np.array(image).shape) < 3:
            image = scipy.misc.imresize(image,(img_size, img_size))
        else:
            image = scipy.misc.imresize(image,(img_size, img_size,3))
            image = rgb2gray(image)
        features = hog(image, orientations=ORIENTATIONS, pixels_per_cell=PIXELS_PER_CELL,
                       cells_per_block=CELLS_PER_BLOCK, visualise=VISUALISE, transform_sqrt=NORMALISE)


        features_file_name = image_path.split('.')[0] + ext
        features_dir_path = feature_dir_path
        features_file_path = os.path.join(features_dir_path, features_file_name)
        joblib.dump(features, features_file_path, compress=3)

        i += 1
        progress_bar.update(1)


def extract_SIFT_features(image_dir_path, feature_dir_path, n_samples, ext='.feat', img_ext='.jpg', img_size=128):

    progress_bar = tqdm(total=n_samples)
    i = 0
    for image_path in os.listdir(image_dir_path):
        if i == n_samples:
            break
        img = Image.open(os.path.join(image_dir_path, image_path))
        img = img.resize((img_size, img_size)) ##Faster
        new_image = os.path.join(feature_dir_path, image_path)
        img.save(new_image)
        cmd = "%s %s -detector densesampling --descriptor sift --outputFormat binary --output %s --ds_spacing 10 > output.log" % (str(SIFT_LOC), new_image, new_image.replace(img_ext, ext))
        os.system(cmd)
        os.remove(new_image)
        progress_bar.update(1)

def extract_FashionNet_boundingBoxes(df_box_file_loc, image_dir, image_reg_exp):
    print('Recommended to use absolute paths for matlab part. Paths for extraction in FashionNet are relative on preprocessing folder.')
    print('Image_dir:  %s' % image_dir)
    print('Reg exp:  %s%s' % (image_dir, image_reg_exp))
    dir_cmd = "cd preprocessing"
    matlab_cmd = "matlab -nodesktop -nojvm -r 'try extract(\"%s\", \"%s\", \"%s\"); catch; end; quit';" % (df_box_file_loc, image_dir, image_reg_exp)
    os.system('%s && %s' % (dir_cmd, matlab_cmd))