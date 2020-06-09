import numpy as np
import glob
import os

from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm, trange
from preprocessing.utils import histogram
from preprocessing.config import *

def train(clf_type, save_model_loc, pos_feat_path, neg_feat_path, descriptor, img_size):
    # Parse the command line arguments
    POS_SAMPLES = 1000000
    NEG_SAMPLES = 1000000


    # Classifiers supported

    if clf_type == 'LIN_SVM':
        print('Training a Linear SVM classifier:')
        X = []
        y = []

        print('Loading positive samples...')
        progress_bar = tqdm(total=len(glob.glob(os.path.join(pos_feat_path, '*.feat'))))
        i = 0
        for feat_path in glob.glob(os.path.join(pos_feat_path, '*.feat')):
            if descriptor == 'HOG':
                x , j = joblib.load(feat_path)
            else:
                from preprocessing.colordescriptors40.DescriptorIO import readDescriptors
                j, x = readDescriptors(feat_path) #points, descriptors
            X.append(x)
            y.append(1)
            i += 1
            progress_bar.update(1)

        print('Loading negative samples...')
        progress_bar = tqdm(total=len(glob.glob(os.path.join(neg_feat_path, '*.feat'))))
        i = 0
        for feat_path in glob.glob(os.path.join(neg_feat_path, '*.feat')):
            if descriptor == 'HOG':
                x, _ = joblib.load(feat_path)
            else:
                j, x = readDescriptors(feat_path)  # points, descriptors
            X.append(x)
            y.append(0)

            i += 1
            progress_bar.update(1)
        X_train = np.array(list(X))


        y_train = np.array(list(y))

        del X
        del y


        if descriptor == 'SIFT':
            X_train = X_train.reshape((X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
            X_des = X_train
            batch_size = 20
            kmeans = MiniBatchKMeans(n_clusters=int(VOCAB_SIZE), batch_size=batch_size)
            kmeans.fit(X_des)
            if not os.path.isdir(os.path.split(save_model_loc)[0]):
                os.makedirs(os.path.split(save_model_loc)[0])
            joblib.dump(kmeans, os.path.join(os.path.split(save_model_loc)[0], 'kmeans.clf'), compress=3)
            X_train = []
            for x in X_des:
                test = np.array([x])
                hist = np.array(histogram(kmeans, test, int(VOCAB_SIZE)))
                X_train.append(hist.reshape(hist.shape[1]))
            X_train = np.array(X_train)
            #labels = kmeans.labels_
            #codewords = kmeans.cluster_centers_

        print('Training a Linear SVM Classifier...')
        clf = LinearSVC(random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)

    elif clf_type == 'SGD':
        print('Training a SGDClassifier:')
        clf = SGDClassifier(random_state=RANDOM_STATE)

        samples = []
        print('Loading positive samples...')
        progress_bar = tqdm(total=POS_SAMPLES)
        i = 0
        for feat_path in glob.glob(os.path.join(pos_feat_path, '*.feat')):
            if i == POS_SAMPLES:
                break

            samples.append((feat_path, 1))

            i += 1
            progress_bar.update(1)

        print('Loading negative samples...')
        progress_bar = tqdm(total=NEG_SAMPLES)
        i = 0
        for feat_path in glob.glob(os.path.join(neg_feat_path, '*.feat')):
            if i == NEG_SAMPLES:
                break

            samples.append((feat_path, 0))

            i += 1
            progress_bar.update(1)

        random.shuffle(samples)

        print('Training classifier...')
        progress_bar = tqdm(total=POS_SAMPLES + NEG_SAMPLES)
        i = 0
        for i in range(len(samples)):
            feat_path, label = samples[i]

            if descriptor == 'HOG':
                x, j = joblib.load(feat_path)
            else:
                from preprocessing.colordescriptors40.DescriptorIO import readDescriptors
                j, x = readDescriptors(feat_path) #points, descriptors

            print('-------')
            clf.partial_fit([x], [label], classes=[0, 1])

    # If model directory doesn't exist, create one
    if not os.path.isdir(os.path.split(save_model_loc)[0]):
        os.makedirs(os.path.split(save_model_loc)[0])
    joblib.dump(clf, save_model_loc, compress=3)
    print('Classifier saved to {}'.format(save_model_loc))
