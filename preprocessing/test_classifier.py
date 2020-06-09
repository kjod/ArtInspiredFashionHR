import os
import cv2
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.externals import joblib
from preprocessing.config import *
from preprocessing.utils import sliding_window, pyramid, non_max_suppression, rgb2gray, histogram


class Detector:
    def __init__(self, downscale=PYRAMID_DOWNSCALE, window_size=(WINDOW_SIZE, WINDOW_SIZE), window_step_size=WINDOW_STEP_SIZE, threshold=THRESHOLD, img_size=128,
                 model_path='preprocessing/saved_models/test.clf', img_ext='.jpg', ext='.feat'):
        self.clf = joblib.load(model_path)
        self.downscale = downscale
        self.window_size = window_size
        self.window_step_size = window_step_size
        self.threshold = threshold
        self.img_size = img_size
        self.img_ext = img_ext
        self.ext = ext
        self.kmeans_loc = os.path.join(os.path.split(model_path)[0], 'kmeans.clf')

    def detect(self, image, descriptor):

        if len(np.array(image).shape) < 3:
            image = scipy.misc.imresize(image, (self.img_size, self.img_size))
            clone = image.copy()
        else:
            image = scipy.misc.imresize(image, (self.img_size, self.img_size, 3))
            clone = image.copy()
            image = rgb2gray(image)

        # list to store the detections
        detections = []
        # current scale of the image
        downscale_power = 0

        # downscale the image and iterate
        for im_scaled in pyramid(image, downscale=self.downscale, min_size=self.window_size):
            # if the width or height of the scaled image is less than
            # the width or height of the window, then end the iterations
            if im_scaled.shape[0] < self.window_size[1] or im_scaled.shape[1] < self.window_size[0]:
                break
            for (x, y, im_window) in sliding_window(im_scaled, self.window_step_size,
                                                    self.window_size):
                if im_window.shape[0] != self.window_size[1] or im_window.shape[1] != self.window_size[0]:
                    continue
                if descriptor == 'HOG':
                    # calculate the HOG features
                    feature_vector, _ = hog(im_window, orientations=ORIENTATIONS, pixels_per_cell=PIXELS_PER_CELL,
                                            cells_per_block=CELLS_PER_BLOCK, visualise=VISUALISE, transform_sqrt=NORMALISE)
                    #feature_vector = hog(im_window)
                    X = np.array([feature_vector])
                else:
                    from preprocessing.colordescriptors40.DescriptorIO import readDescriptors
                    kmeans = joblib.load(self.kmeans_loc)
                    new_image = 'temp_image.jpg'
                    Image.fromarray(np.uint8(im_window)).save(new_image)
                    cmd = "%s %s -detector densesampling --descriptor sift --outputFormat binary --output %s --ds_spacing 10 > output.log" % (
                        str(SIFT_LOC), new_image, new_image.replace(self.img_ext, self.ext))
                    os.system(cmd)
                    os.remove(new_image)
                    j, feature_vector = readDescriptors(new_image.replace(self.img_ext, self.ext))  # points, descriptors
                    os.remove(new_image.replace(self.img_ext, self.ext))
                    X = np.array([feature_vector])
                    X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
                    hist = np.array(histogram(kmeans, X, int(VOCAB_SIZE)))
                    X = hist

                prediction = self.clf.predict(X)
                if prediction == 1:
                    print('Prediction!!')
                    x1 = int(x * (self.downscale ** downscale_power))
                    y1 = int(y * (self.downscale ** downscale_power))
                    detections.append((x1, y1,
                                       x1 + int(self.window_size[0] * (
                                               self.downscale ** downscale_power)),
                                       y1 + int(self.window_size[1] * (
                                               self.downscale ** downscale_power))))

            # Move the the next scale
            downscale_power += 1

        # Display the results before performing NMS
        clone_before_nms = clone.copy()
        for (x1, y1, x2, y2) in detections:
            # Draw the detections
            cv2.rectangle(clone_before_nms, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

        # Perform Non Maxima Suppression
        detections = non_max_suppression(np.array(detections), self.threshold)

        clone_after_nms = clone
        # Display the results after performing NMS
        coordinates = None
        for (x1, y1, x2, y2) in detections:
            # Draw the detections
            coordinates = (x1, x2, y1, y2)
            cv2.rectangle(clone_after_nms, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

        return clone_before_nms, clone_after_nms, coordinates


def test(image_dir_path, load_model_loc, img_size, descriptor):
    # Parse the command line arguments

    detector = Detector(downscale=PYRAMID_DOWNSCALE, window_size=WINDOW_SIZE,
                        window_step_size=WINDOW_STEP_SIZE, threshold=THRESHOLD, img_size=img_size, model_path=load_model_loc)

    for image_name in os.listdir(image_dir_path):
        if image_name == '.DS_Store':
            continue

        # Read the image
        image = scipy.misc.imread(os.path.join(image_dir_path, image_name))

        # detect faces and return 2 images - before NMS and after
        image_before_nms, image_after_nms, coordinates = detector.detect(image, descriptor)

        plt.imshow(image_before_nms)
        plt.xticks([]), plt.yticks([])
        plt.show()

        plt.imshow(image_after_nms)
        plt.xticks([]), plt.yticks([])
        plt.show()