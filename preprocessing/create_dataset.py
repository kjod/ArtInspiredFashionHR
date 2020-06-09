import os
import numpy as np
import pandas as pd
import scipy.misc
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from PIL import ImageFile
from preprocessing.test_classifier import Detector
from preprocessing.config import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

def apply_mask(dataset_loc, img_size=128, mask_loc='data/redbubble/masks/', json_file='data/redbubble/data_dresses.json'):
    mask_files = os.listdir(mask_loc)
    data_df = pd.read_json(json_file)

    print('Applying masking')
    progress_bar = tqdm(total=len(data_df['pictures']))

    for i in range(len(data_df['pictures'])):
        mask_img = Image.open(os.path.join(mask_loc, data_df['type'][i] + '_2.png'))
        #mask_img = mask_img.resize((img_size, img_size))
        mask_img_arr = ~np.array(mask_img)
        nor = (lambda x: int((x - 0) // (255 - 0)))
        mask_img_arr[mask_img_arr > 250] = 255
        mask_img_arr[mask_img_arr < 40] = 0
        mask_img_arr = np.vectorize(nor)(mask_img_arr)
        new_mask = mask_img_arr.copy()
        new_mask[new_mask == 0] = 2
        new_mask[new_mask == 1] = 0
        new_mask[new_mask == 2] = 255

        image_path = os.path.join(dataset_loc, data_df['pictures'].iloc[i][2])

        try:
            img = Image.open(image_path)
        except Exception as e:
            print('%s not found!' % str(image_path))
            continue
        
        #img = img.resize((img_size, img_size))
        img = np.array(img)

        img = (img * mask_img_arr)
        img = (img + new_mask).astype('uint8')
        img = Image.fromarray(img)
        #img.show()
        img.save(image_path)
        #cropped = img[290:970, 260:710]
        progress_bar.update(1)

def get_coordinates_detector(image_dir_path, image_name, detector, descriptor='HOG'):
    image = scipy.misc.imread(os.path.join(image_dir_path, image_name))
    image_before_nms, image_after_nms, coordinates = detector.detect(image, descriptor)
    return coordinates, image_after_nms


def get_coordinates_fashionNet(image_dir_path, image_name, fashion_df, descriptor=None):
    coordinate_entry = fashion_df[fashion_df['file_name'] == image_name]
    if len(coordinate_entry.index) > 0:
        coordinates = [int(coordinate_entry['x1']), int(coordinate_entry['x2']), int(coordinate_entry['y1']), int(coordinate_entry['y2'])]
    else:
        print('Missing %s '% image_name)
        coordinates = None
    #draw cooordinates
    return coordinates, []


def create(image_dir_path, load_model_loc, img_size, fashionNet_loc=None, descriptor='HOG'):
    print('Resizing images')
    # Parse the command line arguments
    if fashionNet_loc:
        detector = pd.read_csv(fashionNet_loc)
        coord_f = get_coordinates_fashionNet
    else:
        detector = Detector(downscale=PYRAMID_DOWNSCALE, window_size=WINDOW_SIZE,
                            window_step_size=WINDOW_STEP_SIZE, threshold=THRESHOLD, img_size=img_size,
                            model_path=load_model_loc)
        coord_f = get_coordinates_detector

    progress_bar = tqdm(total=len(os.listdir(image_dir_path)))
    count = 0
    for image_name in os.listdir(image_dir_path):
        if image_name == '.DS_Store':
            progress_bar.update(1)
            continue

        # Read the image
        if '_2.jpg' in image_name or '_2.png' in image_name:
            if '.jpg' in image_name:
                extension = '.jpg'
                other = '.png'
            else:
                extension = '.png'
                other = '.jpg'


            coordinates, image_with_box = coord_f(image_dir_path, image_name, detector, descriptor)    

            # Alter painting based on bounding box
            if coordinates != None:
                count+=1
                temp_name = image_name
                if '_big_2' in image_name:
                    temp_name = image_name.replace('_big_2', '_2')
                
                painting_name = temp_name.split('_')
                painting_name = painting_name[0:len(painting_name)-1]
                painting_name = '_'.join(painting_name)
                path = os.path.join(image_dir_path, painting_name + '_1' + extension)
                
                try:
                    painting = Image.open(path)
                except Exception as e:
                    path = path.replace(extension, other)
                    painting = Image.open(path)

                dress = Image.open(os.path.join(image_dir_path, image_name))
                #dress.show()
                painting = painting.resize((128, 128))
                #painting = painting.resize((dress.height, dress.width)) #TODO check resize for SIFT and HOG

                array = np.linspace(1, 1, painting.width * painting.height * 3)
                mat = np.reshape(array, (painting.height, painting.width, 3))
                img = Image.fromarray(np.uint8(mat * 255))

                image_copy = img.copy()
                painting = painting.resize((abs(coordinates[1] - coordinates[0]) , abs(coordinates[3] - coordinates[2])))
                #painting.show()
                position = (int(coordinates[0]), int(coordinates[2]))
                image_copy.paste(painting, position)
                
                #image_copy.show()
                image_copy.save(path)
        
        progress_bar.update(1)
        painting = Image.open(os.path.join(image_dir_path, image_name))
        painting = painting.resize((img_size, img_size))
        #painting.show()
        painting.save(os.path.join(image_dir_path, image_name))
    print('Final count: ', count)