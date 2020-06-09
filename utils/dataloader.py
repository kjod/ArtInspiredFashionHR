import torch
import json
import random
import PIL.ImageOps
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class RedbubbleImageDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None, should_invert=False, train=True, n='all', 
                 clothing_types = ['A-Line Dress'], box_loc='data/redbubble/bounding_box_coordinates.csv',
                 use_bb=True, dataset_loc='data/redbubble/data_dresses.json'):
        """
            Loads training / test image set into an interator.

                    ImageFolderDataset: image folder directory

                    transform:          composed pytorch transform to be applied
                                        on every image

                    should_invert:      use inverted image

                    train:              retrieve training set, test set otherwise
        """
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert
        self.n = n
        self.dresses_dict = {}
        mask_df = pd.read_csv(box_loc)

        #with open('data/redbubble/data_dresses.json', 'r') as f:
        json_file = pd.read_json(dataset_loc)
        for i, line in json_file.iterrows():
            #line = json.loads(line)
            
            if line['type'] in clothing_types:
                if use_bb == True:
                    mask_line = mask_df[mask_df['file_name'] == line['pictures'][2]]
                    line['bb'] = [mask_line['x1'], mask_line['x2'], mask_line['y1'], mask_line['y2']]
                self.dresses_dict[line['id']] = line
        self.keys = list(self.dresses_dict.keys())

        if (n != 'all'):
            if(len(self.keys) < int(n)):
                n = len(self.keys)
            self.keys = random.sample(self.keys, int(n))
            new_dresses_dict = {the_chosen_key: self.dresses_dict[the_chosen_key] for the_chosen_key in self.keys}
            self.dresses_dict = new_dresses_dict

        # split for cross-validation
        if train:
            print('---------------- Data Summary ----------------')
            print('Labels used in dataset ', clothing_types)
            self.keys = self.keys[:round(len(self.keys) * 0.8)]
            print("Size of training set %d" % len(self.keys))
        else:
            self.keys = self.keys[round(len(self.keys) * 0.8):]
            print("Size of validation set %d" % len(self.keys))
                
        self.ImageFolderIndexer = iter(self.keys)

    def __getitem__(self, index):
        """
            Retrieves and preprocesses the next image pair
        """

        img_id = next(self.ImageFolderIndexer)
        img_painting = "data/redbubble/images/" + self.dresses_dict[img_id]['pictures'][1]
        img_model = "data/redbubble/images/" + self.dresses_dict[img_id]['pictures'][2]
        bb = self.dresses_dict[img_id]['bb']

        img0 = Image.open(img_painting)
        img1 = Image.open(img_model)
        
        bb_box_black = Image.new("RGB", img1.size, (0, 0, 0)) # we fill with blue
        bb_size = (int(abs(bb[1] - bb[0])), int(abs(bb[3] - bb[2])))
        bb_box_white = Image.new("RGB", (bb_size), (255, 255, 255))
        offset = (int(bb[0]) , int(bb[2]))
        bb_box_black.paste(bb_box_white, offset)
        
        img0 = img0.convert('RGB')
        img1 = img1.convert('RGB')
        bb_box_black = bb_box_black.convert('RGB')

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            bb_box_black = self.transform(bb_box_black)

        #create bounding box array

        return {'Painting': img0, 'Dress': img1, 'Bb': bb_box_black}

    def __len__(self):
        return len(self.keys)


class UnNormalize(object):
            def __init__(self, mean, std):
                self.mean = mean
                self.std = std
        
            def __call__(self, tensor):
                """
                Args:
                    tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
                Returns:
                    Tensor: Normalized image.
                """
                
                for t, m, s in zip(tensor, self.mean, self.std):
                    t.mul_(s).add_(m)
                    # The normalize code -> t.sub_(m).div_(s)
                return tensor