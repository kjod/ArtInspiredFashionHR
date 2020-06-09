"""
Set config variables
"""

import configparser as cp
import json
import random

import numpy as np

config = cp.RawConfigParser()
config.read('preprocessing/config.cfg')

WINDOW_SIZE = json.loads(config.get('HOG', 'window_size'))
WINDOW_STEP_SIZE = config.getint('HOG', 'window_step_size')
ORIENTATIONS = config.getint('HOG', 'orientations')
PIXELS_PER_CELL = json.loads(config.get('HOG', 'pixels_per_cell'))
CELLS_PER_BLOCK = json.loads(config.get('HOG', 'cells_per_block'))
VISUALISE = config.getboolean('HOG', 'visualise')

NORMALISE = config.get('HOG', 'normalise')
if NORMALISE == 'None':
    NORMALISE = None

THRESHOLD = config.getfloat('nms', 'threshold')

MODEL_PATH = config.get('paths', 'model_path')

PYRAMID_DOWNSCALE = config.getfloat('general', 'pyramid_downscale')
POS_SAMPLES = config.getint('general', 'pos_samples')
NEG_SAMPLES = config.getint('general', 'neg_samples')

RANDOM_STATE = 31
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

SIFT_LOC = config.get('SIFT', 'SIFT_extractor_loc')
VOCAB_SIZE = config.get('SIFT', 'vocab_size')
