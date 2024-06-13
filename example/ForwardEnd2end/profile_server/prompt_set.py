import os
import time
import numpy as np
import sys
from random import randint
import random

sys.path.append("../")
from preprocess_data import load_data_from_path

class PromptSet(object):
    def __init__(self, dataset_path,
                 config={'dummy_data':False}):
        self.config = config
        self.dataset_path = dataset_path
        random.seed(self.config['seed'])
        self.data = load_data_from_path(dataset_path)