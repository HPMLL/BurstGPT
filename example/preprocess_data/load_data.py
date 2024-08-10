import json
import copy
from random import shuffle
import random

import pandas as pd

def load_data_from_path(file_path:str):
    with open(file_path, "r") as f:
        data = json.load(f)
        f.close()

    return data


# def load_data_from_path(file_path:str):
#     df = pd.read_csv(file_path)
#     return df