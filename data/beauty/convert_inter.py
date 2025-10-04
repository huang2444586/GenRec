import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

dataset = "beauty"

data = {}

with open(f"./data/{dataset}/handle/inter_seq.txt", 'r') as f:
    for line in tqdm(f):
        line_data = line.rstrip().split(' ')
        user_id = line_data[0]
        line_data.pop(0)    # delete user_id
        data[user_id] = line_data

with open(f"./data/{dataset}/handled/inter.txt", 'w') as f:
    for user, item_list in tqdm(data.items()):
        for item in item_list:
            u = int(user)
            i = int(item)
            f.write('%s %s\n' % (u, i))

