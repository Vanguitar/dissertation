import numpy as np
import os
import cv2
import random
from PIL import Image

import imageio
'''加载数据
'''


def load_CWRU1D(data_path = './4运行/datasets'):

    import h5py
    import numpy as np

    with h5py.File(data_path + '.h5','r') as hf:
        x = hf.get('data')[:]
        y = hf.get('labels')[:]

    a = set(y)
    u = 0

    for id in a:
        y[y==id] = u
        u = u + 1

    print(x.shape)
    return x, y

def load_wind1D(data_path='./4运行/datasets'):
    import h5py
    import numpy as np

    with h5py.File(data_path + '.h5', 'r') as hf:
        x = hf.get('data')[:]
        y = hf.get('labels')[:]

    a = set(y)
    u = 0

    for id in a:
        y[y == id] = u
        u = u + 1

    print(x.shape)
    return x, y




