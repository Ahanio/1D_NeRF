import numpy as np
import torch


def to_numpy(x):
    return x.detach().numpy()


def place_objects(min_x, max_x):
    objects = []
    for i in range(1):
        # x = np.random.uniform(min_x, max_x)
        # y = np.random.uniform(x, max_x)
        # objects.append((x, y))
        objects.append((3, 5))
    return objects
