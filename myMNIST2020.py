import torch
import matplotlib.pyplot as plt
import numpy as np
# from PIL import Image
import pandas as pd
from PIL import Image


from matplotlib import image

data = image.imread('data/MNISTDataSet/train/0.png')
print(data.dtype, data.shape)
my_data = torch.FloatTensor(data)
# print(my_data)
# print(my_data[:, :, 1] == my_data[:, :, 2])
print(my_data.narrow(2, 0, 1)[15])
# print(my_data.narrow(2, 0, 3))

# plt.imshow(data)
# plt.show()

# from os import walk
# f = []
# mypath = 'data/MNISTDataSet/train'
# for (dirpath, dirnames, filenames) in walk(mypath):
#     f.extend(filenames)
# print(f)
