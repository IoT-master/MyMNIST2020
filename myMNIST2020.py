import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from os import walk
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import image

data = image.imread('data/MNISTDataSet/train/0.png')


def normalize(np_array):
    avg = np_array.sum()/np_array.size
    std = (np_array - avg).sum()/np_array.size
    return avg, std

# df = pd.DataFrame()
# my_data = torch.FloatTensor(data)
# print(my_data)
# print(my_data[:, :, 1] == my_data[:, :, 2])
# print(my_data.narrow(2, 0, 1))


# plt.imshow(data)
# plt.show()

# f = []
# mypath = 'data/MNISTDataSet/train'
# for (dirpath, dirnames, filenames) in walk(mypath):
#     f.extend(filenames)
# print(dirpath, f)

df = pd.read_csv('data/MNISTDataSet/test.csv')
print(df.sample(10))


class MNISTDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        my_image = image.imread(img_name)
        label = self.landmarks_frame.iloc[idx, 1:].to_numpy(dtype=np.int)
        sample = {'image': my_image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


my_dataset = MNISTDataset(
    'data/MNISTDataSet/test.csv', 'data/MNISTDataSet/test')


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}


transformed_dataset = MNISTDataset(
    'data/MNISTDataSet/test.csv', 'data/MNISTDataSet/test', transforms.Compose([
        ToTensor()
    ]))


print(my_dataset.__getitem__(3)['label'])

dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)
