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


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transforms.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'label': label}


transformed_dataset = MNISTDataset(
    'data/MNISTDataSet/test.csv', 'data/MNISTDataSet/test', transforms.Compose([
        ToTensor()
    ]))


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'label': label}


def get_sample(index):
    return my_dataset.__getitem__(index)


print(get_sample(7))

batch_loader_params = {
    "batch_size": 4,
    "shuffle": True,
    "num_workers": 4
}
dataloader = DataLoader(transformed_dataset, **batch_loader_params)


def get_batch_sample():
    return iter(dataloader)


print(get_batch_sample().next())

epochs = 1
