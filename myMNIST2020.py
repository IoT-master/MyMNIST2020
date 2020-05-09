import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from os import walk
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import image

torch.set_printoptions(linewidth=120)
print(torch.__version__)
print(torchvision.__version__)

data = image.imread('data/MNISTDataSet/train/0.png')


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


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
    'data/MNISTDataSet/train.csv', 'data/MNISTDataSet/train', transforms.Compose([
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


# print(get_sample(7))
print(get_sample(7)['image'].shape)

batch_loader_params = {
    "batch_size": 6,
    "shuffle": True,
    "num_workers": 4
}
dataloader = DataLoader(transformed_dataset, **batch_loader_params)

batch_samples = iter(dataloader)

samples = batch_samples.next()
# # print labels
# print(' '.join('%5s' % samples['label'][j] for j in range(4)))

# # show images
# imshow(torchvision.utils.make_grid(samples['image']))

# Conv2d params
# in_channels, => channels of your pixels, 3rd element in the size
# out_channels, => How many layers you want for feature extraction
# kernel_size, => L, H of pixel size
# stride = 1,
# padding = 0,
# dilation = 1,
# groups = 1,
# bias = True,
# padding_mode = builtins.str

# MaxPool2d
# kernel_size,
# stride = None,
# padding = 0,
# dilation = 1,
# return_indices = False,
# ceil_mode = False

# Pooling cut size by 2


def spatial_size(input_size: int, kernel_size: int, stride: int = 1, padding: int = 0):
    # https://cs231n.github.io/convolutional-networks/
    spatial_size = (input_size - kernel_size + 2 * padding)/stride + 1
    assert spatial_size % 1 == 0
    assert spatial_size > 0
    return int(spatial_size)


print(spatial_size(28, 5))
print(spatial_size(24, 5))
print(spatial_size(12, 5))
# Start with an image that is 28x28*4. The first convolution layer as a 5x5 kernel size with no padding
# Output image is smaller. The Output image has a width and height that is smaller by 4 pixel
# 7x7 image is reduced to a 3 x 3 image
# Applying a 5x5 kernel will remove 2 pixel layers from the top, left, right and bottom
# Input 28 x 28 becomes 24x24 in conv1
# it is fed to a 2x2 pooling layer, which cuts the image size by half, so you end up with 12x12
# next covolution layer reduces it to 8x8 with a 5x5 kernel
# next pooling layer reduces it to a 4x4 (with 50 output channels)... * 4 color channels


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 20, 5, 1)  # 28x28x4 with batch of 6, 18816
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 60)
        self.fc2 = nn.Linear(60, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 24x24x20 with at batch of 6
        x = F.max_pool2d(x, 2, 2)  # 12x12x20 w/ batch of 6
        x = F.relu(self.conv2(x))  # 8x8x56 w/ batch of 6
        x = F.max_pool2d(x, 2, 2)  # 4x4x50 w/ batch of 6
        x = x.view(-1, 4*4*50)     # 800 w/ batch of 6
        x = F.relu(self.fc1(x))    # 60 w/ batch of 6
        x = self.fc2(x)            # 10 w/ batch of 6
        # # There's no activation at the final layer because of the criterion of CEL
        return x


net = Net()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(data['image'])
        loss = criterion(outputs, data['label'].squeeze())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    print('[%d, %5d] loss: %.5f' %
          (epoch + 1, i + 1, running_loss / (epoch*i+1)))

print('Finished Training')

torch.save(net, 'torchsave')
