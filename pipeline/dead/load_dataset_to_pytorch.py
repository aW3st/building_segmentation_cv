import sys, os

import numpy as np

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F

from PIL import Image

def mytransform(image, mask):
    image = transforms.functional.pad(image, padding=20, padding_mode='reflect')
    crop_loc = np.random.randint(0, 728, 2)
    image = transforms.functional.crop(image, *crop_loc, 296, 296)
    mask = transforms.functional.crop(mask, *crop_loc, 256, 256)
    image = transforms.functional.to_tensor(image)
    mask = transforms.functional.to_tensor(mask)
    return image, mask

class MyDataset(Dataset):
    def __init__(self, path, transforms=None):
        self.path = path
        self.transforms = transforms
        self.images = list(sorted(os.listdir(os.path.join(path, 'images'))))
        self.masks = list(sorted(os.listdir(os.path.join(path, 'masks'))))

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.path, 'images', self.images[index]))
        mask = Image.open(os.path.join(self.path, 'masks', self.masks[index]))
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
            return (image, mask)

    def __len__(self):
      return len(self.images)


path = './data/train'

itsmydata = MyDataset(path, transforms=mytransform)

myDataloader = DataLoader(itsmydata, shuffle=True, batch_size=5)

