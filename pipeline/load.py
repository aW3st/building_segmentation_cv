

import glob
import os
import re
import numpy as np
import pdb
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

colorjitter = transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25)
# ---- Image Utitilies ----

CITY_REGION_CTS = {
    'znz': {
        '_total': 19717,
        '076995': 1701,
        '75cdfa': 2304,
        '425403': 1444,
        '33cae6': 1106,
        '06f252': 2387,
        'e52478': 601,
        'c7415c': 1346,
        'bc32f1': 1521,
        '3f8360': 1086,
        'aee7fd': 1521,
        '9b8638': 1482,
        'bd5c14': 1369,
        '3b20d4': 1849},
    'dar': {
        '_total': 12356,
        '353093': 1364,
        'f883a0': 4094,
        '0a4c40': 1604,
        '42f235': 2257,
        'a017f9': 1642,
        'b15fce': 1395},
    'acc': {
        '_total': 10603,
        '665946': 6982,
        'a42435': 1318,
        'ca041a': 1566,
        'd41d81': 737},
    'ptn': {'_total': 44, 'abe1a3': 31, 'f49f31': 13},
    'kam': {'_total': 868, '4e7c7f': 868},
    'mon': {
        '_total': 803,
        '401175': 192,
        '493701': 280,
        'f15272': 205,
        '207cc7': 126},
    'nia': {'_total': 65, '825a50': 65}
    }

def is_valid_loc(basename, split):
    '''
    Filter basenames according to allowed regions.
    '''
    basename = os.path.basename(basename)
    region, x_pos, y_pos, ext = basename.split('_')

    train_regions = {
        
        # ZNZ - Zanzibar 
        '076995': {'skip_region': []},
        '75cdfa': {'skip_region': []},
        '425403': {'skip_region': []},
        '33cae6': {'skip_region': []},
        '06f252': {'skip_region': []},
        'e52478': {'skip_region': []},
        'c7415c': {'skip_region': []},
        'bc32f1': {'skip_region': []},
        '3f8360': {'skip_region': []},
        'aee7fd': {'skip_region': []},
        '9b8638': {'skip_region': []},
        'bd5c14': {'skip_region': []},
        '3b20d4': {'skip_region': []},

        # ACC - Accra
        '665946': {'skip_region': []},
        'a42435': {'skip_region': []},
        'ca041a': {'skip_region': []},
        'd41d81': {'skip_region': [
            (-1024, np.inf , -1024, np.inf) # TEMPORARY EXCLUSION OF ENTIRE SCENE
            ]},

        # PTN
        'abe1a3': {'skip_region': []},
        'f49f31': {'skip_region': []},

        # KAM
        '4e7c7f': {'skip_region': []},

        # MON
        '401175': {'skip_region': []},
        '493701': {'skip_region': []},
        'f15272': {'skip_region': []},
        '207cc7': {'skip_region': []},

        # NIA
        '825a50': {'skip_region': []},

    }


    
    val_regions = {

        # DAR
        '353093': {'skip_region': []},
        'f883a0': {'skip_region': []},
        '0a4c40': {'skip_region': []},
        '42f235': {'skip_region': []},
        'a017f9': {'skip_region': []},
        'b15fce': {'skip_region': []}
        
    }

    def is_excluded(x_0, x_1, y_0, y_1):
        return (x_0 - 512 <= int(x_pos) <= x_1  - 512) and (y_0  - 512 <= int(y_pos) <= y_1 - 512)

    if split == 'train' or split == 'clean': 
        for reg, d in train_regions.items():
            # Check all matching regions.
            if reg == region:
                # Skip regions that are badly labeled.
                for x_0, x_1, y_0, y_1 in d['skip_region']:
                    # print(x_0, x_1, y_0, y_1, "___", x_pos, y_pos)
                    if is_excluded(x_0, x_1, y_0, y_1):
                        return False
                return True
    
    if split == 'test' or split == 'clean': 
        for reg, d in val_regions.items():
            # Check all matching regions.
            if reg == region:
                # Skip regions that are badly labeled.
                for x_0, x_1, y_0, y_1 in d['skip_region']:
                    # print(x_0, x_1, y_0, y_1, "___", x_pos, y_pos)
                    if is_excluded(x_0, x_1, y_0, y_1):
                        return False
                return True
                
    return False

    


def train_transform(image, mask):
    '''
    Custom Pytorch randomized preprocessing of training image and mask.
    '''
    image = transforms.functional.pad(image, padding=0, padding_mode='reflect')
    crop_size = 460
    crop_loc = np.random.randint(0, 1024 - crop_size, 2)
    image = transforms.functional.crop(image, *crop_loc, crop_size, crop_size)
    mask = transforms.functional.crop(mask, *crop_loc, crop_size, crop_size)
    image = colorjitter(image)
    # image = transforms.functional.pad(image, padding=3, fill=0, padding_mode='constant')
    image = transforms.functional.to_tensor(image)
    mask = transforms.functional.to_tensor(mask)
    return image, mask

def val_transform(image, mask):
    image = transforms.functional.center_crop(image, 800)
    mask = transforms.functional.center_crop(mask, 800)
    image = transforms.functional.to_tensor(image)
    mask = transforms.functional.to_tensor(mask)
    return image, mask

# ---- Dataset Class ----
class DatasetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y, z = self.subset[index]
        if self.transform:
            x,y = self.transform(x,y)
        return x, y, z
        
    def __len__(self):
        return len(self.subset)

class MyDataset(Dataset):
    '''
    Custom PyTorch Dataset class.
    '''
    def __init__(self, in_dir=None, custom_transforms=None, load_test=False, split=None, batch_trim=False, compressed=False, region=None):

        self.transforms = custom_transforms
        self.load_test = load_test
        self.compressed = compressed

        if in_dir is None:
            in_dir = 'training_data'
            print(f'Pulling from directory {in_dir}')
        
        if self.load_test:
            print('Loading Test')
            if compressed:
                self.path = 'submission_data/compressed'
            else:
                self.path = 'submission_data/test'
            self.images = glob.glob(os.path.join(self.path, '*'))
            self.images = [os.path.basename(g) for g in self.images]

        else:
            self.path = in_dir
            print(self.path)
            pattern = os.path.join(self.path, 'images', '*.jpg')
            print(pattern)
            self.images = glob.glob(pattern)

            if split is None:
                self.basenames = [os.path.basename(g) for g in self.images]

            elif split =='clean' or split == 'train' or split == 'test':
                self.basenames = [os.path.basename(g) for g in self.images if is_valid_loc(g, split)]

            if region is not None:
                sample_ct = 100
                self.basenames = [b for b in self.basenames if region in b]
                self.basenames = np.random.choice(self.basenames, size=sample_ct)


            self.images = []
            self.masks = []
            for basename in self.basenames:
                img = os.path.join(self.path, 'images', basename)
                mask = os.path.join(self.path, 'masks', basename.replace('_i.jpg', '_mask.jpg'))
                if (os.path.exists(img) and os.path.exists(mask)):
                    self.images.append(img)
                    self.masks.append(mask)

            self.coordinates = None
        
        # Option to dataset for speedy development training to subset of batches
        if batch_trim:
            if batch_trim * 16 > len(self.images):
                pass
            else:
                self.images, self.masks = self.images[:int(batch_trim)*16], self.masks[:int(batch_trim)*16]

    def __getitem__(self, index):
        # print(index)
        if self.load_test:
            img_name = self.images[index]
            if self.compressed:
                image = Image.open(os.path.join(self.path, img_name + '.jpg'))
            else:
                image = Image.open(os.path.join(self.path, img_name, img_name + '.tif'))
            image_tensor = transforms.functional.to_tensor(image)[:3]
            return image_tensor, img_name
        else:
            image = Image.open(self.images[index])
            mask = Image.open(self.masks[index])
            img_name = self.images[index]
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        return (image, mask, img_name)

    def __len__(self):
        return len(self.images)


# ---- Load Dataset ----

def get_dataloader(in_dir=None, load_test=False, batch_size=16, batch_trim=False, overwrite=False, out_dir=None, split=None, region=None):
    '''
    Load pytorch batch data loader only
    '''

    def filter_written(name):
        img_path = '{}/{}.tif'.format(out_dir, name)
        if os.path.exists(img_path):
            return False
        else:
            return True
    
    print('Getting dataset.')
    if split == 'train' or split == 'random' or split == 'clean':
        custom_transforms = train_transform
    elif split == 'test':
        custom_transforms = val_transform
    else:
        custom_transforms = val_transform
    dataset = MyDataset(
        in_dir=in_dir, custom_transforms=custom_transforms, region=region,
        load_test=load_test, batch_trim=batch_trim, split=split
        )
    
    # Check if images have been written.

    if split=='random':
        train_len = int(len(dataset)*0.8)
        lengths = [train_len, len(dataset)-train_len]
        train_subset, val_subset = torch.utils.data.random_split(dataset, lengths)
        val_dataset = DatasetWrapper(val_subset, transform=val_transform)
        train_dataset = DatasetWrapper(train_subset, transform=train_transform)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True,num_workers=3)
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size//4, pin_memory=True, num_workers=3)
        return train_loader, val_loader
    else:
        return DataLoader(
                dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=3
                )


