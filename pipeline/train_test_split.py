import os, fnmatch, glob

import pdb

import sys
import re

def split_regions(images, masks, split):
    '''
    Filter images according to their train / test split.
    '''

    if split == 'train':
        regions = {
            'd41d81': {
                'pattern': '_1024_',
                'bad_label_reg': [
                    # (1000, 3000 , 9000, 11000) # x lim, y lim
                ]
            }
        }
    elif split == 'test':
        regions = {
            'd41d81': {
                'pattern': '_2048_',
                'bad_label_reg': [
                    (1000, 3000 , 9000, 11000) # x lim, y lim
                    
                ]
            }
        }


    def filter_regions(filename):
        '''
        Filter filenames according to allowed regions.
        '''
        region, x_pos, y_pos, ext = filename.split('_')
    
        for key, values in regions.items():
            if re.search(values['pattern'], filename):
                for x_0, x_1, y_0, y_1 in values['bad_label_reg']:
                    print(x_0, x_1, y_0, y_1, "___", x_pos, y_pos)
                    if x_0 <= int(x_pos) <= x_1:
                        return False
                    if y_0 <= int(y_pos) <= y_1:
                        return False
                return True
        return False

    filtered_imgs = list(filter(filter_regions, images))
    filtered_masks = list(filter(filter_regions, masks))

    assert len(filtered_imgs) == len(filtered_masks)

    return filtered_imgs, filtered_masks

    
    
