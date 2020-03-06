import os, fnmatch, glob

import pdb

import sys
import re

def split_regions(images, masks, split, tier=1):
    '''
    Filter images according to their train / test split.
    '''

    # 6 of 7 cities used for train region
    # Training regs are 86.0% of total area
    # Training regs are 65.0% of unique blocks
    # Train scene ID list (tier 1): ['f883a0' '4e7c7f' 'f15272' '825a50' 'f49f31' 'e52478']
    # Test scene ID list (tier 1): ['d41d81']

    if split == 'train':
        regions = {
            'f883a0': {
                'skip_region': [
                    # (x_0, x_1 , y_0, y_1), # x lim, y lim
                    # (x_0, x_1 , y_0, y_1), # x lim, y lim
                ]
            },
            '4e7c7f': {
                'skip_region': [
                    # (x_0, x_1 , y_0, y_1), # x lim, y lim
                    # (x_0, x_1 , y_0, y_1), # x lim, y lim
                ]
            },
            'f15272': {
                'skip_region': [
                    # (x_0, x_1 , y_0, y_1), # x lim, y lim
                    # (x_0, x_1 , y_0, y_1), # x lim, y lim
                ]
            },
            '825a50': {
                'skip_region': [
                    # (x_0, x_1 , y_0, y_1), # x lim, y lim
                    # (x_0, x_1 , y_0, y_1), # x lim, y lim
                ]
            },
            'f49f31': {
                'skip_region': [
                    # (x_0, x_1 , y_0, y_1), # x lim, y lim
                    # (x_0, x_1 , y_0, y_1), # x lim, y lim
                ]
            },
            'e52478': {
                'skip_region': [
                    # (x_0, x_1 , y_0, y_1), # x lim, y lim
                    # (x_0, x_1 , y_0, y_1), # x lim, y lim
                ]
            }
        }
    elif split == 'test':
        regions = {
            'd41d81': {
                'skip_region': [
                    # (x_0, x_1 , y_0, y_1), # x lim, y lim
                    # (x_0, x_1 , y_0, y_1), # x lim, y lim
                ]
            }
        }


    def filter_regions(filename):
        '''
        Filter filenames according to allowed regions.
        '''
        region, x_pos, y_pos, ext = filename.split('_')
    
        for key, values in regions.items():
            
            # Check all matching regions.
            if re.search(key, filename):
                # Skip regions that are badly labeled.
                for x_0, x_1, y_0, y_1 in values['skip_region']:
                    # print(x_0, x_1, y_0, y_1, "___", x_pos, y_pos)
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

    
    
