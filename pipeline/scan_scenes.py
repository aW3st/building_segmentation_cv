from data.ingest import Tile, generate_tile_and_mask, get_hosted_urls, get_scene_and_labels
from tqdm import tqdm
from os import listdir
from os.path import isfile, join

import pandas as pd

import logging
logging.basicConfig()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

metadata = pd.read_csv('https://s3.amazonaws.com/drivendata/data/60/public/train_metadata.csv')
base_url = 'https://drivendata-competition-building-segmentation.s3-us-west-1.amazonaws.com/'


def update_scan_log():
    '''
    Scan local training directory for scenes already scanned.
    
    Update the local log.
    '''

    print('Updating local scan log.')
    with open('./data/scan_log.txt') as file:
        scanned_scenes = file.read().splitlines()

    # loop through directory and find out which scenes have been scanned
    scanned_files = set(
        [f.split('_')[0] for f in listdir('data/train')
         if isfile(join('data/train', f))]
         )
    print(scanned_files)

    return None


def get_scene_ids():
    '''
    Retrieve a list of scene_id's from the metadata list.
    '''

    scene_ids = [row.split('/')[-2] for row in metadata['img_uri'].values]

    print(scene_ids)

    return scene_ids


def save_scene_tiles(scene_id, path='./data/train'):
    '''Scan entire scene for a given scene id.'''
    scene, labels = get_scene_and_labels(scene_id=scene_id)

    for x_pos in tqdm(range(0, scene.height, 1024), desc='x_pos' ,position=0):
        for y_pos in range(0, scene.width, 1024):
            tile = Tile(scene, labels, x_pos, y_pos, scene_id)
            tile.get_mask()
            if (tile.tile is not None) and (tile.mask is not None):
                if tile.tile[0].shape != tile.mask[0].shape:
                    print(f'tile and shape mismatch at {(x_pos, y_pos)}: {tile.tile[0].shape} != {tile.mask[0].shape}')
                    continue
                tile.write_data('data/train/')
            else:
                continue
    scene.close()

    return True