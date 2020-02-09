# –––––––––––––––––––––––––––––––––––––––––
# –––––––––––––––– MAIN––––––––––––––––––––
# –––––––––––––––––––––––––––––––––––––––––

# This module also is meant to make scratchpad work easier.
# To this end, from interactive shell, run:
#   `from scripts import *`
# This will give you access to the top-level functions in this module.

# Purpose of this file is to collect top-level management scripts into neat methods.
# To test basic functions, run pytest from command-line on this file.

import pytest
from tqdm import tqdm
import json

from pipeline.ingest import get_scene_and_labels
from pipeline.scan_scenes import update_scan_log, get_scene_ids, save_scene_tiles
from pipeline.model import *
from pipeline.

# Getting rid of those damn CRS warnings.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def collect_scene_information():
    '''
    Store certain scene information for better analytics.
    '''
    scene_ids = get_scene_ids()
    scenes = {}

    for scene_id in scene_ids:
        logger.info('Collecting info on scene',scene_id)
        scene, labels = get_scene_and_labels(scene_id)
        
        scene_url = scene.name
        scene_info = {
            'scene_id':scene_id,
            'shape': scene.shape,
            'size': str(scene.shape[0]*scene.shape[1]),
            'blocks': str(scene.shape[0]//1024 * scene.shape[1]//1024),
            'lnglat': str(scene.lnglat()),
            'scene_url': scene_url,
            'city': scene_url.split('/')[-3],
            'tier': scene_url.split('/')[-4].split('_')[-1]
            }
        scenes[scene_id] = scene_info

    with open('data/scene_log.json','w') as file:
        json.dump(scenes, file, ensure_ascii=False, indent=4)

    return scenes


def scan_scenes_to_local(limit=10):
    '''
    Scan scenes, ignoring ones already scanned.

    Only scan as many scenes as specified in limit arg.
    '''

    # Scenes scanned in this directory so far.
    # confirm registry is properly updated with local scenes
    scanned_scenes = update_scan_log()

    # All available scenes.
    scene_list = get_scene_ids()

    # A json with a few pieces of helpful information.
    with open('data/scene_log.json','r') as file:
        scene_info = json.load(file)

    count = 0
    for scene_id in scene_list:
        if count > limit:
            break
        if scene_id not in scanned_scenes:
            logger.info(f'{scene_id} already scanned')
        else:
            # Get basic scene info.
            logger.info('Looking up scene id:', scene_id)
            s_info = scene_info[scene_id]
            blk_ct = s_info['blocks']
            logger.info(f'{scene_id} has {blk_ct} blocks')
            if int(s_info['blocks']) < 2000:
                save_scene_tiles(scene_id)
                count += 1
            else:
                continue

    return True


def scratch():
    update_scan_log()
    get_scene_ids()



# ––––––––––––––––––––––––––––––––––––––––
# –––––––– TESTING FUNCTIONS –––––––––––––
# ––––––––––––––––––––––––––––––––––––––––
#
# Run tests from command line with pytest:
#  $ pytest main.py
#
#  All functions with the test_ prefix below will run tests.


def test_tile_and_mask_write():
    '''
    Testing function for tile and mask writing.
    '''
    metadata, base_url = get_hosted_urls()

    scene, labels = get_scene_and_labels(scene_id='d41d81')

    tile = Tile(scene, labels, scene.height//2, scene.width//2, 'blahblah')
    tile.get_mask()
    tile.plot(mask=True)
    tile.write_data('data/test')


# ––––––––––––––––––––––––––––––––––––––––
# –––––––– MAIN FUNCTION –––––––––––––––––
# ––––––––––––––––––––––––––––––––––––––––

# Use this to quickly test functions.


if __name__=='__main__':
    run_model_5()
