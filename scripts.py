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
from data.ingest import Tile, generate_tile_and_mask, get_hosted_urls, get_scene_and_labels
from tqdm import tqdm

from pipeline.scan_scenes import update_scan_log, get_scene_ids

import json

# Getting rid of those damn CRS warnings.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def collect_scene_information():
    '''
    Store certain scene information for better analytics.
    '''
    scene_ids = get_scene_ids()
    scenes = {}

    for scene_id in scene_ids:
        print('collecting info on scene',scene_id)
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


def scan_scenes_to_local(limit=2):
    '''
    Scan scenes, ignoring ones already scanned.

    Only scan as many scenes as limit.
    '''

    # Scenes scanned in this directory so far.
    # confirm registry is properly updated with local scenes
    scanned_scenes = update_scan_log()

    # All available scenes.
    scene_list = get_scene_ids()

    # A json with a few pieces of helpful information.
    scene_info = json.load('data/scene_log.json')

    count = 0
    for scene_id in scene_list:
        if count > limit:
            break
        print('scene id', scene_id)
        s_info = scene_info[scene_id]

        blk_ct = s_info['blocks']

        print(f'{scene_id} has {blk_ct} blocks')
        if s_info['blocks'] < 1000:
            if scene_id not in scanned_scenes:
                save_scene_tiles(scene_id)
                count += 1
            else:
                print('already scanned')
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
    metadata, base_url = get_hosted_urls()

    scene, labels = get_scene_and_labels(scene_id='d41d81')

    tile = Tile(scene, labels, scene.height//2, scene.width//2, 'blahblah')
    tile.get_mask()
    tile.plot(mask=True)
    tile.write_data('data/test')