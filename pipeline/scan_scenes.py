from data.ingest import Scene
from tqdm import tqdm
from os import listdir, remove
from os.path import isfile, join
import pandas as pd, logging
from pipeline.gcloud import upload_blob
import pdb
import os
logging.basicConfig(level=(logging.INFO))
logger = logging.getLogger()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
metadata = pd.read_csv('https://s3.amazonaws.com/drivendata/data/60/public/train_metadata.csv')
base_url = 'https://drivendata-competition-building-segmentation.s3-us-west-1.amazonaws.com/'


def update_scan_log():
    """
    Scan local training directory for scenes already scanned.
    
    Update the local log.
    """
    logger.info('Updating local scan log.')
    with open('./data/local_scene_registry.txt') as (file):
        scanned_scenes = file.read().splitlines()
    scanned_scenes = set([f.split('_')[0] for f in listdir('data/train') if isfile(join('data/train', f))])
    print('Previously scanned scenes:\n', scanned_scenes)
    with open('./data/local_scene_registry.txt', 'w') as (file):
        file.write(str(scanned_scenes))
    return scanned_scenes


def get_scene_ids():
    """
    Retrieve a list of scene_id's from the metadata list.
    """
    scene_ids = [row.split('/')[(-2)] for row in metadata['img_uri'].values]
    # print(scene_ids)
    return scene_ids


def scan_scenes(path):
    """
    scans each scene in metadata for tiles and uploads tiles to GCP bucket
    """
    scene_ids = get_scene_ids()
    for idx in scene_ids:
        print(f'scanning {idx}')
        image = Scene(idx)
        for x_pos in tqdm((range(0, image.scene.height, 1024)), desc='x_pos', position=0):
            for y_pos in range(0, image.scene.width, 1024):
                tile = image.get_tile(x_pos, y_pos, 1024)
                tile.get_mask()
                if tile.alpha_pct>0.5:
                    logger.info(f'Not enough data in tile {x_pos, y_pos}')
                    continue
                try:
                    assert tile.tile[0].shape==(1024,1024)
                except:
                    logger.info(f'wrong tile shape at {x_pos,y_pos}: {tile.tile.shape}')
                    continue
                try:
                    assert tile.tile.shape[1:]==tile.mask.shape[:2]
                except:
                    logger.info(f'tile/mask mismatch at {x_pos, y_pos}')
                    pdb.set_trace()
                    continue
                filename = idx+'_'+str(x_pos)+"_"+str(y_pos)
                tile.write_data(path)
                upload_blob(bucketname, os.path.join(path, 'images', filename+'_i.jpg'))
                upload_blob(bucketname, os.path.join(path, 'masks', filename+'_mask.jpg'))
                remove(os.path.join(path, 'images', filename+'_i.jpg'))
                remove(os.path.join(path, 'masks', filename+'_mask.jpg'))

        image.scene.close()


if __name__=='__main__':
    bucketname = "satellite_tiles2"
    scan_scenes('data/train')


