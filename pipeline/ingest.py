# ----------------------------- #
# STAC and Geospatial Ingest
# ----------------------------- #

import os
from urllib.parse import urlparse
import requests
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
import numpy as np
import rasterio.plot
from rasterio.mask import raster_geometry_mask
from rasterio.windows import Window, bounds
from shapely.geometry import Polygon, box
import pdb
# Tensorflow stuff
# import tensorflow as tf
from tqdm import tqdm

# from IPython.display import clear_output
import matplotlib.pyplot as plt

# assert tf.version.VERSION[0] == '2', "Must use TF Version 2.x"

# ----------------------------- #
# Hosted dataset URLs + prefixes
# ----------------------------- #


metadata = pd.read_csv('https://s3.amazonaws.com/drivendata/data/60/public/train_metadata.csv')
base_url = 'https://drivendata-competition-building-segmentation.s3-us-west-1.amazonaws.com/'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Idea: pull rasters + labels iteratively from dataframe,
# then iterate over rasters in 1024x1024 blocks,
# saving each block to a new file that includes the mask for each tile.


def get_hosted_urls():
    '''
    Return metadata, base_url of hosted data.
    '''
    return metadata, base_url
  

def get_scene_and_labels(scene_id):
    '''
    Returns opened scene reader and labels
    '''
    # search for row
    row = metadata[metadata['img_uri'].str.contains(scene_id)]

    img_uri, label_uri = row['img_uri'].values[0], row['label_uri'].values[0]
    base_url = 'https://drivendata-competition-building-segmentation.s3-us-west-1.amazonaws.com/'
    scene_path = base_url+img_uri
    scene = rasterio.open(scene_path)

    labels = gpd.read_file(base_url+label_uri)
    #transform to same CRS as raster file
    labels = labels.to_crs(scene.crs)
    return scene, labels


def get_tile_at_idx(idx=10, x_pos=None, y_pos=None):
    '''
    Get a single tile from a scene at the specified index.
    Defaults to the middle of the 1st scene unless specified.
    '''
    scene = rasterio.open(base_url+metadata['img_uri'].iloc[idx])

    if x_pos is None:
       x_pos = scene.height//2

    if y_pos is None:
        y_pos = scene.width//2

    window = Window(x_pos, y_pos,1024,1024)
    window_transform = rasterio.windows.transform(window, scene.transform)
    tile = scene.read(window=window)

    # Confirm there is actually data at this tile
    assert tile.any(), "No data in at position in scene."

    rasterio.plot.show(tile, transform=window_transform);
    return tile


def generate_tile_and_mask(scene, labels, x_pos, y_pos, plot=False):
    '''
    Generate a tile and mask from a scene.
    '''
    #build window 
    window = Window(y_pos,x_pos,1024,1024)
    window_transform = rasterio.windows.transform(window, scene.transform)
    tile = scene.read(window=window)

    if plot:
        rasterio.plot.show(tile, transform=window_transform)
    
    #get window coordinates to make bounding box
    minx,miny,maxx,maxy = bounds(window, scene.transform)
    boundingbox = box(minx, miny, maxx, maxy)
    # crop labels to area of raster tile
    mask = labels.intersection(boundingbox).is_empty == False
    
    label_intersection = labels.intersection(boundingbox)[mask]
    #if there are no buildings in the tile, we need to make our own mask
    
    # Return None, None if there are too many alpha tiles, 
    if np.count_nonzero(tile[0]) < tile[0].size/2:
        # print('skipping tile')
        return None, None
    if label_intersection.empty:
        print('no buildings!')
        return tile, np.zeros((1024,1024))
    else:
        mask = rasterio.mask.raster_geometry_mask(scene, label_intersection, invert=True)
        # raster_geometry_mask returns an array of shape (img.height, img.width), need to slice the portion we want
        mask = mask[0][window.row_off:window.row_off+1024, window.col_off:window.col_off+1024]
        return tile, mask.astype(np.float32)


def generate_tf_tiles_from_scene(scene_id, metadata, limit=None, chunks=False, write_to=None):
    '''
    Iterate through a scene and generate a dataset of tile and label masks.

    Returns a tf.data.Dataset() class object contains data from a single scene.
    '''

    if chunks:
        raise NotImplementedError
      
    scene = get_scene(scene_id, metadata)

    tiles = []
    masks = []
    x = []
    y = []

    num_data = 0
    skip_count = 0

    for x_pos in tqdm(range(0, scene.height, 1024), desc='x_pos' ,position=0):
        print(num_data, 'tiles collected\n')
        if num_data > limit:
            break

        for y_pos in range(0, scene.width, 1024):

          num_data += 1
          if num_data > limit:
              break

          # print(x_pos, y_pos)
          tile = Tile(scene, label, xpos, ypos, scene_id)
          if tile.tile is not None:
            if tile.tile[0].shape != tile.mask.shape:
              print(f'tile and shape mismatch at {(x_pos, y_pos)}: {tile[0].shape} != {mask.shape}')
              continue

          else:
            skip_count += 1
            continue
    
    if write_to:
        raise NotImplementedError

    data = ({
        'scene_id': [scene_id]*len(x),
        'x': x,
        'y': y,
        'tile': tiles,
        'mask': masks
    })

    # pdb.set_trace()

    print(f'{num_data} of {num_data + skip_count} possible tile candidates stored')
    # print(len(x))
    dataset = tf.data.Dataset.from_tensor_slices(data)

    scene.close()

    return dataset


def ingest_scenes(scene_ids, path_out='/scenes'):
    '''
    Ingest scenes in given list,
    and write to directory as tf.data.dataset objects. 
    '''

    if type(scene_ids) != list:
        scene_ids = list(scene_ids)

    for scene_id in scene_ids:
        generate_tf_tiles_from_scene(scene_id, metadata, limit=10, write_to=path_out)


class Tile():

    def __init__(self, scene, labels, xpos, ypos, scene_id):
        self.scene = scene
        self.labels = labels.to_crs(self.scene.crs)
        self.xpos = xpos
        self.ypos = ypos
        self.scene_id = scene_id
        self.window = Window(xpos, ypos, 1024,1024)
        self.tile = self.scene.read(window=self.window)[:3]
        self.alpha_pct = 1 - np.count_nonzero(self.tile[0]) / self.tile[0].size
        self.window_transform = rasterio.windows.transform(self.window, self.scene.transform)

    def get_mask(self):
        window_coords = bounds(self.window, self.scene.transform)
        boundingbox = box(*window_coords)
        boolmask = self.labels.intersects(boundingbox)
        self.label_intersection = self.labels.intersection(boundingbox)[boolmask]

        #if there are no buildings in tile, mask should be all zeros
        if self.label_intersection.empty:
            self.mask = np.zeros((1, 1024, 1024))
            return self.mask
        else:
            self.mask = rasterio.mask.raster_geometry_mask(self.scene, self.label_intersection, invert=True)
            # raster_geometry_mask returns an array of shape (img.height, img.width), need to slice the portion we want
            self.mask = np.expand_dims(self.mask[0][self.ypos:self.ypos + 1024, self.xpos:self.xpos + 1024].astype(np.float32), axis=0)
            return self.mask

    def plot(self, mask=False):
        fig, ax = plt.subplots(figsize=(15,15))
        rasterio.plot.show(self.tile, transform = self.window_transform, ax = ax)
        if mask:
            if self.mask is None:
                self.get_mask()
            self.label_intersection.plot(ax=ax)

    def write_data(self, path):
        image = tf.keras.preprocessing.image.array_to_img(self.tile, data_format='channels_first')
        mask = tf.keras.preprocessing.image.array_to_img(self.mask, data_format='channels_first')
        image.save(path+self.scene_id+"_"+str(self.xpos)+"_"+str(self.ypos)+"_i.jpg")
        mask.save(path+self.scene_id+"_"+str(self.xpos)+"_"+str(self.ypos)+"_mask.jpg")








