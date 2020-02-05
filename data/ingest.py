# ----------------------------- #
# Hosted dataset URLs + prefixes
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
import tensorflow as tf
from tqdm import tqdm

from IPython.display import clear_output
import matplotlib.pyplot as plt

assert tf.version.VERSION[0] == '2', "Must use TF Version 2.x"


metadata = pd.read_csv('https://s3.amazonaws.com/drivendata/data/60/public/train_metadata.csv')
base_url = 'https://drivendata-competition-building-segmentation.s3-us-west-1.amazonaws.com/'

# Idea: pull rasters + labels iteratively from dataframe,
# then iterate over rasters in 1024x1024 blocks,
# saving each block to a new file that includes the mask for each tile.

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

tile = get_tile_at_idx(10)
print("tile shape:", tile.shape)


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

  row = metadata[metadata['img_uri'].str.contains(scene_id)]
  img_uri, label_uri = row['img_uri'].values[0], row['label_uri'].values[0]
  base_url = 'https://drivendata-competition-building-segmentation.s3-us-west-1.amazonaws.com/'
  
  path = base_url+img_uri
    
  with rasterio.open(path) as scene:
    labels = gpd.read_file(base_url+label_uri)
    #transform to same CRS as raster file
    labels = labels.to_crs(scene.crs.data)

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
        tile, mask = generate_tile_and_mask(scene, labels, x_pos, y_pos)

        if tile is not None:
          # print(tile.shape, mask.shape)
          if tile[0].shape != mask.shape:
            print(f'tile and shape mismatch at {(x_pos, y_pos)}: {tile[0].shape} != {mask.shape}')
            continue
          tiles.append(tile)
          masks.append(mask)
          x.append(x_pos)
          y.append(y_pos)

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
