# Save entire scene's worth of tiles to local directory.

from data.ingest import Tile, generate_tile_and_mask, get_hosted_urls, get_scene_and_labels
from tqdm import tqdm

def save_scene_tiles(scene_id):
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

save_scene_tiles('abe1a3')