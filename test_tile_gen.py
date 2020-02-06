import pytest

from data.ingest import Tile, generate_tile_and_mask, get_hosted_urls, get_scene_and_labels

def test_tile_and_mask_write():
    metadata, base_url = get_hosted_urls()

    scene, labels = get_scene_and_labels(scene_id='d41d81')

    tile = Tile(scene, labels, scene.height//2, scene.width//2, 'blahblah')
    tile.get_mask()
    tile.plot(mask=True)
    tile.write_data('data/test')

if __name__ == '__main__':
    test_tile_and_mask_write()