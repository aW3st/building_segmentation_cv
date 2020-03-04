import os
import pdb
import glob
from PIL import Image
import numpy as np

if __name__ == "__main__":
    
    IMG_DIR = 'models/04-03-2020__Wed__03-04__five_epoch_single_region/predictions/'

    # IMG_PATH = os.path.join(IMG_DIR, '0a0ab4.tif')
    # IMG = Image.open(IMG_PATH).convert(mode='1')
    # NP_IMG = np.array(IMG)

    # pdb.set_trace()

    IMAGES = glob.glob(IMG_DIR + '*.tif')
    print(IMAGES)

    for img in IMAGES:
        # print(img)
        IMG = Image.open(img).convert(mode='1')
        IMG.save(img)