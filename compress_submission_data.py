from PIL import Image
import glob
import os
import numpy as np

SUBMISSION_DIR = 'submission_data/test'

def alpha_composite(front, back):
    """Alpha composite two RGBA images.

    Source: http://stackoverflow.com/a/9166671/284318

    Keyword Arguments:
    front -- PIL RGBA Image object
    back -- PIL RGBA Image object

    """
    front = np.asarray(front)
    back = np.asarray(back)
    result = np.empty(front.shape, dtype='float')
    alpha = np.index_exp[:, :, 3:]
    rgb = np.index_exp[:, :, :3]
    falpha = front[alpha] / 255.0
    balpha = back[alpha] / 255.0
    result[alpha] = falpha + balpha * (1 - falpha)
    old_setting = np.seterr(invalid='ignore')
    result[rgb] = (front[rgb] * falpha + back[rgb] * balpha * (1 - falpha)) / result[alpha]
    np.seterr(**old_setting)
    result[alpha] *= 255
    np.clip(result, 0, 255)
    # astype('uint8') maps np.nan and np.inf to 0
    result = result.astype('uint8')
    result = Image.fromarray(result, 'RGBA')
    return result

def alpha_composite_with_color(image, color=(255, 255, 255)):
    """Alpha composite an RGBA image with a single color image of the
    specified color and the same size as the original image.

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)

    """
    back = Image.new('RGBA', size=image.size, color=color + (255,))
    return alpha_composite(image, back)

def pure_pil_alpha_to_color_v2(image, color=(255, 255, 255)):
    """Alpha composite an RGBA Image with a specified color.

    Simpler, faster version than the solutions above.

    Source: http://stackoverflow.com/a/9459208/284318

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)

    """
    image.load()  # needed for split()
    background = Image.new('RGB', image.size, color)
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return background


def filter_written(name):
    if name == 'catalog.json':
        return False
    img_path = '{}/{}.jpg'.format('submission_data/compressed', name)
    if os.path.exists(img_path):
        return False
    else:
        return True


if __name__ == "__main__":
    
    PATH = os.path.join(SUBMISSION_DIR, '*')
    print(PATH)
    IMAGES = glob.glob(PATH)
    print(IMAGES[:10])
    IMAGES = [os.path.basename(g) for g in IMAGES]
    print(IMAGES[:10])

    IMAGES = list(filter(filter_written, IMAGES))
    print(IMAGES)

    for path in IMAGES:
        img = Image.open(os.path.join(SUBMISSION_DIR, path, path + '.tif')).convert('')
        img = pure_pil_alpha_to_color_v2(img)
        img.save(f'submission_data/compressed/{path}.jpg', "JPEG", quality=80)