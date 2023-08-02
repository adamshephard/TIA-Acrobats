from time import time
from functools import wraps

from skimage import color, exposure, measure, morphology
import numpy as np
from scipy import ndimage

# https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func:%r args:[%r, %r] took: %2.4f sec" % (f.__name__, args, kw, te - ts))
        return result

    return wrap

def preprocess_image(image):
    """This function converts the RGB image to grayscale image and
    improves the contrast by linearly rescaling the values.
    """
    image = color.rgb2gray(image)
    image = exposure.rescale_intensity(
        image, in_range=tuple(np.percentile(image, (0.5, 99.5)))
    )
    image = image * 255
    return image.astype(np.uint8)

def post_processing_mask(mask):
    mask = ndimage.binary_fill_holes(mask, structure=np.ones((3, 3))).astype(int)
    # remove all the objects while keep the biggest object only
    label_img = measure.label(mask)
    if len(np.unique(label_img)) > 2:
        regions = measure.regionprops(label_img)
        mask = mask.astype(bool)
        all_area = [i.area for i in regions]
        second_max = max([i for i in all_area if i != max(all_area)])
        mask = morphology.remove_small_objects(mask, min_size=second_max + 1)
    return mask.astype(np.uint8)
