from time import time
from functools import wraps

from skimage import color, exposure, measure, morphology
import numpy as np
from scipy import ndimage
import cv2

# Disable logging at warning level
import logging
from tiatoolbox import logger
logging.getLogger().setLevel(logging.CRITICAL)
logger.setLevel(logging.CRITICAL)

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

def preprocess_image(image, exposure_percentile=(0, 90), apply_adaptive_thresholding=False):
    """This function converts the RGB image to grayscale image and
    improves the contrast by linearly rescaling the values.
    """
    if apply_adaptive_thresholding:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        output_threshold_value = 255
        output_threshold_type = cv2.ADAPTIVE_THRESH_MEAN_C # cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        threshold_method = cv2.THRESH_BINARY
        pixel_neighborhood = 91
        finetune_constant = 3
        image = cv2.adaptiveThreshold(image, output_threshold_value, output_threshold_type, threshold_method, pixel_neighborhood, finetune_constant)
    else:
        image = color.rgb2gray(image)
        image = exposure.rescale_intensity(
            image, in_range=tuple(np.percentile(image, exposure_percentile))
        )
        image = image * 255
    image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2)
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

def convert_pytorch_checkpoint(net_state_dict):
    variable_name_list = list(net_state_dict.keys())
    is_in_parallel_mode = all(v.split(".")[0] == "module" for v in variable_name_list)
    if is_in_parallel_mode:
        net_state_dict = {
            ".".join(k.split(".")[1:]): v for k, v in net_state_dict.items()
        }
    return net_state_dict