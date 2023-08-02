import gc

from pathlib import Path
import click
import shutil

from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
from tiatoolbox.tools.registration.wsi_registration import (
    DFBRegister,
    match_histograms,
)
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.utils.misc import imread, imwrite 
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

from utils import preprocess_image, post_processing_mask

@click.command()
@click.option("--moving_image_path", type=Path, required=True)
@click.option("--fixed_image_path", type=Path, required=True)
@click.option("--intermediate_path", type=Path, required=True)
@click.option("--output_path", type=Path, required=True)
@click.option("--proc_res", type=float, default=0.1563)
@click.option("--out_res", type=float, default=1.25)
def registration(moving_image_path, fixed_image_path, intermediate_path, output_path, proc_res=0.1563, out_res=1.25):
    
    # moving_image_path = "/data/ACROBAT/tia-acrobats/test_images/moving_images/moving_image.tif"
    # fixed_image_path = "/data/ACROBAT/tia-acrobats/test_images/fixed_images/fixed_image.tif"
    fixed_name = os.path.basename(fixed_image_path).split(".")[0]
    moving_name = os.path.basename(moving_image_path).split(".")[0]
    # intermediate_path = "/data/ACROBAT/tia-acrobats/test_images/output/tmpoutput"

    # output_path = "/data/ACROBAT/tia-acrobats/test_images/output/"

    fixed_wsi_reader = WSIReader.open(input_img=fixed_image_path)
    fixed_image_rgb = fixed_wsi_reader.slide_thumbnail(resolution=proc_res, units="power")
    moving_wsi_reader = WSIReader.open(input_img=moving_image_path)
    moving_image_rgb = moving_wsi_reader.slide_thumbnail(resolution=proc_res, units="power")

    moving_image = imread(os.path.join(intermediate_path, f"{moving_name}.png"))[:,:,0]
    fixed_image = imread(os.path.join(intermediate_path, f"{fixed_name}.png"))[:,:,0]
    moving_mask = imread(os.path.join(intermediate_path, f"{moving_name}_mask.png"))
    fixed_mask = imread(os.path.join(intermediate_path, f"{fixed_name}_mask.png"))

    # Registration using DFBR
    dfbr_fixed_image = np.repeat(np.expand_dims(fixed_image, axis=2), 3, axis=2)
    dfbr_moving_image = np.repeat(np.expand_dims(moving_image, axis=2), 3, axis=2)

    df = DFBRegister()
    dfbr_transform = df.register(
        dfbr_fixed_image, dfbr_moving_image, fixed_mask, moving_mask
    )

    # Visualisation of warp
    original_moving = cv2.warpAffine(
        moving_image, np.eye(2, 3), fixed_image.shape[:2][::-1]
    )
    dfbr_registered_image = cv2.warpAffine(
        moving_image, dfbr_transform[0:-1], fixed_image.shape[:2][::-1]
    )
    dfbr_registered_mask = cv2.warpAffine(
        moving_mask, dfbr_transform[0:-1], fixed_image.shape[:2][::-1]
    )
    imwrite(os.path.join(intermediate_path, f"{moving_name}_registered.png"), dfbr_registered_image)
    imwrite(os.path.join(intermediate_path, f"{moving_name}_registered_mask.png"), dfbr_registered_mask)

    before_overlay = np.dstack((original_moving, fixed_image, original_moving))
    imwrite(os.path.join(intermediate_path, f"{moving_name}_overlay_pre_registration.png"), before_overlay)

    dfbr_overlay = np.dstack((dfbr_registered_image, fixed_image, dfbr_registered_image))
    imwrite(os.path.join(intermediate_path, f"{moving_name}_overlay_post_registration.png"), dfbr_overlay)

    # Now get thumb at required resolution and warp/save
    out_moving_image_rgb = moving_wsi_reader.slide_thumbnail(resolution=out_res, units="power")
    out_fixed_image = fixed_wsi_reader.slide_thumbnail(resolution=out_res, units="power")
    out_moving_mask = cv2.resize((moving_mask[:,:,0]/255).astype('uint8'), (out_moving_image_rgb.shape[1], out_moving_image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    out_fixed_mask = cv2.resize((fixed_mask[:,:,0]/255).astype('uint8'), (out_fixed_image.shape[1], out_fixed_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    out_fixed_image = preprocess_image(out_fixed_image)
    out_moving_image = preprocess_image(out_moving_image_rgb)
    out_fixed_image, out_moving_image = match_histograms(out_fixed_image, out_moving_image)
    # out_fixed_image = np.repeat(np.expand_dims(out_fixed_image, axis=2), 3, axis=2)
    # out_moving_image = np.repeat(np.expand_dims(out_moving_image, axis=2), 3, axis=2)

    # original_moving = cv2.warpAffine(
    #     out_moving_image, np.eye(2, 3), out_fixed_image.shape[:2][::-1]
    # )
    # dfbr_registered_image = cv2.warpAffine(
    #     out_moving_image, dfbr_transform[0:-1], out_fixed_image.shape[:2][::-1]
    # )
    # dfbr_registered_mask = cv2.warpAffine(
    #     out_moving_mask, dfbr_transform[0:-1], out_fixed_image.shape[:2][::-1]
    # )
    # Not scaling??? this is because you cannot scale up the affine matrix (owing to the constant in column 3?)
    # Images will be in correct position but shifted by an unknown amount
    
    # INSTEAD need to scale up the affine matrix (owing to the constant in column 3)
    
    # Also perform transformation for higher res image
    dfbr_fixed_mask = np.repeat(np.expand_dims(out_fixed_mask, axis=2), 3, axis=2)*255
    dfbr_moving_mask = np.repeat(np.expand_dims(out_moving_mask, axis=2), 3, axis=2)*255
    dfbr_fixed_image = np.repeat(np.expand_dims(out_fixed_image, axis=2), 3, axis=2)
    dfbr_moving_image = np.repeat(np.expand_dims(out_moving_image, axis=2), 3, axis=2)

    df_new = DFBRegister()
    dfbr_transform_new = df_new.register(
        dfbr_fixed_image, dfbr_moving_image, dfbr_fixed_mask, dfbr_moving_mask
    )

    original_moving = cv2.warpAffine(
        out_moving_image, np.eye(2, 3), out_fixed_image.shape[:2][::-1]
    )
    dfbr_registered_image = cv2.warpAffine(
        out_moving_image, dfbr_transform_new[0:-1], out_fixed_image.shape[:2][::-1]
    )
    dfbr_registered_rgb_image = cv2.warpAffine(
        out_moving_image_rgb, dfbr_transform_new[0:-1], out_fixed_image.shape[:2][::-1]
    )
    dfbr_registered_mask = cv2.warpAffine(
        out_moving_mask, dfbr_transform_new[0:-1], out_fixed_image.shape[:2][::-1]
    )

    before_overlay = np.dstack((original_moving, out_fixed_image, original_moving))
    dfbr_overlay = np.dstack((dfbr_registered_image, out_fixed_image, dfbr_registered_image))

    # plt.figure();plt.imshow(dfbr_registered_image); plt.show(block=False)
    imwrite(os.path.join(output_path, f"{moving_name}_registered.tiff"), dfbr_registered_image)
    imwrite(os.path.join(output_path, f"{moving_name}_registered_rgb.tiff"), dfbr_registered_rgb_image)
    temp = np.repeat(np.expand_dims(dfbr_registered_mask, axis=2), 3, axis=2)*255
    imwrite(os.path.join(output_path, f"{moving_name}_registered_mask.tiff"), temp)
    
    # Save transformation matric to csv
    np.savetxt(os.path.join(intermediate_path, "transform.csv"), dfbr_transform_new, delimiter=",")
    gc.collect() 
    print("Finished registration!")


if __name__ == "__main__":
    registration()