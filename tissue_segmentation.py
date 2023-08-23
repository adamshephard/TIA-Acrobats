import os
from pathlib import Path
import click
import shutil
import gc

from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
from tiatoolbox.models.architecture.unet import UNetModel
from tiatoolbox.utils.misc import imwrite
import torch
from wsi_registration_local import match_histograms
from tiatoolbox.wsicore.wsireader import WSIReader

import numpy as np
from matplotlib import pyplot as plt

from utils import preprocess_image, post_processing_mask, convert_pytorch_checkpoint

unet_model_path = 'models/unet-acrobat-v3-01.pth'
foreground_mask_class = 1
artefact_mask_class = 2
apply_adaptive_thresholding = True

@click.command()
@click.option("--moving_image_path", type=Path, required=True)
@click.option("--fixed_image_path", type=Path, required=True)
@click.option("--output_path", type=Path, required=True)
@click.option("--resolution", type=float, default=1.25, required=False)
def tissue_segmentation(moving_image_path, fixed_image_path, output_path, resolution=1.25):
    fixed_name = os.path.basename(fixed_image_path).split(".")[0]
    moving_name = os.path.basename(moving_image_path).split(".")[0]
    fixed_wsi_reader = WSIReader.open(input_img=fixed_image_path)
    fixed_image_rgb = fixed_wsi_reader.slide_thumbnail(resolution=resolution, units="power")
    moving_wsi_reader = WSIReader.open(input_img=moving_image_path)
    moving_image_rgb = moving_wsi_reader.slide_thumbnail(resolution=resolution, units="power")

    # Preprocessing fixed and moving images
    fixed_image = preprocess_image(fixed_image_rgb, apply_adaptive_thresholding=apply_adaptive_thresholding)
    moving_image = preprocess_image(moving_image_rgb, apply_adaptive_thresholding=apply_adaptive_thresholding)
    # fixed_image, moving_image = match_histograms(fixed_image, moving_image)

    imwrite(os.path.join(output_path, f"{fixed_name}.png"), fixed_image)
    imwrite(os.path.join(output_path, f"{moving_name}.png"), moving_image)

    # tissue segmentation
    save_dir = os.path.join(output_path, "tissue_mask")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir, ignore_errors=False, onerror=None)

    pretrained = torch.load(unet_model_path, map_location='cpu')
    pretrained = convert_pytorch_checkpoint(pretrained)
    model = UNetModel(num_input_channels = 3, num_output_channels = 3)
    model.load_state_dict(pretrained)

    segmentor = SemanticSegmentor(
        model=model,
        pretrained_model="unet_tissue_mask_tsef",
        num_loader_workers=4,
        batch_size=4,
    )

    output = segmentor.predict(
        [
            os.path.join(output_path, f"{fixed_name}.png"),
            os.path.join(output_path, f"{moving_name}.png"),
        ],
        save_dir=save_dir,
        mode="tile",
        resolution=1.0,
        units="baseline",
        patch_input_shape=[1024, 1024],
        patch_output_shape=[512, 512],
        stride_shape=[512, 512],
        on_gpu=True,
        crash_on_exception=True,
    )

    # post proc and mask visualization
    fixed_mask = np.load(output[0][1] + ".raw.0.npy")
    moving_mask = np.load(output[1][1] + ".raw.0.npy")

    # Simple processing of the raw prediction to generate semantic segmentation task
    fixed_mask = np.argmax(fixed_mask, axis=-1) == foreground_mask_class 
    moving_mask = np.argmax(moving_mask, axis=-1) == foreground_mask_class 
    # artefact_fixed_mask = np.argmax(fixed_mask, axis=-1) == artefact_mask_class
    # artefact_moving_mask = np.argmax(moving_mask, axis=-1) == artefact_mask_class
    
    # Combine foreground and artefact masks
    # fixed_mask = np.logical_or(foreground_fixed_mask, artefact_fixed_mask)
    # moving_mask = np.logical_or(foreground_moving_mask, artefact_moving_mask)

    fixed_mask = post_processing_mask(fixed_mask)
    moving_mask = post_processing_mask(moving_mask)

    # Next save masks
    fixed_mask_rgb = np.repeat(np.expand_dims(fixed_mask, axis=2), 3, axis=2)*255
    imwrite(os.path.join(output_path, f"{fixed_name}_mask.png"), fixed_mask_rgb)
    moving_mask_rgb = np.repeat(np.expand_dims(moving_mask, axis=2), 3, axis=2)*255
    imwrite(os.path.join(output_path, f"{moving_name}_mask.png"), moving_mask_rgb)
    
    shutil.rmtree(save_dir)
    gc.collect() 
    print("Finished tissue segmentation!")

if __name__ == "__main__":
    tissue_segmentation()