import gc

from pathlib import Path
import click
import shutil

from tiatoolbox.wsicore.wsireader import WSIReader
import os
import cv2
import numpy as np
import pandas as pd


@click.command()
@click.option("--landmarks_path", type=Path, required=True)
@click.option("--moving_image_path", type=Path, required=True)
@click.option("--intermediate_path", type=Path, required=True)
@click.option("--output_path", type=Path, required=True)
@click.option("--out_res", type=float, default=1.25)
def landmark_registration(landmarks_path, moving_image_path, intermediate_path, output_path, out_res=1.25):
    """
    Script cureenlty not working correctly - needs fixing!
    """
    moving_wsi_reader = WSIReader.open(input_img=moving_image_path)
    moving_image_rgb = moving_wsi_reader.slide_thumbnail(resolution=out_res, units="power")
    moving_base_mpp = moving_wsi_reader.info.mpp
    moving_base_dims = moving_wsi_reader.slide_dimensions(units='mpp', resolution=moving_base_mpp[0])
    moving_proc_dims = moving_wsi_reader.slide_dimensions(units='power', resolution=out_res)
    dfbr_transform = np.loadtxt(os.path.join(intermediate_path, "transform.csv"), delimiter=",")
    
    trans = dfbr_transform[0:-1]
    # upscale transform
    scale_factor = moving_base_dims / moving_proc_dims
    trans[:,2] = trans[:,2] * scale_factor
    # Here, [x',y']T = m*x[x,y]T
    
    data = pd.read_csv(landmarks_path)
    
    new_locations = pd.DataFrame(columns=['x_target', 'y_target'])
    # input x,y in microns --> convert to pixels in source res (*source_mpp)
    # then apply transform --> converts pixels to target res
    # then turn to microns --> convert to microns in target res (/target_mpp)
    for idx, info in data.iterrows():
        x = info['x_source'] # in microns
        y = info['y_source'] # in microns
        mpp_source = info['mpp_source']
        mpp_target = info['mpp_target']
        ds = mpp_target/mpp_source
        x_prime = (((trans[0][0]*x + trans[0][1]*y) / mpp_source) + trans[0][2]) * mpp_target
        y_prime = (((trans[1][0]*x + trans[1][1]*y) / mpp_source) + trans[1][2]) * mpp_target
        new_locations.loc[idx] = [x_prime, y_prime]
    
    # should be given in microns not pixels!
    
    # create file with columns x_target, y_target and save to output_path (in microns)
    new_locations.to_csv(os.path.join(output_path, "registered_landmarks.csv"), index=False)

    gc.collect() 
    print("Finished landmark registration!")


if __name__ == "__main__":
    landmark_registration()