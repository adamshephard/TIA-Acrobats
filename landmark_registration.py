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
    
    dfbr_transform = np.loadtxt(os.path.join(intermediate_path, "transform.csv"), delimiter=",")
    
    dfbr = dfbr_transform[0:-1]
    # Here, [x',y']T = m*x[x,y]T
    
    data = pd.read_csv(landmarks_path)
    
    new_locations = pd.DataFrame(columns=['x_target', 'y_target'])
    for idx, info in data.iterrows():
        x = info['x_source']
        y = info['y_source']
        mpp_source = info['mpp_source']
        mpp_target = info['mpp_target']
        ds = mpp_target/mpp_source
        # x_prime = ((dfbr[0][0]*x + dfbr[0][1]*y) * ds) + dfbr[0][2]
        # y_prime = ((dfbr[1][0]*x + dfbr[1][1]*y) * ds) + dfbr[1][2]
        x_prime = ((dfbr[0][0]*x + dfbr[0][1]*y) * mpp_source) + dfbr[0][2]
        y_prime = ((dfbr[1][0]*x + dfbr[1][1]*y) * mpp_source) + dfbr[1][2]
        new_locations.loc[idx] = [x_prime, y_prime]
    
    # should be given in microns not pixels!
    
    # create file with columns x_target, y_target and save to output_path (in microns)
    new_locations.to_csv(os.path.join(output_path, "registered_landmarks.csv"), index=False)

    gc.collect() 
    print("Finished landmark registration!")


if __name__ == "__main__":
    landmark_registration()