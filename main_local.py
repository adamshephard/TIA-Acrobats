"""
Main script for registering an input image to a fixed imafe, at the WSI-level
"""
import traceback
import os
import glob
from pathlib import Path
import shutil

from utils import timing
import gc
import subprocess
import pandas as pd

def print_std(p: subprocess.Popen):

    if p.stderr is not None:
        for line in p.stderr.readlines():
            print(line)

    if p.stdout is not None:
        for line in p.stdout.readlines():
            print(line)

@timing
def run_tissue_segmentation(moving_path, fixed_path, output_path, resolution=0.1563):

    print("Running tissue segmentation")
    cmd = [
        "python3",
        "-u",
        "-m",
        "tissue_segmentation",
        f"--moving_image_path={moving_path}",
        f"--fixed_image_path={fixed_path}",
        f"--output_path={output_path}",
        f"--resolution={resolution}",
    ]

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    p.wait()
    print_std(p)
    
@timing
def run_registration(moving_path, fixed_path, intermediate_path, output_path, proc_res=0.1563, out_res=1.25):

    print("running registration")
    cmd = [
        "python3",
        "-u",
        "-m",
        "registration",
        f"--moving_image_path={moving_path}",
        f"--fixed_image_path={fixed_path}",
        f"--intermediate_path={intermediate_path}",
        f"--output_path={output_path}",
        f"--proc_res={proc_res}",
        f"--out_res={out_res}",
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    p.wait()
    print_std(p)
    
@timing
def run_landmark_registration(landmarks_path, moving_image_path, intermediate_path, output_path, out_res=1.25):

    print("running landmark registration")
    cmd = [
        "python3",
        "-u",
        "-m",
        "landmark_registration",
        f"--landmarks_path={landmarks_path}",
        f"--moving_image_path={moving_image_path}",
        f"--intermediate_path={intermediate_path}",
        f"--output_path={output_path}",
        f"--out_res={out_res}",
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    p.wait()
    print_std(p)

def delete_tmp_files(tmp_folder):
    for filename in os.listdir(str(tmp_folder)):
        file_path = os.path.join(str(tmp_folder), filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


class ACROBATICS(object):
    
    # Change below to take input as CSV file...
    
    def __init__(self, input_registration_info="/home/u2271662/tia/projects/acrobat-2023/data/val/validation_set_table.csv",
                       image_folder="/home/u2271662/tia/projects/acrobat-2023/data/val/wsi",#'/input/images/',
                       anno_folder="/home/u2271662/tia/projects/acrobat-2023/data/val/annos",#'/input/images/',
                       output_folder="/home/u2271662/tia/projects/acrobat-2023/data/val/reg-output",#'/output/'):
                       resolution=1.25):
        
        self.input_info = input_registration_info
        self.input_images = image_folder
        self.input_annos = anno_folder
        self.output_folder = output_folder
        self.resolution = resolution

    def process(self):

        """INIT"""
        info_df = pd.read_csv(self.input_info)
        # info_df = info_df.head(8).tail(1)

        # Loop through each image in the input folder (must be tif)
        for _, info in info_df.iterrows():
            moving_path = os.path.join(self.input_images, info['wsi_source'])
            # moving_path = moving_path.replace('.tiff', '.tif') # remember to remove on testing!
            fixed_path = os.path.join(self.input_images, info['wsi_target'])
            # fixed_path = fixed_path.replace('.tiff', '.tif') # remember to remove on testing!
            landmarks_csv = os.path.join(self.input_annos, info['landmarks_csv'])
            output_dir = os.path.join(self.output_folder, str(info['output_dir_name']))   

            os.makedirs(output_dir, exist_ok=True)
            intermediate_output_folder = os.path.join(output_dir, 'tmpoutput')
            os.makedirs(intermediate_output_folder, exist_ok=True)
        
            """RUN"""
            try:
                print("Start Tissue Segmentaton")
                run_tissue_segmentation(moving_path, fixed_path, intermediate_output_folder)   
                print('Start Registration')
                run_registration(moving_path, fixed_path, intermediate_output_folder, output_dir, out_res=self.resolution)   
                print('Finished Registration')
                print('Start CSV Registration')
                run_landmark_registration(landmarks_csv, moving_path, intermediate_output_folder, output_dir, out_res=self.resolution)

            except Exception as e:
                print("Exception")
                print(e)
                print(traceback.format_exc())
            # finally:
                # delete_tmp_files('/tempoutput')
            print('Finished')
            print("--------------")
        

if __name__ == '__main__':
    ACROBATICS().process()