# IMPORTS
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openslide import OpenSlide
from tiatoolbox.wsicore.wsireader import WSIReader

# CONSTANTS
path_wsis_base = '/home/u2271662/tia/projects/acrobat-2023/data/val/wsi'
path_df = '/home/u2271662/tia/projects/acrobat-2023/data/val/acrobat_validation_points_public_1_of_1.csv'
reg_out_folder = '/home/u2271662/tia/projects/acrobat-2023/data/val/reg-output-20-08-23-01'
save_folder = os.path.join(reg_out_folder, 'reg_landmarks')
level = 1

# FUNCTIONS
def load_main_landmarks_file(csv_path):
    df = pd.read_csv(csv_path).sort_values(by=['anon_id'])
    return df

def add_he_landmarks_to_file(df, reg_out_folder):
    # Get all csv files in registered landmarks folder named 'registered_landmarks.csv' and fill in he_x and he_y columns in df
    for idx, row in df.iterrows():
        anon_id = row['anon_id']
        point_id = row['point_id']
        # print(f"Processing {anon_id} {point_id}")
        _registered_landmarks_path = os.path.join(reg_out_folder, str(anon_id))
        registered_landmarks_path = os.path.join(_registered_landmarks_path, 'registered_landmarks.csv')
        # Check if file exists and if not, skip
        if not os.path.exists(registered_landmarks_path):
            continue
        registered_landmarks_df = pd.read_csv(registered_landmarks_path)
        he_x = registered_landmarks_df.loc[registered_landmarks_df['point_id'] == point_id]['he_x'].values[0]
        he_y = registered_landmarks_df.loc[registered_landmarks_df['point_id'] == point_id]['he_y'].values[0]
        df.loc[(df['anon_id'] == anon_id) & (df['point_id'] == point_id), 'he_x'] = he_x
        df.loc[(df['anon_id'] == anon_id) & (df['point_id'] == point_id), 'he_y'] = he_y
    return df

def display_landmark_patch(df, patch_size=1000, level=1, limit=10):
    for i in range(len(df) if limit is None else limit):
        # Get first row of the dataframe (not random)
        row = df.iloc[i]
        print(row['anon_filename_ihc'])
        path_ihc = os.path.join(path_wsis_base, os.path.splitext(row['anon_filename_ihc'])[0] + '.tiff')
        ihc_x = int(row['ihc_x']/(row['mpp_ihc_10X']) - size*2**level/2)
        ihc_y = int(row['ihc_y']/(row['mpp_ihc_10X']) - size*2**level/2)
        print(ihc_x, ihc_y)
        wsi_ihc = OpenSlide(path_ihc)
        img_ihc = wsi_ihc.read_region((ihc_x, ihc_y), level, (size, size))    
        
        if (not pd.isnull(row['he_x'])) & (not pd.isnull(row['he_y'])):
            path_he = os.path.join(path_wsis_base, os.path.splitext(row['anon_filename_he'])[0] + '.tiff')
            he_x = int(row['he_x']/(row['mpp_he_10X']) - size*2**level/2)
            he_y = int(row['he_y']/(row['mpp_he_10X']) - size*2**level/2)
            wsi_he = OpenSlide(path_he)
            img_he = wsi_he.read_region((he_x, he_y), level, (size, size))

            fig, axs = plt.subplots(1,2, dpi=300)
            ax = plt.subplot(121)
            ax.imshow(img_ihc)
            ax.scatter(int(size/2), int(size/2))
            plt.axis('off')
            ax = plt.subplot(122)
            ax.imshow(img_he)
            ax.scatter(int(size/2), int(size/2))
            plt.axis('off')
            plt.show()
        else:
            fig, axs = plt.subplots(1,2, dpi=300)
            ax = plt.subplot(121)
            ax.imshow(img_ihc)
            ax.scatter(int(size/2), int(size/2))
            plt.axis('off')
            ax = plt.subplot(122)
            ax.imshow(np.zeros(img_ihc.size))
            plt.axis('off')
            plt.show()

def display_wsi_landmarks(df, anon_id, level=1, marker_size=0.25, save_path=None):
    print('Processing anon_id', anon_id)
    subset = df.loc[df['anon_id'] == anon_id]

    ihc_landmarks = []
    he_landmarks = []

    path_ihc = os.path.join(path_wsis_base, os.path.splitext(subset.iloc[0]['anon_filename_ihc'])[0] + '.tiff')
    path_he = os.path.join(path_wsis_base, os.path.splitext(subset.iloc[0]['anon_filename_he'])[0] + '.tiff')
    ihc_reader = WSIReader.open(path_ihc)
    he_reader = WSIReader.open(path_he)

    ihc_img = ihc_reader.slide_thumbnail(resolution=level, units='level')
    he_img = he_reader.slide_thumbnail(resolution=level, units='level')

    scale_factor = 2**level
    for i in range(0, len(subset)):
        row = subset.iloc[i]
        ihc_x = int(row['ihc_x']/(row['mpp_ihc_10X'] * scale_factor))
        ihc_y = int(row['ihc_y']/(row['mpp_ihc_10X'] * scale_factor))
        he_x = int(row['he_x']/(row['mpp_he_10X'] * scale_factor))
        he_y = int(row['he_y']/(row['mpp_he_10X'] * scale_factor))
        
        ihc_landmarks.append((ihc_x, ihc_y))
        he_landmarks.append((he_x, he_y))

    fig, axs = plt.subplots(1,2, dpi=300)
    ax = plt.subplot(121)
    ax.imshow(ihc_img)
    ax.scatter([x[0] for x in ihc_landmarks], [x[1] for x in ihc_landmarks], s=marker_size, marker="+")
    plt.axis('off')
    ax = plt.subplot(122)
    ax.imshow(he_img)
    ax.scatter([x[0] for x in he_landmarks], [x[1] for x in he_landmarks], s=marker_size, marker="+")
    plt.axis('off')
    if save_path is None: 
        plt.show()
    else:
        save_path = os.path.join(save_path, f"{anon_id}_landmarks.png")
        # Create folder if it doesn't exist
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

def export_all_wsi_landmarks(df, level=1, marker_size=0.25, save_path=None):
    for anon_id in df['anon_id'].unique():
        display_wsi_landmarks(df, anon_id, level=level, marker_size=marker_size, save_path=save_path)

# MAIN
if __name__ == "__main__":
    df = load_main_landmarks_file(path_df)
    df = add_he_landmarks_to_file(df, reg_out_folder)
    # display_wsi_landmarks(df, 0, save_path=save_folder)
    export_all_wsi_landmarks(df, save_path=save_folder)
