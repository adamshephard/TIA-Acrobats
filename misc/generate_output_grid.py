# Generate a photo grid of the output images whose filename end with '_overlay_post_registration.png'

# %%
# IMPORTS
import pathlib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import image as mpimg
from matplotlib import patches as mpatches
from matplotlib import colors
from matplotlib import cm
from matplotlib import rcParams
from matplotlib import rc
from matplotlib import font_manager
from matplotlib import ticker
from matplotlib import transforms
from matplotlib import animation


# %%
# CONSTANTS
path = "/home/u2271662/tia/projects/acrobat-2023/data/val/reg-output/"
output_path = "/home/u2271662/tia/projects/acrobat-2023/data/val/reg-output/output.csv"
output_grid_path = "/home/u2271662/tia/projects/acrobat-2023/data/val/reg-output/output_grid.png"


# %%
# FUNCTIONS
def find_png_files(path, filename="*.png"):
    files = list(pathlib.Path(path).glob('**/{0}'.format(filename)))
    sorted_files = sorted(files, key=lambda x: int(x.name.split('/')[-1].split('_')[0]))
    return sorted_files


def combine_png_files(png_files, output_path=None):
    # Combine all files in the list (don't include first row in each file)
    combined_png = np.concatenate([mpimg.imread(f) for f in png_files[:]], axis=1)
    if output_path:
        # Export to png
        mpimg.imsave(output_path, combined_png)
    else:
        return combined_png

def generate_grid(png_files, output_path=None):
    # Generate grid of images 
    # Number of rows = number of images / 4

    # Set up figure
    fig = plt.figure(figsize=(50, 50))
    gs = gridspec.GridSpec(10, 10, figure=fig)
    gs.update(wspace=0.025, hspace=0.025)
    for i, png_file in enumerate(png_files):
        ax = plt.subplot(gs[i])
        ax.imshow(mpimg.imread(png_file))
        ax.set_title(os.path.basename(png_file).split("_")[0], fontsize=20)
        ax.axis('off')
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    else:
        plt.show()

# %% 
# MAIN
if __name__ == "__main__":
    files = find_png_files(path, filename="*_overlay_post_registration.png")
    generate_grid(files, output_path=output_grid_path)