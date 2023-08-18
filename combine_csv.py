# IMPORTS
import pathlib

import pandas as pd

# CONSTANTS
path = "/home/u2271662/tia/projects/acrobat-2023/data/val/reg-output/"
output_path = "/home/u2271662/tia/projects/acrobat-2023/data/val/reg-output/output.csv"

# FUNCTIONS
def find_csv_files(path, filename="*.csv"):
    return sorted(pathlib.Path(path).glob('**/{0}'.format(filename)))

def combine_csv_files(csv_files, output_path=None):
    # Combine all files in the list (don't include first row in each file)
    combined_csv = pd.concat([pd.read_csv(f, header=None, skiprows=1) for f in csv_files[:]])
    if output_path:
        # Export to csv
        # Add header
        header = ["anon_id","point_id","he_x","he_y"]
        combined_csv.to_csv(output_path, index=False, header=header)
    else:
        return combined_csv

# MAIN
if __name__ == "__main__":
    files = find_csv_files(path, filename="registered_landmarks.csv")
    combine_csv_files(files, output_path=output_path)