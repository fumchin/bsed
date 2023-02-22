import pandas as pd
import numpy as np
import os, os.path
from glob import glob
import config as cfg

if __name__ == "__main__":
    annotation_dir = cfg.annotation_dir
    annotation_list = glob(os.path.join(annotation_dir, "*.txt"))
    for file_count, annotation_file in enumerate(annotation_list):
        current_df = pd.read_csv(annotation_file, sep="\t")
        species_set =  set(list(current_df["Species"]))
        print(species_set)