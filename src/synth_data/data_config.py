import os, os.path

clip_duration = 10.0
sample_rate = 32000
ref_db = -55
n_soundscapes = 100
random_seed = 2023
# pitch_shift = ("uniform", -3, 3)
bg_folder = os.path.join("/home/fumchin/data/bsed/dataset", "SYN", "background")
fg_folder = os.path.join("/home/fumchin/data/bsed/dataset", "SYN", "foreground")