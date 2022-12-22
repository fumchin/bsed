import os
import os.path
import numpy as np
import pandas as pd
from glob import glob
import torch
import data.config as cfg
from data.Transforms import get_transforms
# import config as cfg
# from Transforms import get_transforms

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class ENA_Dataset(Dataset):
    def __init__(self, preprocess_dir, transform, compute_log=False):
        self.sample_rate = cfg.sr
        self.preprocess_dir = preprocess_dir
        # self.encode_function = encode_function
        self.pooling_time_ratio = cfg.pooling_time_ratio
        self.n_frames = cfg.max_frames // self.pooling_time_ratio
        self.hop_size = cfg.hop_size
        self.transform = transform
        
        self.annotation_dir = os.path.join(self.preprocess_dir, "annotation")
        self.feature_dir = os.path.join(self.preprocess_dir, "wav")
        self.feature_file_list = glob(os.path.join(self.feature_dir, "*.npy"))
        self.labels = cfg.bird_list
        
    def __len__(self):
        return len(self.feature_file_list)
        
    def __getitem__(self, index):
        # get selected file
        selected_file_path = self.feature_file_list[index]
        features = np.load(selected_file_path)
        
        # get its annotation
        feature_file_name = os.path.splitext(os.path.basename(selected_file_path))[0]
        annotation_file_path = glob(os.path.join(self.annotation_dir, feature_file_name + ".txt"))[0]
        # read with pandas
        df = pd.read_csv(annotation_file_path, sep="\t")
        target = self.encode(df)
        if self.transform:
            sample = self.transform((features, target))
        return sample
        
        
        pass
    
    def encode(self, label_df):
        """Encode a list (or pandas Dataframe or Serie) of strong labels, they correspond to a given filename

        Args:
            label_df: pandas DataFrame or Series, contains filename, onset (in frames) and offset (in frames)
                If only filename (no onset offset) is specified, it will return the event on all the frames
                onset and offset should be in frames
        Returns:
            numpy.array
            Encoded labels, 1 where the label is present, 0 otherwise
        """

        # assert self.n_frames is not None, "n_frames need to be specified when using strong encoder"
        # if type(label_df) is str:
        #     if label_df == 'empty':
        #         y = np.zeros((self.n_frames, len(self.labels))) - 1
        #         return y
        y = np.zeros((self.n_frames, len(self.labels)))
        if type(label_df) is pd.DataFrame:
            # if {"onset", "offset", "event_label"}.issubset(label_df.columns):
            for _, row in label_df.iterrows():
                i = self.labels.index(row["Species"])
                onset = int(row["Begin Time (s)"] * self.sample_rate // self.hop_size // self.pooling_time_ratio)
                offset = int(row["End Time (s)"] * self.sample_rate // self.hop_size // self.pooling_time_ratio)
                y[onset:offset, i] = 1  # means offset not included (hypothesis of overlapping frames, so ok)
        return y
        # elif type(label_df) in [pd.Series, list, np.ndarray]:  # list of list or list of strings
        #     if type(label_df) is pd.Series:
        #         if {"onset", "offset", "event_label"}.issubset(label_df.index):  # means only one value
        #             if not pd.isna(label_df["event_label"]):
        #                 i = self.labels.index(label_df["event_label"])
        #                 onset = int(label_df["onset"])
        #                 offset = int(label_df["offset"])
        #                 y[onset:offset, i] = 1
        #             return y

        #     for event_label in label_df:
        #         # List of string, so weak labels to be encoded in strong
        #         if type(event_label) is str:
        #             if event_label is not "":
        #                 i = self.labels.index(event_label)
        #                 y[:, i] = 1

        #         # List of list, with [label, onset, offset]
        #         elif len(event_label) == 3:
        #             if event_label[0] is not "":
        #                 i = self.labels.index(event_label[0])
        #                 onset = int(event_label[1])
        #                 offset = int(event_label[2])
        #                 y[onset:offset, i] = 1

        #         else:
        #             raise NotImplementedError("cannot encode strong, type mismatch: {}".format(type(event_label)))

        # else:
        #     raise NotImplementedError("To encode_strong, type is pandas.Dataframe with onset, offset and event_label"
        #                             "columns, or it is a list or pandas Series of event labels, "
        #                             "type given: {}".format(type(label_df)))
        # return y
if __name__ == "__main__":
    file_name = "Fuji_train_list.txt"
    root = "/home/fumchin/data/cv/final/dataset"
    n_channel = 1
    add_axis_conv = 0
    transforms = get_transforms(cfg.max_frames, None, add_axis_conv,
                                noise_dict_params={"mean": 0., "snr": cfg.noise_snr})
    ENA = ENA_Dataset(preprocess_dir="../../dataset/ENA/preprocess", transform=transforms, compute_log=False)
    train_dataloader = DataLoader(ENA, batch_size=6, shuffle=True)
    a, b = next(iter(train_dataloader))
    print(a, b)
    print('f')