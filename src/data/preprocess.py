import os, os.path
import glob
from glob import glob
import threading

import librosa
import pandas as pd
import numpy as np


import config as cfg

def preprocess(audio, compute_log=False):
    ham_win = np.hamming(cfg.n_window)
    mel_min_max_freq = (cfg.mel_f_min, cfg.mel_f_max)
    spec = librosa.stft(
        audio,
        n_fft=cfg.n_window,
        hop_length=cfg.hop_size,
        window=ham_win,
        center=True,
        pad_mode='reflect'
    )
    
    mel_spec = librosa.feature.melspectrogram(
        S=np.abs(spec),  # amplitude, for energy: spec**2 but don't forget to change amplitude_to_db.
        sr=cfg.sr,
        n_mels=cfg.n_mels,
        fmin=mel_min_max_freq[0], 
        fmax=mel_min_max_freq[1],
        htk=False, 
        norm=None
    )
    
    if compute_log:
        mel_spec = librosa.amplitude_to_db(mel_spec)  # 10 * log10(S**2 / ref), ref default is 1
    mel_spec = mel_spec.T
    mel_spec = mel_spec.astype(np.float32)
    
    return mel_spec

def overlap(df, time):
    if(len(df.index) == 0):
        return df
    
    df_overlap = df.loc[(df["Begin Time (s)"] < time) & (df["End Time (s)"] > time)]
    df_non_overlap = df.loc[~((df["Begin Time (s)"] < time) & (df["End Time (s)"] > time))]
    
    df_overlap = pd.concat([df_overlap]*2, ignore_index=True)
    df_overlap = df_overlap.sort_values(by=['Species'])
    
    for i in range(len(df_overlap.index)):
        if (i % 2) == 0:
            df_overlap.loc[i, "End Time (s)"] = time - 10**(-6)
        else:
            df_overlap.loc[i , "Begin Time (s)"] = time
    
    df_result = pd.concat([df_non_overlap, df_overlap], axis=0, ignore_index=True)
    
    return df_result

if __name__ == '__main__':
    dataset_root = cfg.dataset_root
    
    annotation_path = os.path.join(dataset_root, "annotation")
    recording_path = os.path.join(dataset_root, "wav")
    domain_name_list = [name for name in os.listdir(annotation_path) if "Recording" in name]
    
    saved_path = os.path.join(cfg.dataset_root, "preprocess")
    
    mel_saved_path = os.path.join(saved_path, "wav")
    annotation_saved_path = os.path.join(saved_path, "annotation")
    
    if not os.path.exists(mel_saved_path):
        os.makedirs(mel_saved_path)
    
    if not os.path.exists(annotation_saved_path):
        os.makedirs(annotation_saved_path)
        
        
    # iterate through all domain and preprocess the data inside
    for domain_name in domain_name_list:
        current_annotation_path = os.path.join(annotation_path, domain_name)
        current_recording_path = os.path.join(recording_path, domain_name)
        
        audio_files_path_list = glob(os.path.join(current_recording_path, "*.wav"))
        # iterate through all wave files in the domain folder
        for current_audio_files_path in audio_files_path_list:
            # find corresponding annotation file
            # 1. get base name and eliminate extention
            wav_name = os.path.splitext(os.path.basename(current_audio_files_path))[0]
            current_annotation_file_path = glob(os.path.join(current_annotation_path, wav_name + "*.txt"))[0]
            
            audio, sr = librosa.load(current_audio_files_path, sr=cfg.sr)
            annotation_df = pd.read_csv(current_annotation_file_path, sep="\t")
            
            # eliminate the bird not in the bird list
            annotation_df = annotation_df[annotation_df["Species"].isin(cfg.bird_list)]
            
            # cut audio data every 10 sec, puting into list
            audio_seg_list = librosa.util.frame(audio, frame_length=cfg.seg_sec*sr, hop_length=cfg.seg_sec*sr, axis=0)
            
            
            df_current = annotation_df[["Begin Time (s)", "End Time (s)", "Species"]]
            for count, audio_seg in enumerate(audio_seg_list):
                # 10 sec every time
                # mel spectrogram
                mel = preprocess(audio=audio_seg, compute_log=False)
                
                # get corresponding annotation
                current_time_min = count * cfg.seg_sec
                current_time_max = (count + 1) * cfg.seg_sec
                
                # cases that cross the segment
                df_current = overlap(df = df_current, time = current_time_max)
                # df_overlap = df_current.loc[(df_current["Begin Time (s)"] < current_time_max) & (df_current["End Time (s)"] > current_time_max)]
                
                
                df_current_filter = df_current.loc[(df_current["Begin Time (s)"] >= current_time_min) & (df_current["End Time (s)"] < current_time_max)]
                df_current_filter["Begin Time (s)"] = df_current_filter["Begin Time (s)"] - current_time_min
                df_current_filter["End Time (s)"] = df_current_filter["End Time (s)"] - current_time_min
                
                df_current_filter = df_current_filter.sort_values(by=['Begin Time (s)'])
                df_current_filter = df_current_filter.drop_duplicates()
                # save mel spectrogram
                np.save(os.path.join(mel_saved_path, wav_name + "_" + str(count)), mel)
                
                # save annotation file
                df_current_filter.to_csv(os.path.join(annotation_saved_path, wav_name + "_" + str(count) + ".txt"), sep="\t", index=False)
            
            # print(domain_name + " done")
    print("end")