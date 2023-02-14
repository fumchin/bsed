import pandas as pd
import os, os.path
from glob import glob
import librosa
import shutil
import json
import numpy as np
import scaper
from desed.generate_synthetic import SoundscapesGenerator
from desed.logger import create_logger
from desed.post_process import rm_high_polyphony, post_process_txt_labels
from desed.utils import create_folder
import data_config as cfg

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

def syn_preprocess(generated_folder, syn_preprocess_folder):
    preprocess_mel_folder = os.path.join(syn_preprocess_folder, "wav")
    preprocess_annotation_folder = os.path.join(syn_preprocess_folder, "annotation")
    
    if not os.path.exists(preprocess_mel_folder):
        os.makedirs(preprocess_mel_folder)
    if not os.path.exists(preprocess_annotation_folder):
        os.makedirs(preprocess_annotation_folder)

    audio_file_list = glob(os.path.join(generated_folder, "*.wav"))
    annotation_file = glob(os.path.join(generated_folder, "output.tsv"))[0]
    annotation_df = pd.read_csv(annotation_file, sep="\t")
    # df = pd.read_csv('output.tsv', sep="\t")
    for file_count, audio_file_path in enumerate(audio_file_list):

        audio_file = os.path.basename(audio_file_path)
        audio_file_name = os.path.splitext(audio_file)[0]

        audio, sr = librosa.load(audio_file_path, sr=cfg.sr)
        mel = preprocess(audio=audio, compute_log=False)
        np.save(os.path.join(preprocess_mel_folder, audio_file_name), mel)

        
        print(audio_file)
        df_extract = annotation_df.loc[annotation_df["filename"] == audio_file]
        df_extract = df_extract.drop(["filename"],axis=1)
        df_extract.to_csv(os.path.join(preprocess_annotation_folder, audio_file_name), sep="\t", index=False)
    # print(annotation_df)
    # print(audio_file_list)
    pass

if __name__ == '__main__':
    # synth_annotation_file = os.path.join(cfg.synth_annotation_dir, "nips4b_birdchallenge_train_labels.csv")
    # synth_wav_file_dir = os.path.join(cfg.synth_audio_dir)
    nip4_dataset_root = "/home/fumchin/data/bsed/dataset/NIP4"
    nip4_annotation_file = os.path.join(nip4_dataset_root, "annotation","nips4b_birdchallenge_train_labels.csv")
    nip4_audio_dir = os.path.join(nip4_dataset_root, "NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV", "train")

    # bg_folder = os.path.join("/home/fumchin/data/bsed/dataset", "SYN", "background")
    # fg_folder = os.path.join("/home/fumchin/data/bsed/dataset", "SYN", "foreground")
    syn_folder = cfg.syn_folder
    bg_folder = cfg.bg_folder
    fg_folder = cfg.fg_folder
    generated_folder = os.path.join(syn_folder,  "generated")
    out_tsv = os.path.join(generated_folder, "output.tsv")
    default_json_path = os.path.join(syn_folder, "metadata", "event_occurences", "event_occurences_train.json")
    
    
    # if not os.path.exists(default_json_path):
    #     os.makedirs(default_json_path)

    with open(default_json_path) as json_file:
        co_occur_dict = json.load(json_file)
    
    
    
    if not os.path.exists(bg_folder):
        os.makedirs(bg_folder)

        df = pd.read_csv(nip4_annotation_file, skiprows=2)
        df_empty = df[df["Empty"]==1]
        empty_filename_list = df_empty["Filename"].tolist()
        print(empty_filename_list)

        for file_count, empty_filename in enumerate(empty_filename_list):
            empty_file_path = os.path.join(nip4_audio_dir, empty_filename)
            shutil.copy(empty_file_path, bg_folder)
    else:
        print("backgreound folder already created")

    clip_duration = cfg.clip_duration
    sample_rate = cfg.sr
    ref_db = cfg.ref_db
    n_soundscapes = cfg.n_soundscapes
    random_state = cfg.random_seed
    # pitch_shift = cfg.pitch_shift

    # OUTPUT FOLDER
    outfolder = generated_folder
    if not os.path.exists(generated_folder):
        os.makedirs(generated_folder)
        sg = SoundscapesGenerator(duration=clip_duration,
                                fg_folder=fg_folder,
                                bg_folder=bg_folder,
                                ref_db=ref_db,
                                samplerate=sample_rate,
                                random_state=random_state)
        sg.generate_by_label_occurence(label_occurences=co_occur_dict,
                                        number=n_soundscapes,
                                        out_folder=generated_folder,
                                        save_isolated_events=True)

        # ##
        # Post processing
        rm_high_polyphony(generated_folder, max_polyphony=2)
        # concat same labels overlapping
        post_process_txt_labels(generated_folder,
                                output_folder=generated_folder,
                                output_tsv=out_tsv, rm_nOn_nOff=True)
    else:
        print("synthetic data already generated")
    
    
    syn_preprocess(generated_folder, cfg.syn_preprocess_folder)
