import pandas as pd
import os, os.path
import shutil
import json
import numpy as np
import scaper
from desed.generate_synthetic import SoundscapesGenerator
from desed.logger import create_logger
from desed.post_process import rm_high_polyphony, post_process_txt_labels
from desed.utils import create_folder
import data_config as cfg

if __name__ == '__main__':
    # synth_annotation_file = os.path.join(cfg.synth_annotation_dir, "nips4b_birdchallenge_train_labels.csv")
    # synth_wav_file_dir = os.path.join(cfg.synth_audio_dir)
    synth_dataset_root = "/home/fumchin/data/bsed/dataset/NIP4"
    synth_annotation_file = os.path.join(synth_dataset_root, "annotation","nips4b_birdchallenge_train_labels.csv")
    synth_audio_dir = os.path.join(synth_dataset_root, "NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV", "train")

    bg_folder = os.path.join("/home/fumchin/data/bsed/dataset", "SYN", "background")
    fg_folder = os.path.join("/home/fumchin/data/bsed/dataset", "SYN", "foreground")
    generated_folder = os.path.join("/home/fumchin/data/bsed/dataset", "SYN", "generated")
    out_tsv = os.path.join(generated_folder, "output.tsv")
    default_json_path = os.path.join("/home/fumchin/data/bsed/dataset", "SYN", "metadata", "event_occurences", "event_occurences_train.json")
    
    
    # if not os.path.exists(default_json_path):
    #     os.makedirs(default_json_path)

    with open(default_json_path) as json_file:
        co_occur_dict = json.load(json_file)
    
    if not os.path.exists(generated_folder):
        os.makedirs(generated_folder)
    
    if not os.path.exists(bg_folder):
        os.makedirs(bg_folder)

        df = pd.read_csv(synth_annotation_file, skiprows=2)
        df_empty = df[df["Empty"]==1]
        empty_filename_list = df_empty["Filename"].tolist()
        print(empty_filename_list)

        for file_count, empty_filename in enumerate(empty_filename_list):
            empty_file_path = os.path.join(synth_audio_dir, empty_filename)
            shutil.copy(empty_file_path, bg_folder)
    else:
        print("backgreound folder already created")

    clip_duration = cfg.clip_duration
    sample_rate = cfg.sample_rate
    ref_db = cfg.ref_db
    n_soundscapes = cfg.n_soundscapes
    random_state = cfg.random_seed
    # pitch_shift = cfg.pitch_shift

    # OUTPUT FOLDER
    outfolder = generated_folder

    # n_soundscapes = 1000
    # ref_db = -50
    # duration = 10.0 

    # min_events = 1
    # max_events = 9

    # event_time_dist = 'truncnorm'
    # event_time_mean = 5.0
    # event_time_std = 2.0
    # event_time_min = 0.0
    # event_time_max = 10.0

    # source_time_dist = 'const'
    # source_time = 0.0

    # event_duration_dist = 'uniform'
    # event_duration_min = 0.5
    # event_duration_max = 4.0

    # snr_dist = 'uniform'
    # snr_min = 6
    # snr_max = 30

    # pitch_dist = 'uniform'
    # pitch_min = -3.0
    # pitch_max = 3.0

    # time_stretch_dist = 'uniform'
    # time_stretch_min = 0.8
    # time_stretch_max = 1.2
        
    # # Generate 1000 soundscapes using a truncated normal distribution of start times

    # for n in range(n_soundscapes):
        
    #     print('Generating soundscape: {:d}/{:d}'.format(n+1, n_soundscapes))
        
    #     # create a scaper
    #     sc = scaper.Scaper(duration, fg_folder, bg_folder)
    #     sc.protected_labels = []
    #     sc.ref_db = ref_db
        
    #     # add background
    #     sc.add_background(label=('const', 'NIP4'), 
    #                     source_file=('choose', []), 
    #                     source_time=('const', 0))

    #     # add random number of foreground events
    #     n_events = np.random.randint(min_events, max_events+1)
    #     for _ in range(n_events):
    #         sc.add_event(label=('choose', []), 
    #                     source_file=('choose', []), 
    #                     source_time=(source_time_dist, source_time), 
    #                     event_time=(event_time_dist, event_time_mean, event_time_std, event_time_min, event_time_max), 
    #                     event_duration=(event_duration_dist, event_duration_min, event_duration_max), 
    #                     snr=(snr_dist, snr_min, snr_max),
    #                     pitch_shift=(pitch_dist, pitch_min, pitch_max),
    #                     time_stretch=(time_stretch_dist, time_stretch_min, time_stretch_max))
        
    #     # generate
    #     audiofile = os.path.join(outfolder, "soundscape_unimodal{:d}.wav".format(n))
    #     jamsfile = os.path.join(outfolder, "soundscape_unimodal{:d}.jams".format(n))
    #     txtfile = os.path.join(outfolder, "soundscape_unimodal{:d}.txt".format(n))
        
    #     sc.generate(audiofile, jamsfile,
    #                 allow_repeated_label=True,
    #                 allow_repeated_source=False,
    #                 reverb=0.1,
    #                 disable_sox_warnings=True,
    #                 no_audio=False,
    #                 txt_path=txtfile)
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
