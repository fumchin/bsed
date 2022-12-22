import math
# path related
dataset_root = "../../dataset/ENA"
feature_dir = "../../dataset/ENA/preprocess"
# audio
# mel dim (1255, 128)
sr = 32000
seg_sec = 10
n_window = 2048
hop_size = 255
n_mels = 128
mel_f_min = 0.
mel_f_max = 16000.
max_len_seconds = 10.

max_frames = math.ceil(max_len_seconds * sr / hop_size)
pooling_time_ratio = 4

noise_snr = 30
median_window_s = 0.45
out_nb_frames_1s = sr / hop_size / 4 # 4 for pooling_time_ratio
median_window_s_classwise = [0.45, 0.45, 0.45, 0.45, 0.45, 2.7, 2.7, 2.7, 0.45, 2.7] # [0.3, 0.9, 0.9, 0.3, 0.3, 2.7, 2.7, 2.7, 0.9, 2.7] 
median_window = [max(int(item * out_nb_frames_1s), 1) for item in median_window_s_classwise]
# max_frames = math.ceil(max_len_seconds * sample_rate / hop_size)

# bird list
bird_list = \
[
    "EATO", "WOTH", "BCCH", "BTNW", "TUTI", 
    "NOCA", "REVI", "AMCR", "BLJA", "OVEN", 
    "COYE", "BGGN", "SCTA", "AMRE", "KEWA", 
    "BHCO", "BHVI", "HETH", "RBWO", "BAWW", 
    "HOWA", "NOFL", "AMGO", "CARW", "BWWA", 
    "LOWA", "RCKI", "YBCU", "SWTH", "WBNU"
]