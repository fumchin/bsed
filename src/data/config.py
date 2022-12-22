# path related
dataset_root = "../../dataset/ENA"

# audio
# mel dim (1255, 128)
sr = 32000
seg_sec = 10
n_window = 2048
hop_size = 255
n_mels = 128
mel_f_min = 0.
mel_f_max = 16000.
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