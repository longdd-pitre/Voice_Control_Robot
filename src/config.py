class Config:
    train_size = 0.8
    n_states = 5  # Số trạng thái ẩn
    covariance_type = 'diag'  # Loại ma trận hiệp phương sai
    n_mfcc = 13
    hop_length = 220
    n_frame = 12
    sr = 22050
    n_fft = 512
    state_dict = {
    'phai': 45,
    'trai': 50,    
    'di': 46,  
    'thang': 39,
    'lui': 47,
    're': 46,
    'sil': 25
    }
    