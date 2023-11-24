import os
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
import librosa
from config import Config

c = Config()

def reduce_mfccs_state(df):
    def kmean_reduce(mfcc_norma, label, state_dict):

        if label == 'sil':
            return mfcc_norma

        n_state, b = mfcc_norma.shape
        if n_state > state_dict[label]:
            n_state = state_dict[label]

        # Gom nhóm các frame sử dụng KMeans
        clustering = KMeans(n_clusters=n_state, n_init=12, random_state=0)
        clustering.fit(mfcc_norma)
        state = clustering.labels_

        # Xác định các frame giống nhau
        idx_arr_of_states = []
        start_idx = 0

        for i in range(1, len(state)):
            if state[i] != state[i - 1]:
                idx_arr_of_states.append(np.arange(start_idx, i))
                start_idx = i

        idx_arr_of_states.append(np.arange(start_idx, len(state)))

        # Tính mean của các frame giống nhau
        mfcc_feat = [np.mean(mfcc_norma[arr], axis=0) for arr in idx_arr_of_states]

        return np.array(mfcc_feat)

    df['mfcc_feat'] = df.apply(lambda x: kmean_reduce(x.mfcc_norma, x.label, c.state_dict), axis=1)
    return df

def feature_normalize(df):
    def normalize_mfcc(mfcc):
        return librosa.util.normalize(mfcc, norm=1)
    df['mfcc_norma'] = df['mfcc_origin'].apply(normalize_mfcc)
    return df


def calculate_mfcc_feature(df):
    def extract_mfcc_feature(x, sound, sr, hop_length):
        # Kiểm tra thời điểm bắt đầu và kết thúc hợp lý
        start_sample = int(x.loc["start"] * sr)
        end_sample = int(x.loc["end"] * sr)

        if start_sample < 0:
            start_sample = 0
        if end_sample > len(sound):
            end_sample = len(sound)

        # Tính toán MFCC
        mfcc = librosa.feature.mfcc(y=sound[start_sample:end_sample], sr=sr, hop_length=hop_length)
        delta = librosa.feature.delta(mfcc, width=3)
        delta_2 = librosa.feature.delta(mfcc, order=2, width=3)

        # Kết hợp đặc trưng
        features = np.concatenate((mfcc, delta, delta_2), axis=0)

        return features.T

    fids = df['fid'].unique()
    for fid in fids:
        sound_file_path = os.path.join(fid+".wav")
        # optimize calculation speed by read files only once
        sound, sr = librosa.load(sound_file_path)
        dfi = df[df["fid"] == fid]

        
        # 'apply' function is used dataframe along data vertical axis
        df.loc[df["fid"] == fid, "mfcc_origin"] = dfi.apply(extract_mfcc_feature, args=(sound, sr, c.hop_length), axis=1)
    return df