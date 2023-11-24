import os
import shutil
import copy
import librosa
import numpy as np
from sklearn.cluster import KMeans
from pydub import AudioSegment
from pydub.silence import split_on_silence
from model import GMMHMMModel
from config import Config
from CustomTokenizer import CustomTokenizer
import pyaudio
import wave
from pathlib import Path
import tempfile
import threading
import speech_recognition as sr

c = Config()


def record_audio_to_wav(duration=4):
    recognizer = sr.Recognizer()

    frames = []
    recording = True

    def stop_recording():
        nonlocal recording
        print("Stopping recording...")
        recording = False

    stop_timer = threading.Timer(duration, stop_recording)
    stop_timer.start()

    with sr.Microphone() as source:
        print("Adjusting for ambient noise. Please wait...")
        recognizer.adjust_for_ambient_noise(source, duration=2)

        print("Recording...")
        while recording:
            audio_chunk = recognizer.listen(source)
            frames.append(audio_chunk.frame_data)

    print("Finished recording!")

    # Thay đổi đường dẫn và tên file theo ý muốn của bạn
    test_output = '../testdata/user_audio.wav'
    os.makedirs(os.path.dirname(test_output), exist_ok=True)

    with wave.open(test_output, 'wb') as wf:
        wf.setnchannels(1)  # Số lượng kênh (mono)
        wf.setsampwidth(2)  # 2 byte cho mỗi mẫu âm thanh
        wf.setframerate(44100)  # Tần số lấy mẫu là 44.1 kHz

        # Chuyển đổi dữ liệu âm thanh từ 32-bit PCM (PaInt32) sang 16-bit PCM (PaInt16)
        pcm_data = b''.join(frames)
        wf.writeframes(pcm_data)

        print(test_output)
        return test_output

def split_sounds_in_folder(file_path):
    file_name = os.path.basename(file_path)
    folder_name = os.path.splitext(file_name)[0]

    if os.path.exists(f"../data/user_data/{folder_name}"):
        shutil.rmtree(f"../data/user_data/{folder_name}")

    os.makedirs(f"../data/user_data/{folder_name}")

    # reading from audio mp3 file
    sound = AudioSegment.from_wav(file_path)

    # splitting audio files
    audio_chunks = split_on_silence(sound, min_silence_len=25, silence_thresh=-21)

    # loop is used to iterate over the output list
    for i, chunk in enumerate(audio_chunks):
        file_number = str(i).zfill(3)
        output_file = f"../data/user_data/{folder_name}/chunk{file_number}.wav"
        print("Exporting file", output_file)
        chunk.export(output_file, format="wav")

    return f"../data/user_data/{folder_name}"


def load_model(model_path):
    SpeechModel = GMMHMMModel(n_states=c.n_states, covariance_type=c.covariance_type)
    SpeechModel.load(model_path)
    return SpeechModel


def get_prediction(speech_model, sound_path):
    def normalize_mfcc(mfcc):
        feats = copy.deepcopy(mfcc)
        return librosa.util.normalize(feats, norm=1)

    def reduce_state(mfcc, label):
        feats = copy.deepcopy(mfcc)

        n_state, b = feats.shape
        if n_state > c.state_dict[label]:
            n_state = c.state_dict[label]

        # Gom nhóm các frame sử dụng KMeans
        clustering = KMeans(n_clusters=n_state, n_init=12, random_state=0)
        clustering.fit(feats)
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
        mfcc_feat = [np.mean(feats[arr], axis=0) for arr in idx_arr_of_states]

        return np.array(mfcc_feat)

    folder_name = split_sounds_in_folder(sound_path)
    folder = Path(folder_name)

    file_paths = [file_path for file_path in folder.glob("*.wav")]
    sorted_paths = sorted(file_paths)

    words = []

    for audio_file in sorted_paths:
        y, sr = librosa.load(audio_file)
        mfcc = librosa.feature.mfcc(y=y, sr=c.sr, hop_length=c.hop_length)
        delta = librosa.feature.delta(mfcc, width=3)
        delta_2 = librosa.feature.delta(mfcc, order=2, width=3)

        # Kết hợp đặc trưng
        features = np.concatenate((mfcc, delta, delta_2), axis=0).transpose()
        feat_norma = normalize_mfcc(features)

        scoreList = {}
        for model_label in speech_model.models.keys():
            if model_label == 'sil':
                continue

            model = speech_model.models[model_label]
            feat_reduce = reduce_state(feat_norma, model_label)
            # print(feat_reduce.shape)
            try:
                score = model.score(feat_reduce)
                scoreList[model_label] = score
            except Exception as e:
                print(f"Error scoring model `{model_label}`: {e}")

        predict = max(scoreList, key=scoreList.get)
        words.append(predict)

    return words


if __name__ == '__main__':
    model_path = '../output/model.pkl'
    model = GMMHMMModel(n_states=c.n_states, covariance_type=c.covariance_type)
    model.load(model_path)
    words = get_prediction(model, record_audio_to_wav(duration = 4))   
    sentence = ' '.join(map(str, words))
    tokenizer = CustomTokenizer()
    
    tokens = tokenizer.tokenize(sentence)
    print(tokens)