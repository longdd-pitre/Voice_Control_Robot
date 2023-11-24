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
from pathlib import Path

class SpeechProcessing:
    def __init__(self):
        self.config = Config()

    def split_sounds_in_folder(self, file_path):
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
            #print("Exporting file", output_file)
            chunk.export(output_file, format="wav")

        return f"../data/user_data/{folder_name}"

    def load_model(self, model_path):
        SpeechModel = GMMHMMModel(n_states=self.config.n_states, covariance_type=self.config.covariance_type)
        SpeechModel.load(model_path)
        return SpeechModel

    def get_prediction(self, speech_model, sound_path):
        def normalize_mfcc(mfcc):
            feats = copy.deepcopy(mfcc)
            return librosa.util.normalize(feats, norm=1)

        def reduce_state(mfcc, label):
            feats = copy.deepcopy(mfcc)

            n_state, b = feats.shape
            if n_state > self.config.state_dict[label]:
                n_state = self.config.state_dict[label]

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

        folder_name = self.split_sounds_in_folder(sound_path)
        folder = Path(folder_name)

        file_paths = [file_path for file_path in folder.glob("*.wav")]
        sorted_paths = sorted(file_paths)

        words = []

        for audio_file in sorted_paths:
            y, sr = librosa.load(audio_file)
            mfcc = librosa.feature.mfcc(y=y, sr=self.config.sr, hop_length=self.config.hop_length)
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
                try:
                    score = model.score(feat_reduce)
                    scoreList[model_label] = score
                except Exception as e:
                    print(f"Error scoring model `{model_label}`: {e}")

            predict = max(scoreList, key=scoreList.get)
            words.append(predict)

        return words

    def run(self):
        model_path = '../output/model.pkl' 
        sound_path = '../testdata/2.wav'
        model = self.load_model(model_path)
        words = self.get_prediction(model, sound_path)
        print("Giọng nói chuyển thành văn bản là:",words)
        sentence = ' '.join(map(str, words))
        tokenizer = CustomTokenizer()
        tokens = tokenizer.tokenize(sentence)
        result = ', '.join(map(str, tokens))
        return result

if __name__ == '__main__':
    speech_processor = SpeechProcessing()
    tokens = speech_processor.run()
    print("Tín hiệu truyền đến robot là: ",tokens)
