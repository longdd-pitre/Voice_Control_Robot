from dataset import buildDataDictForMFCC
from model import GMMHMMModel
from train import train_GMMHMM
from config import Config
from dataset import load_data
from preprocess_data import calculate_mfcc_feature, feature_normalize, reduce_mfccs_state
from test import test_GMMHMM

c = Config()


def main():
    # Chuẩn bị dữ liệu, tính toán đặc trưng
    all_member = ['../data/']
    df = load_data(all_member)
    df = calculate_mfcc_feature(df)
    df = feature_normalize(df)
    df = reduce_mfccs_state(df)

    # Mô hình ban đầu định dạng dữ liệu đào tạo/ Kiểm tra
    SpeechModel = GMMHMMModel(n_states=c.n_states, covariance_type=c.covariance_type)
    trainDataDict, testDataDict = buildDataDictForMFCC(df, c.train_size)

    # Giai đoạn đào tạo
    SpeechModel = train_GMMHMM(SpeechModel, trainDataDict)

    print(SpeechModel.models.keys())

    # Giai đoạn test
    test_GMMHMM(SpeechModel, testDataDict)

    # Lưu model
    SpeechModel.save('../output/model.pkl')


if __name__ == '__main__':
    main()