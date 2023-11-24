import numpy as np
from model import GMMHMMModel
from config import Config

c = Config()


def train_GMMHMM(SpeechModel, dataDict):
    for label, data in dataDict.items():
        SpeechModel.add_model_for_label(label)
        sequences, seq_lengths = data
        SpeechModel.fit_for_label(label, sequences, seq_lengths)

    return SpeechModel


if __name__ == '__main__':
    dataDict = {}
    model = train_GMMHMM(dataDict)
    model.save('../output/model.pkl')

