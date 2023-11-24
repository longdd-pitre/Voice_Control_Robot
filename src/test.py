import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from dataset import prepare_test_data


def test_GMMHMM(SpeechModel, testDataDict):
    pairOfTestData = prepare_test_data(testDataDict)
    true_label = []
    pred_label = []

    for l, mfcc_feat in pairOfTestData:
        scoreList = {}
        for model_label in SpeechModel.models.keys():
            model = SpeechModel.models[model_label]
            try:
                score = model.score(mfcc_feat)
                scoreList[model_label] = score
            except Exception as e:
                print(f"Error scoring model {model_label}: {e}")
        predict = max(scoreList, key=scoreList.get)

        true_label.append(l)
        pred_label.append(predict)

    # Đánh giá kết quả sau khi hoàn thành vòng lặp
    true_label = np.array(true_label)
    pred_label = np.array(pred_label)

    accuracy = np.mean(true_label == pred_label)
    print(f"Accuracy: {accuracy}")
    print(classification_report(true_label, pred_label))