import glob
import os
import sklearn
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split

def load_data(all_member):
    df = pd.DataFrame(columns=["fid", 'label', "start", "end"])
    
    for mname in all_member:
        df_mi = pd.DataFrame(columns=["fid", "label", "start", "end"])
        data_dir = mname
        label_fnames = glob.glob(os.path.join(data_dir, "*.txt"))
        label_fnames = [os.path.basename(label_fname) for label_fname in label_fnames]

        for fname in label_fnames:
            df_i = pd.read_csv(os.path.join(data_dir, fname), sep="\t", header=None)
            df_i = df_i.rename(columns={2: "label", 0: "start", 1: "end"})
            df_i["fid"] = [os.path.join(mname, fname.split('.')[0])] * len(df_i)
            df_mi = pd.concat([df_mi, df_i], axis=0, ignore_index=True)

        df = pd.concat([df, df_mi], axis=0, ignore_index=True)

    return df


def buildDataDictForMFCC(df, train_rate):
    trainDataDict = {}
    testDataDict = {}

    set_labels = df['label'].unique()

    for label in set_labels:
        df_temp = df[df['label'] == label]
        msk = np.random.rand(len(df_temp)) < train_rate

        X_tr = df_temp['mfcc_feat'][msk]
        X_te = df_temp['mfcc_feat'][~msk]

        train_seq_lengths = np.array([len(x) for x in X_tr])
        test_seq_lengths = np.array([len(x) for x in X_te])
        X_tr = np.vstack(X_tr)
        X_te = np.vstack(X_te)

        trainDataDict[label] = (X_tr, train_seq_lengths)
        testDataDict[label] = (X_te, test_seq_lengths)

    return trainDataDict, testDataDict


def prepare_test_data(testDataDict):
    pairOfTestData = []
    for l in testDataDict.keys():
        sequence, seq_lengths =  testDataDict[l]
        idx = 0
        for k in seq_lengths:
            pairOfTestData.append((l, sequence[idx:idx+k]))
            idx = idx+k
    return pairOfTestData
