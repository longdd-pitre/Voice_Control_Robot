import hmmlearn
from hmmlearn import hmm
import pickle
import numpy as np
import os

class GMMHMMModel:
    def __init__(self, n_states, covariance_type):
        self.models = {}    # Dict to store HMMs for each label
        self.n_states = n_states
        self.covariance_type = covariance_type

    def add_model_for_label(self, label):
        self.models[label] = hmm.GMMHMM(n_components=self.n_states, covariance_type=self.covariance_type, n_mix=5, n_iter=24)

    # Fit HMM model for label
    def fit_for_label(self, label, sequences, list_sequences_length):
        self.models[label].fit(X=sequences, lengths=list_sequences_length)

    # def predict(self, sequences, list_sequences_length):
    #     for label in self.models.keys():
    #         self.models[label].predict(sequences, list_sequences_length)
    def predict(self, sequences, list_sequences_length):
        predictions = {}
        for label in self.models.keys():
            prediction = self.models[label].predict(sequences, list_sequences_length)
            predictions[label] = prediction
        return predictions
    def load(self, load_path):
        with open(load_path, 'rb') as f:
            self.models = pickle.load(f)
        #print(f"Model loaded from {load_path}")

    def save(self, save_path):
        model_name = "model"
        with open("../output/model.pkl", "wb") as f:
            pickle.dump(self.models, f)
        #print(f"Models saved as {model_name}")
  



