# Model definition and training for the credit scoring model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

class CreditScoringModel:
    def __init__(self):
        self.model = RandomForestClassifier()
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save_model(self, file_path):
        joblib.dump(self.model, file_path)
    
    def load_model(self, file_path):
        self.model = joblib.load(file_path)