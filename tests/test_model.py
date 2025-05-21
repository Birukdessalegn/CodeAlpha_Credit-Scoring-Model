import pytest
from src.model import CreditScoringModel

def test_model_training():
    model = CreditScoringModel()
    model.train(X_train, y_train)
    assert model.is_trained is True

def test_model_prediction():
    model = CreditScoringModel()
    model.train(X_train, y_train)
    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)

def test_model_save_load():
    model = CreditScoringModel()
    model.train(X_train, y_train)
    model.save('test_model.pkl')
    loaded_model = CreditScoringModel.load('test_model.pkl')
    assert loaded_model.is_trained is True

def test_model_performance():
    model = CreditScoringModel()
    model.train(X_train, y_train)
    accuracy = model.evaluate(X_test, y_test)
    assert accuracy >= 0.7  # Assuming 70% accuracy is the threshold for a good model