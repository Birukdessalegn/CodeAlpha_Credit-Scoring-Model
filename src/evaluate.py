# File: /credit-scoring-model/credit-scoring-model/src/evaluate.py

def evaluate_model(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    evaluation_results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc
    }

    return evaluation_results

def print_evaluation_results(results):
    print("Model Evaluation Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")