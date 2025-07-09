import sys
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

def load_data(train_path, test_path, label_column):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data, label_column

def train_model(train_data, label_column, time_limit):
    predictor = TabularPredictor(label=label_column, path="agent_core/models").fit(train_data, time_limit=time_limit)
    return predictor

def generate_leaderboard(predictor):
    leaderboard = predictor.leaderboard()
    leaderboard.to_csv("agent_core/models/model_leaderboard.csv", index=False)
    leaderboard.to_csv("agent_core/leaderboard.csv", index=False)
    return leaderboard

def generate_evaluation_metrics(predictor, test_data, label_column):
    test_pred_proba = predictor.predict_proba(test_data)
    test_labels = test_data[label_column]
    evaluation_metrics = {
        "roc_auc": roc_auc_score(test_labels, test_pred_proba.iloc[:, 1])
    }
    with open("agent_core/models/model_metrics.json", "w") as f:
        import json
        json.dump(evaluation_metrics, f)
    return evaluation_metrics

def generate_feature_importance(predictor, test_data):
    feature_importance = predictor.feature_importance(test_data)
    return feature_importance

def plot_feature_importance(feature_importance):
    plt.bar(feature_importance.index, feature_importance.values)
    plt.title("Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.savefig("agent_core/models/feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_roc_curve(test_labels, test_pred_proba, model_name):
    fpr, tpr, _ = roc_curve(test_labels, test_pred_proba)
    plt.plot(fpr, tpr, label=model_name)
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.savefig(f"agent_core/models/roc_{model_name}.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_precision_recall_curve(test_labels, test_pred_proba, model_name):
    precision, recall, _ = precision_recall_curve(test_labels, test_pred_proba)
    plt.plot(recall, precision, label=model_name)
    plt.title("Precision Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig(f"agent_core/models/pr_{model_name}.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_confusion_matrix(test_labels, test_pred, model_name):
    cm = confusion_matrix(test_labels, test_pred)
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.savefig(f"agent_core/models/cm_{model_name}.png", dpi=300, bbox_inches="tight")
    plt.close()

def main():
    if len(sys.argv) != 4:
        print("Usage: python generated_pipeline.py <train_path> <test_path> <label_column>")
        return
    train_path, test_path, label_column = sys.argv[1], sys.argv[2], sys.argv[3]
    train_data, test_data, label_column = load_data(train_path, test_path, label_column)
    predictor = train_model(train_data, label_column, time_limit=60)
    leaderboard = generate_leaderboard(predictor)
    leaderboard_sorted = leaderboard.sort_values(by="score_val", ascending=False)
    model_names = leaderboard_sorted["model"].head(3).tolist()
    evaluation_metrics = generate_evaluation_metrics(predictor, test_data, label_column)
    feature_importance = generate_feature_importance(predictor, test_data)
    plot_feature_importance(feature_importance)
    colors = ["red", "green", "blue"]
    for i, model_name in enumerate(model_names):
        test_pred = predictor.predict(test_data, model=model_name)
        test_pred_proba = predictor.predict_proba(test_data, model=model_name)
        plot_roc_curve(test_data[label_column], test_pred_proba.iloc[:, 1], model_name)
        plot_precision_recall_curve(test_data[label_column], test_pred_proba.iloc[:, 1], model_name)
        plot_confusion_matrix(test_data[label_column], test_pred, model_name)

if __name__ == "__main__":
    main()