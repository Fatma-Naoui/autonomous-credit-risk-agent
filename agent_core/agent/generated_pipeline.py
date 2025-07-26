import sys
import os
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix, auc, roc_curve, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def load_data(train_csv, test_csv):
    train_data = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)
    return train_data, test_data

def train_model(train_data, label_column, output_dir):
    predictor = TabularPredictor(label=label_column, path=output_dir).fit(train_data, time_limit=60)
    return predictor

def generate_leaderboard(predictor):
    leaderboard = predictor.leaderboard()
    leaderboard.to_csv(os.path.join(predictor.path, "model_leaderboard.csv"), index=False)
    return leaderboard

def generate_evaluation_metrics(predictor, test_data, label_column):
    test_pred = predictor.predict(test_data)
    test_pred_proba = predictor.predict_proba(test_data)
    evaluation_metrics = {
        "roc_auc": roc_auc_score(test_data[label_column], test_pred_proba.iloc[:, 1]),
        "precision": precision_score(test_data[label_column], test_pred),
        "recall": recall_score(test_data[label_column], test_pred),
        "f1": f1_score(test_data[label_column], test_pred)
    }
    return evaluation_metrics

def generate_feature_importance(predictor, test_data):
    feature_importance = predictor.feature_importance(test_data)
    return feature_importance

def plot_feature_importance(feature_importance, output_dir):
    feature_importance.plot(kind="bar")
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_roc_curve(test_data, test_pred_proba, model_name, output_dir):
    fpr, tpr, _ = roc_curve(test_data.iloc[:, -1], test_pred_proba.iloc[:, 1])
    plt.plot(fpr, tpr, label=model_name)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"roc_{model_name}.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_precision_recall_curve(test_data, test_pred_proba, model_name, output_dir):
    precision, recall, _ = precision_recall_curve(test_data.iloc[:, -1], test_pred_proba.iloc[:, 1])
    plt.plot(recall, precision, label=model_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"pr_{model_name}.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_confusion_matrix(test_data, test_pred, model_name, output_dir):
    cm = confusion_matrix(test_data.iloc[:, -1], test_pred)
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(set(test_data.iloc[:, -1])))
    plt.xticks(tick_marks, ["0", "1"])
    plt.yticks(tick_marks, ["0", "1"])
    thresh = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, "{:,}".format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(os.path.join(output_dir, f"cm_{model_name}.png"), dpi=300, bbox_inches="tight")
    plt.close()

def main():
    if len(sys.argv) != 5:
        print("Usage: python generated_pipeline.py <train_csv> <test_csv> <label_column> <output_dir>")
        return
    train_csv = sys.argv[1]
    test_csv = sys.argv[2]
    label_column = sys.argv[3]
    output_dir = sys.argv[4]
    os.makedirs(output_dir, exist_ok=True)
    train_data, test_data = load_data(train_csv, test_csv)
    predictor = train_model(train_data, label_column, output_dir)
    leaderboard = generate_leaderboard(predictor)
    evaluation_metrics = generate_evaluation_metrics(predictor, test_data, label_column)
    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
        import json
        json.dump(evaluation_metrics, f)
    feature_importance = generate_feature_importance(predictor, test_data)
    plot_feature_importance(feature_importance, output_dir)
    leaderboard_sorted = leaderboard.sort_values(by="score_val", ascending=False)
    model_names = leaderboard_sorted["model"].head(3).tolist()
    colors = ["red", "green", "blue"]
    for i, model_name in enumerate(model_names):
        test_pred = predictor.predict(test_data, model=model_name)
        test_pred_proba = predictor.predict_proba(test_data, model=model_name)
        plot_roc_curve(test_data, test_pred_proba, model_name, output_dir)
        plot_precision_recall_curve(test_data, test_pred_proba, model_name, output_dir)
        plot_confusion_matrix(test_data, test_pred, model_name, output_dir)

if __name__ == "__main__":
    main()