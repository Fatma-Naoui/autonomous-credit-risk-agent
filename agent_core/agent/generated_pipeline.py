import sys
import os
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.metrics import roc_auc_score, precision_score, f1_score, accuracy_score, confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import json

def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def train_model(train_data, label_column, output_dir):
    predictor = TabularPredictor(label=label_column, path=output_dir).fit(train_data, time_limit=60)
    return predictor

def generate_leaderboard(predictor):
    leaderboard = predictor.leaderboard()
    leaderboard.to_csv(os.path.join(predictor.path, 'model_leaderboard.csv'), index=False)
    return leaderboard

def generate_evaluation_metrics(predictor, test_data, label_column):
    test_pred = predictor.predict(test_data, model=predictor.model_names()[0])
    test_pred_proba = predictor.predict_proba(test_data, model=predictor.model_names()[0])
    evaluation_metrics = {
        'f1': f1_score(test_data[label_column], test_pred),
        'precision': precision_score(test_data[label_column], test_pred),
        'auc': roc_auc_score(test_data[label_column], test_pred_proba.iloc[:, 1]),
        'gini': 2 * roc_auc_score(test_data[label_column], test_pred_proba.iloc[:, 1]) - 1,
        'accuracy': accuracy_score(test_data[label_column], test_pred)
    }
    return evaluation_metrics

def generate_feature_importance(predictor, test_data):
    feature_importance = predictor.feature_importance(test_data)
    return feature_importance

def plot_feature_importance(feature_importance, output_dir):
    feature_importance.plot(kind='bar')
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches="tight")
    plt.close()

def plot_roc_curve(test_data, test_pred_proba, label_column, model_name, output_dir):
    fpr, tpr, _ = roc_curve(test_data[label_column], test_pred_proba.iloc[:, 1])
    plt.plot(fpr, tpr, label=model_name)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'roc_curve_{model_name}.png'), dpi=300, bbox_inches="tight")
    plt.close()

def plot_precision_recall_curve(test_data, test_pred_proba, label_column, model_name, output_dir):
    precision, recall, _ = precision_recall_curve(test_data[label_column], test_pred_proba.iloc[:, 1])
    plt.plot(recall, precision, label=model_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'precision_recall_curve_{model_name}.png'), dpi=300, bbox_inches="tight")
    plt.close()

def plot_confusion_matrix(test_data, test_pred, label_column, model_name, output_dir):
    conf_mat = confusion_matrix(test_data[label_column], test_pred)
    plt.imshow(conf_mat, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(set(test_data[label_column])))
    plt.xticks(tick_marks)
    plt.yticks(tick_marks)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    thresh = conf_mat.max() / 2
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        plt.text(j, i, format(conf_mat[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_mat[i, j] > thresh else "black")
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name}.png'), dpi=300, bbox_inches="tight")
    plt.close()

def main():
    if len(sys.argv) != 5:
        print("Usage: python generated_pipeline.py <train_path> <test_path> <label_column> <output_dir>")
        return
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    label_column = sys.argv[3]
    output_dir = sys.argv[4]
    os.makedirs(output_dir, exist_ok=True)
    train_data, test_data = load_data(train_path, test_path)
    predictor = train_model(train_data, label_column, output_dir)
    leaderboard = generate_leaderboard(predictor)
    evaluation_metrics = generate_evaluation_metrics(predictor, test_data, label_column)
    feature_importance = generate_feature_importance(predictor, test_data)
    plot_feature_importance(feature_importance, output_dir)
    leaderboard_sorted = leaderboard.sort_values(by="score_val", ascending=False)
    model_names = leaderboard_sorted["model"].head(3).tolist()
    colors = ['b', 'g', 'r']
    for model_name, color in zip(model_names, colors):
        test_pred = predictor.predict(test_data, model=model_name)
        test_pred_proba = predictor.predict_proba(test_data, model=model_name)
        plot_roc_curve(test_data, test_pred_proba, label_column, model_name, output_dir)
        plot_precision_recall_curve(test_data, test_pred_proba, label_column, model_name, output_dir)
        plot_confusion_matrix(test_data, test_pred, label_column, model_name, output_dir)
        model_metrics = {
            'f1': f1_score(test_data[label_column], test_pred),
            'precision': precision_score(test_data[label_column], test_pred),
            'auc': roc_auc_score(test_data[label_column], test_pred_proba.iloc[:, 1]),
            'gini': 2 * roc_auc_score(test_data[label_column], test_pred_proba.iloc[:, 1]) - 1,
            'accuracy': accuracy_score(test_data[label_column], test_pred)
        }
        with open(os.path.join(output_dir, f'model_metrics_{model_name}.json'), 'w') as f:
            json.dump(model_metrics, f)

if __name__ == "__main__":
    import itertools
    main()