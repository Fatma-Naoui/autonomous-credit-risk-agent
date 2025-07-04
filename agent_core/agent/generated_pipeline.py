import pandas as pd
import matplotlib.pyplot as plt
import os
import sklearn
from autogluon.tabular import TabularPredictor
import json

def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def train_model(train_data, save_path):
    predictor = TabularPredictor(label='loan_status', path=save_path).fit(train_data, time_limit=60)
    return predictor

def generate_leaderboard(predictor, save_path):
    leaderboard = predictor.leaderboard()
    leaderboard.to_csv(os.path.join(save_path, 'model_leaderboard.csv'), index=False)
    leaderboard.to_csv(os.path.join('../', 'leaderboard.csv'), index=False)

def generate_evaluation_metrics(predictor, test_data, save_path):
    y_pred = predictor.predict(test_data)
    y_pred_proba = predictor.predict_proba(test_data)
    positive_class = predictor.class_labels[-1]
    y_test = test_data['loan_status']
    y_test_proba = y_pred_proba.iloc[:, -1]
    metrics = {
        'roc_auc': sklearn.metrics.roc_auc_score(y_test, y_test_proba)
    }
    with open(os.path.join(save_path, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f)

def generate_feature_importance(predictor, test_data, save_path):
    feature_importance = predictor.feature_importance(test_data)
    feature_importance.to_csv(os.path.join(save_path, 'feature_importance.csv'), index=False)
    return feature_importance

def plot_feature_importance(feature_importance, save_path):
    feature_importance.plot(kind='bar')
    plt.savefig(os.path.join(save_path, 'feature_importance.png'))
    plt.close()

def plot_roc_curve(y_test, y_pred_proba, model_name, save_path, color):
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, color=color, label=model_name)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'{model_name}_roc_curve.png'))
    plt.close()

def plot_precision_recall_curve(y_test, y_pred_proba, model_name, save_path, color):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recall, precision, color=color, label=model_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'{model_name}_precision_recall_curve.png'))
    plt.close()

def plot_confusion_matrix(y_test, y_pred, model_name, save_path, color):
    cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(os.path.join(save_path, f'{model_name}_confusion_matrix.png'))
    plt.close()

def plot_shap_summary(predictor, test_data, save_path):
    shap_values = predictor.shap_values(test_data)
    shap.summary_plot(shap_values, test_data, plot_type='bar')
    plt.savefig(os.path.join(save_path, 'shap_summary.png'))
    plt.close()

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data')
    train_path = os.path.join(data_path, 'train_data.csv')
    test_path = os.path.join(data_path, 'test_data.csv')
    model_path = os.path.join(project_root, 'models')
    os.makedirs(model_path, exist_ok=True)

    train_data, test_data = load_data(train_path, test_path)
    predictor = train_model(train_data, model_path)
    generate_leaderboard(predictor, model_path)
    generate_evaluation_metrics(predictor, test_data, model_path)
    feature_importance = generate_feature_importance(predictor, test_data, model_path)
    plot_feature_importance(feature_importance, model_path)

    leaderboard = predictor.leaderboard()
    top_models = leaderboard['model'].iloc[:3]
    colors = ['red', 'green', 'blue']

    for i, model_name in enumerate(top_models):
        y_pred = predictor.predict(test_data, model=model_name)
        y_pred_proba_df = predictor.predict_proba(test_data, model=model_name)
        positive_class = predictor.class_labels[-1]
        y_pred_proba = y_pred_proba_df[positive_class] if positive_class in y_pred_proba_df else y_pred_proba_df.iloc[:, -1]

        plot_roc_curve(test_data['loan_status'], y_pred_proba, model_name, model_path, colors[i])
        plot_precision_recall_curve(test_data['loan_status'], y_pred_proba, model_name, model_path, colors[i])
        plot_confusion_matrix(test_data['loan_status'], y_pred, model_name, model_path, colors[i])

if __name__ == '__main__':
    main()