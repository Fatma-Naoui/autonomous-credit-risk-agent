from autogluon.tabular import TabularPredictor
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

train_data = pd.read_csv('agent_core/data/train_data.csv')
test_data = pd.read_csv('agent_core/data/test_data.csv')

predictor = TabularPredictor(label='loan_status').fit(train_data)

y_test = test_data['loan_status']
y_pred = predictor.predict(test_data.drop('loan_status', axis=1))

leaderboard = predictor.leaderboard(test_data)
leaderboard.to_csv('agent_core/models/model_leaderboard.csv', index=False)
leaderboard.to_csv('agent_core/leaderboard.csv', index=False)

with open('agent_core/models/model_metrics.json', 'w') as f:
    json.dump(predictor.evaluate(test_data), f)
with open('agent_core/evaluation_metrics.json', 'w') as f:
    json.dump(predictor.evaluate(test_data), f)

predictor.save('agent_core/models/')

importance = predictor.feature_importance(test_data)
importance.plot(kind='bar')
plt.savefig('agent_core/models/shap_summary.png')
plt.close()