import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import sys
sys.path.append('/content/drive/MyDrive/Data_Project_1/')
from Python_Code.Data_project_1_Automation import isf_model

def test_preds_dtype_range_len(kmeans_df, test_config):
  df_isf = isf_model(kmeans_df, test_config)
  df_isf_num_cols = df_isf.select_dtypes(include = 'number')

  scaler = StandardScaler().set_output(transform = 'pandas')
  df_isf_scaled = scaler.fit_transform(df_isf_num_cols)
  assert str(df_isf['predictions'].dtype).startswith('int')
  assert len(df_isf['predictions']) == len(df_isf_scaled)
  assert set(df_isf['predictions']).issubset({-1, 1})

def test_model_perfomance(kmeans_df, test_config):
  df_isf = isf_model(kmeans_df, test_config)
  df_isf_pred = (df_isf['predictions'] == -1).astype(int)
  assert precision_score(df_isf['is_fraud'].astype(int), df_isf_pred) >= 0.35
  assert recall_score(df_isf['is_fraud'].astype(int), df_isf_pred) >= 0.35


