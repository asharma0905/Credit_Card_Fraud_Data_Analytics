import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

import sys
sys.path.append('/content/drive/MyDrive/Data_Project_1/')
from Python_Code.Data_project_1_Automation import data_eda, plt_bar_plots, save_plots

def test_no_nulls_feature_dtypes(cleaned_df, plots_dir):
  eda_df = data_eda(cleaned_df, plots_dir)
  assert eda_df.isnull().sum().sum() == 0
  assert str(eda_df['transaction_datetime'].dtype).startswith('datetime64[ns, UTC]')
  assert str(eda_df['transaction_month'].dtype).startswith('int')
  assert str(eda_df['transaction_dayofweek'].dtype).startswith('int')
  assert str(eda_df['transaction_hour'].dtype).startswith('int')
  assert str(eda_df['is_international'].dtype).startswith('int')
  assert str(eda_df['is_online'].dtype).startswith('int')

def test_corr_1hotenc(cleaned_df, plots_dir):
  eda_df = data_eda(cleaned_df, plots_dir)
  num_cols_eda_df = eda_df.select_dtypes(include = 'number')
  corr_matrix = num_cols_eda_df.corr(method = 'kendall')
  assert ((corr_matrix.values >= -1) & (corr_matrix.values <= 1)).all()
  np.fill_diagonal(corr_matrix.values, 0.0)
  assert not np.logical_or((corr_matrix.values == -1.0), (corr_matrix.values == 1.0)).any()
  list_col_labels = list(eda_df.columns)
  list_tr_type = [x for x in list_col_labels if x.startswith('transaction_type_')]
  print('length', len(list_tr_type))
  assert (len(list_tr_type) >= 1)