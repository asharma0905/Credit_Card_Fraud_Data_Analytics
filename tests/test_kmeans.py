import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import random
import pytest

import sys
sys.path.append('/content/drive/MyDrive/Data_Project_1/')
from Python_Code.Data_project_1_Automation import kmeans_clustering, plt_bar_plots, save_plots

def test_scaling_std_mean(eda_df, test_config, plots_dir):
  kmeans_df = kmeans_clustering(eda_df, test_config, plots_dir)
  kmeans_df_num_cols = kmeans_df.select_dtypes(include = 'number')

  scaler = StandardScaler().set_output(transform = 'pandas')
  kmeans_df_scaled = scaler.fit_transform(kmeans_df_num_cols)
  test_feature = random.choice(list(kmeans_df_num_cols.columns))
  mean_val = kmeans_df_scaled[test_feature].mean()
  std_val = kmeans_df_scaled[test_feature].std(ddof=0)
  assert mean_val == pytest.approx(0.0, abs=1e-1)
  assert std_val == pytest.approx(1.0, rel=0.1)

def test_cluster_labels(eda_df, test_config, plots_dir):
  kmeans_df = kmeans_clustering(eda_df, test_config, plots_dir)
  assert 'cluster_labels' in list(kmeans_df.columns)
  assert len(set(kmeans_df['cluster_labels'])) == test_config.n_clusters
  assert kmeans_df['cluster_labels'].notna().all()
  assert str(kmeans_df['cluster_labels'].dtype).startswith('int')












  