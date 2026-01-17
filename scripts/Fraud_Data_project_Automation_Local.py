import argparse
from dataclasses import dataclass
import os

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

@dataclass
class Config:
  input_path : str
  output_dir : str = "outputs"

  contamination : float = 0.15
  random_state : int = 42
  max_samples : str | int = "auto"

  n_clusters : int = 4
  pca_components: int = 2

  final_csv_output : str = "final_credit_card_data.csv"

def read_data(path):
  if path.lower().endswith(".csv"):
    return pd.read_csv(path)
  else:
    return ValueError("Unsupported File Type!")

def inspect_clean_data(df):
  df.head()
  df.info()
  df.describe().map(lambda x : f"{x:.5f}")
  df = df[~df.duplicated()]
  df.isnull().sum()

  (df['transaction_city'].isnull() & df['transaction_country'].isnull()).sum()
  df = df[~(df['transaction_country'].isnull())]

  df.fillna({'merchant_category' : df[~(df['merchant_category'].isnull())
& (df['transaction_type'] == 'withdrawal')]['merchant_category'].iloc[0]}, inplace = True)
  df.info()
  df.fillna('NA', inplace=True)

  df['transaction_datetime'] = df['transaction_datetime'].str.replace('Z','')
  df['transaction_datetime'] = pd.to_datetime(df['transaction_datetime'], format = 'ISO8601', utc = True, errors = 'coerce')
  
  df.info()
  
  df['is_online_fraud'] = df['is_online'] & df['is_fraud']
  df = df[~(df['is_international'] == 'NA')]
  df['is_international'] = df['is_international'].astype(bool)
  df.reset_index(drop=True)
  new_index = [i for i in range(len(df['transaction_id']))]
  df.index = new_index
  df['is_international_fraud'] = df['is_international'] & df['is_fraud']
  return df


def save_plots(path, dpi : int = 200):
  plt.tight_layout()
  plt.savefig(path, dpi=dpi, bbox_inches='tight')
  plt.close()

def plt_bar_plots(df, group_col, bool_col):
  return df.groupby(group_col)[bool_col].sum().plot(kind='bar')

def data_eda(df, plots_dir):
  df['transaction_month'] = df['transaction_datetime'].dt.month
  df['transaction_dayofweek'] = df['transaction_datetime'].dt.dayofweek
  df['transaction_hour'] = df['transaction_datetime'].dt.hour

  plt_bar_plots(df, 'transaction_month', 'is_fraud')
  save_plots(os.path.join(plots_dir, "transaction_month_bar_plot.png"))
  plt_bar_plots(df, 'transaction_dayofweek', 'is_fraud')
  save_plots(os.path.join(plots_dir, "transaction_dayofweek_bar_plot.png"))
  plt_bar_plots(df, 'transaction_hour', 'is_fraud')
  save_plots(os.path.join(plots_dir, "transaction_hour_bar_plot.png"))

  plt.scatter(df['cardholder_age'], df['transaction_amount'])
  save_plots(os.path.join(plots_dir, "sactter_plot_tr_crd_age.png"))
  df[['cardholder_age', 'transaction_amount']].corr(method = 'kendall')
  df[['is_fraud', 'transaction_amount']].corr(method = 'kendall')

  df['is_international'] = df['is_international'].astype(int)
  df['is_online'] = df['is_online'].astype(int)
  plt_bar_plots(df, 'merchant_category', 'is_fraud')
  save_plots(os.path.join(plots_dir, "merchant_category_bar_plot.png"))
  plt_bar_plots(df, 'transaction_type', 'is_fraud')
  save_plots(os.path.join(plots_dir, "transaction_type_bar_plot.png"))
  plt_bar_plots(df, 'merchant_name', 'is_fraud')
  save_plots(os.path.join(plots_dir, "merchant_name_bar_plot.png"))

  df_eda_type_cats = pd.get_dummies(df['transaction_type'], dtype = int, prefix = 'transaction_type')
  df = pd.concat([df, df_eda_type_cats], axis=1)
  df_eda_mer_cats = pd.get_dummies(df['merchant_category'], prefix = 'merchant_category', dtype = int)
  df = pd.concat([df, df_eda_mer_cats], axis = 1)
  df_kmeans_em_cat = pd.get_dummies(df['entry_mode'], dtype = int, prefix = 'entry_mode')
  df = pd.concat([df, df_kmeans_em_cat], axis = 1)

  num_cols_df = df.select_dtypes(include = 'number')
  corr_matrix = num_cols_df.corr(method = 'kendall')
  sns.heatmap(corr_matrix)
  save_plots(os.path.join(plots_dir, "corr_matrix_heatmap.png"))
  return df


def kmeans_clustering(df, cfg : Config, plots_dir):
  df = df.fillna('NA')
  df_num_cols = df.select_dtypes(include = 'number')

  scaler = StandardScaler().set_output(transform = 'pandas')
  df_scaled = scaler.fit_transform(df_num_cols)

  pca = PCA(n_components = cfg.pca_components)
  df_pca= pca.fit_transform(df_scaled)
  pca.explained_variance_ratio_

  kmeans = KMeans(n_clusters = cfg.n_clusters, random_state=cfg.random_state)
  df['cluster_labels'] = kmeans.fit_predict(df_pca)
  kmeans.cluster_centers_

  fig, ax = plt.subplots()
  ax.scatter(df_pca[:, 0], df_pca[:, 1], c=df['cluster_labels'])
  ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='blue', marker = 'X')
  centroids = kmeans.cluster_centers_
  for i, coord in enumerate(centroids):
    plt.text(coord[0], coord[1], s=f'Cluster {i}',
              fontsize=12, fontweight='bold', color='black',
              bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

  save_plots(os.path.join(plots_dir, "kmeans_clustering.png"))

  silhouette_score(df_pca, kmeans.labels_)
  pca_loadings = pd.DataFrame(pca.components_.T, index=df_num_cols.columns, columns = ['PC1', 'PC2'])

  df.groupby('cluster_labels')['transaction_amount'].mean()
  df.groupby('cluster_labels')['cardholder_age'].mean()
  df.groupby('cluster_labels')['is_fraud'].sum()
  return df

def isf_model(df, cfg : Config):
  df = df.fillna('NA')
  df_num_cols = df.select_dtypes(include = 'number')
  df_num_cols = df_num_cols.drop(columns = ['cluster_labels'])

  scaler = StandardScaler().set_output(transform = 'pandas')
  df_scaled = scaler.fit_transform(df_num_cols)

  n_est_list = [50, 100, 150, 200, 250, 2000]
  predictions_list = []
  isf_accuracy_max = -np.inf
  max_accu_n_est = 0
  for i in n_est_list:
    isf_model = IsolationForest(n_estimators = i, contamination = cfg.contamination, random_state = cfg.random_state)
    df['predictions'] = isf_model.fit_predict(df_scaled)
    isf_model_accuracy = len(df[(df['predictions'] == -1) & (df['is_fraud'] == True)]) / len(df[df['is_fraud'] == True])
    if isf_model_accuracy > isf_accuracy_max:
      max_accu_n_est = i
    predictions_list.append(isf_model_accuracy*100.0)
    df = df.drop(columns = ['predictions'])

  isf_model = IsolationForest(n_estimators = max_accu_n_est, contamination = cfg.contamination, random_state = cfg.random_state)
  df['predictions'] = isf_model.fit_predict(df_scaled)
  df_pred = [1 if x == -1 else 0 for x in df['predictions']]
  print(classification_report(df['is_fraud'].astype(int), df_pred))

  return df

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", required = True, help = 'Path to raw file')
  parser.add_argument("--output", default = "outputs", help = 'Folder to write outputs')
  parser.add_argument("--contamination", type=float, default=0.15)
  args = parser.parse_args()

  cfg = Config (
      input_path = args.input,
      output_dir = args.output,
      contamination = args.contamination
  )

  os.makedirs(cfg.output_dir, exist_ok = True)
  plots_dir = os.path.join(cfg.output_dir, "plots")
  os.makedirs(plots_dir, exist_ok = True)

  raw_df = read_data(cfg.input_path)

  cleaned_df = inspect_clean_data(raw_df)

  df_eda = data_eda(cleaned_df, plots_dir)

  df_kmeans = kmeans_clustering(df_eda, cfg, plots_dir)

  df_model = isf_model(df_kmeans, cfg)

  final_csv_path = os.path.join(cfg.output_dir, cfg.final_csv_output)
  df_model.to_csv(final_csv_path, index=False)

  print(f"Saved all plots: {plots_dir}")
  print(f"Saved final csv: {final_csv_path}")

if __name__ == "__main__":
  main()



