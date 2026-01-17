import pandas as pd
import pytest
import os
from dataclasses import dataclass

raw_df = pd.read_csv('/content/drive/MyDrive/Data_Project_1/credit_card_data_combined.csv')

import sys
sys.path.append('/content/drive/MyDrive/Data_Project_1/')
from Python_Code.Data_project_1_Automation import inspect_clean_data
from Python_Code.Data_project_1_Automation import data_eda, plt_bar_plots, save_plots
from Python_Code.Data_project_1_Automation import kmeans_clustering

sample_df = raw_df[raw_df['transaction_country'].isnull() | raw_df['merchant_category'].isnull() | 
                   raw_df['transaction_datetime'].str.contains('Z') | raw_df['is_international'].isnull()]

@dataclass
class Config:
  contamination : float = 0.15
  random_state : int = 42
  max_samples : str | int = "auto"

  n_clusters : int = 4
  pca_components: int = 2

cfg = Config()

@pytest.fixture
def dirty_df():
  return sample_df

@pytest.fixture
def cleaned_df():
  cleaned_df = inspect_clean_data(sample_df)
  return cleaned_df

@pytest.fixture
def plots_dir():
  plots_dir = os.path.join("/content", "plots")
  os.makedirs(plots_dir, exist_ok = True)
  return plots_dir

@pytest.fixture
def eda_df():
  cleaned_df = inspect_clean_data(sample_df)
  plots_dir = os.path.join("/content", "plots")
  os.makedirs(plots_dir, exist_ok = True)
  eda_df = data_eda(cleaned_df, plots_dir)
  return eda_df

@pytest.fixture
def kmeans_df():
  cleaned_df = inspect_clean_data(sample_df)
  plots_dir = os.path.join("/content", "plots")
  os.makedirs(plots_dir, exist_ok = True)
  eda_df = data_eda(cleaned_df, plots_dir)
  kmeans_df = kmeans_clustering(eda_df, cfg, plots_dir)
  return kmeans_df

@pytest.fixture
def test_config():
  @dataclass
  class Config:
    contamination : float = 0.15
    random_state : int = 42
    max_samples : str | int = "auto"

    n_clusters : int = 4
    pca_components: int = 2
  
  cfg = Config()
  return cfg



