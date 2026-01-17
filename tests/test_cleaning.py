#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd


# In[24]:


import sys
sys.path.append('/content/drive/MyDrive/Data_Project_1/')
from Python_Code.Data_project_1_Automation import inspect_clean_data


# In[25]:


def test_clean_no_nulls(dirty_df):
  clean_df = inspect_clean_data(dirty_df)
  assert ~clean_df.duplicated().any()
  assert clean_df['transaction_country'].notna().all()
  assert clean_df['merchant_category'].notna().all()


# In[26]:


def test_clean_correct_dtypes(dirty_df):
  clean_df = inspect_clean_data(dirty_df)
  assert str(clean_df['transaction_datetime'].dtype).startswith('datetime')
  assert str(clean_df['is_international_fraud'].dtype).startswith('bool')

