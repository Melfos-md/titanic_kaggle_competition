#%%
import pandas as pd


# %%
data = pd.read_csv('data/train.csv')
# %%
data['Survived'].value_counts()
# %%
data.dtypes
# %%
