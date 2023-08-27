#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import get_ticket_number_length
# %%
data = pd.read_csv('data/train.csv')
# %%
# #### EXPLORING TICKET FEATURE ####
len_ticket_numbers = get_ticket_number_length(data['Ticket'])
df = pd.DataFrame({'Ticket_Length': len_ticket_numbers, 'Survived': data['Survived']})
grouped_data = df.groupby('Ticket_Length').mean().reset_index()
# %%
# Statistical test with CHI²
from scipy.stats import chi2_contingency

# Generate a contingency table
# Count the number of survived and not-survived for each ticket length
contingency_table = pd.crosstab(df['Ticket_Length'], df['Survived'])

# Perform Chi-Squared test
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Display results
chi2, p, dof, expected
# %%
plt.figure(figsize=(10,6))
plt.bar(grouped_data['Ticket_Length'], grouped_data['Survived'], color='blue', alpha=0.7)
plt.xlabel('Ticket Number Length')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Ticket Number Length')
plt.xticks(grouped_data['Ticket_Length'])
plt.yticks(np.arange(0, 1.1, 0.1))
plt.grid(axis='y')

plt.text(5, 0.91, f"Chi-Squared Value: {chi2:.2f}", fontsize=12)
plt.text(5, 0.86, f"p-value: {p:.2e}", fontsize=12)
plt.text(5, 0.81, f"Degrees of Freedom: {dof}", fontsize=12)
plt.savefig('ticket_length_survival.png')
#plt.show()
# %%
# #### EXPLORING NAME FEATURE ####

data['Name'].head(10)
# %%
def extract_title(name_series):
    return name_series.str.extract(' ([A-Za-z]+)\.', expand=False)

# Test
names_series = pd.Series([
    "Braund, Mr. Owen Harris",
    "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
    "Heikkinen, Miss. Laina",
    "Futrelle, Mrs. Jacques Heath (Lily May Peel)",
    "Allen, Mr. William Henry",
    "Moran, Mr. James",
    "McCarthy, Mr. Timothy J",
    "Palsson, Master. Gosta Leonard",
    "Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)",
    "Nasser, Mrs. Nicholas (Adele Achem)"
])

list(extract_title(names_series))
# %%
import tensorflow as tf
from preprocessing import preprocess_model
_, titanic_preprocessing, _, _ = preprocess_model('data/train.csv', live=False)
tf.keras.utils.plot_model(model=titanic_preprocessing, to_file='preprocessing_pipeline.png', rankdir="LR", dpi=72, show_shapes=True)
# %%
