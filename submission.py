#%%
import tensorflow as tf
import pandas as pd
import numpy as np
from preprocessing import preprocess_model

#%%
model = tf.keras.models.load_model('doc/model_1.0/model_1.0.tf')
# %%

usecols = ['PassengerId','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
titanic_features = pd.read_csv('data/test.csv', usecols=usecols)
titanic_features['Embarked'].fillna("Missing", inplace=True)  
titanic_features['Pclass'] = titanic_features['Pclass'].astype(str)
passenger_id = titanic_features.pop('PassengerId')

features_dict =  {name: np.array(value) for name, value in titanic_features.items()}

# %%
predictions = model.predict(features_dict)
# %%
predicted_classes = (predictions > 0.5).astype(int).flatten()
# %%
submission_df = pd.DataFrame({
    'PassengerId': passenger_id,
    'Survived': predicted_classes
})
# %%
submission_df.to_csv('doc/model_1.0/submission.csv',index=False)
# %%
