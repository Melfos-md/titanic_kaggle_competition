#%%
import tensorflow as tf
import pandas as pd
import numpy as np
from preprocessing import preprocess_model

#%%
model = tf.keras.models.load_model('doc/1.1/1.1.tf')
# %%
inputs,titanic_preprocessing, titanic_features, titanic_id = preprocess_model('data/test.csv', live=True)
features_dict =  {name: np.array(value) for name, value in titanic_features.items()}

# %%
predictions = model.predict(features_dict)
# %%
predicted_classes = (predictions > 0.5).astype(int).flatten()
# %%
submission_df = pd.DataFrame({
    'PassengerId': titanic_id,
    'Survived': predicted_classes
})
# %%
submission_df.to_csv('doc/1.1/submission.csv',index=False)
# kaggle competitions submit -c [COMPETITION] -f [FILE] -m [MESSAGE]
# kaggle competitions submit -c titanic -f submission.csv -m wow
# %%
