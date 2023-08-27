# TODO:
# - mettre un seuil de prediction ? il doit y avoir un lien avec from_logits
# Du coup je suis pas trop sûr de ce qui se passe en matière de prédiction là... puisque je prédis des probabilités
#%%
from preprocessing import preprocess_model
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

#%%
def titanic_model(preprocessing_head, inputs):
    body = tf.keras.Sequential([
        layers.Dense(1, activation='sigmoid')
    ])
    preprocessed_inputs = preprocessing_head(inputs)
    result = body(preprocessed_inputs)
    model = tf.keras.Model(inputs, result, name='simple_sigmoid_model')
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
                  metrics=['accuracy', 'Precision', 'Recall'])
    return model

inputs,titanic_preprocessing, titanic_features, titanic_labels = preprocess_model()
model = titanic_model(titanic_preprocessing, inputs)

#model.summary()
#tf.keras.utils.plot_model(model=model, rankdir="LR", dpi=72, show_shapes=True, to_file='simple_sigmoid_model.png')
#with open('modelsummary.txt', 'w') as f:
#    model.summary(print_fn=lambda x: f.write(x + '\n'))

#%%
# split 60-20-20: ici 80/20 et dans model.fit 75-25
X_train, X_test, y_train, y_test = train_test_split(titanic_features,titanic_labels, test_size=0.2, random_state=42)
titanic_train_features_dict = {name: np.array(value) for name, value in X_train.items()}

#%%
history = model.fit(x=titanic_train_features_dict, y=y_train, 
                    epochs=50, validation_split=0.25)
#%%
from utils import plot_loss_metrics
plot_loss_metrics(history, to_file=False)
# %%
titanic_test_features_dict = {name: np.array(value) for name, value in X_test.items()}
results = model.evaluate(titanic_test_features_dict, y_test)

print("metrics: ", model.metrics_names)
print("results: ", results)


#%%
#model.save('doc/model_1.0/model_1.0.tf')
# %%
