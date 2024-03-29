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
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  metrics=['accuracy'])
    return model

inputs,titanic_preprocessing, titanic_features, titanic_labels = preprocess_model('data/train.csv', live=False)
model = titanic_model(titanic_preprocessing, inputs)

#%%
#model.summary()
#with open('modelsummary.txt', 'w') as f:
#    model.summary(print_fn=lambda x: f.write(x + '\n'))

#tf.keras.utils.plot_model(model=model, rankdir="LR", dpi=72, show_shapes=True)

#%%
# split 60-20-20: ici 80/20 et dans model.fit 75-25
X_train, X_test, y_train, y_test = train_test_split(titanic_features,titanic_labels, test_size=0.2, random_state=42)
titanic_train_features_dict = {name: np.array(value) for name, value in X_train.items()}

#%%
history = model.fit(x=titanic_train_features_dict, y=y_train, 
                    epochs=100, validation_split=0.25)
#%%
from utils import plot_loss_metrics
plot_loss_metrics(history, to_file=False)
# %%
titanic_test_features_dict = {name: np.array(value) for name, value in X_test.items()}
results = model.evaluate(titanic_test_features_dict, y_test)

print("metrics: ", model.metrics_names)
print("results: ", results)


#%%
#model.save('doc/1.1/1.1.tf')
# %%
