from preprocessing import preprocess_model
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

def titanic_model(preprocessing_head, inputs):
    body = tf.keras.Sequential([
        layers.Dense(1, activation='sigmoid')
    ])
    preprocessed_inputs = preprocessing_head(inputs)
    result = body(preprocessed_inputs)
    model = tf.keras.Model(inputs, result, name='simple_sigmoid_model')
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer=tf.keras.optimizers.Adam())
    return model

inputs,titanic_preprocessing, titanic_features, titanic_labels = preprocess_model()
model = titanic_model(titanic_preprocessing, inputs)

#model.summary()
#tf.keras.utils.plot_model(model=model, rankdir="LR", dpi=72, show_shapes=True, to_file='simple_sigmoid_model.png')
#with open('modelsummary.txt', 'w') as f:
#    model.summary(print_fn=lambda x: f.write(x + '\n'))

# split 60-20-20
X_train, X_test, y_train, y_test = train_test_split(titanic_features,titanic_labels, test_size=0.2, random_state=42)
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

titanic_train_features_dict = {name: np.array(value) for name, value in X_train.items()}

features_dict = {name:values[:1] for name, values in titanic_train_features_dict.items()}
titanic_preprocessing(features_dict)

model.fit(x=titanic_train_features_dict, y=y_train, epochs=10)
