# TODO:
# - Remake preprocess.py following README.md
# - Re test image generation of preprocessing_pipeline model
# excluded features for the first model : ['PassengerId', 'Name', 'Ticket', 'Cabin']
from preprocessing import preprocess_model

titanic_preprocessing = preprocess_model()
tf.keras.utils.plot_model(model=titanic_preprocessing, to_file='preprocessing_pipeline.png', rankdir="LR", dpi=300, show_shapes=True)