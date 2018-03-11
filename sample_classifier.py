# Example classifier to run on dataset

# NEEDS deep_models in local directory
from deep_models import models
import tensorflow as tf 


# Set verbosity
tf.logging.set_verbosity('INFO')

# Get derived model inputs.
#

# Create the class
test_screen_shot_model = models.DeepModel('van',2, 
                            input_shape = (28,28,3),
                            num_classes=10)
# Train and eval
test_screen_shot_model.train_and_eval('TFRecords','traintest',
                    train_steps=2,
                    eval_steps=3)
#Predict
model_path = test_screen_shot_model.exp_dir
test_screen_shot_model.predict('/home/jay/Network-Comparisons/otest_images/')

