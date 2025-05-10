import tensorflow as tf
import os

#  Your custom layer
class Cast(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.identity(inputs)  # Equivalent to tf.cast, but simpler

#  Load the Keras model
h5_model_path = "D:/Real Time Testing/engagement_model_89.h5"
tflite_model_path = "D:/Real Time Testing/engagement_model_89.tflite"

model = tf.keras.models.load_model(
    h5_model_path,
    custom_objects={"Cast": Cast},
    compile=False
)

#  Set up the converter with more control
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: Reduce model size (not required, but can help)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Optional: Enable float16 if your model supports it
# converter.target_spec.supported_types = [tf.float16]

#  Add fallback for unsupported ops
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Use TFLite native ops
    tf.lite.OpsSet.SELECT_TF_OPS     # Fallback to full TF ops
]

#  Allow custom operations (in case the Cast layer causes problems)
converter.allow_custom_ops = True

#  Convert
try:
    tflite_model = converter.convert()
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(" Successfully converted and saved TFLite model at:", tflite_model_path)
except Exception as e:
    print(" Conversion failed with error:\n", str(e))
