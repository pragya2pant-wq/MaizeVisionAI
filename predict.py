import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

model = tf.keras.models.load_model(
    "models/maize_disease_model.keras"
)

class_names = [
    "Blight",
    "Common_Rust",
    "Gray_Leaf_Spot",
    "Healthy"
]

img = load_img(
    r"D:\genai\gui\cnn_maize_disease\Corn_Blight (1).jpeg",
    target_size=(128, 128)
)

img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
print(prediction)
predicted_index = np.argmax(prediction)
confidence = np.max(prediction)

print("Prediction:", class_names[predicted_index])
print("Confidence:", round(float(confidence) * 100, 2), "%")