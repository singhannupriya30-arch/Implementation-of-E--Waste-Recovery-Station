from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model("ewaste_model.h5")

# Class labels 
classes = ['Battery', 'Bulb', 'Keyboard', 'Microwave', 'Mobile', 'Mouse', 'Non-E-Waste',
           'PCB', 'Television', 'Washing Machine']

# Load image
img_path = "dataset/test/Random/Washing_Machine_24.jpg"   # change this
img = image.load_img(img_path, target_size=(128,128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Prediction
pred = model.predict(img_array)[0]
confidence = np.max(pred)
pred_class = np.argmax(pred)

if confidence < 0.6:   # threshold for unknown
    print("Prediction: Unknown / Non-E-waste")
else:
    print("Prediction:", classes[pred_class], "Confidence:", confidence)

