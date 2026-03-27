from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = load_model("dog_cat_model.h5")

# Load image
img = image.load_img('test.jpg', target_size=(64,64))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img / 255.0

# Predict
result = model.predict(img)

if result[0][0] > 0.5:
    print("Dog 🐶")
else:
    print("Cat 🐱")