from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = load_model("dog_cat_model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    if file:
        filepath = os.path.join("static", file.filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(64,64))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0

        result = model.predict(img)

        if result[0][0] > 0.5:
            prediction = "Dog 🐶"
        else:
            prediction = "Cat 🐱"

        return render_template('index.html', prediction=prediction, img_path=filepath)

    return "Error"

if __name__ == "__main__":
    #app.run(debug=True)
     app.run(host="0.0.0.0", port=10000)