from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import numpy as np

app = Flask(__name__)

model = load_model('model.h5')

class_names = ['Bags', 'Caps', 'Dresses', 'Shoes']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['file']

    filepath = "static/" + file.filename

    file.save(filepath)

    img = image.load_img(filepath,
                         target_size=(128,128))

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]

    return render_template(
        'index.html',
        prediction=predicted_class,
        img_path=filepath
    )

if __name__ == '__main__':
    app.run(debug=True)