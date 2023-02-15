from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model('imageclassifier.h5')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        file_name = file.filename
        file_ext = file_name.split('.')[-1]
        if file_ext not in ['jpg', 'jpeg', 'png']:
            return "Invalid file type."
        file.save(file_name)  # save the uploaded file with the correct extension
        img = image.load_img(file_name, target_size=(128, 128), color_mode='grayscale')
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)[0][0]
        if prediction < 0.5:
            return "The image is of a cat."
        else:
            return "The image is of a dog."

if __name__ == '__main__':
    app.run()
