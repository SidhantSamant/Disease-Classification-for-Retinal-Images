# Imports
import os
import numpy as np
import PIL
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model


# Load Load and Assign labels
model = load_model("models/model_ResNet18.hdf5")
labels = {0: "Diabetic Retinopathy", 1: "Glaucoma", 2: "Healthy"}


# Define a flask app
app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    # Main page
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        # Get the file from post request
        f = request.files["file"]

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, "uploads", secure_filename(f.filename))
        f.save(file_path)

        # Open Image and prediction
        img = PIL.Image.open(file_path)
        img = img.resize((256, 256))
        # converting image to array
        img = np.asarray(img, dtype=np.float32)
        # normalizing the image
        img = img / 255
        # reshaping the image in to a 4D array
        img = img.reshape(-1, 256, 256, 3)
        # making prediction of the model
        predict = model.predict(img)
        predict = np.argmax(predict)

        result = labels[predict]

        return result

    return None


if __name__ == "__main__":
    app.run(debug=False)
