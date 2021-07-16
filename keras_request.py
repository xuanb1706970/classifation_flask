import base64
import numpy as np
import io
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)
def get_model():
    global model
    model = load_model('model.h5')
    print(" * Model loaded!")

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image
print(" * load keras model...")

get_model()
@app.route("/predict", methods=["POST"])
def predict():

	message = request.get_json(force=True)
	encoded	= message['image']
	decoded = base64.b64decode(encoded)
	image = Image.open(io.BytesIO(decoded))
	processed_image = preprocess_image(image, target_size=(224, 224))
	prediction = model.predict(processed_image).tolist()
	
	respone = {
		'prediction':{
			'billgates':prediction[0][0],
			'CanhXuan':prediction[0][1],
			'markZuchkerberg':prediction[0][2],
			'TuanKha':prediction[0][3],
		}
	}
	return jsonify(respone)

if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	app.run(host='0.0.0.0', port='6868')

