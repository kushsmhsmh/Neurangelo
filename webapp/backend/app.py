# backend/app.py

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
from model_converter import convert_image_to_3d

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/convert", methods=["POST"])
def convert():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(image_path)

    model_path = convert_image_to_3d(image_path, OUTPUT_FOLDER)

    return send_file(model_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
