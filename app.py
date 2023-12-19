import os
from flask import Flask, jsonify, request
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import re
import json
import torch
from donut import JSONParseEvaluator
import csv
from werkzeug.utils import secure_filename
import tensorflow 
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle 
import numpy as np

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'json'])
app.config['UPLOAD_FOLDER_IMAGES'] = 'static/uploads/images'
app.config['UPLOAD_FOLDER_JSON'] = 'static/uploads/json'
app.config['UPLOAD_FOLDER_FINAL_JSON'] = 'static/uploads/final-json'

# Load Model 1
processor = DonutProcessor.from_pretrained("Vasettha/Donut_Cord")
model1 = VisionEncoderDecoderModel.from_pretrained("Vasettha/Donut_Cord")
device = "cuda" if torch.cuda.is_available() else "cpu"
model1.eval()
model1.to(device)

# Load Model 2
model2 = load_model('model.h5', compile=False)
max_length = 10
trunc_type = 'post'
padding_type = 'post'
# Load tokenizer, label_encoder for Model 2
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pkl', 'rb') as label_handle:
    label_encoder = pickle.load(label_handle)

# Function to process Model 1
def process_model1(image):
    pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    decoder_input_ids = decoder_input_ids.to(device)

    outputs = model1.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=model1.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    seq = processor.batch_decode(outputs.sequences)[0]
    seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    seq = re.sub(r"<.*?>", "", seq, count=1).strip()
    seq = processor.token2json(seq)
    # ... perform inference with model 1 and get seq ...

    # Save output of model 1 to JSON file
    json_file_path = 'menu_data.json'
    with open(os.path.join(app.config['UPLOAD_FOLDER_JSON'], json_file_path), 'w') as jsonfile:
        json.dump(seq, jsonfile, indent=2)
    return seq

# Function to process Model 2
def process_model2(json_data):
    # ... process json_data with Model 2 ...
    items = []
    
    raw_json = json_data.get('raw_json', {})
    menu = raw_json.get('menu', [])

    if isinstance(menu, dict):
        items.append(menu.get('nm', ''))
    elif isinstance(menu, list):
        for item in menu:
            items.append(item.get('nm', ''))
    return items

@app.route("/")
def index():
    return "message: DuitDojo ML Model API is here!!!"

@app.route("/donut", methods=["POST"])
def donut_route():
    if request.method == "POST":
        # Receive uploaded image
        image = request.files["image"]
        if image and '.' in image.filename and image.filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']:
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config["UPLOAD_FOLDER_IMAGES"], filename))
            image_path = os.path.join(app.config["UPLOAD_FOLDER_IMAGES"], filename)

            # Inference with Model 1
            image = Image.open(image_path).convert("RGB")
            seq = process_model1(image)

            # Process output of Model 1 with Model 2
            with open(os.path.join(app.config['UPLOAD_FOLDER_JSON'], 'menu_data.json')) as jsonfile:
                json_data = json.load(jsonfile)
            items = process_model2(json_data)

            # Save output of Model 2 to final JSON file
            final_json_file_path = 'final_data.json'
            with open(os.path.join(app.config['UPLOAD_FOLDER_FINAL_JSON'], final_json_file_path), 'w') as final_jsonfile:
                json.dump(items, final_jsonfile, indent=2)

            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Success predicting"
                },
                "data": {
                    "final_data": items,
                }
            }), 200
        else:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Invalid file format. Please upload a JPG, JPEG, or PNG image."
                },
                "data": None,
            }), 400
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed"
            },
            "data": None,
        }), 405

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
