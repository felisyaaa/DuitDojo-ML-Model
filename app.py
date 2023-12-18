<<<<<<< HEAD
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

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Fetch model & processor
processor = DonutProcessor.from_pretrained("Vasettha/Donut_Cord")
model = VisionEncoderDecoderModel.from_pretrained("Vasettha/Donut_Cord")
# processor = DonutProcessor.from_pretrained("https://huggingface.co/vasettha/Donut_Cord/tree/main")
# model = VisionEncoderDecoderModel.from_pretrained("https://huggingface.co/vasettha/Donut_Cord/tree/main")
# processor = DonutProcessor.from_pretrained('donut_model')
# model = VisionEncoderDecoderModel.from_pretrained('donut_model')
device = "cuda" if torch.cuda.is_available() else "cpu"

# Assuming your model and processor are already defined
model.eval()
model.to(device)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route("/")
def index():
    return "Hello World!"

@app.route("/donut", methods=["POST"])
def donut_route():
    if request.method == "POST":
        image = request.files["image"]
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            # Inference code
            image = Image.open(image_path).convert("RGB")
            pixel_values = processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)
            task_prompt = "<s_cord-v2>"
            decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
            decoder_input_ids = decoder_input_ids.to(device)

            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=model.decoder.config.max_position_embeddings,
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

            # CSV OUTPUT FOR CATEGORIZATION MODEL
            if isinstance(seq['menu'], list):
                menu_items = seq['menu']
            else:
                menu_items = [seq['menu']]

            # JSON OUTPUT FOR APP
            total_price = seq['total']['total_price']
            data_dict = {'menu_items': []}

            for item in menu_items:
                data_dict['menu_items'].append({
                    'Item': item.get('nm', ''),
                    'Quantity': item.get('cnt', ''),
                    'Price': item.get('price', '')
                })

            data_dict['total'] = {'Item': 'Total', 'Quantity': '', 'Price': total_price}

            # Writing data to JSON
            json_file_path = 'menu_data.json'
            with open(json_file_path, 'w') as jsonfile:
                json.dump(data_dict, jsonfile, indent=2)

            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Success predicting"
                },
                "data": {
                    "menu_data": data_dict,
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
    app.run(debug=True,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8080)))
=======
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

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Fetch model & processor
processor = DonutProcessor.from_pretrained("Vasettha/Donut_Cord")
model = VisionEncoderDecoderModel.from_pretrained("Vasettha/Donut_Cord")
# processor = DonutProcessor.from_pretrained("https://huggingface.co/vasettha/Donut_Cord/tree/main")
# model = VisionEncoderDecoderModel.from_pretrained("https://huggingface.co/vasettha/Donut_Cord/tree/main")
# processor = DonutProcessor.from_pretrained('donut_model')
# model = VisionEncoderDecoderModel.from_pretrained('donut_model')
device = "cuda" if torch.cuda.is_available() else "cpu"

# Assuming your model and processor are already defined
model.eval()
model.to(device)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route("/")
def index():
    return "Hello World!"

@app.route("/donut", methods=["POST"])
def donut_route():
    if request.method == "POST":
        image = request.files["image"]
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            # Inference code
            image = Image.open(image_path).convert("RGB")
            pixel_values = processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)
            task_prompt = "<s_cord-v2>"
            decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
            decoder_input_ids = decoder_input_ids.to(device)

            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=model.decoder.config.max_position_embeddings,
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

            # CSV OUTPUT FOR CATEGORIZATION MODEL
            if isinstance(seq['menu'], list):
                menu_items = seq['menu']
            else:
                menu_items = [seq['menu']]

            # JSON OUTPUT FOR APP
            total_price = seq['total']['total_price']
            data_dict = {'menu_items': []}

            for item in menu_items:
                data_dict['menu_items'].append({
                    'Item': item.get('nm', ''),
                    'Quantity': item.get('cnt', ''),
                    'Price': item.get('price', '')
                })

            data_dict['total'] = {'Item': 'Total', 'Quantity': '', 'Price': total_price}

            # Writing data to JSON
            json_file_path = 'menu_data.json'
            with open(json_file_path, 'w') as jsonfile:
                json.dump(data_dict, jsonfile, indent=2)

            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Success predicting"
                },
                "data": {
                    "menu_data": data_dict,
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
    app.run(debug=True,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8080)))
>>>>>>> f636b010cf867eb06180c6d538e6646fd73f5c2a
