from flask import Flask, request, jsonify, render_template
import os
import ctypes
import json
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
from aux_functions import clean_text
from pred import predict_text
from ensemble_pred import predict_text_rfc
import sklearn

liblouis_bin_path = os.path.abspath('liblouis-bin')
os.environ['LD_LIBRARY_PATH'] = liblouis_bin_path
liblouis_so = os.path.join(liblouis_bin_path, 'liblouis.so.20')
ctypes.CDLL(liblouis_so)

from liblouis.python import louis

app = Flask(__name__)
CORS(app)

# load lstm model
model = tf.keras.models.load_model('english-v1.h5')
all_chars = np.load(os.path.join(os.getcwd(), 'encoded_chars.npy'))
type_chars = len(all_chars)
encoded = dict((c, i) for i, c in enumerate(all_chars))


# loading in RFC with pickle, and fivegram with probabilities with json
with open("fivegram_lexicon_all_prob.json", "r") as file:
    fivegram_model_prob = json.load(file)

with open('rfc_model.pkl', 'rb') as file:
    rfc_model = pickle.load(file)



TABLES_PATH = os.path.join(os.path.dirname(__file__), "liblouis", "tables")

# load fivegram model
with open("fivegram_lexicon.json") as l:
    fivegram_model = json.load(l)

def next_char(previous, data):
    key = previous[-5:]

    if key in data:
        return data[key]
    else:
        return "No available predictions"


@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    text = data.get('text', '')
    tableList = data.get('tableList', ["en-us-g2.ctb"])
    tableList = [os.path.join(TABLES_PATH, t) for t in tableList]
    braille = louis.translate(tableList, text)[0]
    return jsonify({"braille": braille})

@app.route('/backtranslate', methods=['POST'])
def backtranslate():
    data = request.get_json()
    braille = data.get('braille', '')
    tableList = data.get('tableList', ["en-us-g2.ctb"])
    tableList = [os.path.join(TABLES_PATH, t) for t in tableList]
    text = louis.backTranslate(tableList, braille)[0]
    return jsonify({"text": text})

@app.route('/lstm', methods=['POST'])
def lstm():
    data = request.get_json()
    text = data.get('text', '')
    # tableList = data.get('tableList', ["en-us-g2.ctb"])
    cleaned_text = clean_text(text)
    pred = predict_text(cleaned_text, model, all_chars, type_chars, encoded)
    return jsonify({"pred": pred})

@app.route('/fivegram', methods=['POST'])
def fivegram():
    data = request.get_json()
    text = data.get('text', '')
    # tableList = data.get('tableList', ["en-us-g2.ctb"])
    cleaned_text = clean_text(text)
    pred = next_char(cleaned_text, fivegram_model)
    return jsonify({"pred": pred})

@app.route('/randomforest', methods=['POST'])
def randomforest():
    data = request.get_json()
    text = data.get('text', '')
    # tableList = data.get('tableList', ["en-us-g2.ctb"])
    cleaned_text = clean_text(text)
    pred = predict_text_rfc(
        cleaned_text,
        rfc_model,
        model,
        fivegram_model_prob,
        all_chars,
        type_chars,
        encoded,
        temperature=0.8
    )
    return jsonify({"pred": pred})


@app.route('/demo', methods=['GET'])
def demo():
    return render_template('demo.html')

if __name__ == '__main__':
    app.run(debug=True)
