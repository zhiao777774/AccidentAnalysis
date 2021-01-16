import re
import json
import pathlib
import logging
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template, redirect, url_for
from flask_socketio import SocketIO
from flask_cors import CORS

from utils.preprocess import TMA1, TMA2, TMA3
import utils.analyse as analyse
import utils.model as model


app = Flask(__name__)
CORS(app)
socketio = SocketIO(app)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


_ROOT_PATH = str(pathlib.PurePath(__file__).parent)

_DATA = [TMA1, TMA2, TMA3]
_JSON_DATA = [df.head(10000).to_json(orient='table', index=False)
              for df in _DATA]

_DEAD_KNN_MODEL, _INJURED_KNN_MODEL = analyse.KNN(TMA1)
# _DEAD_KNN_MODEL, _INJURED_KNN_MODEL = analyse.KNN(pd.concat([TMA1, TMA2], ignore_index=True))
_SVM_MODEL = analyse.SVM(pd.concat([TMA1, TMA2], ignore_index=True))
_RNN_MODEL = model.load_model('./static/data/model')
_KMEANS_MODEL = analyse.kmeans(TMA1)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', dataSize=[len(d) for d in (TMA1, TMA2, TMA3)])

'''
@app.route('/buttons', methods=['GET'])
def buttons():
    return render_template('buttons.html')
'''
'''
@app.route('/cards', methods=['GET'])
def cards():
    return render_template('cards.html')
'''

@app.route('/chart', methods=['GET'])
def chart():
    return render_template('chart.html', data=_JSON_DATA)


@app.route('/table', methods=['GET', 'POST'])
def table():
    if request.method == 'POST':
        data_idx = request.args.get('data') or 0
        position = request.args.get('position') or 'head'
        fmt = request.args.get('format') or 'table'

        df = (TMA1, TMA2, TMA3)[int(data_idx)]
        df = df.head(int(10000)) if position == 'head' else df.tail(int(10000))
        json_data = df.to_json(orient=fmt, index=False)
        return jsonify(json_data)
    else:
        tma1_json = TMA1.head(int(10000)).to_json(orient='table', index=False)
        return render_template('table.html', data=tma1_json)
        '''
        tables_html = []
        for df in (TMA1, TMA2, TMA3):
            html = df.head(10).to_html(classes='table table-bordered', 
                header='true', index=False, table_id='tma-df', justify='center')
            html = re.sub(r'<table([^>]*)>', r'<table\1 width="100%" cellspacing="0">', html)
            tables_html.append(html)
        return render_template('table.html', data = tables_html)
        '''


@app.route('/calculateRisks', methods=['POST'])
def calculate_risks():
    inp = request.get_json()['input']
    sex = inp.pop(0)
    age = inp.pop(0)
    dead_probs = _DEAD_KNN_MODEL.predict_proba([inp])
    injured_probs = _INJURED_KNN_MODEL.predict_proba([inp])
    predicted_probs = [dead_probs, injured_probs]

    weights = [[1, 1.3, 1.6], [1, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5]]
    risks = []
    for idx, prob in enumerate(predicted_probs):
        for p in prob:
            risks.append(np.sum(p * weights[idx]))

    risks = np.sum(risks) / round(np.sum(np.sum(weights)), 2)
    risks *= 1 if sex == 0 else 1.3
    risks *= [2, 1, 1, 1.2, 1.5, 1.2 * 2.9][age]
    probs = json.loads(pd.Series(predicted_probs).to_json(orient='values'))
    probs = [probs[0][0], probs[1][0]]
    res = {
        'risks': risks,
        'probs': probs
    }

    return jsonify(res)


@app.route('/calculateInjuredProbs', methods=['POST'])
def calculate_injured_probs():
    inp = request.get_json()['input']
    predicted_probs = [
        _INJURED_KNN_MODEL.predict_proba([inp]),
        _SVM_MODEL.predict_proba([inp]),
        [_RNN_MODEL.predict([inp])[0][:8]]
    ]

    probs = json.loads(pd.Series(predicted_probs).to_json(orient='values'))
    probs = [probs[0][0], probs[1][0], probs[2][0]]
    res = {
        'probs': probs
    }

    return jsonify(res)


@ socketio.on('connect')
def socket_connect():
    print(f'socket connect id: {request.sid}')


@ socketio.on('disconnect')
def socket_disconnect():
    print(f'socket disconnect id: {request.sid}')


if __name__ == '__main__':
    port = 8090
    app.run(port=port, debug=True)