import numpy as np
import pandas as pd
import joblib
import json
import websocket
from threading import Thread
from flask import Flask, render_template, redirect, url_for, request, jsonify
from sklearn.preprocessing import MinMaxScaler

from search_alg import analyze

socket = "wss://stream.binance.com:9443/ws/btcusdt@trade"
#socket_info = []
records = []
transaction_window_size = 1000
recording = False
model = joblib.load('../models/sklearn_svm/model_10_04_23_1.joblib')

# Websocket operations
def on_close(ws):
    print("closed")

def on_error(ws, message):
    print(message)

def on_message(ws, message):
    global records
    global transaction_window_size

    if recording:
        msg_str = str(message)
        #pd.read_json('[' + msg_str + ']', orient='records')
        msg_json = json.loads(msg_str)
        #data_over_limit = records.shape[0] - transaction_window_size
        records.append(msg_json)
        #socket_info.append(msg_str)

# Run application
app = Flask(__name__)
ws = websocket.WebSocketApp(socket, on_message=on_message, on_close=on_close, on_error=on_error)
ws_thread = Thread(target=ws.run_forever, args=(None, None, 60, 30))
ws_thread.start()

# Flask endpoints
@app.route("/")
def hello_world():
    return render_template('index.html', df_shape = len(records))

@app.route("/info")
def info():
    return str(records.shape)

@app.route("/toggleRecording")
def toggle_recording():
    global recording
    recording = not recording
    return redirect('/')

@app.route("/analyze")
def analyzeRecordedData():
    result = analyze(pd.DataFrame(records), model)
    return jsonify(result.to_dict('records'))

'''
Пользователь нажимает начать запись
Данные записываются в orig_df
Далее из него берутся последние transaction_window_size записей
Они проходят препроцессинг (каждый раз новый MinMaxScaler) и отправляются на анализ
Далее возвращаются результаты анализа и сопрягаются с транзакциями
Пользователь нажимает стоп и ему выводтся запись результатов
'''


