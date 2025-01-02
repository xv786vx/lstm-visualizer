from flask_cors import CORS
from flask import Flask, request, jsonify, send_from_directory, Response
import os
import time
import json
from lstm_strategy import lstm_strategy, fetch_stock_data, create_sequences, build_model, predict, plot_results
import yfinance as yf

app = Flask(__name__)
CORS(app)

request_data = {}

@app.route('/train', methods=['POST', 'OPTIONS'])
def train_model():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    try:
        # parse request data
        data = request.get_json()
        ticker = data.get('ticker').strip().upper()
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        seq_length = data.get('seq_length', 30)
        units = data.get('units', 12)
        epochs_num = data.get('epochs_num', 8)

        if not ticker or not start_date or not end_date:
            return jsonify({'error': 'Missing required fields.'}), 400

        # Store request data
        request_data['ticker'] = ticker
        request_data['start_date'] = start_date
        request_data['end_date'] = end_date
        request_data['seq_length'] = seq_length
        request_data['units'] = units
        request_data['epochs_num'] = epochs_num

        return jsonify({'status': 'Request received'}), 200
    except Exception as e:
        print("Error occurred:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/train/stream', methods=['GET'])
def stream_updates():
    try:
        ticker = request_data.get('ticker')
        start_date = request_data.get('start_date')
        end_date = request_data.get('end_date')
        seq_length = request_data.get('seq_length')
        units = request_data.get('units')
        epochs_num = request_data.get('epochs_num')

        def generate():
            yield 'data: {"progress": "Fetching stock data..."}\n\n'
            time.sleep(1)  # Simulate time delay
            data, X = fetch_stock_data(ticker, start_date, end_date)
            
            yield 'data: {"progress": "Creating sequences..."}\n\n'
            time.sleep(1)  # Simulate time delay
            X_seq, y_seq = create_sequences(data, seq_length)
            
            yield 'data: {"progress": "Building and training model..."}\n\n'
            time.sleep(1)  # Simulate time delay
            model, X_test, y_test = build_model(X_seq, y_seq, units, epochs_num)
            
            yield 'data: {"progress": "Making predictions..."}\n\n'
            time.sleep(1)  # Simulate time delay
            predicted_prices, actual_prices = predict(model, X_test, y_test, X)
            
            yield 'data: {"progress": "Plotting results..."}\n\n'
            time.sleep(1)  # Simulate time delay
            plot_path = plot_results(ticker, actual_prices, predicted_prices, start_date, end_date)
            
            yield f'data: {{"plot_url1": "/static/{plot_path[0]}", "plot_url2": "/static/{plot_path[1]}"}}\n\n'

        return Response(generate(), content_type='text/event-stream')
    except Exception as e:
        print("Error occurred:", str(e))
        return jsonify({'error': str(e)}), 500


@app.route('/earliest-start-date', methods=['GET'])
def earliest_start_date():
    try:
        ticker = request.args.get('ticker').strip().upper()
        stock = yf.Ticker(ticker)
        hist = stock.history(period='max')
        earliest_date = hist.index.min().strftime('%Y-%m-%d')
        return jsonify({'earliest_date': earliest_date}), 200
    except Exception as e:
        print("Error occurred:", str(e))
        return jsonify({'error': str(e)}), 500



# serve static files for plots
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')

    # run app
    # app.run(host='0.0.0.0', port=8080, debug=True)
    app.run(host='0.0.0.0', port=4999, debug=True)

