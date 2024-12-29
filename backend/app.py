from flask_cors import CORS
from flask import Flask, request, jsonify, send_from_directory
import os
from lstm_strategy import lstm_strategy

# init flask app
app = Flask(__name__, static_folder='static')
CORS(app) # allow requests from React frontend

# route for training LSTM model
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
        units = data.get('units', 32)
        epochs_num = data.get('epochs_num', 20)

        if not ticker or not start_date or not end_date:
            return jsonify({'error': 'Missing required fields.'}), 400

        # run lstm strategy
        plot_filename1, plot_filename2 = lstm_strategy(ticker, start_date, end_date, seq_length, units, epochs_num)

        # return result as JSON response
        return jsonify({
            'plot_url1': f"/static/{plot_filename1}",
            'plot_url2': f"/static/{plot_filename2}",
        })
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
    app.run(host='0.0.0.0', port=5000, debug=True)

