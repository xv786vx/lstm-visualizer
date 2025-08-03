from flask_cors import CORS
from flask import Flask, request, jsonify, send_from_directory, Response
import os
import time
import json
from lstm_strategy_v2 import LSTMStockPredictor
import yfinance as yf

app = Flask(__name__)
CORS(app)

# Global predictor instance
predictor = None
request_data = {}

@app.route('/train', methods=['POST', 'OPTIONS'])
def train_model():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    try:
        # parse request data
        data = request.get_json()
        tickers = data.get('tickers', [])  # List of tickers
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        seq_length = data.get('seq_length', 30)

        if not tickers or not start_date or not end_date:
            return jsonify({'error': 'Missing required fields.'}), 400

        # Validate ticker count
        if len(tickers) > 8:
            return jsonify({'error': 'Maximum 8 stocks allowed.'}), 400

        # Store request data
        request_data['tickers'] = tickers
        request_data['start_date'] = start_date
        request_data['end_date'] = end_date
        request_data['seq_length'] = seq_length

        return jsonify({'status': 'Request received'}), 200
    except Exception as e:
        print("Error occurred:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/train/stream', methods=['GET'])
def stream_updates():
    global predictor
    try:
        tickers = request_data.get('tickers')
        start_date = request_data.get('start_date')
        end_date = request_data.get('end_date')
        seq_length = request_data.get('seq_length')

        def generate():
            global predictor
            
            yield 'data: {"progress": "Initializing LSTM v2 predictor..."}\n\n'
            time.sleep(1)
            predictor = LSTMStockPredictor('models/lstm_50_stocks_model.keras', seq_length)
            
            yield 'data: {"progress": "Training/Loading 50-stock model..."}\n\n'
            time.sleep(1)
            
            yield 'data: {"progress": "Training/Loading 50-stock model..."}\n\n'
            time.sleep(1)
            
            # For v2, train on all 50 stocks if needed (model will reuse existing if compatible)
            X_test, y_test = predictor.train_model_on_all_50_stocks(start_date, end_date)
            
            yield 'data: {"progress": "Generating backtest results..."}\n\n'
            time.sleep(1)
            
            # Generate results for each ticker
            plot_urls = []
            metrics = {}
            for ticker in tickers:
                results = predictor.backtest_stock(ticker, start_date, end_date, tickers)
                if results is not None:
                    # Save plots to static folder
                    plot_filename = f'{ticker}_results.png'
                    plot_path = os.path.join('static', plot_filename)
                    
                    # Create and save the plot
                    ticker_metrics = predictor.plot_results(results, ticker, start_date, end_date, plot_path)
                    metrics[ticker] = ticker_metrics
                    
                    plot_urls.append(f'/static/{plot_filename}')
            
            yield f'data: {{"plot_urls": {json.dumps(plot_urls)}, "metrics": {json.dumps(metrics)}}}\n\n'

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

@app.route('/validate-tickers', methods=['POST'])
def validate_tickers():
    try:
        data = request.get_json()
        tickers = data.get('tickers', [])
        
        valid_tickers = []
        invalid_tickers = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker.strip().upper())
                hist = stock.history(period='1d')
                if not hist.empty:
                    valid_tickers.append(ticker.strip().upper())
                else:
                    invalid_tickers.append(ticker)
            except:
                invalid_tickers.append(ticker)
        
        return jsonify({
            'valid_tickers': valid_tickers,
            'invalid_tickers': invalid_tickers
        }), 200
    except Exception as e:
        print("Error occurred:", str(e))
        return jsonify({'error': str(e)}), 500



# Health check endpoint for Render
@app.route('/')
@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'service': 'lstm-visualizer-backend'}), 200

# serve static files for plots
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')

    # Get port from environment variable (for production) or use default for development
    port = int(os.environ.get('PORT', 4999))
    
    # run app
    app.run(host='0.0.0.0', port=port, debug=False)

