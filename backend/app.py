from flask_cors import CORS
from flask import Flask, request, jsonify, send_from_directory, Response
import os
import time
import json
from lstm_strategy_v2 import LSTMStockPredictor
from lstm_strategy_Vertige import LSTMStockPredictor as VertigeLSTMPredictor
import yfinance as yf
import threading
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Production mode detection
IS_PRODUCTION = os.getenv('RENDER') is not None or os.getenv('RAILWAY_ENVIRONMENT') is not None

# Global predictor instance and startup status
predictor = None
request_data = {}
startup_status = {
    'is_ready': False,
    'current_step': 'Starting backend...',
    'progress_percent': 0,
    'start_time': datetime.now(),
    'steps_completed': 0,
    'total_steps': 6
}

@app.route('/train', methods=['POST', 'OPTIONS'])
def train_model():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    # Check if backend is ready
    if not startup_status['is_ready']:
        return jsonify({
            'error': 'Backend is still initializing. Please wait...',
            'current_step': startup_status['current_step'],
            'progress_percent': startup_status['progress_percent']
        }), 503  # Service Unavailable
    
    try:
        # parse request data
        data = request.get_json()
        model_type = data.get('model_type', 'lstm_v2')  # Default to v2
        tickers = data.get('tickers', [])  # List of tickers
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        seq_length = data.get('seq_length', 30)

        if not tickers or not start_date or not end_date:
            return jsonify({'error': 'Missing required fields.'}), 400

        # In production, check if we can use existing models
        if IS_PRODUCTION:
            try:
                if model_type == 'lstm_v2':
                    predictor_check = LSTMStockPredictor('models/lstm_50_stocks_model.keras', seq_length)
                    if not predictor_check.can_predict_without_training(tickers, start_date, end_date):
                        return jsonify({
                            'error': f'Model training disabled in production. Please use date ranges within the trained model bounds: {predictor_check.training_start_date} to {predictor_check.training_end_date}',
                            'training_start_date': predictor_check.training_start_date,
                            'training_end_date': predictor_check.training_end_date,
                            'supported_tickers': predictor_check.training_tickers
                        }), 400
                        
                elif model_type == 'lstm_vertige':
                    predictor_check = VertigeLSTMPredictor('models/lstm_vertige_model.keras', seq_length)
                    if not predictor_check.can_predict_without_training(tickers, start_date, end_date):
                        return jsonify({
                            'error': 'Vertige model requires exact matching tickers and dates within training range in production.',
                            'training_tickers': predictor_check.training_tickers,
                            'training_start_date': predictor_check.training_start_date,
                            'training_end_date': predictor_check.training_end_date
                        }), 400
            except Exception as e:
                return jsonify({'error': f'Error validating model compatibility: {str(e)}'}), 500

        # Validate ticker count based on model type
        if model_type == 'lstm_vertige':
            if len(tickers) > 7:
                return jsonify({'error': 'Maximum 7 stocks allowed for Vertige model.'}), 400
            # Validate that tickers are from the allowed Vertige set
            allowed_vertige_stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'META']
            invalid_tickers = [t for t in tickers if t not in allowed_vertige_stocks]
            if invalid_tickers:
                return jsonify({'error': f'Invalid tickers for Vertige model: {invalid_tickers}. Allowed: {allowed_vertige_stocks}'}), 400
        elif model_type == 'lstm_v2':
            if len(tickers) > 10:
                return jsonify({'error': 'Maximum 10 stocks allowed for V2 model.'}), 400
        else:
            return jsonify({'error': f'Unknown model type: {model_type}'}), 400

        # Store request data
        request_data['model_type'] = model_type
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
        model_type = request_data.get('model_type', 'lstm_v2')
        tickers = request_data.get('tickers')
        start_date = request_data.get('start_date')
        end_date = request_data.get('end_date')
        seq_length = request_data.get('seq_length')

        def generate():
            global predictor
            
            if model_type == 'lstm_vertige':
                yield 'data: {"progress": "Initializing LSTM Vertige predictor..."}\n\n'
                time.sleep(1)
                predictor = VertigeLSTMPredictor('models/lstm_vertige_model.keras', seq_length)
                
                # Store current dates for metadata
                predictor._current_start_date = start_date
                predictor._current_end_date = end_date
                
                # Check if we can use existing model for prediction only
                if predictor.can_predict_without_training(tickers, start_date, end_date):
                    yield 'data: {"progress": "Using existing Vertige model - no training needed!"}\n\n'
                    time.sleep(0.5)
                else:
                    yield 'data: {"progress": "Training/Loading Vertige model..."}\n\n'
                    time.sleep(1)
                    
                    # For Vertige, train on the 7 specified stocks
                    data = predictor.fetch_and_prepare_data(tickers, start_date, end_date)
                    normalized_data = predictor.normalize_data(data)
                    X, y, _ = predictor.create_sequences(normalized_data, tickers)
                    X_test, y_test = predictor.train_save_model(X, y, tickers)
                
            else:  # lstm_v2
                yield 'data: {"progress": "Initializing LSTM v2 predictor..."}\n\n'
                time.sleep(1)
                predictor = LSTMStockPredictor('models/lstm_50_stocks_model.keras', seq_length)
                
                # Check if we can use existing model for prediction only
                if predictor.can_predict_without_training(tickers, start_date, end_date):
                    yield 'data: {"progress": "Using existing 50-stock model - no training needed!"}\n\n'
                    time.sleep(0.5)
                else:
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
                if model_type == 'lstm_vertige':
                    results = predictor.backtest_stock(ticker, start_date, end_date, tickers)
                else:  # lstm_v2
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
@app.route('/model/compatibility', methods=['POST'])
def check_model_compatibility():
    """Check if requested parameters are compatible with existing models"""
    try:
        data = request.get_json()
        model_type = data.get('model_type', 'lstm_v2')
        tickers = data.get('tickers', [])
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        seq_length = data.get('seq_length', 60)
        
        # Create temporary predictor instance to check compatibility
        if model_type == 'lstm_vertige':
            temp_predictor = VertigeLSTMPredictor('models/lstm_vertige_model.keras', seq_length)
        else:
            temp_predictor = LSTMStockPredictor('models/lstm_50_stocks_model.keras', seq_length)
        
        can_predict = temp_predictor.can_predict_without_training(tickers, start_date, end_date)
        
        response = {
            'compatible': can_predict,
            'model_exists': temp_predictor.model is not None,
            'training_tickers': temp_predictor.training_tickers,
            'training_start_date': getattr(temp_predictor, 'training_start_date', None),
            'training_end_date': getattr(temp_predictor, 'training_end_date', None)
        }
        
        if not can_predict and temp_predictor.model is not None:
            # Provide specific reasons why it's not compatible
            reasons = []
            if model_type == 'lstm_v2':
                if not all(ticker in temp_predictor.TOP_50_SP500 for ticker in tickers):
                    invalid_tickers = [t for t in tickers if t not in temp_predictor.TOP_50_SP500]
                    reasons.append(f"Invalid tickers for v2 model: {invalid_tickers}")
            else:  # vertige
                if not temp_predictor._tickers_match(tickers):
                    reasons.append(f"Tickers don't match training set: {temp_predictor.training_tickers}")
            
            if not temp_predictor._date_range_compatible(start_date, end_date):
                reasons.append(f"Date range {start_date} to {end_date} is outside training range {temp_predictor.training_start_date} to {temp_predictor.training_end_date}")
            
            response['incompatibility_reasons'] = reasons
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    import os
    model_files_exist = {
        'v2_model': os.path.exists('models/lstm_50_stocks_model.keras'),
        'v2_metadata': os.path.exists('models/lstm_50_stocks_model_metadata.json'),
        'vertige_model': os.path.exists('models/lstm_vertige_model.keras'),
        'vertige_metadata': os.path.exists('models/lstm_vertige_model_metadata.json'),
        'models_dir': os.path.exists('models')
    }
    
    return jsonify({
        'status': 'healthy' if startup_status['is_ready'] else 'initializing',
        'service': 'lstm-visualizer-backend',
        'environment': 'production' if IS_PRODUCTION else 'development',
        'ready': startup_status['is_ready'],
        'current_step': startup_status['current_step'],
        'progress_percent': startup_status['progress_percent'],
        'steps_completed': startup_status['steps_completed'],
        'total_steps': startup_status['total_steps'],
        'model_files': model_files_exist
    }), 200

# Startup progress streaming endpoint
@app.route('/startup-progress')
def startup_progress():
    def generate():
        while not startup_status['is_ready']:
            # Create a JSON-serializable copy of startup_status
            status_copy = startup_status.copy()
            if 'start_time' in status_copy:
                status_copy['start_time'] = status_copy['start_time'].isoformat()
            yield f"data: {json.dumps(status_copy)}\n\n"
            time.sleep(1)  # Send update every second
        
        # Send final ready status
        status_copy = startup_status.copy()
        if 'start_time' in status_copy:
            status_copy['start_time'] = status_copy['start_time'].isoformat()
        yield f"data: {json.dumps(status_copy)}\n\n"
    
    return Response(generate(), content_type='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*'
    })

def update_startup_progress(step_description, step_number):
    """Update the startup progress status"""
    startup_status['current_step'] = step_description
    startup_status['steps_completed'] = step_number
    startup_status['progress_percent'] = int((step_number / startup_status['total_steps']) * 100)
    print(f"Startup Progress: {startup_status['progress_percent']}% - {step_description}")

def initialize_backend():
    """Initialize the backend components with progress updates"""
    global predictor
    
    try:
        update_startup_progress("Loading Python dependencies...", 1)
        time.sleep(0.5)  # Simulate loading time
        
        update_startup_progress("Initializing TensorFlow/Keras...", 2)
        # This is where TensorFlow actually loads (usually the slowest part)
        import tensorflow as tf
        time.sleep(1)
        
        update_startup_progress("Setting up model directory...", 3)
        if not os.path.exists('models'):
            os.makedirs('models')
        if not os.path.exists('static'):
            os.makedirs('static')
        time.sleep(0.5)
        
        update_startup_progress("Loading LSTM predictor class...", 4)
        # Pre-initialize the predictor class (but don't load model yet)
        time.sleep(1)
        
        update_startup_progress("Checking yfinance connectivity...", 5)
        # Quick test to ensure yfinance is working
        test_ticker = yf.Ticker("AAPL")
        test_data = test_ticker.history(period="1d")
        if test_data.empty:
            raise Exception("Cannot connect to Yahoo Finance")
        time.sleep(0.5)
        
        update_startup_progress("Backend ready!", 6)
        startup_status['is_ready'] = True
        
        print("Backend initialization complete!")
        
    except Exception as e:
        startup_status['current_step'] = f"Error during initialization: {str(e)}"
        startup_status['progress_percent'] = -1  # Error state
        print(f"Backend initialization failed: {str(e)}")

# Start backend initialization in a separate thread
initialization_thread = threading.Thread(target=initialize_backend)
initialization_thread.daemon = True
initialization_thread.start()

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

