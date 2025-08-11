#!/usr/bin/env python3
"""
Re-train models with explicit TensorFlow compatibility settings
This script addresses the LSTM layer compatibility issues seen on Render
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Import our strategy classes
from lstm_strategy_v2 import LSTMStockPredictor as LSTMPredictor50
from lstm_strategy_Vertige import LSTMStockPredictor as LSTMPredictorVertige

def create_compatible_50_stock_model():
    """Create a TensorFlow-compatible 50-stock model"""
    print("=" * 60)
    print("TRAINING COMPATIBLE 50-STOCK LSTM MODEL")
    print("=" * 60)
    
    # Use explicit constructor to ensure compatibility
    predictor = LSTMPredictor50('models/lstm_50_stocks_model.h5', seq_length=30)
    
    # Clear any existing model
    predictor._clear_model()
    
    # Force training from scratch with full date range
    predictor._current_start_date = "2010-01-01"
    predictor._current_end_date = "2024-12-31"
    
    # Train on first stock to trigger model creation
    X_test, y_test = predictor.train_model_on_all_50_stocks(start_date='2010-01-01', end_date='2024-12-31')
    
    print(f"Model saved successfully!")
    print(f"Model file size: {os.path.getsize(predictor.model_path) / 1024:.1f} KB")
    
    return predictor

def create_compatible_vertige_model():
    """Create a TensorFlow-compatible Vertige model"""
    print("\n" + "=" * 60)
    print("TRAINING COMPATIBLE VERTIGE LSTM MODEL")
    print("=" * 60)
    
    # Use explicit constructor to ensure compatibility
    predictor = LSTMPredictorVertige('models/lstm_vertige_model.h5', seq_length=30)
    
    # Clear any existing model
    predictor._clear_model()
    
    # Define Vertige 7 stocks
    vertige_stocks = ['AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA', 'TSLA']
    
    # Force training from scratch with full date range
    predictor._current_start_date = "2010-01-01"
    predictor._current_end_date = "2024-12-31"
    
    # Train the model using the correct workflow
    print("Fetching and preparing data...")
    data = predictor.fetch_and_prepare_data(vertige_stocks, "2010-01-01", "2024-12-31")
    normalized_data = predictor.normalize_data(data)
    predictor.training_tickers = vertige_stocks  # Store training tickers
    X, y, _ = predictor.create_sequences(normalized_data, vertige_stocks)
    X_test, y_test = predictor.train_save_model(X, y, vertige_stocks)
    
    print(f"Model saved successfully!")
    print(f"Model file size: {os.path.getsize(predictor.model_path) / 1024:.1f} KB")
    
    return predictor

def main():
    """Main training function"""
    print("Starting compatible model training...")
    print(f"Python version: {sys.version}")
    
    # Import TensorFlow to check version
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    try:
        # Train both models
        model_50 = create_compatible_50_stock_model()
        model_vertige = create_compatible_vertige_model()
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print("Both models have been retrained with TensorFlow compatibility fixes.")
        print("Models should now work correctly on Render's TensorFlow environment.")
        print("\nNext steps:")
        print("1. Commit and push these new model files")
        print("2. Wait for Render to deploy")
        print("3. Test the frontend to verify green compatibility status")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
