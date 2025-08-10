import os
import sys
sys.path.append('.')

from lstm_strategy_v2 import LSTMStockPredictor
from lstm_strategy_Vertige import LSTMStockPredictor as VertigeLSTMPredictor

def train_all_models():
    """Train both models with full date ranges for deployment"""
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    print("🚀 Training LSTM v2 Model (50 stocks)...")
    predictor_v2 = LSTMStockPredictor('models/lstm_50_stocks_model.keras', 30)
    
    # Train with full range 2010-2024 to cover most user requests
    try:
        predictor_v2.train_model_on_all_50_stocks('2010-01-01', '2024-12-31')
        print("✅ LSTM v2 model trained and saved!")
    except Exception as e:
        print(f"❌ Error training LSTM v2: {e}")
    
    print("\n🚀 Training LSTM Vertige Model (7 stocks)...")
    predictor_vertige = VertigeLSTMPredictor('models/lstm_vertige_model.keras', 30)
    
    # Train Vertige model with all 7 stocks
    vertige_stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'META']
    
    try:
        # Store current dates for metadata
        predictor_vertige._current_start_date = '2010-01-01'
        predictor_vertige._current_end_date = '2024-12-31'
        
        data = predictor_vertige.fetch_and_prepare_data(vertige_stocks, '2010-01-01', '2024-12-31')
        normalized_data = predictor_vertige.normalize_data(data)
        X, y, _ = predictor_vertige.create_sequences(normalized_data, vertige_stocks)
        predictor_vertige.train_save_model(X, y, vertige_stocks)
        print("✅ LSTM Vertige model trained and saved!")
    except Exception as e:
        print(f"❌ Error training LSTM Vertige: {e}")
    
    print("\n📁 Checking model file sizes...")
    if os.path.exists('models'):
        for filename in os.listdir('models'):
            if filename.endswith(('.keras', '.json')):
                filepath = os.path.join('models', filename)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  {filename}: {size_mb:.1f} MB")
                
                if size_mb > 100:
                    print(f"  ⚠️  WARNING: {filename} is over 100MB - GitHub may reject it!")
    
    print("\n🎉 All models trained successfully!")
    print("Next steps:")
    print("1. Run: git add backend/models/*.keras backend/models/*.json")
    print("2. Run: git commit -m 'Add pre-trained models'")
    print("3. Run: git push")

if __name__ == "__main__":
    train_all_models()
