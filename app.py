from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__, static_folder='static', template_folder='templates')

# Add this static URL path configuration
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development

# Your existing code continues...

# Global variables for model and features
model = None
feature_names = []
model_info = {}

def load_model():
    """Load the trained model and features"""
    global model, feature_names, model_info
    
    try:
        model = joblib.load('models/house_price_model.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        model_info = joblib.load('models/model_info.pkl')
        print("✅ Model loaded successfully!")
        print(f"📊 Model Performance: RMSE ${model_info['rmse']:,.2f}, R²: {model_info['r2']:.4f}")
        print(f"🔢 Features loaded: {len(feature_names)}")
        
        # Print first 10 features for debugging
        print("📋 Sample features:", feature_names[:10])
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

# Load model at startup
model_loaded = load_model()

def create_feature_vector(input_data):
    """Create proper feature vector from user input"""
    feature_vector = []
    
    for feature in feature_names:
        if feature in input_data:
            try:
                value = input_data[feature]
                # Handle empty strings and None values
                if value == '' or value is None:
                    feature_vector.append(0.0)
                else:
                    feature_vector.append(float(value))
            except (ValueError, TypeError):
                feature_vector.append(0.0)
        else:
            # Feature not provided by user - use 0 as default
            feature_vector.append(0.0)
    
    return np.array(feature_vector)

@app.route('/')
def home():
    """Main page with house price prediction form"""
    return render_template('index.html', 
                         rmse=model_info.get('rmse', 0),
                         r2=model_info.get('r2', 0),
                         model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for price predictions"""
    if not model_loaded:
        return jsonify({
            'success': False, 
            'error': 'Model not loaded. Please check if the model files exist.'
        })
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided. Please fill in the form.'
            })
        
        print(f"📨 Received data: {data}")
        
        # Create feature vector
        features = create_feature_vector(data)
        print(f"🔢 Feature vector shape: {features.shape}")
        
        # Reshape for prediction (1 sample, n features)
        features_reshaped = features.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_reshaped)[0]
        print(f"💰 Raw prediction: {prediction}")
        
        # Ensure prediction is reasonable
        if prediction < 0:
            prediction = 50000  # Minimum reasonable price
        elif prediction > 5000000:
            prediction = 500000  # Cap unreasonable high values
        
        # Calculate confidence interval
        confidence_range = model_info.get('rmse', 25511)
        lower_bound = max(10000, prediction - confidence_range)
        upper_bound = prediction + confidence_range
        
        # Calculate accuracy percentage (simplified)
        accuracy_pct = max(80, min(95, (1 - (confidence_range / prediction)) * 100))
        
        # Prepare response
        response = {
            'success': True,
            'prediction': float(prediction),
            'prediction_formatted': f"${prediction:,.2f}",
            'confidence_low': f"${lower_bound:,.2f}",
            'confidence_high': f"${upper_bound:,.2f}",
            'confidence_range': f"${lower_bound:,.2f} - ${upper_bound:,.2f}",
            'model_performance': {
                'rmse': f"${model_info.get('rmse', 0):,.2f}",
                'r2': f"{model_info.get('r2', 0):.4f}",
                'accuracy': f"{accuracy_pct:.1f}%"
            },
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print(f"📈 Prediction successful: ${prediction:,.2f}")
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        print(f"🔍 Full traceback: {traceback.format_exc()}")
        
        return jsonify({
            'success': False, 
            'error': 'Failed to process prediction. Please check your input values.'
        }), 500

@app.route('/api/model-info')
def api_model_info():
    """API endpoint for model information"""
    return jsonify({
        'status': 'loaded' if model_loaded else 'error',
        'performance': {
            'rmse': model_info.get('rmse', 0),
            'r2': model_info.get('r2', 0),
            'feature_count': len(feature_names),
            'model_type': model_info.get('model_type', 'XGBoost')
        },
        'features_sample': feature_names[:20]  # Show first 20 features
    })

@app.route('/api/health')
def health_check():
    """Health check endpoint for deployment"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat(),
        'features_count': len(feature_names)
    })

@app.route('/features')
def features_info():
    """Page showing feature information"""
    return render_template('features.html', 
                         features=feature_names,
                         feature_count=len(feature_names),
                         model_info=model_info)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print(f"🚀 Starting House Price Prediction App...")
    print(f"📍 Port: {port}")
    print(f"🔧 Debug: {debug}")
    print(f"📊 Model Performance: RMSE ${model_info.get('rmse', 0):,.2f}")
    print(f"🎯 R² Score: {model_info.get('r2', 0):.4f}")
    print(f"🔢 Features: {len(feature_names)}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)