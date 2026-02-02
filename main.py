"""
Flask API Server for LSTM Fitness Prediction Model
Menerima data kesehatan dari Huawei Watch dan mengembalikan 5 skor kebugaran
"""

import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # Enable CORS untuk Flutter app

# Load model saat startup
MODEL_PATH = os.environ.get('MODEL_PATH', 'fitbit_complete_model.keras')

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Output labels sesuai spesifikasi
OUTPUT_LABELS = [
    'sleep_score',
    'hrv_score', 
    'rhr_score',
    'recovery_score',
    'readiness_score'
]

# Input labels untuk validasi
INPUT_LABELS = ['StepTotal', 'Calories', 'heart_rate', 'stress']


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'input_shape': '(1, 6, 4) - 6 timesteps x 4 features',
        'output_shape': '(1, 5) - 5 fitness scores',
        'input_features': INPUT_LABELS,
        'output_scores': OUTPUT_LABELS
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict fitness scores dari data kesehatan
    
    Expected JSON body:
    {
        "data": [
            [steps, calories, heart_rate, stress],  // timestep 1
            [steps, calories, heart_rate, stress],  // timestep 2
            [steps, calories, heart_rate, stress],  // timestep 3
            [steps, calories, heart_rate, stress],  // timestep 4
            [steps, calories, heart_rate, stress],  // timestep 5
            [steps, calories, heart_rate, stress]   // timestep 6
        ]
    }
    
    Returns:
    {
        "success": true,
        "predictions": {
            "sleep_score": 0.85,
            "hrv_score": 0.72,
            "rhr_score": 0.68,
            "recovery_score": 0.79,
            "readiness_score": 0.81
        }
    }
    """
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Check server logs.'
        }), 500
    
    try:
        # Get data from request
        data = request.json
        
        if 'data' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "data" field in request body'
            }), 400
        
        input_data = data['data']
        
        # Validate input shape
        if len(input_data) != 6:
            return jsonify({
                'success': False,
                'error': f'Expected 6 timesteps, got {len(input_data)}'
            }), 400
        
        for i, row in enumerate(input_data):
            if len(row) != 4:
                return jsonify({
                    'success': False,
                    'error': f'Timestep {i+1} should have 4 features, got {len(row)}'
                }), 400
        
        # Convert to numpy array dan reshape untuk model
        # Model expects shape: (batch_size, timesteps, features) = (1, 6, 4)
        input_array = np.array(input_data, dtype=np.float32).reshape(1, 6, 4)
        
        # Run prediction
        predictions = model.predict(input_array, verbose=0)
        
        # Format output - handle both single output and multi-output models
        result = {
            'success': True,
            'predictions': {}
        }
        
        # Check if predictions is a list (multi-output model) or single array
        if isinstance(predictions, list):
            # Multi-output model: each output is a separate array
            for i, label in enumerate(OUTPUT_LABELS):
                if i < len(predictions):
                    # Each prediction[i] is shape (1, 1) or (1,)
                    val = predictions[i]
                    if hasattr(val, 'flatten'):
                        val = val.flatten()[0]
                    result['predictions'][label] = float(val)
                else:
                    result['predictions'][label] = 0.0
        else:
            # Single output model: predictions is shape (1, 5)
            for i, label in enumerate(OUTPUT_LABELS):
                if i < predictions.shape[-1]:
                    result['predictions'][label] = float(predictions[0][i])
                else:
                    result['predictions'][label] = 0.0
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction untuk multiple samples
    
    Expected JSON body:
    {
        "samples": [
            {"data": [[...], [...], [...], [...], [...], [...]]},
            {"data": [[...], [...], [...], [...], [...], [...]]}
        ]
    }
    """
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    try:
        samples = request.json.get('samples', [])
        results = []
        
        for sample in samples:
            input_data = sample.get('data', [])
            if len(input_data) == 6 and all(len(row) == 4 for row in input_data):
                input_array = np.array(input_data, dtype=np.float32).reshape(1, 6, 4)
                predictions = model.predict(input_array, verbose=0)
                
                result = {}
                for i, label in enumerate(OUTPUT_LABELS):
                    result[label] = float(predictions[0][i])
                results.append({'success': True, 'predictions': result})
            else:
                results.append({'success': False, 'error': 'Invalid input shape'})
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'true').lower() == 'true'
    
    print(f"🚀 Starting Flask API server on port {port}")
    print(f"   Debug mode: {debug}")
    print(f"   Endpoints:")
    print(f"     GET  /health  - Health check")
    print(f"     POST /predict - Single prediction")
    print(f"     POST /predict/batch - Batch prediction")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
