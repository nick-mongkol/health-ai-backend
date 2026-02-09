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
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS untuk Flutter app secara explicit

# Load model saat startup
MODEL_PATH = os.environ.get('MODEL_PATH', 'model_multitask_lstm2.keras')

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Input labels (features)
INPUT_LABELS = ['steps', 'calories', 'heart_rate', 'stress']

# Output labels sesuai spesifikasi
OUTPUT_LABELS = [
    'sleep_score',
    'hrv_score', 
    'rhr_score',
    'recovery_score',
    'readiness_score'
]

# Normalization parameters (based on typical health data ranges)
# [min, max] for each feature
NORMALIZATION_PARAMS = {
    'steps': [0, 20000],
    'calories': [0, 4000],
    'heart_rate': [40, 180],
    'stress': [0, 100]
}

def normalize_data(data):
    """
    Min-Max normalization untuk input data
    data: shape (1, 6, 4) or similar
    """
    data_norm = np.copy(data)
    
    # Iterate over features (last dimension)
    # 0: steps, 1: calories, 2: heart_rate, 3: stress
    features_indices = [0, 1, 2, 3] # Indices corresponding to INPUT_LABELS
    feature_keys = ['steps', 'calories', 'heart_rate', 'stress']
    
    for i, key in zip(features_indices, feature_keys):
        min_val, max_val = NORMALIZATION_PARAMS[key]
        # Normalize: (x - min) / (max - min)
        # Clip values to range [min, max] first to avoid outliers
        data_norm[:, :, i] = np.clip(data_norm[:, :, i], min_val, max_val)
        data_norm[:, :, i] = (data_norm[:, :, i] - min_val) / (max_val - min_val)
        
    return data_norm


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'input_shape': '(1, 6, 4) - 6 timesteps x 4 features',
        'output_shape': '(1, 5) - 5 fitness scores',
        'input_features': INPUT_LABELS,
        'output_scores': OUTPUT_LABELS,
        'normalization': 'MinMax Scaling applied on server (High values -> High scores)'
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict fitness scores dari data kesehatan
    ...
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
        
        # Model expects shape: (batch_size, timesteps, features) = (1, 6, 4)
        input_array = np.array(input_data, dtype=np.float32).reshape(1, 6, 4)
        
        # Apply normalization
        input_array_scaled = normalize_data(input_array)
        
        print(f"DEBUG: Input shape: {input_array.shape}")
        print(f"DEBUG: Raw Data Stats - Min: {np.min(input_array):.2f}, Max: {np.max(input_array):.2f}, Mean: {np.mean(input_array):.2f}")
        print(f"DEBUG: Scaled Data Stats - Min: {np.min(input_array_scaled):.2f}, Max: {np.max(input_array_scaled):.2f}, Mean: {np.mean(input_array_scaled):.2f}")
        
        # Run prediction
        predictions = model.predict(input_array_scaled, verbose=0)
        print(f"DEBUG: Raw predictions output: {predictions}")
        
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
                    
                    val = float(val)
                    # Heuristic: if <= 1.0, assume 0-1 and scale. Else assume 0-100.
                    if val <= 1.0 and val > 0: # Check >0 to avoid scaling 0
                         result['predictions'][label] = val * 100
                    else:
                         result['predictions'][label] = val
                else:
                    result['predictions'][label] = 0.0
        else:
            # Single output model: predictions is shape (1, 5)
            for i, label in enumerate(OUTPUT_LABELS):
                if i < predictions.shape[-1]:
                    val = float(predictions[0][i])
                    # Heuristic: if small value, assume 0-1 range and scale to percentage
                    if val <= 1.0 and val > 0:
                        result['predictions'][label] = val * 100
                    else:
                        result['predictions'][label] = val
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
