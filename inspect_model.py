import numpy as np
import tensorflow as tf
import os

MODEL_PATH = 'model_multitask_lstm2.keras'

# Normalization Params (Manual copy from main.py to emulate server)
NORMALIZATION_PARAMS = {
    'steps': [0, 20000],
    'calories': [0, 4000],
    'heart_rate': [40, 180],
    'stress': [0, 100]
}

def get_normalized_input(steps, cal, hr, stress):
    # Create (1, 6, 4) array
    data = np.zeros((1, 6, 4), dtype=np.float32)
    # 0: steps, 1: cal, 2: hr, 3: stress
    
    # Normalize
    n_steps = (steps - 0) / 20000
    n_cal = (cal - 0) / 4000
    n_hr = (hr - 40) / 140
    n_stress = (stress - 0) / 100
    
    # Fill array
    data[:, :, 0] = n_steps
    data[:, :, 1] = n_cal
    data[:, :, 2] = n_hr
    data[:, :, 3] = n_stress
    
    # Clip to 0-1
    return np.clip(data, 0, 1)

try:
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded.")

    def print_readiness(predictions, label):
        # predictions is a list of 5 arrays. Readiness is index 4.
        # Each array shape (1, 1).
        val = predictions[4][0][0]
        print(f"[{label}] Readiness Score: {val:.4f}")

    # Test 1: Zeros
    input_zeros = get_normalized_input(0, 0, 0, 0)
    pred_zeros = model.predict(input_zeros, verbose=0)
    print_readiness(pred_zeros, "Zeros (Norm 0)")
    
    # Test 2: High (Norm 1)
    input_high = get_normalized_input(20000, 4000, 70, 10)
    pred_high = model.predict(input_high, verbose=0)
    print_readiness(pred_high, "High (Norm 1)")
    
    # Test 6: RAW (20k)
    raw_data = np.zeros((1, 6, 4), dtype=np.float32)
    raw_data[:, :, 0] = 20000
    raw_data[:, :, 1] = 3000
    raw_data[:, :, 2] = 70
    raw_data[:, :, 3] = 10
    pred_raw = model.predict(raw_data, verbose=0)
    print_readiness(pred_raw, "RAW (20k)")

except Exception as e:
    print(f"❌ Error: {e}")
