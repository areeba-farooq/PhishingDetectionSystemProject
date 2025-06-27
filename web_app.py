import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
from feature_extractor import URLFeatureExtractor

app = Flask(__name__)
CORS(app)

# Load trained model and scaler
try:
    model = joblib.load('best_phishing_model.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    model_name = joblib.load('best_model_name.pkl')
    print(f"Loaded model: {model_name}")
except:
    print("Warning: Model files not found. Please run phishing_detection.py first.")
    model = None
    scaler = None

# Initialize feature extractor
extractor = URLFeatureExtractor()

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')
@app.route('/style.css')
def css():
    return send_from_directory(os.path.join(app.root_path, 'templates'), 'style.css')
@app.route('/predict', methods=['POST'])
def predict():
    """Predict if URL is phishing"""
    try:
        data = request.get_json()
        url = data.get('url', '')
        
        if not url:
            return jsonify({
                'error': 'No URL provided',
                'success': False
            }), 400
        
        # Extract features
        features = extractor.extract_features(url)
        features_array = np.array(features).reshape(1, -1)
        
        if model and scaler:
            # Scale features
            features_scaled = scaler.transform(features_array)
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            
            confidence = 0.85 
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features_scaled)[0]
                confidence = max(proba)
            
            result = {
                'url': url,
                'is_phishing': bool(prediction),
                'prediction': 'PHISHING' if prediction == 1 else 'LEGITIMATE',
                'confidence': float(confidence),
                'features': {
                    'ip_address': features[0],
                    'url_length': features[1],
                    'shortening_service': features[2],
                    'at_symbol': features[3],
                    'double_slash': features[4],
                    'prefix_suffix': features[5],
                    'sub_domains': features[6],
                    'ssl_state': features[7]
                },
                'success': True
            }
        else:
            # Demo mode without model
            is_suspicious = any([
                '192.168' in url,
                '@' in url,
                'bit.ly' in url,
                url.count('.') > 3,
                not url.startswith('https')
            ])
            
            result = {
                'url': url,
                'is_phishing': is_suspicious,
                'prediction': 'PHISHING' if is_suspicious else 'LEGITIMATE',
                'confidence': 0.75 if is_suspicious else 0.85,
                'demo_mode': True,
                'success': True
            }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/stats')
def stats():
    """Get system statistics"""
    stats = {
        'model_type': model_name if model else 'Demo Mode',
        'accuracy': 0.9534, 
        'total_features': 30,
        'training_samples': 11055,
        'status': 'Active' if model else 'Demo Mode'
    }
    return jsonify(stats)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    print("="*50)
    print("PHISHING DETECTION WEB APPLICATION")
    print("ITSOLERA PVT LTD")
    print("="*50)
    print("\nStarting server at http://localhost:5000")
    print("Press Ctrl+C to stop")
    
    app.run(debug=True, port=5000)