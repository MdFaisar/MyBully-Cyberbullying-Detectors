import os

# Disable TensorFlow oneDNN custom ops and suppress INFO logs to avoid
# repeated informational messages that slow startup and page loads.
# These environment variables must be set before importing libraries that may
# import TensorFlow (e.g., transformers). Setting them here (the app entry
# point) ensures they take effect early.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from flask import Flask, render_template, request, jsonify
from model import BullyingDetector
from ocr import analyze_image_bytes
import threading
feedback_log = []

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'

try:
    detector = BullyingDetector(model_dir='models')
    model_loaded = True
except FileNotFoundError as e:
    print(f"Error: {e}")
    model_loaded = False
    detector = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    if not model_loaded:
        return render_template('dashboard.html', error="Model not loaded. Please train the model first.")
    return render_template('dashboard.html', error=None)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.',
            'success': False
        }), 500
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text or text.strip() == '':
            return jsonify({
                'error': 'Please provide text to analyze',
                'success': False
            }), 400
        
        result = detector.analyze(text)
        result['success'] = True
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'success': False
        }), 500

@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.',
            'success': False
        }), 500
    
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts or len(texts) == 0:
            return jsonify({
                'error': 'Please provide texts to analyze',
                'success': False
            }), 400
        
        results = detector.batch_analyze(texts)
        
        return jsonify({
            'results': results,
            'success': True,
            'count': len(results)
        })
    
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'success': False
        }), 500

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded
    })


@app.route('/api/ocr', methods=['POST'])
def ocr_route():
    """Accepts an image file upload (multipart/form-data with key 'image'),
    runs OCR to extract text, and forwards the text to the model for analysis.
    Returns JSON: { extracted_text, analysis, success }
    """
    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.',
            'success': False
        }), 500

    if 'image' not in request.files:
        return jsonify({
            'error': 'No image file part in the request (use key "image").',
            'success': False
        }), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({
            'error': 'No selected file',
            'success': False
        }), 400

    try:
        img_bytes = file.read()
        result = analyze_image_bytes(img_bytes, detector)
        # result already contains extracted_text and analysis keys
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'error': f'An error occurred during OCR or analysis: {str(e)}',
            'success': False
        }), 500

@app.route('/api/correct', methods=['POST'])
def correct():
    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.',
            'success': False
        }), 500

    try:
        data = request.get_json()
        text = data.get('text', '')
        original_prediction = data.get('original_prediction', None)
        user_correction = data.get('user_correction', None)  

        if not text or user_correction not in ['correct', 'wrong']:
            return jsonify({
                'error': 'Missing text or correction value',
                'success': False
            }), 400

        feedback_log.append({
            'text': text,
            'original_prediction': original_prediction,
            'user_correction': user_correction
        })

        def update_model():
            if user_correction == 'correct':
                detector.update_confidence(text, original_prediction)
            else:
                opposite = 'Bullying' if original_prediction == 'Non-Bullying' else 'Non-Bullying'
                detector.update_confidence(text, opposite)

        threading.Thread(target=update_model).start()

        return jsonify({
            'success': True,
            'message': 'Feedback received and model updated.'
        })
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'success': False
        }), 500


if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    
    if not model_loaded:
        print("\n" + "="*60)
        print("WARNING: BERT model not found!")
        print("Please run 'python train_model.py' to train the BERT model first.")
        print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)