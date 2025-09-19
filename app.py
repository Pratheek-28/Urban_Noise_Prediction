import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import joblib
import librosa
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 

# Import VGGish utilities
from vgg_utils import vggish_input
from vgg_utils import vggish_postprocess
from vgg_utils import vggish_params
from vgg_utils import vggish_slim

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.config['ALLOWED_EXTENSIONS'] = {'wav'}

# ESC-50 categories (in order)
CATEGORIES = [
    'dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insects', 'sheep', 'crow',
    'rain', 'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds', 'water_drops',
    'wind', 'pouring_water', 'toilet_flush', 'thunderstorm', 'crying_baby', 'sneezing',
    'clapping', 'breathing', 'coughing', 'footsteps', 'laughing', 'brushing_teeth',
    'snoring', 'drinking_sipping', 'door_wood_knock', 'mouse_click', 'keyboard_typing',
    'door_wood_creaks', 'can_opening', 'washing_machine', 'vacuum_cleaner', 'clock_alarm',
    'clock_tick', 'glass_breaking', 'helicopter', 'chainsaw', 'siren', 'car_horn',
    'engine', 'train', 'church_bells', 'airplane', 'fireworks', 'hand_saw'
]

def load_models():
    """Load all required models and TensorFlow session"""
    try:
        # Load the classifier pipeline (includes scaler)
        model_data = joblib.load('enhanced_audio_classifier.pkl')
        model = model_data['model']  # This is the full pipeline with scaler
        
        # VGGish model paths
        checkpoint_path = 'vggish_model.ckpt'
        pca_params_path = 'vggish_pca_params.npz'
        
        # Initialize VGGish postprocessor
        pproc = vggish_postprocess.Postprocessor(pca_params_path)
        
        # Set up TensorFlow session
        graph = tf.Graph()
        sess = tf.Session(graph=graph)
        
        with graph.as_default():
            vggish_slim.define_vggish_slim(training=False)
            # saver = tf.train.Saver()
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, checkpoint_path)
            features_tensor = graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
            embedding_tensor = graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
        
        return {
            'model': model,
            'session': sess,
            'graph': graph,
            'features_tensor': features_tensor,
            'embedding_tensor': embedding_tensor,
            'postprocessor': pproc,
            'categories': CATEGORIES
        }
    
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise

# Load models at startup
models = load_models()

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_features(audio_path):
    """Extract VGGish features from audio file"""
    try:
        # Load audio with librosa for better control
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Convert to VGGish examples
        examples = vggish_input.waveform_to_examples(y, sr)
        
        if examples.size == 0:
            raise ValueError("Empty feature array from audio file")
        
        # Get embeddings
        [embedding_batch] = models['session'].run(
            [models['embedding_tensor']],
            feed_dict={models['features_tensor']: examples}
        )
        
        # Postprocess embeddings
        postprocessed_batch = models['postprocessor'].postprocess(embedding_batch)
        
        # Temporal pooling - mean across time
        features = np.mean(postprocessed_batch, axis=0)
        
        # Reshape for prediction (1, 128)
        features = features.reshape(1, -1)
        
        return features
    
    except Exception as e:
        print(f"Feature extraction failed: {str(e)}")
        return None

def create_prediction_plot(probabilities, classes):
    """Create a bar plot of prediction probabilities"""
    plt.figure(figsize=(10, 6))
    plt.barh(classes, probabilities)
    plt.xlabel('Probability')
    plt.title('Top Prediction Probabilities')
    plt.tight_layout()
    
    # Save plot to a bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()

    plt.close('all')
    
    # Encode plot to base64 for HTML
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{plot_data}"

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', categories=models['categories'])

@app.route('/predict', methods=['POST'])
def predict():
    """Handle audio file upload and prediction"""
    if 'audio' not in request.files:
        return render_template('index.html', error="No audio file uploaded")
    
    file = request.files['audio']
    if file.filename == '':
        return render_template('index.html', error="No file selected")
    
    if not allowed_file(file.filename):
        return render_template('index.html', error="Invalid file type. Only WAV files are allowed.")
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract features
        features = extract_features(file_path)
        if features is None:
            return render_template('index.html', error="Error processing audio file")
        
        # Make prediction (pipeline handles scaling)
        prediction = models['model'].predict(features)
        predicted_index = prediction[0]
        predicted_label = models['categories'][predicted_index]
        
        # Get probabilities if available
        plot_url = None
        if hasattr(models['model'].steps[-1][1], 'predict_proba'):
            probabilities = models['model'].predict_proba(features)[0]
            
            # Get top 5 predictions
            top_n = 5
            top_indices = np.argsort(probabilities)[-top_n:][::-1]
            top_classes = [models['categories'][i] for i in top_indices]
            top_probabilities = probabilities[top_indices]
            
            # Create visualization
            plot_url = create_prediction_plot(top_probabilities, top_classes)
        
        return render_template('result.html', 
                            prediction=predicted_label,
                            filename=filename,
                            plot_url=plot_url,
                            categories=models['categories'])
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return render_template('index.html', error="An error occurred during processing")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only WAV files are allowed.'}), 400
    
    try:
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join('/tmp', filename)
        file.save(temp_path)
        
        # Extract features
        features = extract_features(temp_path)
        if features is None:
            return jsonify({'error': 'Error processing audio file'}), 500
        
        # Make prediction
        prediction = models['model'].predict(features)
        predicted_index = prediction[0]
        predicted_label = models['categories'][predicted_index]
        
        # Get probabilities if available
        probabilities = None
        if hasattr(models['model'].steps[-1][1], 'predict_proba'):
            probabilities = models['model'].predict_proba(features)[0].tolist()
            class_names = models['categories']
        else:
            class_names = None
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return jsonify({
            'prediction': predicted_label,
            'predicted_index': int(predicted_index),
            'probabilities': probabilities,
            'class_names': class_names
        })
    
    except Exception as e:
        print(f"API prediction error: {str(e)}")
        return jsonify({'error': 'An error occurred during processing'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)




































 