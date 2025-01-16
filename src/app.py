from flask import Flask, render_template, request, jsonify, url_for, session
import os
from werkzeug.utils import secure_filename
from pose_detector import PoseDetector
import logging
import traceback
from threading import Thread
from datetime import datetime

app = Flask(__name__, 
           template_folder='../templates',
           static_folder='../static')
app.secret_key = 'your-secret-key-here'  # Required for session

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Configure upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'videos')
DOWNLOADS_FOLDER = os.path.expanduser("~/Downloads")
ALLOWED_EXTENSIONS = {'mp4'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Store progress globally (in a real app, use proper session handling)
processing_progress = {}

def update_progress(video_id, progress):
    processing_progress[video_id] = progress

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/progress/<video_id>')
def get_progress(video_id):
    return jsonify({
        'progress': processing_progress.get(video_id, 0),
        'status': 'processing' if processing_progress.get(video_id, 0) < 100 else 'complete'
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an MP4 file.'}), 400
        
        # Generate unique ID for this upload
        video_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        processing_progress[video_id] = 0
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Initialize detector and set progress callback
        detector = PoseDetector()
        detector.set_progress_callback(lambda p: update_progress(video_id, p))
        
        # Process video in background thread
        def process_video_task():
            try:
                detector.process_video(input_path)
            except Exception as e:
                app.logger.error(f"Error processing video: {str(e)}")
                processing_progress[video_id] = -1  # Indicate error
        
        Thread(target=process_video_task).start()
        
        return jsonify({
            'video_id': video_id,
            'message': 'Video upload started. Processing...'
        })
        
    except Exception as e:
        app.logger.error(f"Error in upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure the upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True, port=5000)