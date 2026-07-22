from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, send_file
import cv2
import os
import time
import json
import logging
import threading
import pandas as pd

from config import (
    UPLOAD_FOLDER, OUTPUT_FOLDER, WILDLIFE_CLASSES,
    YOLO_CONFIDENCE_THRESHOLD, YOLO_IOU_THRESHOLD,
    SECRET_KEY, HOST, PORT, DEBUG
)
from core.detector import YOLOv11Detector
from core.tracker import DeepSORTTracker
from core.analytics import WildlifeAnalytics
from core.evaluator import AccuracyEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WildlifeApp")

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Global core instances
detector = YOLOv11Detector(
    model_path="yolo11n.pt",
    conf_thresh=YOLO_CONFIDENCE_THRESHOLD,
    iou_thresh=YOLO_IOU_THRESHOLD,
    target_classes=WILDLIFE_CLASSES
)
tracker = DeepSORTTracker(max_cosine_distance=0.3, max_age=60, n_init=3)
analytics = WildlifeAnalytics()
evaluator = AccuracyEvaluator()

# Lock for multi-threaded stream safety
stream_lock = threading.Lock()


def generate_live_stream_frames():
    """
    Generator for real-time video stream (webcam or synthetic test stream).
    Performs frame-by-frame YOLOv11 detection -> DeepSORT tracking -> Analytics annotation.
    """
    # Open camera capture (falls back to synthetic frames if webcam unavailable)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.info("Default webcam not detected. Using synthetic stream generator.")
        cap = None

    frame_count = 0
    t0 = time.time()

    while True:
        with stream_lock:
            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break
            else:
                # Generate synthetic wildlife background video frame (640x480)
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                # Create a natural savanna/woodland background gradient
                for y in range(480):
                    r = int(25 + (y / 480) * 45)
                    g = int(45 + (y / 480) * 60)
                    b = int(20 + (y / 480) * 20)
                    frame[y, :] = [b, g, r]
                # Add foliage noise texture
                cv2.putText(frame, "LIVE CAMERA-TRAP FEED SIMULATION", (140, 460),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 180), 1)

            h, w, _ = frame.shape
            frame_count += 1

            # 1. Detect Wildlife with YOLOv11
            detections = detector.detect(frame)

            # 2. Track with DeepSORT
            tracks = tracker.update(detections, frame)

            # 3. Spatial Analytics & Intrusion Check
            active_stats = analytics.process_tracks(tracks, w, h)

            # 4. Annotate Frame
            annotated_frame = analytics.annotate_frame(frame, tracks, active_stats)

            # Encode as JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)  # Maintain ~30 FPS


@app.route('/')
def index():
    """
    Main Executive Dashboard View.
    """
    stats = {
        'total_unique': len(analytics.tracked_entities),
        'species_counts': analytics.class_counts,
        'intrusions_count': len(analytics.intrusion_events),
        'detector_status': "YOLOv11 Ready",
        'tracker_status': "DeepSORT Active"
    }
    metrics = evaluator.get_benchmark_metrics()
    return render_template('index.html', stats=stats, metrics=metrics)


@app.route('/live')
def live():
    """
    Real-Time Wildlife Monitoring Stream & Virtual Fence Setup View.
    """
    return render_template('live.html')


@app.route('/video_feed')
def video_feed():
    """
    HTTP MJPEG streaming endpoint for live video feed.
    """
    return Response(generate_live_stream_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_analysis', methods=['GET', 'POST'])
def video_analysis():
    """
    Offline Video Analysis: Upload MP4/AVI files, run frame-by-frame YOLOv11 + DeepSORT.
    """
    if request.method == 'POST':
        if 'video_file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['video_file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        filename = f"input_{int(time.time())}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process video in background or synchronous step
        out_filename = f"processed_{filename}"
        out_filepath = os.path.join(app.config['OUTPUT_FOLDER'], out_filename)

        # Run Video Processing Pipeline
        summary = process_video_file(filepath, out_filepath)

        return render_template(
            'video_analysis.html',
            processed_video=url_for('static', filename=f'outputs/{out_filename}'),
            summary=summary
        )

    return render_template('video_analysis.html')


@app.route('/upload_image', methods=['POST'])
def upload_image():
    """
    Single Camera-Trap Photo Analysis Endpoint.
    """
    if 'image_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = f"photo_{int(time.time())}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Read image
    frame = cv2.imread(filepath)
    if frame is None:
        return jsonify({'error': 'Invalid image format'}), 400

    h, w, _ = frame.shape

    # Process single image with YOLOv11 + DeepSORT TTA Boost
    detections = detector.detect(frame, augment=True)
    tracks = tracker.update(detections, frame)
    active_stats = analytics.process_tracks(tracks, w, h)
    annotated = analytics.annotate_frame(frame, tracks, active_stats)

    out_filename = f"annotated_{filename}"
    out_filepath = os.path.join(app.config['OUTPUT_FOLDER'], out_filename)
    cv2.imwrite(out_filepath, annotated)

    return jsonify({
        'status': 'success',
        'result_image': url_for('static', filename=f'outputs/{out_filename}'),
        'total_detected': len(detections),
        'species_list': [d['class_name'] for d in detections],
        'detections': detections
    })


@app.route('/analytics')
def analytics_page():
    """
    Detailed Analytics, Trajectory Logs, and Intrusion History.
    """
    df = analytics.generate_report_df()
    records = df.to_dict(orient='records') if not df.empty else []
    intrusions = analytics.intrusion_events
    species_counts = analytics.class_counts

    return render_template(
        'analytics.html',
        records=records,
        intrusions=intrusions,
        species_counts=species_counts
    )


@app.route('/model_info')
def model_info():
    """
    Deep Learning Architecture Analysis Page: YOLOv11 vs DeepSORT, Accuracy Benchmarks, Equations & Loss functions.
    """
    metrics = evaluator.get_benchmark_metrics()
    return render_template('model_info.html', metrics=metrics)


@app.route('/api/stats')
def api_stats():
    """
    API returning real-time status for AJAX polling in dashboard.
    """
    return jsonify({
        'total_unique': len(analytics.tracked_entities),
        'species_counts': analytics.class_counts,
        'intrusions_count': len(analytics.intrusion_events),
        'recent_alerts': analytics.intrusion_events[-5:],
        'timestamp': time.time()
    })


@app.route('/download_report/<fmt>')
def download_report(fmt):
    """
    Download census report in CSV or JSON format.
    """
    df = analytics.generate_report_df()

    if fmt == 'csv':
        csv_path = os.path.join(app.config['OUTPUT_FOLDER'], 'wildlife_census_report.csv')
        df.to_csv(csv_path, index=False)
        return send_file(csv_path, as_attachment=True, download_name='wildlife_census_report.csv')
    elif fmt == 'json':
        json_path = os.path.join(app.config['OUTPUT_FOLDER'], 'wildlife_census_report.json')
        df.to_json(json_path, orient='records', indent=4)
        return send_file(json_path, as_attachment=True, download_name='wildlife_census_report.json')
    else:
        return "Invalid format specified", 400


def process_video_file(input_path, output_path):
    """
    Helper function to process offline video file with frame-by-frame tracker.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return {'error': 'Could not open input video file'}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    local_tracker = DeepSORTTracker()
    local_analytics = WildlifeAnalytics()

    frame_count = 0
    start_t = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Detect
        detections = detector.detect(frame)
        # Track
        tracks = local_tracker.update(detections, frame)
        # Analytics
        stats = local_analytics.process_tracks(tracks, width, height)
        # Annotate
        annotated = local_analytics.annotate_frame(frame, tracks, stats)

        out.write(annotated)

    cap.release()
    out.release()
    total_time = round(time.time() - start_t, 2)

    return {
        'total_frames': frame_count,
        'processing_time_sec': total_time,
        'average_fps': round(frame_count / max(0.1, total_time), 1),
        'unique_species_tracked': len(local_analytics.tracked_entities),
        'species_breakdown': local_analytics.class_counts,
        'danger_zone_intrusions': len(local_analytics.intrusion_events)
    }


if __name__ == '__main__':
    print("=" * 75)
    print("   STARTING INTELLIGENT WILDLIFE MONITORING SYSTEM (YOLOv11 + DeepSORT)   ")
    print(f"   Listening on http://{HOST}:{PORT}   ")
    print("=" * 75)
    app.run(host=HOST, port=PORT, debug=DEBUG)
