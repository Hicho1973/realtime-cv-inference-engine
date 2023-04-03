
import cv2
import numpy as np
import time
from collections import deque

# --- Configuration --- #
MODEL_PATH = "./models/yolov3.weights"
CONFIG_PATH = "./models/yolov3.cfg"
LABELS_PATH = "./models/coco.names"
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
INPUT_WIDTH = 416
INPUT_HEIGHT = 416

# --- 1. Load YOLO Model and Labels --- #
def load_yolo_model(model_path, config_path, labels_path):
    """Loads the YOLO object detection model and class labels."""
    print("Loading YOLO model...")
    net = cv2.dnn.readNet(model_path, config_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) # Can be changed to DNN_TARGET_CUDA for GPU

    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    
    # Get the names of the output layers
    output_layers = [net.getLayerNames()[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    print("YOLO model loaded successfully.")
    return net, labels, output_layers

# --- 2. Preprocess Frame --- #
def preprocess_frame(frame, input_width, input_height):
    """Preprocesses a single frame for YOLO input."""
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_width, input_height), swapRB=True, crop=False)
    return blob

# --- 3. Postprocess Detections --- #
def postprocess_detections(frame, outs, labels, conf_threshold, nms_threshold):
    """Applies non-maximum suppression and draws bounding boxes on the frame."""
    frame_height, frame_width = frame.shape[:2]
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if len(indices) > 0:
        for i in indices.flatten():
            left, top, width, height = boxes[i]
            label = str(labels[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = (0, 255, 0) # Green
            cv2.rectangle(frame, (left, top), (left + width, top + height), color, 2)
            cv2.putText(frame, f"{label} {confidence}", (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

# --- 4. Real-time Inference Pipeline --- #
def run_inference_pipeline(video_source=0):
    """Runs the real-time object detection inference pipeline."""
    print("Initializing real-time inference pipeline...")
    net, labels, output_layers = load_yolo_model(MODEL_PATH, CONFIG_PATH, LABELS_PATH)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video stream or file.")
        return

    # Frame rate calculation
    frame_times = deque()

    while cv2.waitKey(1) < 0:
        has_frame, frame = cap.read()
        if not has_frame:
            print("End of video stream.")
            break

        start_time = time.time()

        # Preprocess
        blob = preprocess_frame(frame, INPUT_WIDTH, INPUT_HEIGHT)
        net.setInput(blob)

        # Run inference
        outs = net.forward(output_layers)

        # Postprocess
        processed_frame = postprocess_detections(frame, outs, labels, CONF_THRESHOLD, NMS_THRESHOLD)

        # Calculate FPS
        frame_times.append(time.time() - start_time)
        if len(frame_times) > 30: # Keep last 30 frame times
            frame_times.popleft()
        fps = len(frame_times) / sum(frame_times) if sum(frame_times) > 0 else 0
        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display result
        cv2.imshow("Real-time Object Detection", processed_frame)

    cap.release()
    cv2.destroyAllWindows()
    print("Inference pipeline stopped.")

if __name__ == "__main__":
    # Placeholder for model files. In a real scenario, these would be downloaded or trained.
    # For demonstration, ensure these paths exist or mock them.
    import os
    os.makedirs("./models", exist_ok=True)
    # You would typically download yolov3.weights, yolov3.cfg, and coco.names here.
    # For this simulation, we assume they exist or the user understands they need to be provided.
    print("Note: YOLO model files (weights, cfg, names) are required in the ./models directory to run this script.")
    # run_inference_pipeline(0) # Use 0 for webcam, or path to video file
