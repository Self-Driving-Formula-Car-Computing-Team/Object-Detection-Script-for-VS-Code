import cv2
from ultralytics import YOLO

# --- Configuration ---
# You can change 'yolov8n.pt' to 'yolov8s.pt' or 'yolov8m.pt' for better accuracy
# at the cost of speed. 'n' (nano) is best for real-time performance.
MODEL_PATH = "yolov8n.pt"
WEBCAM_INDEX = 0  # 0 is usually the built-in webcam. Change to 1, 2, etc., if you have multiple cameras.
CONFIDENCE_THRESHOLD = 0.40 # Minimum confidence score for a detection to be displayed


def run_webcam_detection():
    """
    Initializes the webcam, loads the YOLOv8 model, and runs the detection loop.
    """
    print("--- YOLOv8 Local Webcam Detector ---")
    
    # 1. Load the YOLOv8 Model
    try:
        model = YOLO(MODEL_PATH)
        print(f" Model loaded successfully: {MODEL_PATH}")
    except Exception as e:
        print(f" Error loading model: {e}")
        print("Please ensure you have a stable internet connection for the first run to download the model.")
        return

    # 2. Initialize Video Capture (Webcam)
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    
    if not cap.isOpened():
        print(f" Error: Could not open webcam with index {WEBCAM_INDEX}.")
        print("Try changing the WEBCAM_INDEX (e.g., to 1 or 2).")
        return
        
    print(f"✅ Webcam opened successfully (Index: {WEBCAM_INDEX}).")
    print("Press 'q' to exit the application window.")

    # 3. Real-Time Detection Loop
    while cap.isOpened():
        # Read a frame from the webcam
        success, frame = cap.read()

        if success:
            # 4. Run YOLOv8 prediction on the frame
            results = model.predict(
                source=frame, 
                conf=CONFIDENCE_THRESHOLD, 
                verbose=False
            )
            
            # 5. Get the annotated frame (with boxes and labels)
            annotated_frame = results[0].plot()

            # 6. Display the annotated frame
            # 'YOLOv8 Detection' is the name of the pop-up window
            cv2.imshow("YOLOv8 Detection", annotated_frame)

            # 7. Check for exit condition (press 'q' or 'Q')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # If frame reading fails
            print("Warning: Failed to read frame from camera.")
            break

    # 8. Cleanup Resources
    cap.release()
    cv2.destroyAllWindows()
    print("\n--- Detection stopped and resources released. ---")


if __name__ == "__main__":
    run_webcam_detection()