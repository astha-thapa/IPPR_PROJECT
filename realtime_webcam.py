import cv2
import time
import joblib
from shape_utils import process_frame

MODEL_PATH = "models/shape_classifier.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"


def main():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        le = joblib.load(ENCODER_PATH)
    except FileNotFoundError:
        print("Trained model not found - run train_classifier.py first. "
              "Falling back to rule-based classifier only.")
        model = scaler = le = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check camera permissions/index.")
        return

    use_ml = False
    prev_time = time.time()
    frame_count = 0
    print("Press 'q' to quit, 's' to save a frame, 'm' to toggle ML classifier.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        output, detections, _ = process_frame(
            frame, model, scaler, le, use_ml=(use_ml and model is not None)
        )

        # FPS counter (matters for the "Speed test" in Week 6 Performance Testing)
        frame_count += 1
        now = time.time()
        if now - prev_time >= 1.0:
            fps = frame_count / (now - prev_time)
            frame_count = 0
            prev_time = now
        else:
            fps = None

        mode_txt = f"Mode: {'ML' if use_ml else 'Rule-based'}"
        cv2.putText(output, mode_txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 128, 255), 2)
        cv2.putText(output, f"Shapes: {len(detections)}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)

        cv2.imshow("Real-Time Shape Recognition", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"outputs/capture_{int(time.time())}.png"
            cv2.imwrite(fname, output)
            print(f"Saved {fname}")
        elif key == ord('m'):
            use_ml = not use_ml
            print(f"Switched to {'ML' if use_ml else 'Rule-based'} classifier")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
