import cv2
import numpy as np
import joblib
from shape_utils import process_frame

model = joblib.load("models/shape_classifier.pkl")
scaler = joblib.load("models/scaler.pkl")
le = joblib.load("models/label_encoder.pkl")

canvas = np.full((400, 900, 3), 255, np.uint8)
cv2.circle(canvas, (120, 200), 80, (0, 0, 0), -1)
cv2.rectangle(canvas, (280, 130), (430, 270), (0, 0, 0), -1)
pts = np.array([[560, 120], [620, 270], [500, 270]], np.int32)
cv2.fillPoly(canvas, [pts], (0, 0, 0))
cv2.rectangle(canvas, (700, 150), (860, 250), (0, 0, 0), -1)  # rectangle (not square)

for name, use_ml in [("rule_based", False), ("ml", True)]:
    out, detections, _ = process_frame(canvas, model, scaler, le, use_ml=use_ml)
    print(f"\n[{name}] detections:")
    for d in detections:
        print("  ", d["label"], d["bbox"])
    cv2.imwrite(f"outputs/quick_test_{name}.png", out)

print("\nSaved annotated results to outputs/quick_test_rule_based.png and outputs/quick_test_ml.png")
