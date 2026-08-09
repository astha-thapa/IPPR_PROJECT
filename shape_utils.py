import cv2
import numpy as np
#preprocessing
def preprocess(image, blur_ksize=5, use_adaptive=False):
   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Remove small noise
    blurred = cv2.GaussianBlur(
        gray,(blur_ksize, blur_ksize),0)

    #gray threshold
    if use_adaptive:
        gray_thresh = cv2.adaptiveThreshold( blurred, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,4)
    else:
        _, gray_thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Saturation channel helps detect colored shapes
    saturation = hsv[:, :, 1]
    _, color_mask = cv2.threshold(saturation, 40,255,cv2.THRESH_BINARY)

    # Combine grayscale and color information
    thresh = cv2.bitwise_or( gray_thresh,color_mask)

    #morphological operations to fill gaps and remove noise
    kernel = np.ones((3, 3), np.uint8)
    # Fill small gaps in the shape
    closed = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations=2)

    # Remove small noise
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN,kernel,iterations=1 )
    return gray, blurred, thresh, closed


#contour detection 
def get_contours(binary_image, min_area=800):
    contours, _ = cv2.findContours(binary_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            valid_contours.append(contour)
    return valid_contours

#feature extraction
def extract_features(contour):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter == 0 or area == 0:
        return None
    # Polygon approximation
    epsilon = 0.01 * perimeter
    approx = cv2.approxPolyDP(contour,epsilon,True)

    num_vertices = len(approx)
#circularity 
    #circularity close to 1.
    circularity = (4 * np.pi * area) / (perimeter * perimeter)

    #box features
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    extent = area / float(w * h)
    #solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        solidity = area / float(hull_area)
    else:
        solidity = 0

    # Compare shape area with the area of its minimum
    # enclosing circle.
    (_, _), radius = cv2.minEnclosingCircle(contour)
    enclosing_circle_area = np.pi * radius * radius

    if enclosing_circle_area > 0:
        circle_ratio = area / enclosing_circle_area
    else:
        circle_ratio = 0
#hu moments
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()
    # Log transformation makes Hu moments easier to use
    hu_log = (-np.sign(hu)* np.log10(np.abs(hu) + 1e-10)
    )
    feature_vector = np.hstack([
        num_vertices,
        circularity,
        aspect_ratio,
        extent,
        solidity,
        hu_log
    ]).astype(np.float32)

    meta = {
        "approx": approx,
        "num_vertices": num_vertices,
        "circularity": circularity,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "solidity": solidity,
        "circle_ratio": circle_ratio,
        "bbox": (x, y, w, h)
    }

    return feature_vector, meta

FEATURE_NAMES = [
    "num_vertices",
    "circularity",
    "aspect_ratio",
    "extent",
    "solidity",
    "hu1",
    "hu2",
    "hu3",
    "hu4",
    "hu5",
    "hu6",
    "hu7"
]
#rule based classification
def classify_rule_based(meta):
   
    v = meta["num_vertices"]
    circ = meta["circularity"]
    ar = meta["aspect_ratio"]
    solidity = meta["solidity"]
    circle_ratio = meta["circle_ratio"]

    #circle
    if (
        circ > 0.85
        and circle_ratio > 0.88
        and solidity > 0.90
        and 0.85 <= ar <= 1.15
    ):
        return "Circle"
    #triangle
    if v == 3:
        return "Triangle"
    #square or rectangle
    if v == 4:
        if 0.88 <= ar <= 1.12:
            return "Square"
        return "Rectangle"
    #pentagon
    if v == 5:
        return "Pentagon"
    #hexagon
    if v == 6:
        return "Hexagon"
    #star
    if v >= 8 and solidity < 0.80:
        return "Star"
    return "Unknown"

#classification using ML model
def classify_ml(feature_vector,model,scaler,label_encoder):
    x = scaler.transform([feature_vector])
    pred = model.predict(x)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0].max()
    else:
        proba = None
    label = label_encoder.inverse_transform([pred])[0]
    return label, proba

#frame processing
def process_frame(frame,model=None,scaler=None,label_encoder=None,use_ml=False):
    output = frame.copy()
    # Preprocess image
    gray, blurred, thresh, closed = preprocess(
        frame,
        use_adaptive=False
    )
    # Find contours
    contours = get_contours(closed)
    detections = []
    # Process each detected object
    for contour in contours:

        result = extract_features(contour)
        if result is None:
            continue
        features, meta = result
        x, y, w, h = meta["bbox"]
        #classification
        if (
            use_ml and model is not None and scaler is not None and label_encoder is not None ):

            label, confidence = classify_ml(features, model,scaler,label_encoder)

            if confidence is not None:
                text = f"{label} ({confidence * 100:.0f}%)"
            else:
                text = label

        else:
            # Use rule-based classification
            label = classify_rule_based(meta)
            text = label

        #drawing contour
        cv2.drawContours(output,[contour],-1,(0, 255, 0),2)
        # Draw bounding box
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            1
        )

        # Draw label
        cv2.putText(
            output,
            text,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        # Store detection information
        detections.append({
            "label": label,
            "bbox": (x, y, w, h),
            "features": features
        })

    # Store intermediate images
    intermediates = {
        "gray": gray,
        "blurred": blurred,
        "thresh": thresh,
        "closed": closed
    }

    return output, detections, intermediates