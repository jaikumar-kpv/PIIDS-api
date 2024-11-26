from flask import Flask, request, jsonify, send_file
import os
import cv2
import pytesseract
import re
from werkzeug.utils import secure_filename
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
app.config['OUTPUT_FOLDER'] = "outputs"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "API is running"})

@app.route('/detect-pii', methods=['POST'])
def detect_pii():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)

    # Process image
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"processed_{filename}")
    image = cv2.imread(input_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Define patterns and detect
    patterns = {
        "Aadhaar": r"\b\d{12}\b",
        "Credit Card": r"(?:\d{4}[-\s]?){3}\d{4}",
        "Phone": r"\b\d{10}\b",
        "License Plate": r"[A-Z0-9]{2,}"
    }
    boxes = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
    for i, text in enumerate(boxes['text']):
        if any(re.search(pattern, text) for pattern in patterns.values()):
            x, y, w, h = boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i]
            roi = image[y:y+h, x:x+w]
            blurred = cv2.GaussianBlur(roi, (65, 45), 0)
            image[y:y+h, x:x+w] = blurred

    cv2.imwrite(output_path, image)

    # Return the processed image
    return send_file(output_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
