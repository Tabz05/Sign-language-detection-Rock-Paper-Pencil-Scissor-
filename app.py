import pickle
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, request, jsonify
import base64
import io
from PIL import Image

app = Flask(__name__)

# Load model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']
labels_dict = {0: 'Rock', 1: 'Paper', 2: 'Pencil', 3: 'Scissor'}  # Adjust to your model

# MediaPipe hands init
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    img_data = data['image']  # base64 image data: "data:image/jpeg;base64,/9j/..."

    # Strip header
    header, encoded = img_data.split(",", 1)

    # Decode base64 image
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    H, W, _ = frame.shape

    # Process with MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        return jsonify({'prediction': 'No hand detected'})

    hand_landmarks = results.multi_hand_landmarks[0]

    x_, y_ = [], []
    data_aux = []

    for lm in hand_landmarks.landmark:
        x_.append(lm.x)
        y_.append(lm.y)

    for lm in hand_landmarks.landmark:
        data_aux.append(lm.x - min(x_))
        data_aux.append(lm.y - min(y_))

    prediction = model.predict([np.asarray(data_aux)])
    predicted_character = labels_dict[int(prediction[0])]

    return jsonify({'prediction': predicted_character})

if __name__ == '__main__':
    app.run(debug=True)
