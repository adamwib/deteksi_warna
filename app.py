from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)
cap = cv2.VideoCapture(0)

lower_range_red = np.array([0, 100, 100])
upper_range_red = np.array([10, 255, 255])

lower_range_yellow = np.array([21, 100, 100])
upper_range_yellow = np.array([30, 255, 255])

lower_range_green = np.array([31, 100, 100])
upper_range_green = np.array([75, 255, 255])

lower_range_blue = np.array([100, 100, 100])
upper_range_blue = np.array([130, 255, 255])

def detect_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red detection
    mask_red = cv2.inRange(hsv, lower_range_red, upper_range_red)
    _, mask1 = cv2.threshold(mask_red, 254, 255, cv2.THRESH_BINARY)
    cnts_red, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for c in cnts_red:
        x = 600
        if cv2.contourArea(c) > x:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            text = "Red"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.putText(frame, text, (x + (w - text_size[0]) // 2, y + h + text_size[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Yellow detection
    mask_yellow = cv2.inRange(hsv, lower_range_yellow, upper_range_yellow)
    _, mask2 = cv2.threshold(mask_yellow, 254, 255, cv2.THRESH_BINARY)
    cnts_yellow, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for c in cnts_yellow:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        text = "Yellow"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(frame, text, (x + (w - text_size[0]) // 2, y + h + text_size[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Green detection
    mask_green = cv2.inRange(hsv, lower_range_green, upper_range_green)
    _, mask3 = cv2.threshold(mask_green, 254, 255, cv2.THRESH_BINARY)
    cnts_green, _ = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for c in cnts_green:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Green"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(frame, text, (x + (w - text_size[0]) // 2, y + h + text_size[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Blue detection
    mask_blue = cv2.inRange(hsv, lower_range_blue, upper_range_blue)
    _, mask4 = cv2.threshold(mask_blue, 254, 255, cv2.THRESH_BINARY)
    cnts_blue, _ = cv2.findContours(mask4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for c in cnts_blue:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        text = "Blue"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(frame, text, (x + (w - text_size[0]) // 2, y + h + text_size[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            detect_color(frame)  # Add color detection
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
