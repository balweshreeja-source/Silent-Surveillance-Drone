import cv2
from ultralytics import YOLO
import serial
import time

# 🔌 Arduino connection
arduino = serial.Serial('COM5', 9600)
time.sleep(2)

# 🧠 Load YOLO model
model = YOLO("yolov8n.pt")

# 🎥 Camera
cap = cv2.VideoCapture(0)

# 📸 Previous frame for motion detection
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # 🔍 YOLO Detection
    results = model(frame)

    person_detected = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]

            # 🎯 Only detect PERSON
            if name == "person":
                person_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, "Person", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # 🧠 Motion Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    frame_diff = cv2.absdiff(prev_gray, gray)
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]

    motion = cv2.countNonZero(thresh)

    # Update previous frame
    prev_gray = gray

    # ⚡ FINAL CONDITION: Person + Motion
    if person_detected and motion > 5000:
        arduino.write(b'1')
        print("ALERT 🚨 Moving person detected")
    else:
        arduino.write(b'0')

    # 🖥️ Show output
    cv2.imshow("AI Motion Security System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# 🧹 Cleanup
cap.release()
cv2.destroyAllWindows()
arduino.close()
