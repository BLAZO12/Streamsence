import cv2
import dlib
import csv
import datetime
import os
import numpy as np
from attention_utils import eye_aspect_ratio, get_head_pose

# Setup
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)

# EAR thresholds
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0

# Camera calibration (approx for laptop webcams)
size = (640, 480)
focal_length = size[1]
center = (size[1]/2, size[0]/2)
cam_matrix = np.array([[focal_length, 0, center[0]],
                       [0, focal_length, center[1]],
                       [0, 0, 1]], dtype="double")
dist_coeffs = np.zeros((4,1))  # assume no lens distortion

# CSV logging
log_file = "logs/attention_log.csv"
if not os.path.exists("logs"):
    os.makedirs("logs")
if not os.path.isfile(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Student_ID", "Status"])

def log_attention(student_id, status):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, student_id, status])

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for i, face in enumerate(faces):
        shape = predictor(gray, face)

        # Eye coordinates
        left_eye = np.array([(shape.part(36).x, shape.part(36).y),
                             (shape.part(37).x, shape.part(37).y),
                             (shape.part(38).x, shape.part(38).y),
                             (shape.part(39).x, shape.part(39).y),
                             (shape.part(40).x, shape.part(40).y),
                             (shape.part(41).x, shape.part(41).y)], np.int32)
        right_eye = np.array([(shape.part(42).x, shape.part(42).y),
                              (shape.part(43).x, shape.part(43).y),
                              (shape.part(44).x, shape.part(44).y),
                              (shape.part(45).x, shape.part(45).y),
                              (shape.part(46).x, shape.part(46).y),
                              (shape.part(47).x, shape.part(47).y)], np.int32)

        leftEAR = eye_aspect_ratio(left_eye)
        rightEAR = eye_aspect_ratio(right_eye)
        ear = (leftEAR + rightEAR) / 2.0

        attentive = True

        # Eye closure detection
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                attentive = False
        else:
            COUNTER = 0

        # Head pose estimation
        rot_vec, trans_vec = get_head_pose(shape, cam_matrix, dist_coeffs)
        yaw = rot_vec[1][0] * 57.3  # rad → deg
        if abs(yaw) > 30:  # turned away
            attentive = False

        # Draw results
        color = (0, 255, 0) if attentive else (0, 0, 255)
        status = "Attentive" if attentive else "Inattentive"
        student_id = f"Student_{i+1}"
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), color, 2)
        cv2.putText(frame, f"{student_id}: {status}", (face.left(), face.top()-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Log
        log_attention(student_id, status)

    cv2.imshow("Classroom Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
